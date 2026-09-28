use std::collections::hash_map;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, Weak};
use std::time::Duration;

use anyhow::Result;
use bumpalo::Bump;
use parking_lot::RwLock;
use quick_cache::sync::{Cache, DefaultLifecycle};
use smallvec::SmallVec;
use ton_types::{ByteOrderRead, CellImpl, UInt256};
use triomphe::ThinArc;

use crate::db::*;
use crate::utils::{spawn_metrics_loop, CellExt, FastDashMap, FastHashMap, FastHasherState};

pub struct CellStorage {
    db: Arc<Db>,
    cells_cache: Arc<FastDashMap<UInt256, Weak<StorageCell>>>,
    raw_cells_cache: Arc<RawCellsCache>,
}

impl CellStorage {
    pub fn new(db: Arc<Db>, cache_size_bytes: u64) -> Arc<Self> {
        let cells_cache = Default::default();
        let raw_cells_cache = Arc::new(RawCellsCache::new(cache_size_bytes));

        let this = Arc::new(Self {
            db,
            cells_cache,
            raw_cells_cache,
        });

        spawn_metrics_loop(&this, Duration::from_secs(5), |this| async move {
            crate::set_metrics! {
                "cells_cache_hits" => this.raw_cells_cache.0.hits(),
                "cells_cache_requests" => this.raw_cells_cache.0.misses() + this.raw_cells_cache.0.hits(),
                "cells_cache_occupied" => this.raw_cells_cache.0.len() ,
                "cells_cache_size_bytes" => this.raw_cells_cache.0.weight(),
                "cells_cache_hits_ratio" => this.raw_cells_cache.hit_ratio(),
            }
        });

        this
    }

    pub fn apply_temp_cell(&self, root: &UInt256) -> Result<()> {
        const MAX_NEW_CELLS_BATCH_SIZE: usize = 10000;

        struct TempCell {
            old_rc: i64,
            additions: u32,
        }

        struct CellHashesIter<'a> {
            data: rocksdb::DBPinnableSlice<'a>,
            offset: usize,
            remaining_refs: u8,
        }

        impl<'a> Iterator for CellHashesIter<'a> {
            type Item = [u8; 32];

            fn next(&mut self) -> Option<Self::Item> {
                if self.remaining_refs == 0 {
                    return None;
                }

                // NOTE: Unwrap is safe here because we have already checked
                // that data can contain all references.
                let item = self.data[self.offset..self.offset + 32].try_into().unwrap();

                self.remaining_refs -= 1;
                self.offset += 32;

                Some(item)
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                let r = self.remaining_refs as usize;
                (r, Some(r))
            }
        }

        enum InsertedCell<'a> {
            New(CellHashesIter<'a>),
            Existing,
        }

        struct Context<'a> {
            cells_cf: BoundedCfHandle<'a>,
            db: &'a Db,
            buffer: Vec<u8>,
            transaction: FastHashMap<[u8; 32], TempCell>,
            new_cells_batch: rocksdb::WriteBatch,
            new_cell_count: usize,
            raw_cache: &'a RawCellsCache,
        }

        impl<'a> Context<'a> {
            fn new(db: &'a Db, raw_cache: &'a RawCellsCache) -> Self {
                Self {
                    cells_cf: db.cells.cf(),
                    db,
                    buffer: Vec::with_capacity(512),
                    transaction: Default::default(),
                    new_cells_batch: rocksdb::WriteBatch::default(),
                    new_cell_count: 0,
                    raw_cache,
                }
            }

            fn load_temp(&self, key: &[u8; 32]) -> Result<CellHashesIter<'a>, CellStorageError> {
                let data = match self.db.temp_cells.get(key) {
                    Ok(Some(data)) => data,
                    Ok(None) => return Err(CellStorageError::CellNotFound),
                    Err(e) => return Err(CellStorageError::Internal(e)),
                };

                let (offset, remaining_refs) = {
                    let data = &mut data.as_ref();

                    let len_before = data.len();
                    let refs = match ton_types::CellData::deserialize(data) {
                        Ok(data) => data.references_count(),
                        Err(_) => return Err(CellStorageError::InvalidCell),
                    };
                    let offset = len_before - data.len();

                    if data.len() < refs * 32 {
                        return Err(CellStorageError::InvalidCell);
                    }

                    (offset, refs as u8)
                };

                Ok(CellHashesIter {
                    data,
                    offset,
                    remaining_refs,
                })
            }

            fn insert_cell(
                &mut self,
                key: &[u8; 32],
            ) -> Result<InsertedCell<'a>, CellStorageError> {
                Ok(match self.transaction.entry(*key) {
                    hash_map::Entry::Occupied(mut entry) => {
                        entry.get_mut().additions += 1; // 1 new reference
                        InsertedCell::Existing
                    }
                    hash_map::Entry::Vacant(entry) => {
                        if let Some(value) =
                            self.db.cells.get(key).map_err(CellStorageError::Internal)?
                        {
                            let (rc, value) = refcount::decode_value_with_rc(value.as_ref());
                            debug_assert!(rc > 0 && value.is_some() || rc == 0 && value.is_none());
                            if value.is_some() {
                                entry.insert(TempCell {
                                    old_rc: rc,
                                    additions: 1, // 1 new reference
                                });
                                return Ok(InsertedCell::Existing);
                            }
                        }

                        // 0 new references (the first one is included in the merge below)
                        entry.insert(TempCell {
                            old_rc: 0,
                            additions: 1,
                        });
                        let iter = self.load_temp(key)?;

                        self.buffer.clear();
                        refcount::add_positive_refount(
                            1,
                            Some(iter.data.as_ref()),
                            &mut self.buffer,
                        );

                        self.new_cells_batch
                            .put_cf(&self.cells_cf, key, self.buffer.as_slice());

                        self.new_cell_count += 1;
                        if self.new_cell_count >= MAX_NEW_CELLS_BATCH_SIZE {
                            self.flush_new_cells()?;
                        }

                        InsertedCell::New(iter)
                    }
                })
            }

            fn flush_new_cells(&mut self) -> Result<(), rocksdb::Error> {
                if self.new_cell_count > 0 {
                    self.db
                        .raw()
                        .write(std::mem::take(&mut self.new_cells_batch))?;
                    self.new_cell_count = 0;
                }
                Ok(())
            }

            fn flush_existing_cells(&mut self) -> Result<(), rocksdb::Error> {
                let mut batch = rocksdb::WriteBatch::default();

                for (key, item) in &self.transaction {
                    let mut refs_diff = item.additions;
                    if item.old_rc == 0 {
                        // 1 reference was added with the data while traversing the tree.
                        refs_diff -= 1;
                    }

                    if refs_diff > 0 {
                        self.buffer.clear();
                        refcount::add_positive_refount(refs_diff, None, &mut self.buffer);
                        batch.merge_cf(&self.cells_cf, key, self.buffer.as_slice());
                    }

                    let new_rc = item.old_rc + item.additions as i64;
                    self.raw_cache.on_insert_cell(&key, new_rc, None);
                }

                self.db.raw().write(batch)
            }
        }

        let mut ctx = Context::new(&self.db, &self.raw_cells_cache);

        let mut stack = Vec::with_capacity(16);
        if let InsertedCell::New(iter) = ctx.insert_cell(root.as_slice())? {
            stack.push(iter);
        }

        'outer: loop {
            let Some(iter) = stack.last_mut() else {
                break;
            };

            for ref child in iter {
                if let InsertedCell::New(iter) = ctx.insert_cell(child)? {
                    stack.push(iter);
                    continue 'outer;
                }
            }

            stack.pop();
        }

        // Clear big chunks of data before finalization
        drop(stack);

        ctx.flush_new_cells()?;
        ctx.flush_existing_cells()?;

        Ok(())
    }

    pub fn store_cell(
        &self,
        batch: &mut rocksdb::WriteBatch,
        root: ton_types::Cell,
    ) -> Result<usize, CellStorageError> {
        struct AddedCell<'a> {
            old_rc: i64,
            additions: u32,
            data: Option<&'a [u8]>,
        }

        struct Context<'a> {
            db: &'a Db,
            raw_cells_cache: &'a RawCellsCache,
            alloc: &'a Bump,
            transaction: FastHashMap<[u8; 32], AddedCell<'a>>,
            buffer: Vec<u8>,
        }

        impl Context<'_> {
            fn insert_cell(
                &mut self,
                key: &[u8; 32],
                cell: &ton_types::Cell,
                depth: usize,
            ) -> Result<bool, CellStorageError> {
                Ok(match self.transaction.entry(*key) {
                    hash_map::Entry::Occupied(mut value) => {
                        value.get_mut().additions += 1;
                        false
                    }
                    hash_map::Entry::Vacant(entry) => {
                        let old_rc = self
                            .raw_cells_cache
                            .get_rc_for_insert(self.db, key, depth)?;

                        let is_new = old_rc == 0;
                        let data = if is_new {
                            self.buffer.clear();
                            if StorageCell::serialize_to(&**cell, &mut self.buffer).is_err() {
                                return Err(CellStorageError::InvalidCell);
                            }
                            Some(self.alloc.alloc_slice_copy(self.buffer.as_slice()) as &[u8])
                        } else {
                            None
                        };
                        entry.insert(AddedCell {
                            old_rc,
                            additions: 1,
                            data,
                        });
                        is_new
                    }
                })
            }

            fn finalize(mut self, batch: &mut rocksdb::WriteBatch) -> usize {
                let total = self.transaction.len();
                let cells_cf = &self.db.cells.cf();
                for (key, item) in self.transaction {
                    self.buffer.clear();
                    refcount::add_positive_refount(item.additions, item.data, &mut self.buffer);
                    batch.merge_cf(cells_cf, key.as_slice(), &self.buffer);

                    let new_rc = item.old_rc + item.additions as i64;
                    self.raw_cells_cache.on_insert_cell(&key, new_rc, item.data);
                }

                total
            }
        }

        let alloc = bumpalo::Bump::new();

        // Prepare context and handles
        let mut ctx = Context {
            db: &self.db,
            raw_cells_cache: &self.raw_cells_cache,
            alloc: &alloc,
            transaction: FastHashMap::with_capacity_and_hasher(128, Default::default()),
            buffer: Vec::with_capacity(512),
        };

        'visit: {
            // Check root cell
            if !ctx.insert_cell(root.repr_hash().as_array(), &root, 0)? {
                break 'visit;
            }
            let mut stack = Vec::with_capacity(16);
            stack.push(root.into_references());

            // Check other cells
            'outer: loop {
                let depth = stack.len();
                let Some(iter) = stack.last_mut() else {
                    break;
                };

                for child in &mut *iter {
                    if ctx.insert_cell(child.repr_hash().as_array(), &child, depth)? {
                        stack.push(child.into_references());
                        continue 'outer;
                    }
                }

                stack.pop();
            }
        }

        // Write transaction to the `WriteBatch`
        Ok(ctx.finalize(batch))
    }

    pub fn load_cell(
        self: &Arc<Self>,
        hash: UInt256,
    ) -> Result<Arc<StorageCell>, CellStorageError> {
        if let Some(cell) = self.cells_cache.get(&hash) {
            if let Some(cell) = cell.upgrade() {
                return Ok(cell);
            }
        }

        let cell = match self.raw_cells_cache.get_raw(&self.db, hash.as_array()) {
            Ok(Some(value)) => match StorageCell::deserialize(self.clone(), &value.slice) {
                Ok(cell) => Arc::new(cell),
                Err(_) => return Err(CellStorageError::InvalidCell),
            },
            Ok(None) => return Err(CellStorageError::CellNotFound),
            Err(e) => return Err(CellStorageError::Internal(e)),
        };

        self.cells_cache.insert(hash, Arc::downgrade(&cell));

        Ok(cell)
    }

    pub fn remove_cell(
        &self,
        batch: &mut rocksdb::WriteBatch,
        alloc: &Bump,
        hash: UInt256,
    ) -> Result<usize, CellStorageError> {
        #[derive(Clone, Copy)]
        struct RemovedCell<'a> {
            old_rc: i64,
            removes: u32,
            refs: &'a [[u8; 32]],
        }

        impl<'a> RemovedCell<'a> {
            fn remove(&mut self) -> Result<Option<&'a [[u8; 32]]>, CellStorageError> {
                self.removes += 1;
                if self.removes as i64 <= self.old_rc {
                    Ok(self.next_refs())
                } else {
                    Err(CellStorageError::CounterMismatch)
                }
            }

            fn next_refs(&self) -> Option<&'a [[u8; 32]]> {
                if self.old_rc > self.removes as i64 {
                    None
                } else {
                    Some(self.refs)
                }
            }
        }

        let cells = &self.db.cells;
        let cells_cf = &cells.cf();

        let mut transaction: FastHashMap<&[u8; 32], RemovedCell<'_>> =
            FastHashMap::with_capacity_and_hasher(128, Default::default());
        let mut buffer = Vec::with_capacity(4);

        let mut stack = Vec::with_capacity(16);
        stack.push(std::slice::from_ref(hash.as_array()).iter());

        // While some cells left
        'outer: loop {
            let Some(iter) = stack.last_mut() else {
                break;
            };

            for cell_id in iter.by_ref() {
                // Process the current cell.
                let refs = match transaction.entry(cell_id) {
                    hash_map::Entry::Occupied(mut v) => v.get_mut().remove()?,
                    hash_map::Entry::Vacant(v) => {
                        let old_rc = self.raw_cells_cache.get_rc_for_delete(
                            &self.db,
                            cell_id,
                            &mut buffer,
                        )?;
                        debug_assert!(old_rc > 0);

                        v.insert(RemovedCell {
                            old_rc,
                            removes: 1,
                            refs: alloc.alloc_slice_copy(buffer.as_slice()),
                        })
                        .next_refs()
                    }
                };

                if let Some(refs) = refs {
                    // And proceed to its refs if any.
                    stack.push(refs.iter());
                    continue 'outer;
                }
            }

            // Drop the current cell when all of its children were processed.
            stack.pop();
        }

        // Clear big chunks of data before finalization
        drop(stack);

        // Write transaction to the `WriteBatch`
        let total = transaction.len();

        for (key, item) in transaction {
            batch.merge_cf(
                cells_cf,
                key.as_slice(),
                refcount::encode_negative_refcount(item.removes),
            );

            let new_rc = item.old_rc - item.removes as i64;
            self.raw_cells_cache.on_remove_cell(key, new_rc);
        }

        Ok(total)
    }

    pub fn drop_cell(&self, hash: &UInt256) {
        self.cells_cache.remove(hash);
    }
}

#[derive(thiserror::Error, Debug)]
pub enum CellStorageError {
    #[error("Cell not found in cell db")]
    CellNotFound,
    #[error("Invalid cell")]
    InvalidCell,
    #[error("Cell counter mismatch")]
    CounterMismatch,
    #[error("Internal rocksdb error")]
    Internal(#[from] rocksdb::Error),
}

pub struct StorageCell {
    _c: countme::Count<Self>,
    cell_storage: Arc<CellStorage>,
    cell_data: ton_types::CellData,
    references: RwLock<SmallVec<[StorageCellReference; 4]>>,
    tree_bits_count: u64,
    tree_cell_count: u64,
}

impl StorageCell {
    pub fn repr_hash(&self) -> UInt256 {
        self.hash(ton_types::MAX_LEVEL)
    }

    pub fn deserialize(boc_db: Arc<CellStorage>, mut data: &[u8]) -> Result<Self> {
        // deserialize cell
        let cell_data = ton_types::CellData::deserialize(&mut data)?;
        let references_count = cell_data.references_count();
        let mut references = SmallVec::with_capacity(references_count);

        for _ in 0..references_count {
            let hash = UInt256::from(data.read_u256()?);
            references.push(StorageCellReference::Unloaded(hash));
        }

        let (tree_bits_count, tree_cell_count) = match data.read_le_u64() {
            Ok(tree_bits_count) => match data.read_le_u64() {
                Ok(tree_cell_count) => (tree_bits_count, tree_cell_count),
                Err(_) => (0, 0),
            },
            Err(_) => (0, 0),
        };

        Ok(Self {
            _c: Default::default(),
            cell_storage: boc_db,
            cell_data,
            references: RwLock::new(references),
            tree_bits_count,
            tree_cell_count,
        })
    }

    pub fn deserialize_references(mut data: &[u8], target: &mut Vec<[u8; 32]>) -> bool {
        let reader = &mut data;

        let references_count = match ton_types::CellData::deserialize(reader) {
            Ok(data) => data.references_count(),
            Err(_) => return false,
        };

        for _ in 0..references_count {
            let Ok(hash) = reader.read_u256() else {
                return false;
            };
            target.push(hash);
        }

        true
    }

    pub fn serialize_to(cell: &dyn CellImpl, target: &mut Vec<u8>) -> Result<()> {
        target.clear();

        // serialize cell
        let references_count = cell.references_count();
        cell.cell_data().serialize(target)?;

        for i in 0..references_count {
            target.extend_from_slice(cell.reference(i)?.repr_hash().as_slice());
        }

        target.extend_from_slice(&cell.tree_bits_count().to_le_bytes());
        target.extend_from_slice(&cell.tree_cell_count().to_le_bytes());

        Ok(())
    }

    pub fn reference(&self, index: usize) -> Result<Arc<StorageCell>> {
        let hash = match &self.references.read().get(index) {
            Some(StorageCellReference::Unloaded(hash)) => *hash,
            Some(StorageCellReference::Loaded(cell)) => return Ok(cell.clone()),
            None => return Err(StorageCellError::AccessingInvalidReference.into()),
        };

        let storage_cell = self.cell_storage.load_cell(hash)?;
        self.references.write()[index] = StorageCellReference::Loaded(storage_cell.clone());

        Ok(storage_cell)
    }
}

impl CellImpl for StorageCell {
    fn data(&self) -> &[u8] {
        self.cell_data.data()
    }

    fn raw_data(&self) -> ton_types::Result<&[u8]> {
        Ok(self.cell_data.raw_data())
    }

    fn cell_data(&self) -> &ton_types::CellData {
        &self.cell_data
    }

    fn bit_length(&self) -> usize {
        self.cell_data.bit_length()
    }

    fn references_count(&self) -> usize {
        self.references.read().len()
    }

    fn reference(&self, index: usize) -> Result<ton_types::Cell> {
        Ok(ton_types::Cell::with_cell_impl_arc(self.reference(index)?))
    }

    fn cell_type(&self) -> ton_types::CellType {
        self.cell_data.cell_type()
    }

    fn level_mask(&self) -> ton_types::LevelMask {
        self.cell_data.level_mask()
    }

    fn hash(&self, index: usize) -> UInt256 {
        self.cell_data.hash(index)
    }

    fn depth(&self, index: usize) -> u16 {
        self.cell_data.depth(index)
    }

    fn store_hashes(&self) -> bool {
        self.cell_data.store_hashes()
    }

    fn tree_bits_count(&self) -> u64 {
        self.tree_bits_count
    }

    fn tree_cell_count(&self) -> u64 {
        self.tree_cell_count
    }
}

impl Drop for StorageCell {
    fn drop(&mut self) {
        self.cell_storage.drop_cell(&self.repr_hash())
    }
}

#[derive(Clone)]
pub enum StorageCellReference {
    Loaded(Arc<StorageCell>),
    Unloaded(UInt256),
}

#[derive(thiserror::Error, Debug)]
enum StorageCellError {
    #[error("Accessing invalid cell reference")]
    AccessingInvalidReference,
}

struct RawCellsCache(Cache<[u8; 32], RawCellsCacheItem, CellSizeEstimator, FastHasherState>);

impl RawCellsCache {
    pub(crate) fn hit_ratio(&self) -> f64 {
        (if self.0.hits() > 0 {
            self.0.hits() as f64 / (self.0.hits() + self.0.misses()) as f64
        } else {
            0.0
        }) * 100.0
    }
}

type RawCellsCacheItem = ThinArc<AtomicI64, u8>;

#[derive(Clone, Copy)]
pub struct CellSizeEstimator;
impl quick_cache::Weighter<[u8; 32], RawCellsCacheItem> for CellSizeEstimator {
    fn weight(&self, key: &[u8; 32], val: &RawCellsCacheItem) -> u32 {
        const STATIC_SIZE: usize = std::mem::size_of::<RawCellsCacheItem>()
            + std::mem::size_of::<i64>()
            + std::mem::size_of::<usize>() * 2; // ArcInner refs + HeaderWithLength length

        let len = key.len() + val.slice.len() + STATIC_SIZE;
        len as u32
    }
}

impl RawCellsCache {
    const RC_NAN: i64 = i64::MAX;

    fn new(size_in_bytes: u64) -> Self {
        // Percentile 0.1%    from 96 to 127  => 1725119 count
        // Percentile 10%     from 128 to 191  => 82838849 count
        // Percentile 25%     from 128 to 191  => 82838849 count
        // Percentile 50%     from 128 to 191  => 82838849 count
        // Percentile 75%     from 128 to 191  => 82838849 count
        // Percentile 90%     from 192 to 255  => 22775080 count
        // Percentile 95%     from 192 to 255  => 22775080 count
        // Percentile 99%     from 192 to 255  => 22775080 count
        // Percentile 99.9%   from 256 to 383  => 484002 count
        // Percentile 99.99%  from 256 to 383  => 484002 count
        // Percentile 99.999% from 256 to 383  => 484002 count

        // from 64  to 95  - 15_267
        // from 96  to 127 - 1_725_119
        // from 128 to 191 - 82_838_849
        // from 192 to 255 - 22_775_080
        // from 256 to 383 - 484_002

        // we assume that 75% of cells are in range 128..191
        // so we can use use 192 as size for value in cache

        const MAX_CELL_SIZE: u64 = 192;
        const KEY_SIZE: u64 = 32;
        const SHARDS: usize = 512;

        let estimated_cell_cache_capacity = size_in_bytes / (KEY_SIZE + MAX_CELL_SIZE);
        tracing::info!(
            estimated_cell_cache_capacity,
            max_cell_cache_size = %bytesize::ByteSize(size_in_bytes),
        );

        let raw_cache = Cache::with_options(
            quick_cache::OptionsBuilder::new()
                .shards(SHARDS)
                .estimated_items_capacity(estimated_cell_cache_capacity as usize)
                .weight_capacity(size_in_bytes)
                .hot_allocation(0.8)
                .build()
                .unwrap(),
            CellSizeEstimator,
            FastHasherState::default(),
            DefaultLifecycle::default(),
        );

        Self(raw_cache)
    }

    fn get_raw(
        &self,
        db: &Db,
        key: &[u8; 32],
    ) -> Result<Option<RawCellsCacheItem>, rocksdb::Error> {
        use quick_cache::sync::GuardResult;

        match self.0.get_value_or_guard(key, None) {
            GuardResult::Value(value) => Ok(Some(value)),
            GuardResult::Guard(g) => {
                let value = db.cells.get(key.as_slice())?;

                Ok(if let Some(value) = value {
                    let (_, data) = refcount::decode_value_with_rc(value.as_ref());
                    data.map(|value| {
                        let value = RawCellsCacheItem::from_header_and_slice(
                            AtomicI64::new(Self::RC_NAN),
                            value,
                        );
                        _ = g.insert(value.clone());
                        value
                    })
                } else {
                    None
                })
            }
            GuardResult::Timeout => unreachable!(),
        }
    }

    fn get_rc_for_insert(
        &self,
        db: &Db,
        key: &[u8; 32],
        depth: usize,
    ) -> Result<i64, CellStorageError> {
        // A constant which tells since which depth we should start to use cache.
        // This method is used mostly for inserting new states, so we can assume
        // that first N levels will mostly be new.
        //
        // This value was chosen empirically.
        const NEW_CELLS_DEPTH_THRESHOLD: usize = 4;

        if depth >= NEW_CELLS_DEPTH_THRESHOLD {
            // NOTE: `get` here is used to affect a "hotness" of the value, because
            // there is a big chance that we will need it soon during state processing
            if let Some(entry) = self.0.get(key) {
                let rc = entry.header.header.load(Ordering::Acquire);
                if rc != Self::RC_NAN {
                    return Ok(rc);
                }
            }
        }

        match db.cells.get(key).map_err(CellStorageError::Internal)? {
            Some(value) => {
                let (rc, value) = refcount::decode_value_with_rc(value.as_ref());

                // TODO: lower to `debug_assert` when sure
                let has_value = value.is_some();
                assert!(has_value && rc > 0 || !has_value && rc == 0);

                Ok(rc)
            }
            None => Ok(0),
        }
    }

    fn get_rc_for_delete(
        &self,
        db: &Db,
        key: &[u8; 32],
        refs_buffer: &mut Vec<[u8; 32]>,
    ) -> Result<i64, CellStorageError> {
        refs_buffer.clear();

        // NOTE: `peek` here is used to avoid affecting a "hotness" of the value
        if let Some(value) = self.0.peek(key) {
            let rc = value.header.header.load(Ordering::Acquire);
            if rc <= 0 {
                return Err(CellStorageError::CellNotFound);
            } else if rc != i64::MAX {
                return StorageCell::deserialize_references(&value.slice, refs_buffer)
                    .then_some(rc)
                    .ok_or(CellStorageError::InvalidCell);
            }
        }

        match db.cells.get(key.as_slice()) {
            Ok(value) => {
                if let Some(value) = value {
                    if let (rc, Some(value)) = refcount::decode_value_with_rc(&value) {
                        return StorageCell::deserialize_references(value, refs_buffer)
                            .then_some(rc)
                            .ok_or(CellStorageError::InvalidCell);
                    }
                }

                Err(CellStorageError::CellNotFound)
            }
            Err(e) => Err(CellStorageError::Internal(e)),
        }
    }

    fn on_insert_cell(&self, key: &[u8; 32], rc: i64, data: Option<&[u8]>) {
        match data {
            None => {
                // NOTE: `get` here is used to affect a "hotness" of the value
                if let Some(v) = self.0.get(key) {
                    v.header.header.store(rc, Ordering::Release);
                }
            }
            Some(data) => self.0.insert(
                *key,
                RawCellsCacheItem::from_header_and_slice(AtomicI64::new(rc), data),
            ),
        }
    }

    fn on_remove_cell(&self, key: &[u8; 32], rc: i64) {
        let v = if rc <= 0 {
            debug_assert_eq!(rc, 0, "too many removed cells");

            match self.0.remove(key) {
                None => return,
                Some((_, v)) => v,
            }
        } else {
            // NOTE: `peek` here is used to avoid affecting a "hotness" of the value
            match self.0.peek(key) {
                None => return,
                Some(v) => v,
            }
        };

        v.header.header.store(rc, Ordering::Release);
    }
}
