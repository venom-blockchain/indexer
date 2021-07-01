use std::sync::{Arc, Weak};

use anyhow::Result;
use dashmap::DashMap;

use super::block_handle::*;
use super::block_meta::*;
use crate::storage::StoredValue;

pub struct BlockHandleStorage {
    cache: Arc<DashMap<ton_block::BlockIdExt, Weak<BlockHandle>>>,
    db: sled::Tree,
}

impl BlockHandleStorage {
    pub fn with_db(db: sled::Tree) -> Self {
        Self {
            cache: Arc::new(Default::default()),
            db,
        }
    }

    pub fn load_handle(
        &self,
        block_id: &ton_block::BlockIdExt,
    ) -> Result<Option<Arc<BlockHandle>>> {
        Ok(loop {
            if let Some(weak) = self.cache.get(block_id) {
                if let Some(handle) = weak.upgrade() {
                    break Some(handle);
                }
            }

            if let Some(meta) = self.db.get(block_id.root_hash.as_slice())? {
                let meta = BlockMeta::from_slice(meta.as_ref())?;
                if let Some(handle) = self.create_handle(block_id.clone(), meta)? {
                    break Some(handle);
                }
            } else {
                break None;
            }
        })
    }

    pub fn store_handle(&self, handle: &Arc<BlockHandle>) -> Result<()> {
        self.db
            .insert(handle.id().root_hash.as_slice(), handle.meta().to_vec()?)?;
        Ok(())
    }

    pub fn create_handle(
        &self,
        block_id: ton_block::BlockIdExt,
        meta: BlockMeta,
    ) -> Result<Option<Arc<BlockHandle>>> {
        use dashmap::mapref::entry::Entry;

        let handle = match self.cache.entry(block_id.clone()) {
            Entry::Vacant(entry) => {
                let handle = Arc::new(BlockHandle::with_values(block_id, meta));
                entry.insert(Arc::downgrade(&handle));
                handle
            }
            Entry::Occupied(_) => return Ok(None),
        };

        self.store_handle(&handle)?;
        Ok(Some(handle))
    }
}