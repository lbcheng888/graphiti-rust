//! Data Replication Implementation

use crate::{DistributedConfig, DistributedResult};

/// Data replicator
pub struct DataReplicator {
    config: DistributedConfig,
}

impl DataReplicator {
    pub async fn new(config: DistributedConfig) -> DistributedResult<Self> {
        Ok(Self { config })
    }
}
