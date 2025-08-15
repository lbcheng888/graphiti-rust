//! Data Sharding Implementation

use crate::{DistributedConfig, ShardingStrategy};

/// Consistent hash sharding
///
/// 使用 blake3 哈希对键进行映射，返回 shard_0..shard_15 的 16 个分片之一。
pub struct ConsistentHashSharding {
    /// 分布式系统配置
    config: DistributedConfig,
}

impl ConsistentHashSharding {
    /// 构造一个一致性哈希分片器
    pub fn new(config: DistributedConfig) -> Self {
        Self { config }
    }

    /// 根据 key 计算其所属的分片标识
    pub fn get_shard(&self, key: &str) -> String {
        // 简单哈希分片：按首字节取模
        let hash = blake3::hash(key.as_bytes());
        format!("shard_{}", hash.as_bytes()[0] % 16)
    }
}
