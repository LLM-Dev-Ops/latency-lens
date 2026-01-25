//! Phase 2 Caching Layer
//!
//! TTL-based cache for historical reads and lineage lookups.
//!
//! # Caching Rules
//!
//! Caching is allowed for:
//! - Historical reads
//! - Lineage lookups
//!
//! TTL: 60-120 seconds

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::debug;
use uuid::Uuid;

/// Default minimum TTL in seconds
pub const DEFAULT_MIN_TTL_SECS: u64 = 60;

/// Default maximum TTL in seconds
pub const DEFAULT_MAX_TTL_SECS: u64 = 120;

/// Cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Minimum TTL in seconds
    pub min_ttl_secs: u64,
    /// Maximum TTL in seconds
    pub max_ttl_secs: u64,
    /// Maximum cache entries
    pub max_entries: usize,
    /// Enable cache statistics
    pub enable_stats: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            min_ttl_secs: DEFAULT_MIN_TTL_SECS,
            max_ttl_secs: DEFAULT_MAX_TTL_SECS,
            max_entries: 1000,
            enable_stats: true,
        }
    }
}

impl CacheConfig {
    /// Create from environment variables
    pub fn from_env() -> Self {
        Self {
            min_ttl_secs: std::env::var("CACHE_MIN_TTL_SECS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(DEFAULT_MIN_TTL_SECS),
            max_ttl_secs: std::env::var("CACHE_MAX_TTL_SECS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(DEFAULT_MAX_TTL_SECS),
            max_entries: std::env::var("CACHE_MAX_ENTRIES")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(1000),
            enable_stats: std::env::var("CACHE_ENABLE_STATS")
                .map(|s| s.to_lowercase() != "false")
                .unwrap_or(true),
        }
    }

    /// Get TTL duration (uses max_ttl_secs)
    pub fn ttl(&self) -> Duration {
        Duration::from_secs(self.max_ttl_secs)
    }
}

/// A cached entry with expiration
#[derive(Debug, Clone)]
struct CacheEntry<T> {
    /// Cached value
    value: T,
    /// When the entry was created
    created_at: Instant,
    /// TTL for this entry
    ttl: Duration,
}

impl<T> CacheEntry<T> {
    fn new(value: T, ttl: Duration) -> Self {
        Self {
            value,
            created_at: Instant::now(),
            ttl,
        }
    }

    fn is_expired(&self) -> bool {
        self.created_at.elapsed() > self.ttl
    }
}

/// Cache statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Number of cache hits
    pub hits: u64,
    /// Number of cache misses
    pub misses: u64,
    /// Number of entries
    pub entries: usize,
    /// Number of evictions
    pub evictions: u64,
}

impl CacheStats {
    /// Calculate hit rate
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

/// Lineage cache for Phase 2 agents
///
/// Provides caching for historical reads and lineage lookups
/// with configurable TTL (60-120 seconds).
pub struct LineageCache {
    /// Configuration
    config: CacheConfig,
    /// Historical event cache
    events: Arc<RwLock<HashMap<Uuid, CacheEntry<serde_json::Value>>>>,
    /// Lineage path cache (event_id -> lineage chain)
    lineage: Arc<RwLock<HashMap<Uuid, CacheEntry<Vec<Uuid>>>>>,
    /// Query result cache
    queries: Arc<RwLock<HashMap<String, CacheEntry<serde_json::Value>>>>,
    /// Cache statistics
    stats: Arc<RwLock<CacheStats>>,
}

impl LineageCache {
    /// Create a new lineage cache with configuration
    pub fn new(config: CacheConfig) -> Self {
        Self {
            config,
            events: Arc::new(RwLock::new(HashMap::new())),
            lineage: Arc::new(RwLock::new(HashMap::new())),
            queries: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(CacheStats::default())),
        }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(CacheConfig::default())
    }

    /// Get configuration
    pub fn config(&self) -> &CacheConfig {
        &self.config
    }

    /// Get cache statistics
    pub async fn stats(&self) -> CacheStats {
        self.stats.read().await.clone()
    }

    /// Get an event from cache
    pub async fn get_event(&self, event_id: &Uuid) -> Option<serde_json::Value> {
        let cache = self.events.read().await;

        if let Some(entry) = cache.get(event_id) {
            if !entry.is_expired() {
                if self.config.enable_stats {
                    let mut stats = self.stats.write().await;
                    stats.hits += 1;
                }
                debug!(event_id = %event_id, "Cache hit for event");
                return Some(entry.value.clone());
            }
        }

        if self.config.enable_stats {
            let mut stats = self.stats.write().await;
            stats.misses += 1;
        }
        debug!(event_id = %event_id, "Cache miss for event");
        None
    }

    /// Put an event in cache
    pub async fn put_event(&self, event_id: Uuid, value: serde_json::Value) {
        let mut cache = self.events.write().await;

        // Check capacity and evict if needed
        if cache.len() >= self.config.max_entries {
            self.evict_expired_entries(&mut cache).await;
        }

        let entry = CacheEntry::new(value, self.config.ttl());
        cache.insert(event_id, entry);

        if self.config.enable_stats {
            let mut stats = self.stats.write().await;
            stats.entries = cache.len();
        }
    }

    /// Get lineage path from cache
    pub async fn get_lineage(&self, event_id: &Uuid) -> Option<Vec<Uuid>> {
        let cache = self.lineage.read().await;

        if let Some(entry) = cache.get(event_id) {
            if !entry.is_expired() {
                if self.config.enable_stats {
                    let mut stats = self.stats.write().await;
                    stats.hits += 1;
                }
                debug!(event_id = %event_id, "Cache hit for lineage");
                return Some(entry.value.clone());
            }
        }

        if self.config.enable_stats {
            let mut stats = self.stats.write().await;
            stats.misses += 1;
        }
        None
    }

    /// Put lineage path in cache
    pub async fn put_lineage(&self, event_id: Uuid, lineage: Vec<Uuid>) {
        let mut cache = self.lineage.write().await;

        if cache.len() >= self.config.max_entries {
            self.evict_expired_lineage(&mut cache).await;
        }

        let entry = CacheEntry::new(lineage, self.config.ttl());
        cache.insert(event_id, entry);
    }

    /// Get a query result from cache
    pub async fn get_query(&self, query_key: &str) -> Option<serde_json::Value> {
        let cache = self.queries.read().await;

        if let Some(entry) = cache.get(query_key) {
            if !entry.is_expired() {
                if self.config.enable_stats {
                    let mut stats = self.stats.write().await;
                    stats.hits += 1;
                }
                debug!(query_key = %query_key, "Cache hit for query");
                return Some(entry.value.clone());
            }
        }

        if self.config.enable_stats {
            let mut stats = self.stats.write().await;
            stats.misses += 1;
        }
        None
    }

    /// Put a query result in cache
    pub async fn put_query(&self, query_key: impl Into<String>, value: serde_json::Value) {
        let mut cache = self.queries.write().await;

        if cache.len() >= self.config.max_entries {
            self.evict_expired_queries(&mut cache).await;
        }

        let entry = CacheEntry::new(value, self.config.ttl());
        cache.insert(query_key.into(), entry);
    }

    /// Invalidate an event
    pub async fn invalidate_event(&self, event_id: &Uuid) {
        let mut cache = self.events.write().await;
        cache.remove(event_id);
    }

    /// Invalidate a lineage
    pub async fn invalidate_lineage(&self, event_id: &Uuid) {
        let mut cache = self.lineage.write().await;
        cache.remove(event_id);
    }

    /// Clear all caches
    pub async fn clear(&self) {
        let mut events = self.events.write().await;
        let mut lineage = self.lineage.write().await;
        let mut queries = self.queries.write().await;

        events.clear();
        lineage.clear();
        queries.clear();

        if self.config.enable_stats {
            let mut stats = self.stats.write().await;
            stats.entries = 0;
        }
    }

    /// Evict expired entries from events cache
    async fn evict_expired_entries(
        &self,
        cache: &mut HashMap<Uuid, CacheEntry<serde_json::Value>>,
    ) {
        let before = cache.len();
        cache.retain(|_, entry| !entry.is_expired());
        let evicted = before - cache.len();

        if self.config.enable_stats && evicted > 0 {
            let mut stats = self.stats.write().await;
            stats.evictions += evicted as u64;
            stats.entries = cache.len();
        }
    }

    /// Evict expired entries from lineage cache
    async fn evict_expired_lineage(&self, cache: &mut HashMap<Uuid, CacheEntry<Vec<Uuid>>>) {
        let before = cache.len();
        cache.retain(|_, entry| !entry.is_expired());
        let evicted = before - cache.len();

        if self.config.enable_stats && evicted > 0 {
            let mut stats = self.stats.write().await;
            stats.evictions += evicted as u64;
        }
    }

    /// Evict expired entries from query cache
    async fn evict_expired_queries(
        &self,
        cache: &mut HashMap<String, CacheEntry<serde_json::Value>>,
    ) {
        let before = cache.len();
        cache.retain(|_, entry| !entry.is_expired());
        let evicted = before - cache.len();

        if self.config.enable_stats && evicted > 0 {
            let mut stats = self.stats.write().await;
            stats.evictions += evicted as u64;
        }
    }
}

impl Clone for LineageCache {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            events: Arc::clone(&self.events),
            lineage: Arc::clone(&self.lineage),
            queries: Arc::clone(&self.queries),
            stats: Arc::clone(&self.stats),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_cache_put_get() {
        let cache = LineageCache::with_defaults();
        let event_id = Uuid::now_v7();
        let value = serde_json::json!({"test": "value"});

        cache.put_event(event_id, value.clone()).await;

        let cached = cache.get_event(&event_id).await;
        assert!(cached.is_some());
        assert_eq!(cached.unwrap(), value);
    }

    #[tokio::test]
    async fn test_cache_miss() {
        let cache = LineageCache::with_defaults();
        let event_id = Uuid::now_v7();

        let cached = cache.get_event(&event_id).await;
        assert!(cached.is_none());
    }

    #[tokio::test]
    async fn test_cache_stats() {
        let cache = LineageCache::with_defaults();
        let event_id = Uuid::now_v7();
        let value = serde_json::json!({"test": "value"});

        // Miss
        let _ = cache.get_event(&event_id).await;

        // Put and hit
        cache.put_event(event_id, value).await;
        let _ = cache.get_event(&event_id).await;

        let stats = cache.stats().await;
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hit_rate(), 0.5);
    }

    #[tokio::test]
    async fn test_lineage_cache() {
        let cache = LineageCache::with_defaults();
        let event_id = Uuid::now_v7();
        let lineage = vec![Uuid::now_v7(), Uuid::now_v7()];

        cache.put_lineage(event_id, lineage.clone()).await;

        let cached = cache.get_lineage(&event_id).await;
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().len(), 2);
    }

    #[test]
    fn test_config_defaults() {
        let config = CacheConfig::default();
        assert_eq!(config.min_ttl_secs, 60);
        assert_eq!(config.max_ttl_secs, 120);
        assert_eq!(config.max_entries, 1000);
    }
}
