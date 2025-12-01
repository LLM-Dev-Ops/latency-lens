//! File I/O operations for benchmark results
//!
//! This module provides utilities for reading and writing benchmark results
//! to the canonical output directories.

use anyhow::{Context, Result};
use std::fs;
use std::path::{Path, PathBuf};

use super::result::BenchmarkResult;

/// Default output directory for benchmark results
pub const DEFAULT_OUTPUT_DIR: &str = "benchmarks/output";

/// Subdirectory for raw JSON results
pub const RAW_OUTPUT_DIR: &str = "benchmarks/output/raw";

/// Filename for the summary markdown file
pub const SUMMARY_FILENAME: &str = "summary.md";

/// Ensure the canonical output directories exist
pub fn ensure_output_dirs() -> Result<()> {
    let dirs = [DEFAULT_OUTPUT_DIR, RAW_OUTPUT_DIR];

    for dir in dirs {
        fs::create_dir_all(dir)
            .with_context(|| format!("Failed to create directory: {}", dir))?;
    }

    Ok(())
}

/// Get the path to the output directory
pub fn output_dir() -> PathBuf {
    PathBuf::from(DEFAULT_OUTPUT_DIR)
}

/// Get the path to the raw output directory
pub fn raw_output_dir() -> PathBuf {
    PathBuf::from(RAW_OUTPUT_DIR)
}

/// Get the path to the summary file
pub fn summary_path() -> PathBuf {
    output_dir().join(SUMMARY_FILENAME)
}

/// Write a single benchmark result to a JSON file
pub fn write_result(result: &BenchmarkResult) -> Result<PathBuf> {
    ensure_output_dirs()?;

    // Generate filename from target_id and timestamp
    let safe_target_id = result.target_id().replace([':', '/', '\\'], "_");
    let timestamp = result.timestamp().format("%Y%m%d_%H%M%S");
    let filename = format!("{}_{}.json", safe_target_id, timestamp);

    let path = raw_output_dir().join(&filename);
    let json = serde_json::to_string_pretty(result)
        .context("Failed to serialize benchmark result")?;

    fs::write(&path, json)
        .with_context(|| format!("Failed to write result to: {}", path.display()))?;

    Ok(path)
}

/// Write multiple benchmark results to JSON files
pub fn write_results(results: &[BenchmarkResult]) -> Result<Vec<PathBuf>> {
    results.iter().map(write_result).collect()
}

/// Read a benchmark result from a JSON file
pub fn read_result(path: &Path) -> Result<BenchmarkResult> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("Failed to read file: {}", path.display()))?;

    serde_json::from_str(&content)
        .with_context(|| format!("Failed to parse JSON from: {}", path.display()))
}

/// Read all benchmark results from the raw output directory
pub fn read_all_results() -> Result<Vec<BenchmarkResult>> {
    let dir = raw_output_dir();

    if !dir.exists() {
        return Ok(Vec::new());
    }

    let mut results = Vec::new();

    for entry in fs::read_dir(&dir)
        .with_context(|| format!("Failed to read directory: {}", dir.display()))?
    {
        let entry = entry.context("Failed to read directory entry")?;
        let path = entry.path();

        if path.extension().map(|e| e == "json").unwrap_or(false) {
            match read_result(&path) {
                Ok(result) => results.push(result),
                Err(e) => {
                    tracing::warn!("Failed to read result from {:?}: {}", path, e);
                }
            }
        }
    }

    // Sort by timestamp (newest first)
    results.sort_by(|a, b| b.timestamp().cmp(&a.timestamp()));

    Ok(results)
}

/// Clean up old benchmark results, keeping the most recent N per target
pub fn cleanup_old_results(keep_per_target: usize) -> Result<usize> {
    use std::collections::HashMap;

    let results = read_all_results()?;
    let mut by_target: HashMap<String, Vec<BenchmarkResult>> = HashMap::new();

    for result in results {
        by_target
            .entry(result.target_id().to_string())
            .or_default()
            .push(result);
    }

    let mut deleted = 0;

    for (target_id, mut target_results) in by_target {
        // Sort by timestamp (newest first)
        target_results.sort_by(|a, b| b.timestamp().cmp(&a.timestamp()));

        // Delete results beyond the keep limit
        for result in target_results.into_iter().skip(keep_per_target) {
            let safe_target_id = target_id.replace([':', '/', '\\'], "_");
            let timestamp = result.timestamp().format("%Y%m%d_%H%M%S");
            let filename = format!("{}_{}.json", safe_target_id, timestamp);
            let path = raw_output_dir().join(&filename);

            if path.exists() {
                fs::remove_file(&path)
                    .with_context(|| format!("Failed to delete: {}", path.display()))?;
                deleted += 1;
            }
        }
    }

    Ok(deleted)
}

/// Write summary content to the summary file
pub fn write_summary(content: &str) -> Result<PathBuf> {
    ensure_output_dirs()?;

    let path = summary_path();
    fs::write(&path, content)
        .with_context(|| format!("Failed to write summary to: {}", path.display()))?;

    Ok(path)
}

/// Read the summary file if it exists
pub fn read_summary() -> Result<Option<String>> {
    let path = summary_path();

    if !path.exists() {
        return Ok(None);
    }

    let content = fs::read_to_string(&path)
        .with_context(|| format!("Failed to read summary from: {}", path.display()))?;

    Ok(Some(content))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::TempDir;

    fn setup_test_dir() -> TempDir {
        tempfile::tempdir().expect("Failed to create temp dir")
    }

    #[test]
    fn test_output_paths() {
        assert_eq!(output_dir(), PathBuf::from("benchmarks/output"));
        assert_eq!(raw_output_dir(), PathBuf::from("benchmarks/output/raw"));
        assert_eq!(
            summary_path(),
            PathBuf::from("benchmarks/output/summary.md")
        );
    }

    #[test]
    fn test_benchmark_result_serialization() {
        let result = BenchmarkResult::new(
            "test:target",
            json!({
                "total_requests": 10,
                "successful_requests": 10
            }),
        );

        let json = serde_json::to_string_pretty(&result).unwrap();
        let parsed: BenchmarkResult = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.target_id(), "test:target");
    }
}
