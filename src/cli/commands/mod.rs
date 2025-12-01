//! Command implementations

pub mod benchmark;
pub mod compare;
pub mod export;
pub mod profile;
pub mod run;
pub mod validate;

use anyhow::Result;
use std::path::Path;

/// Read prompt from file or use provided string
pub fn read_prompt(prompt: &Option<String>, prompt_file: &Option<std::path::PathBuf>) -> Result<String> {
    if let Some(prompt) = prompt {
        Ok(prompt.clone())
    } else if let Some(file) = prompt_file {
        std::fs::read_to_string(file)
            .map_err(|e| anyhow::anyhow!("Failed to read prompt file: {}", e))
    } else {
        anyhow::bail!("Either --prompt or --prompt-file must be provided");
    }
}

/// Write output to file or stdout
pub fn write_output(content: &str, output_path: &Option<std::path::PathBuf>) -> Result<()> {
    if let Some(path) = output_path {
        std::fs::write(path, content)
            .map_err(|e| anyhow::anyhow!("Failed to write output file: {}", e))
    } else {
        println!("{}", content);
        Ok(())
    }
}
