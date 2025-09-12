#![recursion_limit = "512"]
pub mod types;
pub mod handlers;
pub mod middleware;
pub mod app; // renamed from `main` to avoid special_module_name warning
pub mod config;
pub mod qdrant_index;
pub mod utils;
pub mod server;
pub mod models;
pub mod services;
mod ast_parser;
mod serena_tools;
mod serena_ra;
mod serena_symbols;
mod serena_config;
mod project_scanner;
mod learning;
mod learning_endpoints;
mod learning_integration;

pub use app::legacy_main;
pub use crate::main_impl::run;
// 统一对外类型导出，避免与 models::* 产生歧义
pub use crate::types::*;
pub use crate::types::AppState;

mod main_impl {
    use super::*;
    use anyhow::Result;

    pub fn run() -> Result<()> {
        // For now, delegate to old main in main.rs to limit scope.
        // After full split, move logic here.
        legacy_main()
    }
}
