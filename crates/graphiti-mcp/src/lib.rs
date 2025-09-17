#![recursion_limit = "512"]
pub mod app; // renamed from `main` to avoid special_module_name warning
mod ast_parser;
pub mod config;
pub mod handlers;
mod learning;
mod learning_endpoints;
mod learning_integration;
pub mod middleware;
pub mod models;
mod project_scanner;
pub mod qdrant_index;
mod serena_config;
mod serena_ra;
mod serena_symbols;
mod serena_tools;
pub mod server;
pub mod services;
pub mod types;
pub mod utils;

pub use crate::main_impl::run;
pub use app::legacy_main;
// 统一对外类型导出，避免与 models::* 产生歧义
pub use crate::types::AppState;
pub use crate::types::*;

mod main_impl {
    use super::*;
    use anyhow::Result;

    pub fn run() -> Result<()> {
        // For now, delegate to old main in main.rs to limit scope.
        // After full split, move logic here.
        legacy_main()
    }
}
