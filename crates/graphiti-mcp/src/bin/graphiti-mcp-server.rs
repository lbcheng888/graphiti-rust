// Thin binary entry; delegate to library main
fn main() -> anyhow::Result<()> {
    graphiti_mcp::run()
}
