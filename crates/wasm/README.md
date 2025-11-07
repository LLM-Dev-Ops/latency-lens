# llm-latency-lens-wasm

WebAssembly bindings for LLM Latency Lens.

This crate provides the Rust implementation that gets compiled to WebAssembly
for use in JavaScript/TypeScript applications via the `@llm-devops/latency-lens`
npm package.

## Building

To build the WASM artifacts:

```bash
# Install wasm-pack
cargo install wasm-pack

# Build for bundlers (webpack, vite, etc.)
wasm-pack build --target bundler --out-dir ../../npm/dist

# Build for Node.js
wasm-pack build --target nodejs --out-dir ../../npm-node/dist

# Or use the build script
../../scripts/build-wasm.sh --release
```

## Testing

```bash
# Run Rust tests
cargo test

# Run WASM tests in browser
wasm-pack test --headless --chrome
```

## Architecture

The WASM bindings expose:

- `LatencyCollector` - Main metrics collection interface
- `JsProvider` - Provider enum for JavaScript
- Helper types for serialization/deserialization

All timing and statistical calculations are performed in Rust/WASM for
maximum performance and accuracy.

## Publishing

The npm package is published from the `/npm` directory. See the
[npm README](../../npm/README.md) for usage documentation.

```bash
# Build and publish
../../scripts/publish-npm.sh
```

## License

Apache-2.0
