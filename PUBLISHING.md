# Publishing Guide for LLM Latency Lens

This document provides step-by-step instructions for publishing LLM Latency Lens to both crates.io and npm.

## Prerequisites

### For crates.io Publishing
1. Create a crates.io account at https://crates.io/
2. Generate an API token from https://crates.io/me
3. Login to cargo: `cargo login <your-token>`

### For npm Publishing
1. Create an npm account at https://www.npmjs.com/signup
2. Login to npm: `npm login`
3. (Optional) If publishing scoped package (@llm-devops/latency-lens), ensure you have access to the @llm-devops organization

## Publishing to crates.io

The crates must be published in dependency order, as each crate depends on the previous ones.

### Step 1: Publish Core Crate

```bash
cd /workspaces/llm-latency-lens
cargo publish -p llm-latency-lens-core --allow-dirty
```

**Expected output:**
- Package size: ~50KB
- Wait for the crate to be indexed (usually 1-2 minutes)

### Step 2: Publish Dependent Crates

After core is published and indexed, publish the dependent crates:

```bash
# Publish providers (depends on core)
cargo publish -p llm-latency-lens-providers --allow-dirty

# Wait ~1 minute for indexing, then publish metrics (depends on core)
cargo publish -p llm-latency-lens-metrics --allow-dirty

# Wait ~1 minute for indexing, then publish exporters (depends on core, metrics)
cargo publish -p llm-latency-lens-exporters --allow-dirty
```

### Step 3: Verify Publications

Check that all crates are available:
- https://crates.io/crates/llm-latency-lens-core
- https://crates.io/crates/llm-latency-lens-providers
- https://crates.io/crates/llm-latency-lens-metrics
- https://crates.io/crates/llm-latency-lens-exporters

## Publishing to npm

### Step 1: Build WASM Package (Already Done)

The WASM package has been built and is located at:
```
/workspaces/llm-latency-lens/npm/pkg/
```

Package details:
- **Name:** @llm-devops/latency-lens
- **Version:** 0.1.0
- **Size:** 84.0 KB (compressed)
- **Unpacked size:** 211.6 KB

### Step 2: Test Package Locally (Optional)

Install the package locally to test:

```bash
cd /workspaces/llm-latency-lens/npm/pkg
npm install -g llm-devops-latency-lens-0.1.0.tgz
```

### Step 3: Publish to npm

```bash
cd /workspaces/llm-latency-lens/npm/pkg
npm publish --access public
```

**Note:** The `--access public` flag is required for scoped packages (@llm-devops/latency-lens).

### Step 4: Verify Publication

Check that the package is available:
- https://www.npmjs.com/package/@llm-devops/latency-lens

## Post-Publication Tasks

### 1. Tag the Release

```bash
git tag -a v0.1.0 -m "Release v0.1.0"
git push origin v0.1.0
```

### 2. Create GitHub Release

Go to https://github.com/llm-devops/llm-latency-lens/releases and create a new release:
- Tag: v0.1.0
- Title: "LLM Latency Lens v0.1.0 - Initial Release"
- Description: Include key features, installation instructions, and changelog

### 3. Update Documentation

Update the main README.md with installation instructions:

```markdown
## Installation

### Rust

```bash
cargo install llm-latency-lens
```

### npm (WebAssembly)

```bash
npm install @llm-devops/latency-lens
```
```

## Troubleshooting

### crates.io Issues

**Error: "crate not found in registry"**
- Solution: Wait 1-2 minutes for crates.io to index the package before publishing dependent crates

**Error: "authentication required"**
- Solution: Run `cargo login` with your crates.io API token

### npm Issues

**Error: "You do not have permission to publish"**
- Solution: Ensure you're logged in with `npm whoami` and have access to the @llm-devops scope

**Error: "package already exists"**
- Solution: Bump the version in Cargo.toml and rebuild WASM with wasm-pack

## Version Updates

To publish a new version:

1. Update version in all Cargo.toml files (maintain consistency)
2. Update CHANGELOG.md
3. Rebuild all packages:
   ```bash
   cargo build --release
   cd crates/wasm && wasm-pack build --target bundler --out-dir ../../npm/pkg
   ```
4. Follow publishing steps above with new version

## Support

For issues or questions:
- File an issue: https://github.com/llm-devops/llm-latency-lens/issues
- Email: support@llm-devops.com (if applicable)

## License

All published packages are licensed under Apache-2.0.
