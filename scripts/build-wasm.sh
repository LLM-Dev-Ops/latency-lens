#!/bin/bash
#
# Build WebAssembly artifacts for npm distribution
#
# This script builds the WASM bindings for both bundler and Node.js targets
# using wasm-pack.
#
# Usage:
#   ./scripts/build-wasm.sh [--release]
#
# Options:
#   --release    Build with optimizations (default: debug)
#   --help       Show this help message

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Build mode
BUILD_MODE="dev"
WASM_PACK_ARGS=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --release)
            BUILD_MODE="release"
            WASM_PACK_ARGS="--release"
            shift
            ;;
        --help)
            echo "Build WebAssembly artifacts for npm distribution"
            echo ""
            echo "Usage: $0 [--release]"
            echo ""
            echo "Options:"
            echo "  --release    Build with optimizations (default: debug)"
            echo "  --help       Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}=== LLM Latency Lens - WASM Build ===${NC}"
echo ""
echo -e "${YELLOW}Build mode:${NC} $BUILD_MODE"
echo ""

# Check if wasm-pack is installed
if ! command -v wasm-pack &> /dev/null; then
    echo -e "${RED}Error: wasm-pack is not installed${NC}"
    echo ""
    echo "Install wasm-pack with:"
    echo "  curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh"
    echo ""
    echo "Or with cargo:"
    echo "  cargo install wasm-pack"
    exit 1
fi

echo -e "${GREEN}✓ wasm-pack found${NC}"
echo ""

# Build for bundler (webpack, rollup, etc.)
echo -e "${BLUE}Building for bundler target...${NC}"
cd "$PROJECT_ROOT"
wasm-pack build crates/wasm \
    --target bundler \
    --out-dir ../../npm/dist \
    --scope llm-devops \
    $WASM_PACK_ARGS

echo -e "${GREEN}✓ Bundler build complete${NC}"
echo ""

# Build for Node.js
echo -e "${BLUE}Building for Node.js target...${NC}"
mkdir -p "$PROJECT_ROOT/npm-node/dist"
wasm-pack build crates/wasm \
    --target nodejs \
    --out-dir ../../npm-node/dist \
    --scope llm-devops \
    $WASM_PACK_ARGS

echo -e "${GREEN}✓ Node.js build complete${NC}"
echo ""

# Copy TypeScript definitions to dist directories
echo -e "${BLUE}Copying TypeScript definitions...${NC}"
cp "$PROJECT_ROOT/npm/index.d.ts" "$PROJECT_ROOT/npm/dist/index.d.ts"
cp "$PROJECT_ROOT/npm/index.d.ts" "$PROJECT_ROOT/npm-node/dist/index.d.ts"

echo -e "${GREEN}✓ TypeScript definitions copied${NC}"
echo ""

# Display build artifacts
echo -e "${BLUE}Build artifacts:${NC}"
echo ""
echo -e "${YELLOW}Bundler (for webpack, vite, etc.):${NC}"
ls -lh "$PROJECT_ROOT/npm/dist" | tail -n +2
echo ""
echo -e "${YELLOW}Node.js:${NC}"
ls -lh "$PROJECT_ROOT/npm-node/dist" | tail -n +2
echo ""

# Display file sizes
BUNDLER_WASM_SIZE=$(du -h "$PROJECT_ROOT/npm/dist"/*.wasm | cut -f1)
NODE_WASM_SIZE=$(du -h "$PROJECT_ROOT/npm-node/dist"/*.wasm | cut -f1)

echo -e "${GREEN}=== Build Summary ===${NC}"
echo ""
echo -e "  Bundler WASM size: ${YELLOW}$BUNDLER_WASM_SIZE${NC}"
echo -e "  Node.js WASM size: ${YELLOW}$NODE_WASM_SIZE${NC}"
echo -e "  Build mode:        ${YELLOW}$BUILD_MODE${NC}"
echo ""

if [ "$BUILD_MODE" = "dev" ]; then
    echo -e "${YELLOW}Note: This is a debug build. Use --release for production.${NC}"
    echo ""
fi

echo -e "${GREEN}✓ WASM build complete!${NC}"
echo ""
echo "Next steps:"
echo "  1. Test the package: cd npm && npm link"
echo "  2. Publish to npm: ./scripts/publish-npm.sh"
