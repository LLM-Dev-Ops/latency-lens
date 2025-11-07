#!/bin/bash
#
# Publish the npm package to the npm registry
#
# This script builds the WASM artifacts and publishes the package to npm.
# It includes safety checks and version validation.
#
# Usage:
#   ./scripts/publish-npm.sh [--dry-run] [--tag <tag>]
#
# Options:
#   --dry-run    Perform a dry run without actually publishing
#   --tag        Specify npm dist-tag (default: latest)
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

# Default options
DRY_RUN=false
NPM_TAG="latest"
NPM_ARGS=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            NPM_ARGS="--dry-run"
            shift
            ;;
        --tag)
            NPM_TAG="$2"
            shift 2
            ;;
        --help)
            echo "Publish the npm package to the npm registry"
            echo ""
            echo "Usage: $0 [--dry-run] [--tag <tag>]"
            echo ""
            echo "Options:"
            echo "  --dry-run    Perform a dry run without actually publishing"
            echo "  --tag        Specify npm dist-tag (default: latest)"
            echo "  --help       Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}=== LLM Latency Lens - npm Publish ===${NC}"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}Running in DRY RUN mode - no actual publishing will occur${NC}"
    echo ""
fi

# Check if we're logged in to npm
if ! npm whoami &> /dev/null; then
    echo -e "${RED}Error: Not logged in to npm${NC}"
    echo ""
    echo "Please login first:"
    echo "  npm login"
    exit 1
fi

NPM_USER=$(npm whoami)
echo -e "${GREEN}✓ Logged in to npm as: ${YELLOW}$NPM_USER${NC}"
echo ""

# Check git status
if [ -n "$(git status --porcelain)" ]; then
    echo -e "${YELLOW}Warning: Working directory has uncommitted changes${NC}"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${RED}Aborted${NC}"
        exit 1
    fi
fi

# Get version from Cargo.toml
CARGO_VERSION=$(grep "^version" "$PROJECT_ROOT/crates/wasm/Cargo.toml" | head -n1 | cut -d'"' -f2)
NPM_VERSION=$(grep "\"version\"" "$PROJECT_ROOT/npm/package.json" | head -n1 | cut -d'"' -f4)

echo -e "${YELLOW}Cargo version:${NC} $CARGO_VERSION"
echo -e "${YELLOW}npm version:${NC}   $NPM_VERSION"
echo ""

if [ "$CARGO_VERSION" != "$NPM_VERSION" ]; then
    echo -e "${YELLOW}Warning: Version mismatch between Cargo.toml and package.json${NC}"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${RED}Aborted${NC}"
        exit 1
    fi
fi

# Check if this version is already published
if npm view "@llm-devops/latency-lens@$NPM_VERSION" version &> /dev/null; then
    echo -e "${RED}Error: Version $NPM_VERSION is already published${NC}"
    echo ""
    echo "Please update the version in:"
    echo "  - crates/wasm/Cargo.toml"
    echo "  - npm/package.json"
    exit 1
fi

echo -e "${GREEN}✓ Version $NPM_VERSION is available${NC}"
echo ""

# Build WASM artifacts
echo -e "${BLUE}Building WASM artifacts...${NC}"
"$SCRIPT_DIR/build-wasm.sh" --release

echo -e "${GREEN}✓ WASM build complete${NC}"
echo ""

# Run tests if available
if [ -f "$PROJECT_ROOT/crates/wasm/tests" ]; then
    echo -e "${BLUE}Running tests...${NC}"
    cd "$PROJECT_ROOT/crates/wasm"
    cargo test
    echo -e "${GREEN}✓ Tests passed${NC}"
    echo ""
fi

# Verify package contents
echo -e "${BLUE}Package contents:${NC}"
cd "$PROJECT_ROOT/npm"
npm pack --dry-run
echo ""

# Publish confirmation
if [ "$DRY_RUN" = false ]; then
    echo -e "${YELLOW}Ready to publish @llm-devops/latency-lens@$NPM_VERSION${NC}"
    echo -e "${YELLOW}Tag: $NPM_TAG${NC}"
    echo ""
    read -p "Proceed with publish? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${RED}Aborted${NC}"
        exit 1
    fi
fi

# Publish to npm
echo -e "${BLUE}Publishing to npm...${NC}"
cd "$PROJECT_ROOT/npm"
npm publish --tag "$NPM_TAG" --access public $NPM_ARGS

if [ "$DRY_RUN" = false ]; then
    echo ""
    echo -e "${GREEN}=== Publish Complete! ===${NC}"
    echo ""
    echo -e "Package: ${YELLOW}@llm-devops/latency-lens@$NPM_VERSION${NC}"
    echo -e "Tag:     ${YELLOW}$NPM_TAG${NC}"
    echo ""
    echo "View on npm:"
    echo "  https://www.npmjs.com/package/@llm-devops/latency-lens"
    echo ""
    echo "Install with:"
    echo "  npm install @llm-devops/latency-lens"
    echo ""
    echo -e "${GREEN}Next steps:${NC}"
    echo "  1. Create a git tag: git tag v$NPM_VERSION"
    echo "  2. Push the tag: git push origin v$NPM_VERSION"
    echo "  3. Create a GitHub release"
else
    echo ""
    echo -e "${GREEN}✓ Dry run complete - no publishing occurred${NC}"
    echo ""
    echo "To actually publish, run:"
    echo "  ./scripts/publish-npm.sh"
fi
