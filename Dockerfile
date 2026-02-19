# Multi-stage build for LLM-Latency-Lens Phase 2
# Stage 1: Build application
FROM rust:1.93-slim-bookworm AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only dependency files first for better caching
COPY Cargo.toml Cargo.lock ./
COPY crates/core/Cargo.toml crates/core/Cargo.toml
COPY crates/providers/Cargo.toml crates/providers/Cargo.toml
COPY crates/metrics/Cargo.toml crates/metrics/Cargo.toml
COPY crates/exporters/Cargo.toml crates/exporters/Cargo.toml
COPY crates/wasm/Cargo.toml crates/wasm/Cargo.toml

# Create dummy source files for dependency compilation
RUN mkdir -p src crates/core/src crates/providers/src crates/metrics/src crates/exporters/src crates/wasm/src && \
    echo "fn main() {}" > src/main.rs && \
    echo "pub fn placeholder() {}" > src/lib.rs && \
    echo "pub fn placeholder() {}" > crates/core/src/lib.rs && \
    echo "pub fn placeholder() {}" > crates/providers/src/lib.rs && \
    echo "pub fn placeholder() {}" > crates/metrics/src/lib.rs && \
    echo "pub fn placeholder() {}" > crates/exporters/src/lib.rs && \
    echo "pub fn placeholder() {}" > crates/wasm/src/lib.rs

# Build dependencies (this layer will be cached)
RUN cargo build --release 2>/dev/null || true

# Copy actual source code
COPY . .

# Touch source files to invalidate cache
RUN touch src/main.rs src/lib.rs

# Build release binary
RUN cargo build --release --bin llm-latency-lens

# Verify binary exists and is executable
RUN test -f /app/target/release/llm-latency-lens && \
    chmod +x /app/target/release/llm-latency-lens


# Stage 2: Runtime image (distroless for minimal attack surface)
FROM gcr.io/distroless/cc-debian12:nonroot AS runtime

# Labels for metadata
LABEL org.opencontainers.image.title="LLM-Latency-Lens"
LABEL org.opencontainers.image.description="Enterprise-grade LLM performance profiler - Phase 2 Operational Intelligence"
LABEL org.opencontainers.image.vendor="LLM DevOps Team"
LABEL org.opencontainers.image.licenses="Apache-2.0"

# Set working directory
WORKDIR /app

# Copy binary from builder
COPY --from=builder --chown=nonroot:nonroot /app/target/release/llm-latency-lens /usr/local/bin/llm-latency-lens

# Use non-root user (distroless provides 'nonroot' user with UID 65532)
USER nonroot:nonroot

# Expose Cloud Run HTTP port and Prometheus metrics port
EXPOSE 8080 9090

# Set entrypoint
ENTRYPOINT ["/usr/local/bin/llm-latency-lens"]

# Default command - serve mode for Cloud Run
CMD ["serve", "--port", "8080"]
