# Multi-stage build for optimal size and security
# Stage 1: Cache dependencies with cargo-chef
FROM rust:1.92-slim-bookworm AS chef

# Install cargo-chef for dependency caching
RUN cargo install cargo-chef --locked

WORKDIR /app


# Stage 2: Compute dependency recipe
FROM chef AS planner

# Copy all source files
COPY . .

# Generate recipe file for dependency caching
RUN cargo chef prepare --recipe-path recipe.json


# Stage 3: Build dependencies (cached layer)
FROM chef AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

COPY --from=planner /app/recipe.json recipe.json

# Build dependencies - this is the caching Docker layer
RUN cargo chef cook --release --recipe-path recipe.json


# Stage 4: Build application
FROM builder AS app-builder

# Copy source code
COPY . .

# Build release binary with optimizations
RUN cargo build --release --locked

# Verify binary exists and is executable
RUN test -f /app/target/release/llm-latency-lens && \
    chmod +x /app/target/release/llm-latency-lens


# Stage 5: Runtime image (distroless for minimal attack surface)
FROM gcr.io/distroless/cc-debian12:nonroot AS runtime

# Labels for metadata
LABEL org.opencontainers.image.title="LLM-Latency-Lens"
LABEL org.opencontainers.image.description="Enterprise-grade command-line profiler for LLM performance measurement"
LABEL org.opencontainers.image.vendor="LLM DevOps Team"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.source="https://github.com/llm-devops/llm-latency-lens"
LABEL org.opencontainers.image.documentation="https://github.com/llm-devops/llm-latency-lens/blob/main/README.md"

# Set working directory
WORKDIR /app

# Copy binary from builder
COPY --from=app-builder --chown=nonroot:nonroot /app/target/release/llm-latency-lens /usr/local/bin/llm-latency-lens

# Use non-root user (distroless provides 'nonroot' user with UID 65532)
USER nonroot:nonroot

# Expose Prometheus metrics port (if applicable)
EXPOSE 9090

# Health check (adjust based on your actual health endpoint)
# Note: distroless doesn't have shell, so we can't use traditional healthcheck
# This would need to be handled by the orchestrator (Kubernetes, Docker Compose, etc.)

# Set entrypoint
ENTRYPOINT ["/usr/local/bin/llm-latency-lens"]

# Default command (can be overridden)
CMD ["--help"]
