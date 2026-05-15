FROM rust:1-trixie AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    clang \
    cmake \
    protobuf-compiler \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY Cargo.toml ./
COPY src ./src
RUN cargo build --release

FROM debian:trixie-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    poppler-utils \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/target/release/rag-rust-openrouter /usr/local/bin/rag-rust-openrouter

ENV APP_HOST=0.0.0.0
ENV APP_PORT=8080
ENV DOCS_DIR=/app/data/documents
ENV STATE_DIR=/app/data/state

EXPOSE 8080
CMD ["rag-rust-openrouter"]
