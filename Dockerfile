FROM rust:1-slim-bookworm AS builder

# Solo necesita lo mínimo: rust-slim ya incluye cc/gcc para compilar rusqlite (bundled)
# reqwest usa rustls-tls (sin OpenSSL) → cero dependencias extra

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release

FROM debian:bookworm-slim

# Solo poppler-utils para extraer texto de PDFs con pdftotext
# ca-certificates para HTTPS hacia OpenRouter
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/target/release/rag-rust-openrouter /usr/local/bin/rag-rust-openrouter

ENV APP_HOST=0.0.0.0
ENV APP_PORT=8080
ENV DOCS_DIR=/app/data/documents
ENV STATE_DIR=/app/data/state

EXPOSE 8080
CMD ["rag-rust-openrouter"]
