# Rust Local RAG + OpenRouter

Sistema RAG (Retrieval-Augmented Generation) híbrido con Rust, SQLite FTS5/BM25, embeddings opcionales vía OpenRouter y almacenamiento vectorial en Qdrant.

## Qué hace

- **Búsqueda léxica local**: SQLite FTS5 con tokenizador `unicode61 remove_diacritics 2` para español.
- **Expansión de términos**: sinónimos del dominio (salud, legal, contratación, etc.).
- **Búsqueda iterativa**: consultas amplias (OR), focalizadas (términos principales) y por pares de palabras cercanas.
- **Fallback fuzzy**: detección de palabras mal escritas o cortadas con Levenshtein.
- **Fragmentación inteligente**: chunks de ~300 palabras con overlapping de 50, preservando el número de página de origen en PDFs.
- **Expansión de vecinos**: agrega fragmentos anterior/siguiente de los mejores resultados para dar más contexto al LLM.
- **Re-ranking de dos etapas**: puntuación léxica mejorada + reordenamiento fino por densidad de términos e intención.
- **Embeddings opcionales**: chunks indexados como vectores en Qdrant para recuperación semántica.
- **Búsqueda híbrida**: FTS5/BM25 como base, Qdrant como refuerzo semántico.
- **Detección de intención**: clasifica la pregunta (definición, valor/precio, requisito, procedimiento, comparación, normativa, lista, resumen, general) y ajusta la ponderación de términos.
- **Reformulación de preguntas con LLM** (opcional): si `LLM_QUERY_EXPANSION=true`, genera variantes de búsqueda vía OpenRouter antes de consultar. Resultados cacheados en SQLite.
- **Caché FAQ**: respuestas exitosas se cachean para evitar llamadas repetidas al LLM.
- **Streaming SSE**: respuestas del LLM en tiempo real vía Server-Sent Events.
- **Administración web**: subir documentos, pegar texto, gestionar usuarios, ver estado del índice.

## Estructura

```text
rag_rust_openrouter/
├── docker-compose.yml
├── Dockerfile
├── Cargo.toml
├── Cargo.lock
├── .env.example
├── README.md
├── src/
│   └── main.rs
└── data/
    ├── documents/
    │   └── (sus archivos .pdf, .txt, .md aquí)
    └── state/
        └── manifest.sqlite (índice y cachés)
```

## Requisitos

- Docker y Docker Compose.
- Sin GPU ni modelos locales: todo corre en CPU.

## Uso rápido

```bash
cp .env.example .env
```

Edite `.env` y coloque al menos:

```bash
OPENROUTER_API_KEY=sk-or-xxxxxxxx
```

(Opcional) Configure el modelo:

```bash
OPENROUTER_MODEL=anthropic/claude-sonnet-4
```

Coloque sus documentos en:

```bash
./data/documents
```

Levante el sistema:

```bash
docker compose up --build
```

Abra en el navegador:

```text
http://localhost:8080
```

## Variables de entorno

| Variable | Default | Descripción |
|---|---|---|
| `OPENROUTER_API_KEY` | (vacío) | API key de OpenRouter. Vacío = modo privado |
| `OPENROUTER_MODEL` | `openai/gpt-5.2` | Modelo para respuestas y expansión |
| `OPENROUTER_MAX_TOKENS` | `4096` | Máximo de tokens en respuestas del LLM |
| `OPENROUTER_EMBEDDING_MODEL` | `openai/text-embedding-3-small` | Modelo para embeddings |
| `EMBEDDINGS_ENABLED` | `true` | Activa embeddings híbridos |
| `QDRANT_URL` | `http://127.0.0.1:6333` | URL del servicio Qdrant |
| `QDRANT_COLLECTION` | `rag_chunks` | Colección base en Qdrant |
| `QDRANT_ENABLED` | `true` | Activa uso de Qdrant |
| `EMBEDDING_DIMENSION` | `1536` | Dimensión esperada del vector |
| `SEMANTIC_TOP_K` | `24` | Fragmentos semánticos máximos por consulta |
| `HYBRID_LEXICAL_WEIGHT` | `1.0` | Peso del ranking léxico |
| `HYBRID_SEMANTIC_WEIGHT` | `1.0` | Peso del ranking semántico |
| `EMBEDDING_BATCH_SIZE` | `16` | Tamaño de lote para embeddings |
| `LLM_QUERY_EXPANSION` | `false` | Si `true`, reformula la pregunta con LLM antes de buscar |
| `DOCS_DIR` | `/app/data/documents` | Carpeta de documentos |
| `STATE_DIR` | `/app/data/state` | Carpeta de estado (índice SQLite) |
| `INDEX_INTERVAL_SECONDS` | `3600` | Intervalo de reindexación automática |
| `DEFAULT_TOP_K` | `6` | Fragmentos a recuperar por consulta |
| `ANSWER_MAX_WORDS` | `100` | Máximo de palabras en respuesta LLM |
| `ADMIN_USERNAME` | `admin` | Usuario administrador inicial |
| `ADMIN_PASSWORD` | `admin123` | Contraseña del administrador |
| `AUTH_SALT` | `cambie-esta-sal-local` | Salt para hash de contraseñas |
| `SESSION_HOURS` | `24` | Duración de sesión en horas |

## Motor de búsqueda

El sistema usa búsqueda léxica con SQLite FTS5/BM25 y, opcionalmente, recuperación semántica con embeddings en Qdrant:

- No usa fastembed.
- Si `EMBEDDINGS_ENABLED=false`, la semántica se desactiva y queda el modo léxico puro.
- La tokenización está optimizada para español (unicode61 con eliminación de diacríticos).
- Los sinónimos cubren dominios: salud, legal, contratación, interoperabilidad, etc.
- La fragmentación preserva el número de página original de los PDFs.

## Modo privado

Si `OPENROUTER_API_KEY` está vacío, el sistema funciona completamente en modo local:
- La búsqueda y respuesta se hacen sin llamadas externas.
- OpenRouter no recibe ningún dato.

Si `OPENROUTER_API_KEY` está configurada pero `LLM_QUERY_EXPANSION=false`:
- OpenRouter solo recibe los fragmentos recuperados para redactar la respuesta final.

## Formato de streaming

Las respuestas se transmiten vía SSE (Server-Sent Events) por `POST /api/chat-stream`:

```json
{"kind":"variants","payload":{"variants":["consulta externa tarifario","..."],"intent":"ValuePrice"}}
{"kind":"fragments","payload":[{"source":"tarifario.pdf","page_number":35,...}]}
{"kind":"answer","payload":"## Resumen Ejecutivo..."}
```

## Seguridad

- Los documentos nunca se envían a OpenRouter durante la indexación.
- OpenRouter solo recibe fragmentos recuperados + pregunta, si la API key está configurada.
- El servidor web escucha en `127.0.0.1:8080` por defecto.
- Autenticación mediante contraseñas con hash SHA-256 + salt.
- Sesiones con tokens UUID y expiración configurabil.
- Roles de usuario: `admin` y `user`.
