# Rust Local RAG + OpenRouter

MVP para búsqueda semántica privada con Rust, Qdrant y embeddings locales en CPU. OpenRouter se usa únicamente para redactar la respuesta final, cuando se configura `OPENROUTER_API_KEY`.

## Qué hace

- Lee documentos desde `./data/documents`.
- Soporta `.txt`, `.md` y `.pdf`.
- Extrae texto de PDF usando `pdftotext` dentro del contenedor.
- Divide documentos grandes en fragmentos.
- Genera embeddings locales con `fastembed`.
- Guarda vectores en Qdrant local.
- Detecta cambios por hash SHA-256.
- Reindexa solo archivos nuevos o modificados.
- Elimina del índice archivos borrados de la carpeta.
- Permite chatear desde una interfaz web simple.
- Puede funcionar sin OpenRouter en modo privado.

## Estructura

```text
rag_rust_openrouter/
├── docker-compose.yml
├── Dockerfile
├── Cargo.toml
├── .env.example
├── README.md
├── src/
│   └── main.rs
└── data/
    ├── documents/
    │   └── README.md
    └── state/
```

## Uso rápido

```bash
cp .env.example .env
```

Opcionalmente edite `.env` y coloque:

```bash
OPENROUTER_API_KEY=sk-or-xxxxxxxx
OPENROUTER_MODEL=openai/gpt-5.2
OPENROUTER_MAX_TOKENS=4096
```

`OPENROUTER_MAX_TOKENS` controla la longitud máxima de la respuesta del LLM. El valor por defecto es 4096 tokens (~3000 palabras). Increméntelo si necesita informes más extensos.

Coloque sus documentos en:

```bash
./data/documents
```

Levante el sistema:

```bash
docker compose up --build
```

Abra:

```text
http://localhost:8080
```

Qdrant queda disponible solo en localhost:

```text
http://localhost:6333/dashboard
```

## Modo privado

Si `OPENROUTER_API_KEY` está vacío, no se llama al LLM externo. La aplicación solo devuelve los fragmentos relevantes encontrados localmente.

## Actualización periódica

La variable `INDEX_INTERVAL_SECONDS` controla cada cuánto se revisa la carpeta de documentos. Por defecto es cada 3600 segundos.

```bash
INDEX_INTERVAL_SECONDS=3600
```

## Recomendación inicial

Para un documento grande, primero pruebe con un PDF o TXT. Cuando funcione bien, agregue más documentos. El primer arranque puede tardar porque el modelo de embeddings local se descarga y queda cacheado dentro del volumen `./data`/contenedor.

## Seguridad básica

- La documentación no se envía a OpenRouter durante la indexación.
- OpenRouter solo recibe los fragmentos recuperados para responder, si `OPENROUTER_API_KEY` está configurada y el usuario marca la opción de usar LLM.
- Qdrant y la app están publicados en `127.0.0.1`, no en toda la red.

