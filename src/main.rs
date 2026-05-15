use anyhow::{anyhow, Context, Result};
use axum::{
    extract::{DefaultBodyLimit, Multipart, State},
    http::{header, HeaderMap, HeaderValue, StatusCode},
    response::{
        sse::{Event, Sse},
        Html, IntoResponse, Response,
    },
    routing::{get, post},
    Json, Router,
};
use chrono::Utc;
use encoding_rs::WINDOWS_1252;
use reqwest::Client;
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet},
    convert::Infallible,
    env, fs,
    path::{Path, PathBuf},
    process::Command,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::{net::TcpListener, sync::mpsc};
use tokio_stream::{wrappers::ReceiverStream, StreamExt};
use tower_http::{cors::CorsLayer, trace::TraceLayer};
use tracing::{error, info, warn};
use uuid::Uuid;
use walkdir::WalkDir;

const CHUNK_WORDS: usize = 300;
const CHUNK_OVERLAP_WORDS: usize = 50;
const INDEX_PIPELINE_VERSION: &str = "2026-05-15-lexical-bm25-v1";
const COOKIE_NAME: &str = "rag_session";

#[derive(Clone)]
struct AppState {
    config: Arc<Config>,
    http: Client,
    index_lock: Arc<tokio::sync::Mutex<()>>,
}

#[derive(Clone, Debug)]
struct Config {
    docs_dir: PathBuf,
    state_dir: PathBuf,
    openrouter_api_key: Option<String>,
    openrouter_model: String,
    openrouter_max_tokens: u32,
    index_interval_seconds: u64,
    default_top_k: usize,
    app_host: String,
    app_port: u16,
    admin_username: String,
    admin_password: String,
    auth_salt: String,
    session_hours: i64,
}

impl Config {
    fn from_env() -> Self {
        let docs_dir = env::var("DOCS_DIR").unwrap_or_else(|_| "/app/data/documents".to_string());
        let state_dir = env::var("STATE_DIR").unwrap_or_else(|_| "/app/data/state".to_string());
        let openrouter_api_key = env::var("OPENROUTER_API_KEY")
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());

        let openrouter_model = env::var("OPENROUTER_MODEL").unwrap_or_else(|_| "openai/gpt-5.2".to_string());
        let openrouter_max_tokens = env::var("OPENROUTER_MAX_TOKENS").ok().and_then(|v| v.parse().ok()).unwrap_or(4096);
        let index_interval_seconds = env::var("INDEX_INTERVAL_SECONDS").ok().and_then(|v| v.parse().ok()).unwrap_or(3600);
        let default_top_k = env::var("DEFAULT_TOP_K").ok().and_then(|v| v.parse().ok()).unwrap_or(5);
        let app_host = env::var("APP_HOST").unwrap_or_else(|_| "0.0.0.0".to_string());
        let app_port = env::var("APP_PORT").ok().and_then(|v| v.parse().ok()).unwrap_or(8080);

        let admin_username = env::var("ADMIN_USERNAME").unwrap_or_else(|_| "admin".to_string());
        let admin_password = env::var("ADMIN_PASSWORD").unwrap_or_else(|_| "admin123".to_string());
        let auth_salt = env::var("AUTH_SALT").unwrap_or_else(|_| "cambie-esta-sal-local".to_string());
        let session_hours = env::var("SESSION_HOURS").ok().and_then(|v| v.parse().ok()).unwrap_or(24);

        Self {
            docs_dir: PathBuf::from(docs_dir),
            state_dir: PathBuf::from(state_dir),
            openrouter_api_key,
            openrouter_model,
            openrouter_max_tokens,
            index_interval_seconds,
            default_top_k,
            app_host,
            app_port,
            admin_username,
            admin_password,
            auth_salt,
            session_hours,
        }
    }

    fn db_path(&self) -> PathBuf {
        self.state_dir.join("manifest.sqlite")
    }
}

#[derive(Debug, Serialize)]
struct IndexSummary {
    scanned_files: usize,
    indexed_files: usize,
    skipped_files: usize,
    deleted_files: usize,
    chunks_indexed: usize,
    errors: Vec<String>,
}

impl Default for IndexSummary {
    fn default() -> Self {
        Self {
            scanned_files: 0,
            indexed_files: 0,
            skipped_files: 0,
            deleted_files: 0,
            chunks_indexed: 0,
            errors: vec![],
        }
    }
}

#[derive(Debug, Deserialize)]
struct LoginRequest {
    username: String,
    password: String,
}

#[derive(Debug, Serialize)]
struct LoginResponse {
    ok: bool,
    user: AuthUser,
}

#[derive(Debug, Serialize, Clone)]
struct AuthUser {
    username: String,
    role: String,
    is_active: bool,
    blocked_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CreateUserRequest {
    username: String,
    password: String,
    role: Option<String>,
}

#[derive(Debug, Deserialize)]
struct BlockUserRequest {
    username: String,
    blocked: bool,
    reason: Option<String>,
}

#[derive(Debug, Serialize)]
struct UserRow {
    username: String,
    role: String,
    is_active: bool,
    created_at: String,
    blocked_reason: Option<String>,
}

#[derive(Debug, Serialize)]
struct UploadResponse {
    ok: bool,
    files: Vec<String>,
    message: String,
}

#[derive(Debug, Serialize)]
struct DocumentRow {
    path: String,
    indexed_at: String,
    chunk_count: i64,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ChatRequest {
    question: String,
    top_k: Option<usize>,
    use_llm: Option<bool>,
    mode: Option<String>,
    selected_sources: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
struct ChatResponse {
    answer: String,
    used_llm: bool,
    model: Option<String>,
    fragments: Vec<SearchFragment>,
}

#[derive(Debug, Serialize)]
struct StreamEvent<'a, T: Serialize> {
    kind: &'a str,
    payload: T,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct SearchFragment {
    score: f32,
    source: String,
    chunk_index: usize,
    text: String,
}

#[derive(Debug, Serialize)]
struct StatusResponse {
    docs_dir: String,
    search_engine: String,
    openrouter_enabled: bool,
    openrouter_model: String,
    index_interval_seconds: u64,
    manifest_documents: i64,
    user: AuthUser,
}

#[derive(Debug)]
struct AppError {
    status: StatusCode,
    message: String,
}

impl AppError {
    fn new(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }

    fn unauthorized() -> Self {
        Self::new(StatusCode::UNAUTHORIZED, "Debe iniciar sesión.")
    }

    fn forbidden() -> Self {
        Self::new(StatusCode::FORBIDDEN, "No tiene permisos para esta operación.")
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        error!("{}", self.message);
        (self.status, Json(json!({ "error": self.message }))).into_response()
    }
}

impl From<anyhow::Error> for AppError {
    fn from(err: anyhow::Error) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, err.to_string())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string()))
        .init();

    let config = Arc::new(Config::from_env());

    fs::create_dir_all(&config.docs_dir)?;
    fs::create_dir_all(&config.state_dir)?;

    init_db(&config)?;
    ensure_default_admin(&config)?;

    let http = Client::builder()
        .timeout(Duration::from_secs(180))
        .pool_max_idle_per_host(0)
        .build()?;

    info!("Motor RAG léxico activo: SQLite FTS5/BM25, sin embeddings y sin Qdrant.");

    let state = AppState {
        config,
        http,
        index_lock: Arc::new(tokio::sync::Mutex::new(())),
    };

    let state_for_indexer = state.clone();
    tokio::spawn(async move {
        periodic_indexer(state_for_indexer).await;
    });

    let app = Router::new()
        .route("/", get(index_html))
        .route("/api/login", post(login))
        .route("/api/logout", post(logout))
        .route("/api/me", get(me))
        .route("/api/status", get(status))
        .route("/api/chat", post(chat))
        .route("/api/chat-stream", post(chat_stream))
        .route("/api/admin/reindex", post(reindex_now))
        .route("/api/admin/upload", post(upload_documents))
        .route("/api/admin/users", get(list_users).post(create_user))
        .route("/api/admin/users/block", post(block_user))
        .route("/api/admin/documents", get(list_documents))
        .layer(DefaultBodyLimit::max(100 * 1024 * 1024))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state.clone());

    let addr = format!("{}:{}", state.config.app_host, state.config.app_port);
    let listener = TcpListener::bind(&addr).await?;

    info!("Servidor listo en http://{}", addr);

    axum::serve(listener, app).await?;

    Ok(())
}

async fn index_html() -> Html<&'static str> {
    Html(INDEX_HTML)
}

async fn login(
    State(state): State<AppState>,
    Json(req): Json<LoginRequest>,
) -> Result<Response, AppError> {
    let username = req.username.trim().to_lowercase();

    if username.is_empty() || req.password.is_empty() {
        return Err(AppError::new(
            StatusCode::BAD_REQUEST,
            "Usuario y contraseña son obligatorios.",
        ));
    }

    let conn = Connection::open(state.config.db_path()).map_err(anyhow::Error::from)?;

    let row: Option<(i64, String, String, i64, Option<String>)> = conn
        .query_row(
            "SELECT id, password_hash, role, is_active, blocked_reason FROM users WHERE username = ?1",
            params![username],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)),
        )
        .optional()
        .map_err(anyhow::Error::from)?;

    let Some((user_id, password_hash, role, is_active, blocked_reason)) = row else {
        return Err(AppError::new(
            StatusCode::UNAUTHORIZED,
            "Credenciales incorrectas.",
        ));
    };

    if is_active == 0 {
        return Err(AppError::new(
            StatusCode::FORBIDDEN,
            blocked_reason.unwrap_or_else(|| "Usuario bloqueado.".to_string()),
        ));
    }

    if password_hash != hash_password(&state.config, &req.password) {
        return Err(AppError::new(
            StatusCode::UNAUTHORIZED,
            "Credenciales incorrectas.",
        ));
    }

    let token = Uuid::new_v4().to_string();
    let now = Utc::now().timestamp();
    let expires_at = now + state.config.session_hours * 3600;

    conn.execute(
        "INSERT INTO sessions(token, user_id, created_at, expires_at) VALUES (?1, ?2, ?3, ?4)",
        params![token, user_id, now, expires_at],
    )
    .map_err(anyhow::Error::from)?;

    let user = AuthUser {
        username: username.clone(),
        role,
        is_active: true,
        blocked_reason: None,
    };

    let cookie = format!(
        "{}={}; Path=/; HttpOnly; SameSite=Lax; Max-Age={}",
        COOKIE_NAME,
        token,
        state.config.session_hours * 3600
    );

    let mut headers = HeaderMap::new();

    headers.insert(
        header::SET_COOKIE,
        HeaderValue::from_str(&cookie).map_err(|e| anyhow!(e.to_string()))?,
    );

    Ok((headers, Json(LoginResponse { ok: true, user })).into_response())
}

async fn logout(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Response, AppError> {
    if let Some(token) = read_session_cookie(&headers) {
        let conn = Connection::open(state.config.db_path()).map_err(anyhow::Error::from)?;
        let _ = conn.execute("DELETE FROM sessions WHERE token = ?1", params![token]);
    }

    let mut headers = HeaderMap::new();

    headers.insert(
        header::SET_COOKIE,
        HeaderValue::from_static("rag_session=; Path=/; HttpOnly; SameSite=Lax; Max-Age=0"),
    );

    Ok((headers, Json(json!({"ok": true}))).into_response())
}

async fn me(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<AuthUser>, AppError> {
    let user = require_user(&state, &headers)?;
    Ok(Json(user))
}

async fn status(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<StatusResponse>, AppError> {
    let user = require_user(&state, &headers)?;
    let count = count_manifest_documents(&state.config).map_err(anyhow::Error::from)?;

    Ok(Json(StatusResponse {
        docs_dir: state.config.docs_dir.display().to_string(),
        search_engine: "SQLite FTS5 / BM25 léxico local".to_string(),
        openrouter_enabled: state.config.openrouter_api_key.is_some(),
        openrouter_model: state.config.openrouter_model.clone(),
        index_interval_seconds: state.config.index_interval_seconds,
        manifest_documents: count,
        user,
    }))
}

async fn reindex_now(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<IndexSummary>, AppError> {
    require_admin(&state, &headers)?;
    let summary = run_indexer(&state).await.map_err(anyhow::Error::from)?;
    Ok(Json(summary))
}

async fn chat(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ChatRequest>,
) -> Result<Json<ChatResponse>, AppError> {
    require_user(&state, &headers)?;

    let question = req.question.trim();

    if question.is_empty() {
        return Err(AppError::new(
            StatusCode::BAD_REQUEST,
            "La pregunta no puede estar vacía.",
        ));
    }

    let (top_k, max_chars, max_tokens) = (
        req.top_k.unwrap_or(state.config.default_top_k).clamp(1, 30),
        4000usize,
        state.config.openrouter_max_tokens,
    );

    let fragments = lexical_search(&state, question, top_k)
        .await
        .map_err(anyhow::Error::from)?;
    let filtered_fragments = filter_fragments_by_sources(&fragments, req.selected_sources.as_deref());
    let context_fragments = merge_fragments_as_single_context(&filtered_fragments);

    let wants_llm = req.use_llm.unwrap_or(true);
    let can_use_llm = wants_llm && state.config.openrouter_api_key.is_some();

    if can_use_llm {
        if let Some(cached) = check_faq_cache(&state.config, question) {
            return Ok(Json(ChatResponse {
                answer: format!("[Cache] {cached}"),
                used_llm: false,
                model: None,
                fragments,
            }));
        }

        let answer = call_openrouter(&state, question, &context_fragments, max_chars, max_tokens)
            .await
            .map_err(anyhow::Error::from)?;

        let _ = store_faq_cache(&state.config, question, &answer);

        Ok(Json(ChatResponse {
                answer,
                used_llm: true,
                model: Some(state.config.openrouter_model.clone()),
                fragments: filtered_fragments,
            }))
    } else {
        let answer = build_local_response(question, &context_fragments);

        Ok(Json(ChatResponse {
            answer,
            used_llm: false,
            model: None,
            fragments: filtered_fragments,
        }))
    }
}

async fn chat_stream(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ChatRequest>,
) -> Sse<ReceiverStream<Result<Event, Infallible>>> {
    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(64);

    tokio::spawn(async move {
        async fn send_event<T: Serialize>(
            tx: &mpsc::Sender<Result<Event, Infallible>>,
            kind: &'static str,
            payload: T,
        ) {
            let data = match serde_json::to_string(&StreamEvent { kind, payload }) {
                Ok(data) => data,
                Err(e) => {
                    let fallback = json!({
                        "kind": "error",
                        "payload": format!("No se pudo serializar el evento SSE: {e}")
                    })
                    .to_string();
                    let _ = tx.send(Ok(Event::default().data(fallback))).await;
                    return;
                }
            };

            let _ = tx.send(Ok(Event::default().data(data))).await;
        }

        let _user = match require_user(&state, &headers) {
            Ok(u) => u,
            Err(e) => {
                send_event(&tx, "error", e.message).await;
                return;
            }
        };

        let question = req.question.trim().to_string();
        if question.is_empty() {
            send_event(&tx, "error", "La pregunta no puede estar vacía.").await;
            return;
        }

        let (top_k, max_chars, max_tokens) = (req.top_k.unwrap_or(state.config.default_top_k).clamp(1, 30), 4000usize, state.config.openrouter_max_tokens);

        let fragments = match lexical_search(&state, &question, top_k).await {
            Ok(f) => f,
            Err(e) => {
                send_event(&tx, "error", e.to_string()).await;
                return;
            }
        };
        let filtered_fragments =
            filter_fragments_by_sources(&fragments, req.selected_sources.as_deref());
        let context_fragments = merge_fragments_as_single_context(&filtered_fragments);

        send_event(&tx, "fragments", &filtered_fragments).await;

        let wants_llm = req.use_llm.unwrap_or(true);
        let can_use_llm = wants_llm && state.config.openrouter_api_key.is_some();

        if can_use_llm {
            if let Some(cached) = check_faq_cache(&state.config, &question) {
                send_event(&tx, "cached", cached).await;
                return;
            }

            match call_openrouter_stream(&state, &question, &context_fragments, max_chars, max_tokens).await
            {
                Ok(mut orx) => {
                    let mut full = String::new();
                    while let Some(delta) = orx.recv().await {
                        match delta {
                            Ok(chunk) => {
                                full.push_str(&chunk);
                                send_event(&tx, "answer", &full).await;
                            }
                            Err(e) => {
                                if !full.is_empty() {
                                    send_event(&tx, "answer", &full).await;
                                }
                                send_event(&tx, "error", e.to_string()).await;
                                break;
                            }
                        }
                    }
                    let _ = store_faq_cache(&state.config, &question, &full);
                }
                Err(e) => {
                    send_event(&tx, "error", e.to_string()).await;
                }
            }
        } else {
            let answer = build_local_response(&question, &context_fragments);
            send_event(&tx, "local", answer).await;
        }
    });

    Sse::new(ReceiverStream::new(rx))
}

async fn list_users(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<Vec<UserRow>>, AppError> {
    require_admin(&state, &headers)?;

    let conn = Connection::open(state.config.db_path()).map_err(anyhow::Error::from)?;

    let mut stmt = conn
        .prepare(
            "SELECT username, role, is_active, created_at, blocked_reason FROM users ORDER BY username",
        )
        .map_err(anyhow::Error::from)?;

    let rows = stmt
        .query_map([], |r| {
            Ok(UserRow {
                username: r.get(0)?,
                role: r.get(1)?,
                is_active: r.get::<_, i64>(2)? != 0,
                created_at: r.get(3)?,
                blocked_reason: r.get(4)?,
            })
        })
        .map_err(anyhow::Error::from)?;

    let mut out = Vec::new();

    for row in rows {
        out.push(row.map_err(anyhow::Error::from)?);
    }

    Ok(Json(out))
}

async fn create_user(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<CreateUserRequest>,
) -> Result<Json<UserRow>, AppError> {
    require_admin(&state, &headers)?;

    let username = req.username.trim().to_lowercase();
    let role = req.role.unwrap_or_else(|| "user".to_string());

    if username.len() < 3 || req.password.len() < 4 {
        return Err(AppError::new(
            StatusCode::BAD_REQUEST,
            "El usuario debe tener al menos 3 caracteres y la clave al menos 4.",
        ));
    }

    if role != "admin" && role != "user" {
        return Err(AppError::new(
            StatusCode::BAD_REQUEST,
            "El rol debe ser admin o user.",
        ));
    }

    let conn = Connection::open(state.config.db_path()).map_err(anyhow::Error::from)?;
    let now = Utc::now().to_rfc3339();

    conn.execute(
        "INSERT INTO users(username, password_hash, role, is_active, created_at, blocked_reason) VALUES (?1, ?2, ?3, 1, ?4, NULL)",
        params![username, hash_password(&state.config, &req.password), role, now],
    )
    .map_err(|e| {
        AppError::new(
            StatusCode::BAD_REQUEST,
            format!("No se pudo crear el usuario: {e}"),
        )
    })?;

    Ok(Json(UserRow {
        username,
        role,
        is_active: true,
        created_at: now,
        blocked_reason: None,
    }))
}

async fn block_user(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<BlockUserRequest>,
) -> Result<Json<Value>, AppError> {
    let current = require_admin(&state, &headers)?;
    let username = req.username.trim().to_lowercase();

    if username == current.username {
        return Err(AppError::new(
            StatusCode::BAD_REQUEST,
            "El administrador no puede bloquear su propia sesión.",
        ));
    }

    let conn = Connection::open(state.config.db_path()).map_err(anyhow::Error::from)?;
    let active = if req.blocked { 0 } else { 1 };

    let reason = if req.blocked {
        req.reason
            .or_else(|| Some("Bloqueado por administración.".to_string()))
    } else {
        None
    };

    conn.execute(
        "UPDATE users SET is_active = ?1, blocked_reason = ?2 WHERE username = ?3",
        params![active, reason, username],
    )
    .map_err(anyhow::Error::from)?;

    if req.blocked {
        conn.execute(
            "DELETE FROM sessions WHERE user_id IN (SELECT id FROM users WHERE username = ?1)",
            params![username],
        )
        .map_err(anyhow::Error::from)?;
    }

    Ok(Json(json!({"ok": true})))
}

async fn upload_documents(
    State(state): State<AppState>,
    headers: HeaderMap,
    mut multipart: Multipart,
) -> Result<Json<UploadResponse>, AppError> {
    require_admin(&state, &headers)?;

    let mut saved = Vec::new();

    fs::create_dir_all(&state.config.docs_dir).map_err(anyhow::Error::from)?;

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| AppError::new(StatusCode::BAD_REQUEST, e.to_string()))?
    {
        let Some(file_name) = field.file_name().map(|s| s.to_string()) else {
            continue;
        };

        let safe = sanitize_filename(&file_name);

        if !is_supported_filename(&safe) {
            return Err(AppError::new(
                StatusCode::BAD_REQUEST,
                format!("Formato no permitido: {safe}. Use .txt, .md o .pdf"),
            ));
        }

        let bytes = field
            .bytes()
            .await
            .map_err(|e| AppError::new(StatusCode::BAD_REQUEST, e.to_string()))?;

        if bytes.is_empty() {
            continue;
        }

        if bytes.len() > 80 * 1024 * 1024 {
            return Err(AppError::new(
                StatusCode::BAD_REQUEST,
                "El archivo supera 80 MB.",
            ));
        }

        let dest = state.config.docs_dir.join(&safe);
        fs::write(&dest, &bytes).map_err(anyhow::Error::from)?;
        saved.push(safe);
    }

    if saved.is_empty() {
        return Err(AppError::new(
            StatusCode::BAD_REQUEST,
            "No se recibió ningún archivo válido.",
        ));
    }

    let state_clone = state.clone();

    tokio::spawn(async move {
        match run_indexer(&state_clone).await {
            Ok(summary) => info!("Indexación posterior a carga web: {:?}", summary),
            Err(e) => error!("Error indexando después de carga: {e:#}"),
        }
    });

    Ok(Json(UploadResponse {
        ok: true,
        files: saved,
        message: "Archivo(s) subido(s). La indexación se ejecuta sin detener el servicio.".to_string(),
    }))
}

async fn list_documents(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<Vec<DocumentRow>>, AppError> {
    require_admin(&state, &headers)?;

    let conn = Connection::open(state.config.db_path()).map_err(anyhow::Error::from)?;

    let mut stmt = conn
        .prepare("SELECT path, indexed_at, chunk_count FROM documents ORDER BY indexed_at DESC")
        .map_err(anyhow::Error::from)?;

    let rows = stmt
        .query_map([], |r| {
            Ok(DocumentRow {
                path: r.get(0)?,
                indexed_at: r.get(1)?,
                chunk_count: r.get(2)?,
            })
        })
        .map_err(anyhow::Error::from)?;

    let mut out = Vec::new();

    for row in rows {
        out.push(row.map_err(anyhow::Error::from)?);
    }

    Ok(Json(out))
}

fn require_user(state: &AppState, headers: &HeaderMap) -> Result<AuthUser, AppError> {
    get_user_from_headers(state, headers)?.ok_or_else(AppError::unauthorized)
}

fn require_admin(state: &AppState, headers: &HeaderMap) -> Result<AuthUser, AppError> {
    let user = require_user(state, headers)?;

    if user.role == "admin" {
        Ok(user)
    } else {
        Err(AppError::forbidden())
    }
}

fn get_user_from_headers(
    state: &AppState,
    headers: &HeaderMap,
) -> Result<Option<AuthUser>, AppError> {
    let Some(token) = read_session_cookie(headers) else {
        return Ok(None);
    };

    let now = Utc::now().timestamp();
    let conn = Connection::open(state.config.db_path()).map_err(anyhow::Error::from)?;

    let row = conn
        .query_row(
            r#"
            SELECT u.username, u.role, u.is_active, u.blocked_reason
            FROM sessions s
            JOIN users u ON u.id = s.user_id
            WHERE s.token = ?1 AND s.expires_at > ?2
            "#,
            params![token, now],
            |r| {
                Ok(AuthUser {
                    username: r.get(0)?,
                    role: r.get(1)?,
                    is_active: r.get::<_, i64>(2)? != 0,
                    blocked_reason: r.get(3)?,
                })
            },
        )
        .optional()
        .map_err(anyhow::Error::from)?;

    if let Some(user) = row {
        if user.is_active {
            Ok(Some(user))
        } else {
            Ok(None)
        }
    } else {
        Ok(None)
    }
}

fn read_session_cookie(headers: &HeaderMap) -> Option<String> {
    let cookie = headers.get(header::COOKIE)?.to_str().ok()?;

    for part in cookie.split(';') {
        let trimmed = part.trim();

        if let Some(value) = trimmed.strip_prefix(&format!("{}=", COOKIE_NAME)) {
            if !value.trim().is_empty() {
                return Some(value.trim().to_string());
            }
        }
    }

    None
}

async fn periodic_indexer(state: AppState) {
    loop {
        match run_indexer(&state).await {
            Ok(summary) => info!("Indexación periódica: {:?}", summary),
            Err(e) => error!("Error en indexación periódica: {e:#}"),
        }

        tokio::time::sleep(Duration::from_secs(state.config.index_interval_seconds)).await;
    }
}

async fn run_indexer(state: &AppState) -> Result<IndexSummary> {
    let _guard = state.index_lock.lock().await;

    let mut summary = IndexSummary::default();
    let mut seen_paths = HashSet::new();

    let entries = WalkDir::new(&state.config.docs_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .filter(|e| is_supported_file(e.path()))
        .collect::<Vec<_>>();

    for entry in entries {
        summary.scanned_files += 1;

        let path = entry.path().to_path_buf();
        let rel = relative_source(&state.config.docs_dir, &path);

        seen_paths.insert(rel.clone());

        match index_one_file(state, &path, &rel).await {
            Ok(IndexFileOutcome::Indexed(chunks)) => {
                summary.indexed_files += 1;
                summary.chunks_indexed += chunks;
            }
            Ok(IndexFileOutcome::Skipped) => {
                summary.skipped_files += 1;
            }
            Err(e) => {
                let msg = format!("{}: {e:#}", rel);
                warn!("{msg}");
                summary.errors.push(msg);
            }
        }
    }

    let known = list_manifest_paths(&state.config)?;

    for old_path in known {
        if !seen_paths.contains(&old_path) {
            delete_chunks_by_source(&state.config, &old_path)?;
            remove_manifest_document(&state.config, &old_path)?;
            summary.deleted_files += 1;
        }
    }

    Ok(summary)
}

enum IndexFileOutcome {
    Indexed(usize),
    Skipped,
}

async fn index_one_file(
    state: &AppState,
    path: &Path,
    rel_source: &str,
) -> Result<IndexFileOutcome> {
    let bytes = fs::read(path).with_context(|| format!("No se pudo leer {}", path.display()))?;
    let hash = file_hash_with_pipeline(&bytes);
    let modified = file_modified_epoch(path)?;

    if manifest_has_same_hash(&state.config, rel_source, &hash)? {
        return Ok(IndexFileOutcome::Skipped);
    }

    let text = extract_text(path, &bytes)?;
    let chunks = chunk_text(&text, CHUNK_WORDS, CHUNK_OVERLAP_WORDS);

    // Reindexación segura por documento: primero elimina solo los fragmentos de la fuente actual.
    // No toca otros documentos ni borra archivos físicos.
    delete_chunks_by_source(&state.config, rel_source)?;

    let mut total = 0usize;
    for (chunk_index, chunk) in chunks.iter().enumerate() {
        if chunk.trim().is_empty() {
            continue;
        }

        insert_lexical_chunk(&state.config, rel_source, chunk_index, chunk, &hash)?;
        total += 1;
    }

    upsert_manifest_document(&state.config, rel_source, &hash, modified, total)?;

    Ok(IndexFileOutcome::Indexed(total))
}

fn extract_text(path: &Path, bytes: &[u8]) -> Result<String> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "txt" | "md" => Ok(decode_text(bytes)),
        "pdf" => extract_pdf_text(path),
        _ => Err(anyhow!("Formato no soportado: {}", ext)),
    }
}

fn extract_pdf_text(path: &Path) -> Result<String> {
    let output = Command::new("pdftotext")
        .arg("-layout")
        .arg(path)
        .arg("-")
        .output()
        .with_context(|| "No se pudo ejecutar pdftotext. Verifique poppler-utils en el contenedor.")?;

    if !output.status.success() {
        return Err(anyhow!("pdftotext falló para {}", path.display()));
    }

    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

fn decode_text(bytes: &[u8]) -> String {
    match String::from_utf8(bytes.to_vec()) {
        Ok(s) => s,
        Err(_) => {
            let (cow, _, _) = WINDOWS_1252.decode(bytes);
            cow.to_string()
        }
    }
}

fn chunk_text(text: &str, chunk_words: usize, overlap_words: usize) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();

    if words.is_empty() {
        return vec![];
    }

    if words.len() <= chunk_words {
        return vec![words.join(" ")];
    }

    let mut chunks = Vec::new();
    let mut start = 0usize;

    while start < words.len() {
        let end = (start + chunk_words).min(words.len());
        let chunk = words[start..end].join(" ");

        if chunk.trim().len() > 40 {
            chunks.push(chunk);
        }

        if end == words.len() {
            break;
        }

        start = end.saturating_sub(overlap_words);
    }

    chunks
}

async fn lexical_search(
    state: &AppState,
    question: &str,
    top_k: usize,
) -> Result<Vec<SearchFragment>> {
    let query_terms = expand_query_terms(question);

    if query_terms.is_empty() {
        return Ok(Vec::new());
    }

    let conn = Connection::open(state.config.db_path())?;
    let mut fragments = Vec::new();

    // ===== INICIO CAMBIO SEGURO PRODUCCIÓN =====
    // Recuperación iterativa léxica:
    // 1) búsqueda amplia OR con términos originales + sinónimos,
    // 2) búsqueda focalizada con términos principales,
    // 3) fallback por escaneo léxico/fuzzy,
    // 4) expansión con fragmentos vecinos para dar más contexto al LLM.
    for fts_query in build_iterative_fts_queries(question, &query_terms) {
        if fts_query.trim().is_empty() {
            continue;
        }

        let mut found = search_fts_fragments(&conn, &fts_query, (top_k * 6).max(20))?;
        fragments.append(&mut found);
    }

    let mut reranked = rerank_fragments(question, &query_terms, fragments);

    // Aunque FTS5 encuentre resultados, se añade una pasada fallback liviana.
    // Esto mejora palabras mal digitadas, cortadas o no coincidentes exactamente.
    let fallback = lexical_scan_fallback(&conn, question, &query_terms, top_k * 5)?;
    reranked.extend(fallback);
    reranked = dedup_and_sort_fragments(reranked);

    // Agrega fragmentos anterior/siguiente de los mejores resultados.
    // Esto ayuda a que el informe no quede "flaco" cuando la respuesta está repartida
    // en párrafos vecinos del mismo documento.
    let mut expanded = expand_with_neighbor_chunks(&conn, reranked, 1, top_k)?;
    expanded = dedup_and_sort_fragments(expanded);

    // Se devuelve más contexto que el top_k original, pero con límite conservador.
    // Sigue siendo liviano y no usa embeddings.
    let final_limit = (top_k * 2).clamp(top_k, 30);
    expanded.truncate(final_limit);
    Ok(expanded)
    // ===== FIN CAMBIO SEGURO PRODUCCIÓN =====
}

fn search_fts_fragments(
    conn: &Connection,
    fts_query: &str,
    limit: usize,
) -> Result<Vec<SearchFragment>> {
    let sql = r#"
        SELECT source, chunk_index, text, bm25(document_chunks_fts) AS rank
        FROM document_chunks_fts
        WHERE document_chunks_fts MATCH ?1
        ORDER BY rank
        LIMIT ?2
    "#;

    let mut stmt = match conn.prepare(sql) {
        Ok(stmt) => stmt,
        Err(e) => {
            warn!("No se pudo preparar búsqueda FTS5; se omitirá esta pasada: {e}");
            return Ok(Vec::new());
        }
    };

    let rows = match stmt.query_map(params![fts_query, limit as i64], |r| {
        let rank: f64 = r.get(3)?;
        Ok(SearchFragment {
            // FTS5 bm25 devuelve valores menores para mejores coincidencias.
            score: (1.0 / (1.0 + rank.abs())) as f32,
            source: r.get(0)?,
            chunk_index: r.get::<_, i64>(1)? as usize,
            text: r.get(2)?,
        })
    }) {
        Ok(rows) => rows,
        Err(e) => {
            warn!("Búsqueda FTS5 falló; se omitirá esta pasada: {e}");
            return Ok(Vec::new());
        }
    };

    let mut fragments = Vec::new();
    for row in rows {
        fragments.push(row?);
    }

    Ok(fragments)
}

fn build_iterative_fts_queries(question: &str, terms: &[String]) -> Vec<String> {
    let mut queries = Vec::new();

    // Pasada amplia: términos originales + sinónimos.
    let broad = build_fts_query(terms);
    if !broad.trim().is_empty() {
        queries.push(broad);
    }

    // Pasada focalizada: solo términos principales de la pregunta original.
    let mut primary_terms = tokenize_normalized(question)
        .into_iter()
        .filter(|t| t.len() >= 4 && !is_stopword(t))
        .collect::<Vec<_>>();
    primary_terms.sort();
    primary_terms.dedup();

    let focused = build_fts_query(&primary_terms);
    if !focused.trim().is_empty() {
        queries.push(focused);
    }

    // Pasada por pares de palabras cercanas: mejora preguntas con frases como
    // "proteccion datos", "sistema hospitalario", "pistas auditoria".
    for pair in primary_terms.windows(2).take(8) {
        if pair.len() == 2 {
            let q = format!("{}* AND {}*", normalize_token(&pair[0]), normalize_token(&pair[1]));
            queries.push(q);
        }
    }

    queries.sort();
    queries.dedup();
    queries
}

fn expand_with_neighbor_chunks(
    conn: &Connection,
    fragments: Vec<SearchFragment>,
    window: usize,
    seed_limit: usize,
) -> Result<Vec<SearchFragment>> {
    let mut expanded = fragments.clone();

    for base in fragments.iter().take(seed_limit) {
        for offset in 1..=window {
            if let Some(prev_idx) = base.chunk_index.checked_sub(offset) {
                if let Some(mut prev) = load_chunk(conn, &base.source, prev_idx)? {
                    prev.score = (base.score * 0.92).max(0.01);
                    expanded.push(prev);
                }
            }

            let next_idx = base.chunk_index + offset;
            if let Some(mut next) = load_chunk(conn, &base.source, next_idx)? {
                next.score = (base.score * 0.90).max(0.01);
                expanded.push(next);
            }
        }
    }

    Ok(expanded)
}

fn load_chunk(
    conn: &Connection,
    source: &str,
    chunk_index: usize,
) -> Result<Option<SearchFragment>> {
    let found = conn
        .query_row(
            "SELECT source, chunk_index, text FROM document_chunks WHERE source = ?1 AND chunk_index = ?2",
            params![source, chunk_index as i64],
            |r| {
                Ok(SearchFragment {
                    score: 0.0,
                    source: r.get(0)?,
                    chunk_index: r.get::<_, i64>(1)? as usize,
                    text: r.get(2)?,
                })
            },
        )
        .optional()?;

    Ok(found)
}

fn insert_lexical_chunk(
    config: &Config,
    source: &str,
    chunk_index: usize,
    text: &str,
    hash: &str,
) -> Result<()> {
    let conn = Connection::open(config.db_path())?;
    let now = Utc::now().to_rfc3339();

    conn.execute(
        r#"
        INSERT INTO document_chunks(source, chunk_index, text, hash, indexed_at)
        VALUES (?1, ?2, ?3, ?4, ?5)
        ON CONFLICT(source, chunk_index) DO UPDATE SET
            text = excluded.text,
            hash = excluded.hash,
            indexed_at = excluded.indexed_at
        "#,
        params![source, chunk_index as i64, text, hash, now],
    )?;

    conn.execute(
        "INSERT INTO document_chunks_fts(source, chunk_index, text) VALUES (?1, ?2, ?3)",
        params![source, chunk_index as i64, text],
    )?;

    Ok(())
}

fn delete_chunks_by_source(config: &Config, source: &str) -> Result<()> {
    let conn = Connection::open(config.db_path())?;
    conn.execute("DELETE FROM document_chunks WHERE source = ?1", params![source])?;
    conn.execute("DELETE FROM document_chunks_fts WHERE source = ?1", params![source])?;
    Ok(())
}

fn build_fts_query(terms: &[String]) -> String {
    let mut parts = Vec::new();

    for term in terms.iter().take(40) {
        let clean = normalize_token(term);
        if clean.len() < 3 || is_stopword(&clean) {
            continue;
        }

        // Prefix search: ayuda con palabras cortadas como "jurispru" o "hiperten".
        parts.push(format!("{}*", clean));
    }

    parts.sort();
    parts.dedup();
    parts.join(" OR ")
}

fn expand_query_terms(question: &str) -> Vec<String> {
    let mut terms = tokenize_normalized(question)
        .into_iter()
        .filter(|t| t.len() >= 3 && !is_stopword(t))
        .collect::<Vec<_>>();

    let synonyms = domain_synonyms();
    let original = terms.clone();

    for term in original {
        if let Some(expanded) = synonyms.get(term.as_str()) {
            for value in expanded {
                terms.push((*value).to_string());
            }
        }
    }

    terms.sort();
    terms.dedup();
    terms
}

fn domain_synonyms() -> HashMap<&'static str, Vec<&'static str>> {
    HashMap::from([
        ("seguridad", vec!["confidencialidad", "auditoria", "trazabilidad", "acceso", "permisos", "roles"]),
        ("confidencialidad", vec!["seguridad", "reserva", "proteccion", "datos"]),
        ("interoperabilidad", vec!["integracion", "hl7", "fhir", "api", "interfaces", "conexion"]),
        ("integracion", vec!["interoperabilidad", "interfaces", "conexion", "api", "hl7", "fhir"]),
        ("medico", vec!["doctor", "profesional", "salud", "especialista", "clinico"]),
        ("paciente", vec!["usuario", "afiliado", "beneficiario", "atencion"]),
        ("historia", vec!["expediente", "clinica", "registro", "paciente"]),
        ("clinica", vec!["salud", "medica", "historia", "paciente"]),
        ("legal", vec!["juridico", "normativa", "ley", "reglamento", "contrato"]),
        ("juridico", vec!["legal", "normativa", "ley", "reglamento", "derecho"]),
        ("contrato", vec!["contratacion", "convenio", "obligacion", "adjudicacion"]),
        ("demanda", vec!["accion", "reclamo", "pretension", "proceso"]),
        ("responsabilidad", vec!["culpa", "incumplimiento", "obligacion", "danos"]),
        ("encuesta", vec!["formulario", "cuestionario", "respuestas", "evaluacion"]),
        ("carrera", vec!["programa", "profesion", "area", "academica"]),
        ("practicas", vec!["pasantias", "preprofesionales", "internado", "rotacion"]),
        ("demora", vec!["espera", "tardanza", "retraso", "lento", "fila"]),
        ("atencion", vec!["servicio", "trato", "usuario", "paciente"]),
        ("recomendacion", vec!["sugerencia", "mejora", "propuesta", "accion"]),
        ("hta", vec!["hipertension", "presion", "arterial"]),
        ("epoc", vec!["pulmonar", "obstructiva", "cronica", "disnea"]),
        ("diabetes", vec!["diabetico", "glucosa", "dm2", "mellitus"]),
    ])
}

fn rerank_fragments(
    question: &str,
    terms: &[String],
    fragments: Vec<SearchFragment>,
) -> Vec<SearchFragment> {
    let query_tokens = tokenize_normalized(question);
    let scored = fragments
        .into_iter()
        .map(|mut f| {
            let lexical = lexical_score(&f.text, terms, &query_tokens);
            f.score = (f.score * 2.0) + lexical;
            f
        })
        .collect::<Vec<_>>();

    dedup_and_sort_fragments(scored)
}

fn lexical_scan_fallback(
    conn: &Connection,
    question: &str,
    terms: &[String],
    limit: usize,
) -> Result<Vec<SearchFragment>> {
    let query_tokens = tokenize_normalized(question);
    let mut stmt = conn.prepare(
        "SELECT source, chunk_index, text FROM document_chunks ORDER BY source, chunk_index",
    )?;

    let rows = stmt.query_map([], |r| {
        Ok(SearchFragment {
            score: 0.0,
            source: r.get(0)?,
            chunk_index: r.get::<_, i64>(1)? as usize,
            text: r.get(2)?,
        })
    })?;

    let mut scored = Vec::new();
    for row in rows {
        let mut fragment = row?;
        let score = lexical_score(&fragment.text, terms, &query_tokens);
        if score > 0.0 {
            fragment.score = score;
            scored.push(fragment);
        }
    }

    let mut scored = dedup_and_sort_fragments(scored);
    scored.truncate(limit);
    Ok(scored)
}

fn dedup_and_sort_fragments(mut fragments: Vec<SearchFragment>) -> Vec<SearchFragment> {
    let mut seen = HashSet::new();
    fragments.retain(|f| seen.insert((f.source.clone(), f.chunk_index)));
    fragments.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.source.cmp(&b.source))
            .then_with(|| a.chunk_index.cmp(&b.chunk_index))
    });
    fragments
}

fn lexical_score(text: &str, terms: &[String], query_tokens: &[String]) -> f32 {
    let text_tokens = tokenize_normalized(text);

    if text_tokens.is_empty() {
        return 0.0;
    }

    let token_set: HashSet<&str> = text_tokens.iter().map(|s| s.as_str()).collect();
    let mut score = 0.0f32;

    for term in terms {
        if token_set.contains(term.as_str()) {
            score += 4.0;
            continue;
        }

        if text_tokens.iter().any(|t| t.starts_with(term) || term.starts_with(t)) {
            score += 2.0;
            continue;
        }

        if term.len() >= 6 && text_tokens.iter().any(|t| fuzzy_close(term, t)) {
            score += 1.0;
        }
    }

    // Pequeña bonificación para frases cercanas de la pregunta original.
    for window in query_tokens.windows(2) {
        if window.len() == 2 {
            let phrase = format!("{} {}", window[0], window[1]);
            if normalize_text(text).contains(&phrase) {
                score += 3.0;
            }
        }
    }

    score / (1.0 + (text_tokens.len() as f32 / 300.0))
}

fn fuzzy_close(a: &str, b: &str) -> bool {
    let min_len = a.len().min(b.len());
    let max_len = a.len().max(b.len());

    if min_len < 5 || max_len.saturating_sub(min_len) > 2 {
        return false;
    }

    levenshtein_limited(a, b, 2) <= 2
}

fn levenshtein_limited(a: &str, b: &str, limit: usize) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();

    if a_chars.len().abs_diff(b_chars.len()) > limit {
        return limit + 1;
    }

    let mut prev: Vec<usize> = (0..=b_chars.len()).collect();
    let mut curr = vec![0usize; b_chars.len() + 1];

    for (i, ca) in a_chars.iter().enumerate() {
        curr[0] = i + 1;
        let mut row_min = curr[0];

        for (j, cb) in b_chars.iter().enumerate() {
            let cost = if ca == cb { 0 } else { 1 };
            curr[j + 1] = (prev[j + 1] + 1)
                .min(curr[j] + 1)
                .min(prev[j] + cost);
            row_min = row_min.min(curr[j + 1]);
        }

        if row_min > limit {
            return limit + 1;
        }

        std::mem::swap(&mut prev, &mut curr);
    }

    prev[b_chars.len()]
}

fn tokenize_normalized(text: &str) -> Vec<String> {
    normalize_text(text)
        .split_whitespace()
        .map(|s| s.to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

fn normalize_token(text: &str) -> String {
    normalize_text(text).replace(' ', "")
}

fn normalize_text(text: &str) -> String {
    let mut out = String::with_capacity(text.len());

    for c in text.chars() {
        let mapped = match c {
            'á' | 'à' | 'ä' | 'â' | 'Á' | 'À' | 'Ä' | 'Â' => 'a',
            'é' | 'è' | 'ë' | 'ê' | 'É' | 'È' | 'Ë' | 'Ê' => 'e',
            'í' | 'ì' | 'ï' | 'î' | 'Í' | 'Ì' | 'Ï' | 'Î' => 'i',
            'ó' | 'ò' | 'ö' | 'ô' | 'Ó' | 'Ò' | 'Ö' | 'Ô' => 'o',
            'ú' | 'ù' | 'ü' | 'û' | 'Ú' | 'Ù' | 'Ü' | 'Û' => 'u',
            'ñ' | 'Ñ' => 'n',
            _ => c.to_ascii_lowercase(),
        };

        if mapped.is_ascii_alphanumeric() || mapped.is_whitespace() {
            out.push(mapped);
        } else {
            out.push(' ');
        }
    }

    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn is_stopword(token: &str) -> bool {
    matches!(
        token,
        "que" | "como" | "para" | "por" | "con" | "una" | "uno" | "unos" | "unas" |
        "del" | "las" | "los" | "este" | "esta" | "esto" | "ese" | "esa" | "eso" |
        "son" | "ser" | "fue" | "han" | "hay" | "mas" | "muy" | "sin" | "sus" |
        "sobre" | "entre" | "donde" | "cuando" | "cual" | "cuales" | "debe" |
        "deben" | "tiene" | "tienen" | "desde" | "hacia" | "cada" | "todo" |
        "toda" | "todos" | "todas" | "segun" | "base" | "informacion" | "documento"
    )
}


fn filter_fragments_by_sources(
    fragments: &[SearchFragment],
    selected_sources: Option<&[String]>,
) -> Vec<SearchFragment> {
    // ===== INICIO CAMBIO SEGURO PRODUCCIÓN =====
    // None o [] significa: usar todas las fuentes.
    // Solo se filtra cuando el usuario marca fuentes específicas.
    let Some(selected_sources) = selected_sources else {
        return fragments.to_vec();
    };

    if selected_sources.is_empty() {
        return fragments.to_vec();
    }
    // ===== FIN CAMBIO SEGURO PRODUCCIÓN =====

    let selected: HashSet<&str> = selected_sources.iter().map(|s| s.as_str()).collect();

    fragments
        .iter()
        .filter(|f| selected.contains(f.source.as_str()))
        .cloned()
        .collect()
}

fn merge_fragments_as_single_context(fragments: &[SearchFragment]) -> Vec<SearchFragment> {
    if fragments.len() <= 1 {
        return fragments.to_vec();
    }

    let mut ordered = fragments.to_vec();
    ordered.sort_by(|a, b| {
        a.source
            .cmp(&b.source)
            .then_with(|| a.chunk_index.cmp(&b.chunk_index))
    });

    let mut merged_sources = Vec::new();
    let mut seen_sources = HashSet::new();
    let mut merged_text = String::new();

    for fragment in &ordered {
        if seen_sources.insert(fragment.source.clone()) {
            merged_sources.push(fragment.source.clone());
        }

        if !merged_text.is_empty() {
            merged_text.push_str("\n\n");
        }

        merged_text.push_str(&format!(
            "[Fuente: {} | fragmento {}]\n{}",
            fragment.source,
            fragment.chunk_index + 1,
            fragment.text.trim()
        ));
    }

    let best_score = ordered
        .iter()
        .map(|f| f.score)
        .fold(f32::MIN, f32::max);

    vec![SearchFragment {
        score: best_score,
        source: merged_sources.join(" + "),
        chunk_index: 0,
        text: merged_text,
    }]
}

async fn call_openrouter(
    state: &AppState,
    question: &str,
    fragments: &[SearchFragment],
    max_chars_per_fragment: usize,
    max_tokens: u32,
) -> Result<String> {
    let api_key = state
        .config
        .openrouter_api_key
        .as_ref()
        .ok_or_else(|| anyhow!("OPENROUTER_API_KEY no está configurada"))?;

    if fragments.is_empty() {
        return Ok(
            "No se encontró información suficiente en los documentos cargados para responder con fundamento."
                .to_string(),
        );
    }

    let mut context = String::new();
    for (i, f) in fragments.iter().enumerate() {
        context.push_str(&format!(
            "[Evidencia {} | fuente: {}] {}\n",
            i + 1,
            f.source,
            limit_chars(&f.text, max_chars_per_fragment)
        ));
    }

    let _system_prompt = r#"Eres un asistente experto en análisis documental que responde en español.

Produce una respuesta BREVE, precisa y directa con esta estructura:

## Respuesta
Contesta la pregunta en 1-3 frases, sin rodeos.

## Análisis Detallado
Desglosa cada pieza de evidencia relevante. Explica qué dice, qué implica y cómo se relaciona con la pregunta. No omitas detalles importantes.

## Conclusión
Resume las implicaciones prácticas y responde directamente a la pregunta.

Reglas:
- Responde ÚNICAMENTE con base en la evidencia proporcionada.
- Si la evidencia no alcanza para algún punto, dilo explícitamente.
- Sé exhaustivo: es preferible extenderse que omitir información.
- Usa formato Markdown para estructurar (## títulos, **negritas**, - listas)."#;

    let system_prompt = r#"Eres un asistente experto en analisis documental que responde en espanol.

Tu trabajo es redactar un INFORME EXTENDIDO usando unicamente la evidencia entregada.
No inventes datos, autores, leyes, diagnosticos, articulos, cifras ni conclusiones que no esten sustentadas.
Si la evidencia es parcial, indicalo sin bloquear la respuesta completa.

Estructura obligatoria:

## Resumen Ejecutivo
Explica de que trata la documentacion y responde directamente la pregunta.

## Hallazgos Principales
Lista los hallazgos mas importantes, agrupando ideas repetidas de distintas fuentes.

## Analisis Detallado por Temas
Desarrolla los temas encontrados. Relaciona la evidencia con la pregunta del usuario.

## Datos o Evidencias Relevantes
Incluye datos concretos, nombres de modulos, procesos, riesgos, requisitos, actores o conceptos encontrados.

## Limitaciones de la Evidencia
Indica que puntos no pueden afirmarse con seguridad si no aparecen en las fuentes.

## Conclusion
Cierra con una respuesta clara, practica y util.

Reglas:
- Usa Markdown.
- Mantiene tono profesional e institucional.
- No digas que eres IA.
- No uses frases de relleno.
- Prioriza claridad, detalle y utilidad."#;

    let user_prompt = format!(
        "Pregunta del usuario:\n{}\n\nPaquete de evidencia recuperado por Rust:\n{}\n\nRedacta el informe extendido con base estricta en esa evidencia:",
        question, context
    );

    let body = json!({
        "model": state.config.openrouter_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.1,
        "max_tokens": max_tokens.max(500),
        "top_p": 0.8,
        "frequency_penalty": 0.3,
    });

    let value: Value = state
        .http
        .post("https://openrouter.ai/api/v1/chat/completions")
        .bearer_auth(api_key)
        .header("Content-Type", "application/json")
        .header("X-OpenRouter-Title", "Rust Lexical RAG")
        .json(&body)
        .timeout(Duration::from_secs(30))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;

    Ok(value
        .get("choices")
        .and_then(|v| v.as_array())
        .and_then(|arr| arr.first())
        .and_then(|choice| choice.get("message"))
        .and_then(|msg| msg.get("content"))
        .and_then(|v| v.as_str())
        .unwrap_or("No se recibió contenido.")
        .to_string())
}

async fn call_openrouter_stream(
    state: &AppState,
    question: &str,
    fragments: &[SearchFragment],
    max_chars_per_fragment: usize,
    max_tokens: u32,
) -> Result<mpsc::Receiver<Result<String>>> {
    let api_key = state
        .config
        .openrouter_api_key
        .as_ref()
        .ok_or_else(|| anyhow!("OPENROUTER_API_KEY no está configurada"))?;

    if fragments.is_empty() {
        let (tx, rx) = mpsc::channel(1);
        let _ = tx
            .send(Ok(
                "No se encontró información suficiente en los documentos cargados.".to_string(),
            ))
            .await;
        return Ok(rx);
    }

    let mut context = String::new();
    for (i, f) in fragments.iter().enumerate() {
        context.push_str(&format!(
            "[Evidencia {} | fuente: {}] {}\n",
            i + 1,
            f.source,
            limit_chars(&f.text, max_chars_per_fragment)
        ));
    }

    let _system_prompt =
        r#"Eres un asistente experto en análisis documental que responde en español.

Produce siempre un INFORME EXTENDIDO estructurado de la siguiente manera:

## Resumen Ejecutivo
Sintetiza en 2-3 frases el hallazgo principal.

## Análisis Detallado
Desglosa cada pieza de evidencia relevante. Explica qué dice, qué implica y cómo se relaciona con la pregunta. No omitas detalles importantes.

## Conclusión
Resume las implicaciones prácticas y responde directamente a la pregunta.

Reglas:
- Responde ÚNICAMENTE con base en la evidencia proporcionada.
- Si la evidencia no alcanza para algún punto, dilo explícitamente.
- Sé exhaustivo: es preferible extenderse que omitir información.
- Usa formato Markdown para estructurar (## títulos, **negritas**, - listas)."#;

    let system_prompt = r#"Eres un asistente experto en analisis documental que responde en espanol.

Tu trabajo es redactar un INFORME EXTENDIDO usando unicamente la evidencia entregada.
No inventes datos, autores, leyes, diagnosticos, articulos, cifras ni conclusiones que no esten sustentadas.
Si la evidencia es parcial, indicalo sin bloquear la respuesta completa.

Estructura obligatoria:

## Resumen Ejecutivo
Explica de que trata la documentacion y responde directamente la pregunta.

## Hallazgos Principales
Lista los hallazgos mas importantes, agrupando ideas repetidas de distintas fuentes.

## Analisis Detallado por Temas
Desarrolla los temas encontrados. Relaciona la evidencia con la pregunta del usuario.

## Datos o Evidencias Relevantes
Incluye datos concretos, nombres de modulos, procesos, riesgos, requisitos, actores o conceptos encontrados.

## Limitaciones de la Evidencia
Indica que puntos no pueden afirmarse con seguridad si no aparecen en las fuentes.

## Conclusion
Cierra con una respuesta clara, practica y util.

Reglas:
- Usa Markdown.
- Mantiene tono profesional e institucional.
- No digas que eres IA.
- No uses frases de relleno.
- Prioriza claridad, detalle y utilidad."#;

    let user_prompt = format!(
        "Pregunta del usuario:\n{}\n\nPaquete de evidencia recuperado por Rust:\n{}\n\nRedacta el informe extendido con base estricta en esa evidencia:",
        question, context
    );

    let body = json!({
        "model": state.config.openrouter_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.1,
        "max_tokens": max_tokens.max(500),
        "top_p": 0.8,
        "frequency_penalty": 0.3,
        "stream": true
    });

    let response = state
        .http
        .post("https://openrouter.ai/api/v1/chat/completions")
        .bearer_auth(api_key)
        .header("Content-Type", "application/json")
        .header("X-OpenRouter-Title", "Rust Lexical RAG")
        .json(&body)
        .timeout(Duration::from_secs(120))
        .send()
        .await?
        .error_for_status()?;

    let (tx, rx) = mpsc::channel::<Result<String>>(64);

    tokio::spawn(async move {
        let mut stream = response.bytes_stream();
        let mut buffer = String::new();

        while let Some(chunk_result) = stream.next().await {
            match chunk_result {
                Ok(bytes) => {
                    buffer.push_str(&String::from_utf8_lossy(&bytes));

                    while let Some(pos) = buffer.find('\n') {
                        let line = buffer[..pos].trim().to_string();
                        buffer = buffer[pos + 1..].to_string();

                        if let Some(data) = line.strip_prefix("data: ") {
                            if data == "[DONE]" {
                                return;
                            }
                            if let Ok(parsed) = serde_json::from_str::<Value>(data) {
                                if let Some(content) =
                                    parsed["choices"][0]["delta"]["content"].as_str()
                                {
                                    if !content.is_empty() {
                                        let _ = tx.send(Ok(content.to_string())).await;
                                    }
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    let _ = tx.send(Err(anyhow::Error::from(e))).await;
                    return;
                }
            }
        }
    });

    Ok(rx)
}

fn build_local_response(question: &str, fragments: &[SearchFragment]) -> String {
    if fragments.is_empty() {
        return "No se encontró información suficiente en los documentos cargados para responder con fundamento. Puede subir más documentos o ejecutar la reindexación desde la administración.".to_string();
    }

    let mut response = String::new();

    response.push_str("## Respuesta\n");
    response.push_str("Se encontró información relacionada con la consulta en las fuentes recuperadas.\n\n");

    response.push_str(&format!("## Consulta\n{}\n\n", question));

    response.push_str("## Sustento\n");

    for f in fragments.iter().take(2) {
        response.push_str(&format!("- {}\n", clean_excerpt(&f.text, 180)));
    }

    let mut sources: Vec<String> = fragments.iter().map(|f| f.source.clone()).collect();
    sources.sort();
    sources.dedup();

    response.push_str("\n**Fuente consultada:** ");
    response.push_str(&sources.join(", "));

    response
}

fn init_db(config: &Config) -> Result<()> {
    let conn = Connection::open(config.db_path())?;

    conn.execute_batch(
        r#"
        CREATE TABLE IF NOT EXISTS documents (
            path TEXT PRIMARY KEY,
            hash TEXT NOT NULL,
            modified INTEGER NOT NULL,
            indexed_at TEXT NOT NULL,
            chunk_count INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS document_chunks (
            source TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            hash TEXT NOT NULL,
            indexed_at TEXT NOT NULL,
            PRIMARY KEY(source, chunk_index)
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS document_chunks_fts
        USING fts5(
            source UNINDEXED,
            chunk_index UNINDEXED,
            text,
            tokenize = 'unicode61 remove_diacritics 2'
        );

        CREATE INDEX IF NOT EXISTS idx_document_chunks_source ON document_chunks(source);

        CREATE TABLE IF NOT EXISTS faq_cache (
            question_hash TEXT PRIMARY KEY,
            question TEXT NOT NULL,
            response TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('admin','user')),
            is_active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            blocked_reason TEXT
        );

        CREATE TABLE IF NOT EXISTS sessions (
            token TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            created_at INTEGER NOT NULL,
            expires_at INTEGER NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        );

        CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions(expires_at);
        "#,
    )?;

    Ok(())
}

fn ensure_default_admin(config: &Config) -> Result<()> {
    let conn = Connection::open(config.db_path())?;

    let count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM users WHERE role = 'admin'",
        [],
        |r| r.get(0),
    )?;

    if count == 0 {
        conn.execute(
            "INSERT INTO users(username, password_hash, role, is_active, created_at) VALUES (?1, ?2, 'admin', 1, ?3)",
            params![
                config.admin_username.to_lowercase(),
                hash_password(config, &config.admin_password),
                Utc::now().to_rfc3339()
            ],
        )?;

        info!("Administrador inicial creado: {}", config.admin_username);
    }

    Ok(())
}

fn hash_password(config: &Config, password: &str) -> String {
    let mut hasher = Sha256::new();

    hasher.update(config.auth_salt.as_bytes());
    hasher.update(b":");
    hasher.update(password.as_bytes());

    hex::encode(hasher.finalize())
}

fn check_faq_cache(config: &Config, question: &str) -> Option<String> {
    let conn = Connection::open(config.db_path()).ok()?;
    let normalized = normalize_question(question);
    let hash = faq_question_hash(&normalized);

    let mut stmt = conn
        .prepare("SELECT response FROM faq_cache WHERE question_hash = ?1")
        .ok()?;

    stmt.query_row(params![hash], |row| row.get(0)).ok()
}

fn store_faq_cache(config: &Config, question: &str, response: &str) -> Result<()> {
    let conn = Connection::open(config.db_path())?;
    let normalized = normalize_question(question);
    let hash = faq_question_hash(&normalized);

    conn.execute(
        "INSERT OR REPLACE INTO faq_cache(question_hash, question, response, created_at) VALUES (?1, ?2, ?3, ?4)",
        params![hash, normalized, response, Utc::now().to_rfc3339()],
    )?;

    Ok(())
}

fn faq_question_hash(question: &str) -> String {
    let mut hasher = Sha256::new();

    hasher.update(question.as_bytes());

    hex::encode(hasher.finalize())[..16].to_string()
}

fn normalize_question(question: &str) -> String {
    question
        .trim()
        .to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect()
}

fn manifest_has_same_hash(config: &Config, path: &str, hash: &str) -> Result<bool> {
    let conn = Connection::open(config.db_path())?;

    let existing: Option<String> = conn
        .query_row(
            "SELECT hash FROM documents WHERE path = ?1",
            params![path],
            |row| row.get(0),
        )
        .optional()?;

    Ok(existing.map(|h| h == hash).unwrap_or(false))
}

fn upsert_manifest_document(
    config: &Config,
    path: &str,
    hash: &str,
    modified: i64,
    chunk_count: usize,
) -> Result<()> {
    let conn = Connection::open(config.db_path())?;

    conn.execute(
        r#"
        INSERT INTO documents(path, hash, modified, indexed_at, chunk_count)
        VALUES (?1, ?2, ?3, ?4, ?5)
        ON CONFLICT(path) DO UPDATE SET
            hash = excluded.hash,
            modified = excluded.modified,
            indexed_at = excluded.indexed_at,
            chunk_count = excluded.chunk_count
        "#,
        params![
            path,
            hash,
            modified,
            Utc::now().to_rfc3339(),
            chunk_count as i64
        ],
    )?;

    Ok(())
}

fn remove_manifest_document(config: &Config, path: &str) -> Result<()> {
    let conn = Connection::open(config.db_path())?;
    conn.execute("DELETE FROM documents WHERE path = ?1", params![path])?;
    Ok(())
}

fn list_manifest_paths(config: &Config) -> Result<Vec<String>> {
    let conn = Connection::open(config.db_path())?;
    let mut stmt = conn.prepare("SELECT path FROM documents")?;
    let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;

    let mut paths = Vec::new();

    for row in rows {
        paths.push(row?);
    }

    Ok(paths)
}

fn count_manifest_documents(config: &Config) -> Result<i64> {
    let conn = Connection::open(config.db_path())?;
    let count = conn.query_row("SELECT COUNT(*) FROM documents", [], |row| row.get(0))?;
    Ok(count)
}

fn file_hash_with_pipeline(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();

    hasher.update(INDEX_PIPELINE_VERSION.as_bytes());
    hasher.update(bytes);

    hex::encode(hasher.finalize())
}

fn file_modified_epoch(path: &Path) -> Result<i64> {
    let modified = fs::metadata(path)?.modified().unwrap_or(SystemTime::now());

    Ok(modified
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64)
}

fn relative_source(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

fn is_supported_file(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|ext| matches!(ext.to_lowercase().as_str(), "txt" | "md" | "pdf"))
        .unwrap_or(false)
}

fn is_supported_filename(name: &str) -> bool {
    Path::new(name)
        .extension()
        .and_then(|e| e.to_str())
        .map(|ext| matches!(ext.to_lowercase().as_str(), "txt" | "md" | "pdf"))
        .unwrap_or(false)
}

fn sanitize_filename(name: &str) -> String {
    let base = Path::new(name)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("documento.txt");

    base.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '.' || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn limit_chars(text: &str, max_chars: usize) -> String {
    let mut out = String::new();

    for (i, ch) in text.chars().enumerate() {
        if i >= max_chars {
            out.push_str("...");
            break;
        }

        out.push(ch);
    }

    out
}

fn clean_excerpt(text: &str, max_chars: usize) -> String {
    limit_chars(&text.split_whitespace().collect::<Vec<_>>().join(" "), max_chars)
}

const INDEX_HTML: &str = r#"
<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RAG Institucional</title>
<style>
:root{
  --bg:#f4f6fb;
  --card:#fff;
  --text:#172033;
  --muted:#667085;
  --border:#e4e7ec;
  --primary:#1d4ed8;
  --primary2:#1e40af;
  --good:#047857;
  --warn:#b45309;
  --bad:#b42318;
}

*{box-sizing:border-box}

body{
  margin:0;
  font-family:Inter,system-ui,-apple-system,Segoe UI,sans-serif;
  background:linear-gradient(180deg,#eef4ff,#f8fafc 260px);
  color:var(--text);
}

.wrap{
  max-width:1180px;
  margin:0 auto;
  padding:24px;
}

.top{
  display:flex;
  justify-content:space-between;
  gap:14px;
  align-items:center;
  margin-bottom:18px;
}

.brand h1{
  font-size:25px;
  margin:0;
}

.brand p{
  margin:5px 0 0;
  color:var(--muted);
}

.card{
  background:rgba(255,255,255,.94);
  border:1px solid var(--border);
  border-radius:22px;
  padding:18px;
  box-shadow:0 14px 40px rgba(16,24,40,.08);
  margin-bottom:16px;
}

.grid{
  display:grid;
  grid-template-columns:1.2fr .8fr;
  gap:16px;
}

.hidden{
  display:none!important;
}

input,textarea,select{
  width:100%;
  border:1px solid #d0d5dd;
  border-radius:14px;
  padding:12px 13px;
  font-size:15px;
  background:white;
}

textarea{
  min-height:130px;
  resize:vertical;
}

label{
  font-weight:700;
  font-size:13px;
  display:block;
  margin:10px 0 6px;
  color:#344054;
}

button{
  border:0;
  border-radius:14px;
  padding:12px 15px;
  font-weight:800;
  cursor:pointer;
  background:var(--primary);
  color:white;
  box-shadow:0 6px 18px rgba(29,78,216,.18);
}

button:hover{
  background:var(--primary2);
}

button.secondary{
  background:#475467;
}

button.good{
  background:var(--good);
}

button.warn{
  background:var(--warn);
}

button.bad{
  background:var(--bad);
}

button.light{
  background:#eff4ff;
  color:#1d4ed8;
  box-shadow:none;
}

button:disabled{
  opacity:.55;
  cursor:not-allowed;
}

.row{
  display:flex;
  gap:10px;
  align-items:center;
  flex-wrap:wrap;
}

.tabs{
  display:flex;
  gap:8px;
  margin-bottom:14px;
}

.tab{
  background:#fff;
  color:#344054;
  border:1px solid var(--border);
  box-shadow:none;
}

.tab.active{
  background:#1d4ed8;
  color:white;
  border-color:#1d4ed8;
}

.pill{
  display:inline-flex;
  align-items:center;
  gap:6px;
  border-radius:999px;
  padding:5px 10px;
  background:#eef4ff;
  color:#1d4ed8;
  font-size:12px;
  font-weight:800;
}

.muted{
  color:var(--muted);
  font-size:13px;
}

.answer{
  line-height:1.65;
  font-size:16px;
}

.answer h1,.answer h2,.answer h3{
  margin:14px 0 8px;
}

.answer p{
  margin:9px 0;
}

.answer ul{
  padding-left:22px;
}

.answer blockquote{
  border-left:4px solid #bfdbfe;
  margin:10px 0;
  padding:7px 12px;
  background:#eff6ff;
  border-radius:10px;
}

.answer code{
  background:#eef2ff;
  border-radius:6px;
  padding:2px 5px;
}

.bubble{
  background:#f8fafc;
  border:1px solid var(--border);
  border-radius:18px;
  padding:14px;
  margin-top:12px;
}

.source{
  border-left:4px solid #d0d5dd;
  background:#f9fafb;
  padding:10px 12px;
  border-radius:12px;
  margin:10px 0;
}

.source .meta{
  font-size:12px;
  color:#667085;
  margin-bottom:5px;
}

.table{
  width:100%;
  border-collapse:collapse;
}

.table th,.table td{
  border-bottom:1px solid var(--border);
  padding:10px;
  text-align:left;
  font-size:14px;
}

.table th{
  color:#475467;
  background:#f9fafb;
}

.login{
  max-width:430px;
  margin:8vh auto;
}

.notice{
  padding:10px 12px;
  border-radius:14px;
  background:#fef3c7;
  color:#92400e;
  margin-top:10px;
}

.ok{
  background:#d1fae5;
  color:#065f46;
}

.err{
  background:#fee4e2;
  color:#991b1b;
}

@media(max-width:850px){
  .grid{
    grid-template-columns:1fr;
  }

  .top{
    align-items:flex-start;
    flex-direction:column;
  }
}
</style>
</head>
<body>
<div class="wrap">
  <div id="loginView" class="login card hidden">
    <div class="brand">
      <h1>Ingreso al RAG institucional</h1>
      <p>Ingrese con el usuario creado por el administrador.</p>
    </div>

    <label>Usuario</label>
    <input id="loginUser" autocomplete="username" value="admin">

    <label>Contraseña</label>
    <input id="loginPass" type="password" autocomplete="current-password" placeholder="Contraseña">

    <div class="row" style="margin-top:14px">
      <button onclick="login()">Ingresar</button>
    </div>

    <div id="loginMsg"></div>
  </div>

  <div id="appView" class="hidden">
    <div class="top">
      <div class="brand">
        <h1>RAG Institucional</h1>
        <p>Consulta documentación local con respuestas naturales y administración de usuarios.</p>
      </div>

      <div class="row">
        <span class="pill" id="userPill">Usuario</span>
        <button class="secondary" onclick="logout()">Salir</button>
      </div>
    </div>

    <div class="tabs">
      <button class="tab active" onclick="showTab('chat')" id="tabChat">Chat</button>
      <button class="tab hidden" onclick="showTab('admin')" id="tabAdmin">Administración</button>
    </div>

    <section id="chatTab">
      <div class="grid">
        <div class="card">
          <label>Pregunta — Informe Extendido</label>

          <textarea id="question" placeholder="Ejemplo: Analiza los resultados de laboratorio del documento cargado"></textarea>

          <div class="row" style="margin-top:10px">
            <label class="row" style="margin:0;font-weight:600">
              <input id="useLlm" type="checkbox" checked style="width:auto"> usar OpenRouter
            </label>

            <button onclick="ask()" id="askBtn">Preguntar</button>
          </div>

          <div id="answerBox" class="bubble hidden">
            <div class="row">
              <span id="answerBadge" class="pill">Informe extendido</span>
            </div>

            <div id="answer" class="answer"></div>
          </div>
        </div>

        <div class="card">
          <h3 style="margin-top:0">Estado</h3>
          <div id="statusBox" class="muted">Cargando...</div>

          <div class="row" style="margin-top:12px">
            <button class="light" onclick="status()">Actualizar estado</button>
          </div>

          <details style="margin-top:14px">
            <summary class="muted">Ver respaldo documental consultado</summary>
            <div class="row" style="margin:10px 0">
              <label class="row" style="margin:0;font-weight:600">
                <input id="restrictToMarked" type="checkbox" style="width:auto"> usar solo fuentes marcadas
              </label>
              <button class="light" type="button" onclick="askWithMarkedSources()">Responder con marcadas</button>
            </div>
            <div id="sources"></div>
          </details>
        </div>
      </div>
    </section>

    <section id="adminTab" class="hidden">
      <div class="grid">
        <div class="card">
          <h3 style="margin-top:0">Subir información</h3>
          <p class="muted">Permite subir .pdf, .txt o .md sin bajar el servicio. Luego se indexa en segundo plano.</p>

          <input id="fileInput" type="file" multiple accept=".pdf,.txt,.md">

          <div class="row" style="margin-top:12px">
            <button class="good" onclick="uploadFiles()">Subir documentos</button>
            <button class="secondary" onclick="reindex()">Reindexar ahora</button>
          </div>

          <div id="uploadMsg"></div>

          <h3>Documentos indexados</h3>
          <div id="docsBox" class="muted">Sin cargar.</div>
        </div>

        <div class="card">
          <h3 style="margin-top:0">Usuarios</h3>

          <label>Nuevo usuario</label>
          <input id="newUser" placeholder="usuario">

          <label>Contraseña</label>
          <input id="newPass" type="password" placeholder="contraseña">

          <label>Rol</label>
          <select id="newRole">
            <option value="user">Usuario</option>
            <option value="admin">Administrador</option>
          </select>

          <div class="row" style="margin-top:12px">
            <button onclick="createUser()">Crear usuario</button>
          </div>

          <div id="userMsg"></div>
          <div id="usersBox" style="margin-top:14px"></div>
        </div>
      </div>
    </section>
  </div>
</div>

<script>
let currentUser = null;
let lastFragments = [];

async function api(url, opts = {}) {
  const r = await fetch(url, { credentials: 'same-origin', ...opts });

  let data = null;

  try {
    data = await r.json();
  } catch {}

  if (!r.ok) {
    throw new Error(data?.error || 'Error de servidor');
  }

  return data;
}

function msg(id, text, cls = 'notice') {
  document.getElementById(id).innerHTML = `<div class="${cls}">${escapeHtml(text)}</div>`;
}

async function boot() {
  try {
    currentUser = await api('/api/me');
    showApp();
  } catch {
    document.getElementById('loginView').classList.remove('hidden');
  }
}

function showApp() {
  document.getElementById('loginView').classList.add('hidden');
  document.getElementById('appView').classList.remove('hidden');

  document.getElementById('userPill').textContent = `${currentUser.username} · ${currentUser.role}`;

  if (currentUser.role === 'admin') {
    document.getElementById('tabAdmin').classList.remove('hidden');
    loadUsers();
    loadDocs();
  }

  status();
}

async function login() {
  try {
    const username = document.getElementById('loginUser').value;
    const password = document.getElementById('loginPass').value;

    const data = await api('/api/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    });

    currentUser = data.user;
    showApp();
  } catch (e) {
    msg('loginMsg', e.message, 'notice err');
  }
}

async function logout() {
  await api('/api/logout', { method: 'POST' }).catch(() => {});
  location.reload();
}

function showTab(t) {
  document.getElementById('chatTab').classList.toggle('hidden', t !== 'chat');
  document.getElementById('adminTab').classList.toggle('hidden', t !== 'admin');

  document.getElementById('tabChat').classList.toggle('active', t === 'chat');
  document.getElementById('tabAdmin').classList.toggle('active', t === 'admin');
}

async function status() {
  try {
    const s = await api('/api/status');

    document.getElementById('statusBox').innerHTML =
      `<p><b>Documentos:</b> ${s.manifest_documents}</p>
       <p><b>OpenRouter:</b> ${s.openrouter_enabled ? 'habilitado' : 'modo privado'}</p>
       <p><b>Modelo:</b> ${escapeHtml(s.openrouter_model)}</p>
       <p><b>Búsqueda:</b> ${escapeHtml(s.search_engine || 'RAG léxico local')}</p>
       <p><b>Actualización:</b> cada ${s.index_interval_seconds}s</p>`;
  } catch (e) {
    document.getElementById('statusBox').textContent = e.message;
  }
}

async function ask() {
  await runQuestion(false);
}

async function askWithMarkedSources() {
  await runQuestion(true);
}

function getMarkedSources() {
  return Array.from(document.querySelectorAll('.source-select:checked'))
    .map(input => input.value)
    .filter(Boolean);
}

async function runQuestion(forceMarked) {
  const question = document.getElementById('question').value.trim();

  if (!question) {
    return;
  }

  const askBtn = document.getElementById('askBtn');
  askBtn.disabled = true;
  askBtn.textContent = 'Consultando...';

  document.getElementById('answerBox').classList.remove('hidden');
  document.getElementById('answer').innerHTML = '<p>🔍 Buscando en documentos...</p>';
  document.getElementById('sources').innerHTML = '';

  try {
    const use_llm = document.getElementById('useLlm').checked;
    const top_k = 6;
    const restrictToMarked = document.getElementById('restrictToMarked').checked || forceMarked;
    const selected_sources = restrictToMarked ? getMarkedSources() : [];

    if (restrictToMarked && selected_sources.length === 0) {
      document.getElementById('answer').innerHTML =
        '<div class="notice err">Debe marcar al menos una fuente.</div>';
      return;
    }

    const response = await fetch('/api/chat-stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, use_llm, top_k, selected_sources }),
    });

    if (!response.ok) {
      throw new Error(`Error: ${response.statusText}`);
    }

    let fragments = [];

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      const events = buffer.split('\n\n');
      buffer = events.pop() || '';

      for (const rawEvent of events) {
        const dataLines = rawEvent
          .split('\n')
          .filter(line => line.startsWith('data: '))
          .map(line => line.slice(6));

        if (!dataLines.length) continue;

        let evt = null;

        try {
          evt = JSON.parse(dataLines.join('\n'));
        } catch {
          continue;
        }

        if (evt.kind === 'fragments') {
          fragments = Array.isArray(evt.payload) ? evt.payload : [];
          lastFragments = fragments;
          document.getElementById('answer').innerHTML = '<p>⏳ Procesando con IA...</p>';
        } else if (evt.kind === 'cached') {
          const ans = String(evt.payload || '');
          document.getElementById('answer').innerHTML = renderMarkdown(ans);
          document.getElementById('answerBadge').textContent = '📋 Informe (caché)';
        } else if (evt.kind === 'local') {
          const ans = String(evt.payload || '');
          document.getElementById('answer').innerHTML = renderMarkdown(ans);
          document.getElementById('answerBadge').textContent = '📄 Respuesta local';
        } else if (evt.kind === 'answer') {
          const ans = String(evt.payload || '');
          document.getElementById('answer').innerHTML = renderMarkdown(ans);
          document.getElementById('answerBadge').textContent = '✨ Informe extendido';
        } else if (evt.kind === 'error') {
          document.getElementById('answer').innerHTML =
            `<div class="notice err">${escapeHtml(String(evt.payload || 'Error desconocido'))}</div>`;
        }
      }
    }

    if (fragments.length > 0) {
      renderSources(fragments);
    }
  } catch (e) {
    document.getElementById('answer').innerHTML =
      `<div class="notice err">${escapeHtml(e.message)}</div>`;
  } finally {
    askBtn.disabled = false;
    askBtn.textContent = 'Preguntar';
  }
}

function renderSources(frags) {
  if (!frags.length) {
    lastFragments = [];
    document.getElementById('sources').innerHTML =
      '<p class="muted">No se encontraron respaldos.</p>';
    return;
  }

  lastFragments = frags;
  document.getElementById('sources').innerHTML = frags
    .map((f, i) =>
      `<div class="source">
        <label class="meta" style="display:flex;align-items:center;gap:8px">
          <input class="source-select" type="checkbox" value="${escapeHtml(f.source)}" checked style="width:auto">
          <span>Documento ${i + 1}: ${escapeHtml(shortSource(f.source))}</span>
        </label>
        <div>${escapeHtml(f.text).slice(0, 650)}...</div>
      </div>`
    )
    .join('');
}

async function uploadFiles() {
  const input = document.getElementById('fileInput');

  if (!input.files.length) {
    msg('uploadMsg', 'Seleccione al menos un archivo.');
    return;
  }

  const fd = new FormData();

  for (const f of input.files) {
    fd.append('files', f);
  }

  try {
    const r = await api('/api/admin/upload', {
      method: 'POST',
      body: fd,
    });

    msg('uploadMsg', r.message, 'notice ok');
    input.value = '';

    setTimeout(() => {
      loadDocs();
      status();
    }, 1500);
  } catch (e) {
    msg('uploadMsg', e.message, 'notice err');
  }
}

async function reindex() {
  try {
    msg('uploadMsg', 'Reindexando, espere...');

    const r = await api('/api/admin/reindex', {
      method: 'POST',
    });

    msg(
      'uploadMsg',
      `Listo. Archivos indexados: ${r.indexed_files}, omitidos: ${r.skipped_files}, chunks: ${r.chunks_indexed}`,
      'notice ok'
    );

    loadDocs();
    status();
  } catch (e) {
    msg('uploadMsg', e.message, 'notice err');
  }
}

async function loadDocs() {
  try {
    const docs = await api('/api/admin/documents');

    document.getElementById('docsBox').innerHTML = docs.length
      ? `<table class="table">
          <tr>
            <th>Documento</th>
            <th>Chunks</th>
            <th>Indexado</th>
          </tr>
          ${docs
            .map(
              d =>
                `<tr>
                  <td>${escapeHtml(d.path)}</td>
                  <td>${d.chunk_count}</td>
                  <td>${escapeHtml(d.indexed_at)}</td>
                </tr>`
            )
            .join('')}
        </table>`
      : 'No hay documentos indexados.';
  } catch (e) {
    document.getElementById('docsBox').textContent = e.message;
  }
}

async function loadUsers() {
  try {
    const users = await api('/api/admin/users');

    document.getElementById('usersBox').innerHTML =
      `<table class="table">
        <tr>
          <th>Usuario</th>
          <th>Rol</th>
          <th>Estado</th>
          <th>Acción</th>
        </tr>
        ${users
          .map(
            u =>
              `<tr>
                <td>${escapeHtml(u.username)}</td>
                <td>${u.role}</td>
                <td>${u.is_active ? 'Activo' : 'Bloqueado'}</td>
                <td>${
                  u.username === currentUser.username
                    ? ''
                    : `<button class="${u.is_active ? 'bad' : 'good'}" onclick="toggleUser('${escapeJs(
                        u.username
                      )}', ${u.is_active})">${u.is_active ? 'Bloquear' : 'Activar'}</button>`
                }</td>
              </tr>`
          )
          .join('')}
      </table>`;
  } catch (e) {
    document.getElementById('usersBox').textContent = e.message;
  }
}

async function createUser() {
  try {
    const username = document.getElementById('newUser').value;
    const password = document.getElementById('newPass').value;
    const role = document.getElementById('newRole').value;

    await api('/api/admin/users', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password, role }),
    });

    msg('userMsg', 'Usuario creado.', 'notice ok');

    document.getElementById('newUser').value = '';
    document.getElementById('newPass').value = '';

    loadUsers();
  } catch (e) {
    msg('userMsg', e.message, 'notice err');
  }
}

async function toggleUser(username, isActive) {
  try {
    let reason = null;

    if (isActive) {
      reason =
        prompt('Motivo del bloqueo:', 'Bloqueado por falta de pago.') ||
        'Bloqueado por administración.';
    }

    await api('/api/admin/users/block', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, blocked: isActive, reason }),
    });

    loadUsers();
  } catch (e) {
    alert(e.message);
  }
}

function renderMarkdown(md) {
  let s = escapeHtml(md || '');

  s = s
    .replace(/^### (.*)$/gm, '<h3>$1</h3>')
    .replace(/^## (.*)$/gm, '<h2>$1</h2>')
    .replace(/^# (.*)$/gm, '<h1>$1</h1>');

  s = s.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  s = s.replace(/^&gt; (.*)$/gm, '<blockquote>$1</blockquote>');

  let lines = s.split('\n');
  let out = '';
  let inList = false;

  for (let line of lines) {
    if (/^\s*[-•] /.test(line)) {
      if (!inList) {
        out += '<ul>';
        inList = true;
      }

      out += '<li>' + line.replace(/^\s*[-•] /, '') + '</li>';
    } else {
      if (inList) {
        out += '</ul>';
        inList = false;
      }

      if (line.trim() === '') {
        out += '';
      } else if (line.startsWith('<h') || line.startsWith('<blockquote')) {
        out += line;
      } else {
        out += '<p>' + line + '</p>';
      }
    }
  }

  if (inList) {
    out += '</ul>';
  }

  return out;
}

function shortSource(s) {
  return String(s || '').split('/').pop();
}

function escapeHtml(str) {
  return String(str ?? '').replace(/[&<>'"]/g, s => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    "'": '&#39;',
    '"': '&quot;',
  }[s]));
}

function escapeJs(str) {
  return String(str ?? '').replace(/\\/g, '\\\\').replace(/'/g, "\\'");
}

boot();
</script>
</body>
</html>
"#;
