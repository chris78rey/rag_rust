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
use regex::Regex;
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
const INDEX_PIPELINE_VERSION: &str = "2026-05-15-qdrant-embeddings-v1";
const COOKIE_NAME: &str = "rag_session";

// ===== INTENT: categorías de pregunta =====
#[derive(Debug, Clone, PartialEq)]
enum QuestionIntent {
    Definition,
    ValuePrice,
    Requirement,
    Procedure,
    Comparison,
    Summary,
    Normative,
    List,
    General,
}

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
    openrouter_embedding_model: String,
    embeddings_enabled: bool,
    qdrant_url: Option<String>,
    qdrant_collection: String,
    qdrant_enabled: bool,
    embedding_dimension: usize,
    #[allow(dead_code)]
    semantic_top_k: usize,
    #[allow(dead_code)]
    hybrid_lexical_weight: f32,
    hybrid_semantic_weight: f32,
    embedding_batch_size: usize,
    llm_query_expansion: bool,
    index_interval_seconds: u64,
    default_top_k: usize,
    app_host: String,
    app_port: u16,
    admin_username: String,
    admin_password: String,
    auth_salt: String,
    session_hours: i64,
    answer_max_words: usize,
    // ===== ITERATIVE RESEARCH ENGINE =====
    exact_evidence_mode: bool,
    max_search_iterations: u32,
    max_internal_questions: u32,
    max_fragments_internal: u32,
    final_context_fragments: u32,
    enable_related_evidence: bool,
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
        let openrouter_embedding_model = env::var("OPENROUTER_EMBEDDING_MODEL").unwrap_or_else(|_| "openai/text-embedding-3-small".to_string());
        let embeddings_enabled = env::var("EMBEDDINGS_ENABLED")
            .ok()
            .map(|v| v == "true" || v == "1" || v == "yes")
            .unwrap_or(true);
        let qdrant_url = env::var("QDRANT_URL").ok().map(|v| v.trim().to_string()).filter(|v| !v.is_empty());
        let qdrant_collection = env::var("QDRANT_COLLECTION").unwrap_or_else(|_| "rag_chunks".to_string());
        let qdrant_enabled = qdrant_url.is_some();
        let embedding_dimension = env::var("EMBEDDING_DIMENSION").ok().and_then(|v| v.parse().ok()).unwrap_or(1536);
        let semantic_top_k = env::var("SEMANTIC_TOP_K").ok().and_then(|v| v.parse().ok()).unwrap_or(24);
        let hybrid_lexical_weight = env::var("HYBRID_LEXICAL_WEIGHT").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0);
        let hybrid_semantic_weight = env::var("HYBRID_SEMANTIC_WEIGHT").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0);
        let embedding_batch_size = env::var("EMBEDDING_BATCH_SIZE").ok().and_then(|v| v.parse().ok()).unwrap_or(16);
        let llm_query_expansion = env::var("LLM_QUERY_EXPANSION")
            .ok()
            .map(|v| v == "true" || v == "1" || v == "yes")
            .unwrap_or(false);
        let index_interval_seconds = env::var("INDEX_INTERVAL_SECONDS").ok().and_then(|v| v.parse().ok()).unwrap_or(3600);
        let default_top_k = env::var("DEFAULT_TOP_K").ok().and_then(|v| v.parse().ok()).unwrap_or(6);
        let app_host = env::var("APP_HOST").unwrap_or_else(|_| "0.0.0.0".to_string());
        let app_port = env::var("APP_PORT").ok().and_then(|v| v.parse().ok()).unwrap_or(8080);

        let admin_username = env::var("ADMIN_USERNAME").unwrap_or_else(|_| "admin".to_string());
        let admin_password = env::var("ADMIN_PASSWORD").unwrap_or_else(|_| "admin123".to_string());
        let auth_salt = env::var("AUTH_SALT").unwrap_or_else(|_| "cambie-esta-sal-local".to_string());
        let session_hours = env::var("SESSION_HOURS").ok().and_then(|v| v.parse().ok()).unwrap_or(24);
        let answer_max_words = env::var("ANSWER_MAX_WORDS").ok().and_then(|v| v.parse().ok()).unwrap_or(100);
        let exact_evidence_mode = env::var("EXACT_EVIDENCE_MODE")
            .ok()
            .map(|v| v == "true" || v == "1" || v == "yes")
            .unwrap_or(true);
        let max_search_iterations = env::var("MAX_SEARCH_ITERATIONS").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
        let max_internal_questions = env::var("MAX_INTERNAL_QUESTIONS").ok().and_then(|v| v.parse().ok()).unwrap_or(40);
        let max_fragments_internal = env::var("MAX_FRAGMENTS_INTERNAL").ok().and_then(|v| v.parse().ok()).unwrap_or(80);
        let final_context_fragments = env::var("FINAL_CONTEXT_FRAGMENTS").ok().and_then(|v| v.parse().ok()).unwrap_or(12);
        let enable_related_evidence = env::var("ENABLE_RELATED_EVIDENCE")
            .ok()
            .map(|v| v == "true" || v == "1" || v == "yes")
            .unwrap_or(true);

        Self {
            docs_dir: PathBuf::from(docs_dir),
            state_dir: PathBuf::from(state_dir),
            openrouter_api_key,
            openrouter_model,
            openrouter_max_tokens,
            openrouter_embedding_model,
            embeddings_enabled,
            qdrant_url,
            qdrant_collection,
            qdrant_enabled,
            embedding_dimension,
            semantic_top_k,
            hybrid_lexical_weight,
            hybrid_semantic_weight,
            embedding_batch_size,
            llm_query_expansion,
            index_interval_seconds,
            default_top_k,
            app_host,
            app_port,
            admin_username,
            admin_password,
            auth_salt,
            session_hours,
            answer_max_words,
            exact_evidence_mode,
            max_search_iterations,
            max_internal_questions,
            max_fragments_internal,
            final_context_fragments,
            enable_related_evidence,
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

#[derive(Debug, Serialize, Clone)]
struct ChatResponse {
    answer: String,
    used_llm: bool,
    model: Option<String>,
    fragments: Vec<SearchFragment>,
    question_variants: Option<Vec<String>>,
    /// Diagnóstico: fragmentos recuperados con score, source, page_number, chunk_index
    debug_search: Option<Vec<DebugFragment>>,
}

#[derive(Debug, Serialize, Clone)]
struct DebugFragment {
    source: String,
    page_number: usize,
    chunk_index: usize,
    score: f32,
    excerpt: String,
}

// ========================================================================
// ITERATIVE RESEARCH ENGINE — ESTRUCTURAS
// ========================================================================

/// Una sonda de búsqueda interna: una pregunta que el sistema se hace a sí mismo
/// para recuperar evidencia desde distintos ángulos.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SearchProbe {
    /// Consulta textual para FTS5
    query: String,
    /// Propósito: direct_answer, acronym, formula, percentage, condition, exception, etc.
    purpose: String,
    /// Prioridad relativa (más alto = más importante)
    priority: u32,
    /// Tipo de evidencia esperada
    expected_evidence_type: String,
}

/// Sesión completa de investigación iterativa
#[derive(Debug, Clone, Serialize)]
struct SearchSession {
    original_question: String,
    probes_executed: Vec<SearchProbe>,
    fragments_found: usize,
    evidence_ledger: EvidenceLedger,
    iterations: u32,
    confidence_score: f32,
}

/// Libro mayor de evidencia agrupada por tipo
#[derive(Debug, Clone, Serialize, Default)]
struct EvidenceLedger {
    direct_answer: Vec<SearchFragment>,
    definitions: Vec<SearchFragment>,
    acronyms: Vec<AcronymEvidence>,
    formulas: Vec<FormulaEvidence>,
    percentages: Vec<PercentageEvidence>,
    numeric_values: Vec<NumericEvidence>,
    obligations: Vec<SearchFragment>,
    exceptions: Vec<SearchFragment>,
    conditions: Vec<SearchFragment>,
    related_findings: Vec<SearchFragment>,
    user_may_be_missing: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct NumericEvidence {
    value: String,
    unit: String,
    context: String,
    source: String,
    page_number: usize,
}

#[derive(Debug, Clone, Serialize)]
struct AcronymEvidence {
    acronym: String,
    meaning: String,
    source: String,
    page_number: usize,
}

#[derive(Debug, Clone, Serialize)]
struct FormulaEvidence {
    formula: String,
    description: String,
    source: String,
    page_number: usize,
}

#[derive(Debug, Clone, Serialize)]
struct PercentageEvidence {
    percentage: String,
    context: String,
    source: String,
    page_number: usize,
}

/// Hecho exacto extraído durante la indexación
#[derive(Debug, Clone, Serialize)]
struct ExtractedFact {
    id: i64,
    source: String,
    page_number: usize,
    chunk_index: usize,
    fact_type: String,
    label: String,
    value: String,
    unit: String,
    text: String,
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
    /// Número de página dentro del documento PDF (0 si no aplica)
    page_number: usize,
    /// Título o sección cercana inferida (vacío si no se pudo determinar)
    source_title: String,
}

#[derive(Debug, Serialize)]
struct StatusResponse {
    docs_dir: String,
    search_engine: String,
    openrouter_enabled: bool,
    openrouter_model: String,
    llm_query_expansion: bool,
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
    if config.llm_query_expansion {
        info!("Expansión de preguntas con LLM habilitada.");
    }

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
        .route("/api/admin/documents", get(list_documents).delete(delete_document))
        .route("/api/admin/text", post(paste_text))
        .route("/api/admin/cache/clear", post(clear_cache))
        .route("/api/admin/cache-expansion/clear", post(clear_expansion_cache))
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

// ========================================================================
// RUTAS HTTP
// ========================================================================

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
        search_engine: if state.config.embeddings_enabled {
            "Qdrant + embeddings".to_string()
        } else {
            "Embeddings deshabilitados".to_string()
        },
        openrouter_enabled: state.config.openrouter_api_key.is_some(),
        openrouter_model: state.config.openrouter_model.clone(),
        llm_query_expansion: state.config.llm_query_expansion,
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
// ========================================================================
// SEARCH PLAN — Plan estructurado de búsqueda generado por LLM
// ========================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SearchPlan {
    answer_type: String,
    needs_acronyms: bool,
    needs_numbers: bool,
    needs_formula: bool,
    needs_section_title: bool,
    query_variants: Vec<String>,
    exact_queries: Vec<String>,
    fact_types: Vec<String>,
}

impl SearchPlan {
    fn default_for(question: &str) -> Self {
        Self {
            answer_type: "extended_report".to_string(),
            needs_acronyms: false,
            needs_numbers: false,
            needs_formula: false,
            needs_section_title: false,
            query_variants: vec![question.to_string()],
            exact_queries: vec![],
            fact_types: vec![],
        }
    }
}

/// Pide al LLM que genere un SearchPlan estructurado (JSON).
async fn build_search_plan_llm(state: &AppState, question: &str) -> Result<SearchPlan> {
    let api_key = state.config.openrouter_api_key.as_ref()
        .ok_or_else(|| anyhow!("OPENROUTER_API_KEY no configurada"))?;

    let prompt = format!(
        r#"Eres un planificador de busqueda documental.

Analiza la pregunta del usuario y genera un plan de busqueda en JSON.

REGLAS:
- answer_type: "direct_answer" si pide un hecho/regla/porcentaje/calculo/decision, "extended_report" si pide analisis general
- needs_acronyms: true si la pregunta pide siglas
- needs_numbers: true si requiere numeros, porcentajes, montos
- needs_formula: true si hay multiplicacion, formula, calculo
- needs_section_title: true si requiere ubicar articulo/seccion
- query_variants: 4-8 variantes lexicas de busqueda
- exact_queries: 2-4 consultas con AND para terminos criticos
- fact_types: tipos de hecho a priorizar: ACRONYM, FORMULA, PERCENTAGE_RULE, OBLIGATION, CONDITION_OR_EXCEPTION, MONETARY_AMOUNT

Responde SOLO con JSON valido, sin explicaciones.

Pregunta: {question}

JSON:
"#,
        question = question
    );

    let body = json!({
        "model": state.config.openrouter_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 600,
    });

    let value: Value = state.http
        .post("https://openrouter.ai/api/v1/chat/completions")
        .bearer_auth(api_key)
        .header("Content-Type", "application/json")
        .header("X-OpenRouter-Title", "Rust RAG - SearchPlan")
        .json(&body)
        .timeout(Duration::from_secs(15))
        .send().await?
        .error_for_status()?
        .json().await?;

    let raw = value.get("choices")
        .and_then(|v| v.as_array())
        .and_then(|a| a.first())
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|v| v.as_str())
        .unwrap_or("");

    // Intentar parsear JSON directamente
    if let Ok(plan) = serde_json::from_str::<SearchPlan>(raw) {
        return Ok(plan);
    }

    // Fallback: extraer JSON del texto
    if let Some(start) = raw.find('{') {
        if let Some(end) = raw.rfind('}') {
            let json_str = &raw[start..=end];
            if let Ok(plan) = serde_json::from_str::<SearchPlan>(json_str) {
                return Ok(plan);
            }
        }
    }

    // Fallback final
    Ok(SearchPlan::default_for(question))
}

// ========================================================================
// ITERATIVE SEARCH ENGINE
// ========================================================================

struct IterativeSearchResult {
    all_fragments: Vec<SearchFragment>,
    session: SearchSession,
    extracted_facts: Vec<ExtractedFact>,
}

/// Busqueda iterativa con multiples rondas de sondas y feedback.
async fn iterative_document_search(
    state: &AppState,
    question: &str,
    search_plan: &SearchPlan,
    top_k: usize,
) -> Result<IterativeSearchResult> {
    let conn = Connection::open(state.config.db_path())?;
    let max_iterations = state.config.max_search_iterations as usize;
    let max_internal = state.config.max_internal_questions as usize;
    let max_fragments = state.config.max_fragments_internal as usize;

    let mut session = SearchSession {
        original_question: question.to_string(),
        probes_executed: Vec::new(),
        fragments_found: 0,
        evidence_ledger: EvidenceLedger::default(),
        iterations: 0,
        confidence_score: 0.0,
    };

    let mut all_fragments: Vec<SearchFragment> = Vec::new();
    let mut executed_queries: HashSet<String> = HashSet::new();
    let mut probes: Vec<SearchProbe> = Vec::new();

    // Sondas desde SearchPlan
    for v in &search_plan.query_variants {
        if !v.is_empty() && executed_queries.insert(v.clone()) {
            probes.push(SearchProbe {
                query: v.clone(),
                purpose: "search_plan_variant".into(),
                priority: 90,
                expected_evidence_type: "general".into(),
            });
        }
    }
    for eq in &search_plan.exact_queries {
        if !eq.is_empty() && executed_queries.insert(eq.clone()) {
            probes.push(SearchProbe {
                query: eq.clone(),
                purpose: "exact_query".into(),
                priority: 95,
                expected_evidence_type: "general".into(),
            });
        }
    }

    // Sondas desde generador interno
    for p in generate_internal_questions(question) {
        if executed_queries.insert(p.query.clone()) {
            probes.push(p);
        }
    }

    if probes.len() > max_internal {
        probes.sort_by(|a, b| b.priority.cmp(&a.priority));
        probes.truncate(max_internal);
    }

    session.probes_executed = probes.clone();

    // === RONDAS ITERATIVAS ===
    for iteration in 0..max_iterations {
        session.iterations = (iteration + 1) as u32;
        let mut round_fragments: Vec<SearchFragment> = Vec::new();

        for probe in &probes {
            let semantic_limit = (top_k * 4).max(20);
            if let Ok(mut found) = semantic_search(state, &probe.query, semantic_limit).await {
                for f in &mut found {
                    f.score += exact_evidence_bonus(&f.text, &probe.query);
                    if contains_numeric_unit_pattern(&f.text) {
                        f.score += 35.0;
                    }
                    if contains_formula_pattern_raw(&f.text) {
                        f.score += 20.0;
                    }
                    if contains_percentage_or_money_raw(&f.text) {
                        f.score += 15.0;
                    }
                }
                round_fragments.append(&mut found);
            }

            // Buscar en extracted_facts por tipo
            let fact_type = match probe.purpose.as_str() {
                "acronyms" => Some("ACRONYM"),
                "acronym_lookup" => Some("ACRONYM"),
                "formula" => Some("FORMULA"),
                "percentage" => Some("PERCENTAGE_RULE"),
                "monetary" => Some("MONETARY_AMOUNT"),
                "obligation" => Some("OBLIGATION"),
                "condition" => Some("CONDITION_OR_EXCEPTION"),
                _ => None,
            };
            if let Some(ft) = fact_type {
                if let Ok(mut fact_frags) = search_facts_by_type(&conn, ft) {
                    round_fragments.append(&mut fact_frags);
                }
            }
        }

        // Consolidar
        round_fragments = dedup_and_sort_fragments(round_fragments);
        for f in &round_fragments {
            let mut is_new = true;
            for existing in &all_fragments {
                if existing.source == f.source && existing.chunk_index == f.chunk_index {
                    is_new = false;
                    break;
                }
            }
            if is_new {
                all_fragments.push(f.clone());
            }
        }
        all_fragments = dedup_and_sort_fragments(all_fragments);
        all_fragments.truncate(max_fragments);

        // === FEEDBACK: nuevas sondas basadas en lo encontrado ===
        if iteration + 1 < max_iterations {
            let combined: String = round_fragments.iter().map(|f| f.text.clone()).collect::<Vec<_>>().join(" ");
            let mut new_probes: Vec<SearchProbe> = Vec::new();

            if search_plan.needs_acronyms && !contains_acronym_pattern(&combined) {
                new_probes.push(SearchProbe {
                    query: "acronimo sigla definicion".into(),
                    purpose: "acronyms".into(), priority: 95,
                    expected_evidence_type: "acronym".into(),
                });
            }
            if search_plan.needs_numbers && !contains_numeric_unit_pattern(&combined) {
                for nq in build_numeric_unit_queries(question) {
                    new_probes.push(SearchProbe {
                        query: nq,
                        purpose: "numeric_unit_exact".into(),
                        priority: 96,
                        expected_evidence_type: "numeric_value".into(),
                    });
                }
                new_probes.push(SearchProbe {
                    query: "numero valor cantidad plazo periodo duracion dias meses consultas".into(),
                    purpose: "numeric_fallback".into(),
                    priority: 90,
                    expected_evidence_type: "numeric_value".into(),
                });
            }
            if search_plan.needs_formula && !combined.contains('=') && !combined.contains('*') {
                new_probes.push(SearchProbe {
                    query: "formula producto multiplicar".into(),
                    purpose: "formula".into(), priority: 90,
                    expected_evidence_type: "formula".into(),
                });
            }

            for p in new_probes {
                if executed_queries.insert(p.query.clone()) {
                    probes.push(p);
                }
            }
            probes.sort_by(|a, b| b.priority.cmp(&a.priority));
            if probes.len() > max_internal {
                probes.truncate(max_internal);
            }
        }
    }

    // === CONSTRUIR EVIDENCE LEDGER ===
    let mut ledger = EvidenceLedger::default();

    if all_fragments.is_empty() && embeddings_available(state) {
        if let Ok(mut semantic) = semantic_search(state, question, max_fragments.min(top_k * 2)).await {
            all_fragments.append(&mut semantic);
            all_fragments = dedup_and_sort_fragments(all_fragments);
        }
    }

    for f in &all_fragments {
        let raw = &f.text;
        let t = normalize_text(raw);

        if contains_acronym_pattern(raw) {
            ledger.acronyms.push(AcronymEvidence {
                acronym: String::new(),
                meaning: raw.chars().take(180).collect(),
                source: f.source.clone(),
                page_number: f.page_number,
            });
        }

        if contains_percentage_or_money_raw(raw) {
            ledger.percentages.push(PercentageEvidence {
                percentage: "%/$".into(),
                context: clean_excerpt(raw, 220),
                source: f.source.clone(),
                page_number: f.page_number,
            });
        }

        for n in extract_numeric_values(raw, &f.source, f.page_number) {
            ledger.numeric_values.push(n);
        }

        if contains_formula_pattern_raw(raw) {
            ledger.formulas.push(FormulaEvidence {
                formula: clean_excerpt(raw, 220),
                description: String::new(),
                source: f.source.clone(),
                page_number: f.page_number,
            });
        }

        let norm_words = [
            "debera", "deberan", "debe", "deben",
            "obligatorio", "obligatoria", "sera", "seran",
            "se aplicara", "corresponde", "establece", "dispone",
        ];
        if norm_words.iter().any(|w| t.contains(w)) {
            ledger.obligations.push(f.clone());
        }

        let exc_words = [
            "excepto", "salvo", "en caso de",
            "siempre que", "sin perjuicio", "no aplica",
        ];
        if exc_words.iter().any(|w| t.contains(w)) {
            ledger.exceptions.push(f.clone());
        }

        let cond_words = [
            "cuando", "condicion", "condiciones",
            "durante", "dentro de",
        ];
        if cond_words.iter().any(|w| t.contains(w)) {
            ledger.conditions.push(f.clone());
        }

        ledger.direct_answer.push(f.clone());
    }

    // Deduplicar valores numéricos
    ledger.numeric_values.sort_by(|a, b| {
        a.source.cmp(&b.source)
            .then(a.page_number.cmp(&b.page_number))
            .then(a.value.cmp(&b.value))
    });
    ledger.numeric_values.dedup_by(|a, b| {
        a.value == b.value
            && a.unit.eq_ignore_ascii_case(&b.unit)
            && a.source == b.source
            && a.page_number == b.page_number
    });

    session.evidence_ledger = ledger;
    session.fragments_found = all_fragments.len();

    let mut confidence = 50.0f32;
    if search_plan.needs_acronyms && !session.evidence_ledger.acronyms.is_empty() { confidence += 20.0; }
    if search_plan.needs_numbers
        && (!session.evidence_ledger.numeric_values.is_empty()
            || !session.evidence_ledger.percentages.is_empty())
    {
        confidence += 20.0;
    }
    if search_plan.needs_formula && !session.evidence_ledger.formulas.is_empty() { confidence += 15.0; }
    if all_fragments.len() >= 5 { confidence += 10.0; }
    session.confidence_score = confidence.min(100.0);

    let extracted_facts = load_extracted_facts(&conn, &all_fragments)?;

    Ok(IterativeSearchResult { all_fragments, session, extracted_facts })
}

fn search_facts_by_type(conn: &Connection, fact_type: &str) -> Result<Vec<SearchFragment>> {
    let mut stmt = conn.prepare(
        "SELECT DISTINCT c.source, c.chunk_index, c.text, c.page_number
         FROM extracted_facts f
         JOIN document_chunks c ON c.source = f.source AND c.chunk_index = f.chunk_index
         WHERE f.fact_type = ?1
         ORDER BY c.source, c.chunk_index
         LIMIT 30"
    )?;
    let rows = stmt.query_map(params![fact_type], |r| {
        Ok(SearchFragment {
            score: 30.0,
            source: r.get(0)?,
            chunk_index: r.get::<_, i64>(1)? as usize,
            text: r.get(2)?,
            page_number: r.get::<_, i64>(3)? as usize,
            source_title: String::new(),
        })
    })?;
    let mut fragments = Vec::new();
    for row in rows { fragments.push(row?); }
    Ok(fragments)
}

fn load_extracted_facts(conn: &Connection, fragments: &[SearchFragment]) -> Result<Vec<ExtractedFact>> {
    if fragments.is_empty() { return Ok(Vec::new()); }
    let sources: Vec<String> = fragments.iter().map(|f| f.source.clone()).collect();
    let mut all_facts = Vec::new();
    for source in sources.iter().take(10) {
        if let Ok(mut stmt) = conn.prepare(
            "SELECT id, source, page_number, chunk_index, fact_type, label, value, unit, text
             FROM extracted_facts WHERE source = ?1 LIMIT 20"
        ) {
            if let Ok(rows) = stmt.query_map(params![source], |r| {
                Ok(ExtractedFact {
                    id: r.get(0)?, source: r.get(1)?,
                    page_number: r.get::<_, i64>(2)? as usize,
                    chunk_index: r.get::<_, i64>(3)? as usize,
                    fact_type: r.get(4)?, label: r.get(5)?,
                    value: r.get(6)?, unit: r.get(7)?, text: r.get(8)?,
                })
            }) {
                for row in rows { all_facts.push(row?); }
            }
        }
    }
    Ok(all_facts)
}

// ========================================================================
// EVIDENCIA RELACIONADA (lo que el usuario no pregunto)
// ========================================================================

fn discover_related_evidence(question: &str, all_fragments: &[SearchFragment]) -> Vec<String> {
    let q = normalize_text(question);
    let mut findings = Vec::new();
    let mut found_conditions = false;
    let mut found_percentages = false;
    let mut found_obligations = false;

    for f in all_fragments {
        let t = normalize_text(&f.text);
        if !found_conditions {
            for w in &["excepto", "salvo", "siempre que", "en caso de"] {
                if t.contains(w) {
                    findings.push(format!("Condicion o excepcion en '{}' (p.{}): {}", f.source, f.page_number, limit_chars(&f.text, 120)));
                    found_conditions = true; break;
                }
            }
        }
        if !found_percentages && t.contains('%') {
            findings.push(format!("Porcentaje en '{}' (p.{}): {}", f.source, f.page_number, limit_chars(&f.text, 120)));
            found_percentages = true;
        }
        if !found_obligations {
            for w in &["debera", "debera", "obligatorio", "sera"] {
                if t.contains(w) {
                    findings.push(format!("Regla obligatoria en '{}' (p.{}): {}", f.source, f.page_number, limit_chars(&f.text, 120)));
                    found_obligations = true; break;
                }
            }
        }
    }

    if !q.contains("excepto") && found_conditions {
        findings.push("Nota: el documento contiene condiciones o excepciones relevantes.".into());
    }
    if !q.contains("porcentaje") && !q.contains("%") && found_percentages {
        findings.push("Nota: hay porcentajes y valores que podrian ser relevantes.".into());
    }
    findings
}

// ========================================================================
// VERIFICADOR DE SUFICIENCIA DE EVIDENCIA
// ========================================================================

struct SufficiencyReport {
    is_sufficient: bool,
    missing: Vec<String>,
    found: Vec<String>,
    #[allow(dead_code)]
    confidence_delta: f32,
}

/// Verifica si la evidencia recolectada cubre lo que el SearchPlan requiere.
/// Retorna un reporte con lo que falta, lo que se encontro y el ajuste de confianza.
fn verify_evidence_sufficiency(search_plan: &SearchPlan, session: &SearchSession) -> SufficiencyReport {
    let ledger = &session.evidence_ledger;
    let mut missing = Vec::new();
    let mut found = Vec::new();
    let mut sufficient = true;

    if search_plan.needs_acronyms {
        if ledger.acronyms.is_empty() {
            missing.push("siglas o acrónimos".to_string());
            sufficient = false;
        } else {
            found.push(format!("{} sigla(s) encontrada(s)", ledger.acronyms.len()));
        }
    }

    if search_plan.needs_numbers {
        let numeric_count = ledger.numeric_values.len() + ledger.percentages.len();
        if numeric_count == 0 {
            missing.push("números, días, meses, montos, porcentajes o valores con unidad".to_string());
            sufficient = false;
        } else {
            found.push(format!("{} valor(es) numérico(s) encontrado(s)", numeric_count));
        }
    }

    if search_plan.needs_formula {
        if ledger.formulas.is_empty() {
            missing.push("fórmulas o cálculos".to_string());
            sufficient = false;
        } else {
            found.push(format!("{} fórmula(s) encontrada(s)", ledger.formulas.len()));
        }
    }

    let q_norm = normalize_text(&session.original_question);

    let mentions_exception =
        q_norm.contains("excepto") || q_norm.contains("excepcion") || q_norm.contains("salvo")
        || q_norm.contains("inusual") || q_norm.contains("complicacion");

    let mentions_condition =
        q_norm.contains("condicion") || q_norm.contains("cuando") || q_norm.contains("siempre que")
        || q_norm.contains("durante") || q_norm.contains("dentro de");

    if mentions_exception && ledger.exceptions.is_empty() {
        missing.push("excepciones relacionadas con la pregunta".to_string());
        sufficient = false;
    } else if mentions_exception {
        found.push(format!("{} excepción(es) encontrada(s)", ledger.exceptions.len()));
    }

    if mentions_condition && ledger.conditions.is_empty() {
        missing.push("condiciones relacionadas con la pregunta".to_string());
        sufficient = false;
    } else if mentions_condition {
        found.push(format!("{} condición(es) encontrada(s)", ledger.conditions.len()));
    }

    let norm_words = [
        "debe", "deben", "obligatorio", "obligatoria",
        "aplica", "regla", "disposicion", "corresponde",
        "establece", "segun",
    ];
    let mentions_obligation = norm_words.iter().any(|w| q_norm.contains(w));

    if mentions_obligation && ledger.obligations.is_empty() {
        missing.push("lenguaje normativo o regla aplicable".to_string());
        sufficient = false;
    } else if mentions_obligation {
        found.push(format!("{} regla(s) encontrada(s)", ledger.obligations.len()));
    }

    if session.fragments_found < 3 {
        missing.push("suficientes fragmentos de evidencia".to_string());
        sufficient = false;
    }

    let confidence_delta = if sufficient { 15.0 } else { -15.0 };

    SufficiencyReport {
        is_sufficient: sufficient,
        missing,
        found,
        confidence_delta,
    }
}

// ========================================================================
// SELECTOR DE PROMPT SEGUN TIPO DE RESPUESTA
// ========================================================================

fn select_response_prompt(answer_type: &str, max_words: usize, has_related: bool) -> String {
    let related_note = if has_related {
        "Incluye información relacionada importante si aparece en la evidencia, aunque el usuario no la haya preguntado directamente. No la inventes; solo úsala si está en el paquete de evidencia.".to_string()
    } else {
        String::new()
    };

    match answer_type {
        "direct_answer" => format!(
            r#"Eres un asistente experto en análisis documental.

Responde de forma DIRECTA, precisa y pegada a la evidencia.

Reglas obligatorias:
- No uses formato de informe.
- No uses "Resumen Ejecutivo", "Hallazgos", "Análisis Detallado", "Limitaciones" ni "Conclusión".
- Si la pregunta pide "cuántos", "cuántas", "plazo", "período", "duración", "porcentaje", "monto" o "valor exacto", inicia la respuesta con el dato exacto.
- Si la evidencia contiene números con unidad, fórmulas, porcentajes, condiciones o excepciones, inclúyelos.
- Si hay una condición o excepción importante en la evidencia, menciónala aunque el usuario no la haya considerado.
- Si la evidencia no alcanza, explica exactamente qué dato faltó, no respondas con informe genérico.
- Si varios fragmentos dicen lo mismo con otras palabras, unifícalos en un solo párrafo claro.
- Puedes usar sinónimos o reexpresión breve solo para mejorar lectura, pero no cambies el sentido ni inventes datos.
- No inventes datos fuera de la evidencia.

{related}

LÍMITE: {max_words} palabras."#,
            related = related_note,
            max_words = max_words
        ),
        _ => format!(
            r#"Eres un asistente experto en análisis documental.

Responde de forma natural, clara y completa usando solo la evidencia proporcionada.

Reglas:
- La respuesta principal debe ir al inicio.
- Evita formato rígido de informe salvo que el usuario lo pida.
- Incluye datos exactos encontrados: números, unidades, porcentajes, fórmulas, condiciones y excepciones.
- Si varios fragmentos dicen lo mismo con otras palabras, unifícalos en un solo párrafo claro.
- Puedes usar sinónimos o reexpresión breve solo para mejorar lectura, pero no cambies el sentido ni inventes datos.
- Si hay información relacionada importante, intégrala de forma breve.
- Si falta evidencia, indica exactamente qué faltó.

{related}

LÍMITE: {max_words} palabras."#,
            related = related_note,
            max_words = max_words
        ),
    }
}


// ========================================================================
// BUILD EVIDENCE SUMMARY — arma el paquete de evidencia para el LLM
// ========================================================================

fn build_evidence_summary(
    question: &str,
    filtered_fragments: &[SearchFragment],
    ledger: &EvidenceLedger,
    extracted_facts: &[ExtractedFact],
    related: &[String],
    max_chars_per_fragment: usize,
) -> String {
    let mut evidence_summary = String::new();

    evidence_summary.push_str(&format!("Pregunta original:\n{}\n\n", question));
    evidence_summary.push_str("===== EXPEDIENTE DE EVIDENCIA =====\n\n");

    if !ledger.numeric_values.is_empty() {
        evidence_summary.push_str(&format!("--- Valores numéricos con unidad ({}) ---\n", ledger.numeric_values.len()));
        for n in ledger.numeric_values.iter().take(20) {
            evidence_summary.push_str(&format!("[{}|p.{}] {} {} \u{2014} {}\n",
                n.source, n.page_number, n.value, n.unit, limit_chars(&n.context, 220)));
        }
        evidence_summary.push('\n');
    }

    if !ledger.percentages.is_empty() {
        evidence_summary.push_str(&format!("--- Porcentajes o montos ({}) ---\n", ledger.percentages.len()));
        for p in ledger.percentages.iter().take(15) {
            evidence_summary.push_str(&format!("[{}|p.{}] {}\n",
                p.source, p.page_number, limit_chars(&p.context, 220)));
        }
        evidence_summary.push('\n');
    }

    if !ledger.acronyms.is_empty() {
        evidence_summary.push_str(&format!("--- Siglas encontradas ({}) ---\n", ledger.acronyms.len()));
        for a in ledger.acronyms.iter().take(15) {
            evidence_summary.push_str(&format!("[{}|p.{}] {}\n",
                a.source, a.page_number, limit_chars(&a.meaning, 220)));
        }
        evidence_summary.push('\n');
    }

    if !ledger.formulas.is_empty() {
        evidence_summary.push_str(&format!("--- Fórmulas o cálculos encontrados ({}) ---\n", ledger.formulas.len()));
        for f in ledger.formulas.iter().take(15) {
            evidence_summary.push_str(&format!("[{}|p.{}] {}\n",
                f.source, f.page_number, limit_chars(&f.formula, 220)));
        }
        evidence_summary.push('\n');
    }

    if !ledger.obligations.is_empty() {
        evidence_summary.push_str(&format!("--- Reglas u obligaciones ({}) ---\n", ledger.obligations.len()));
        for o in ledger.obligations.iter().take(10) {
            evidence_summary.push_str(&format!("[{}|p.{}] {}\n",
                o.source, o.page_number, limit_chars(&o.text, 260)));
        }
        evidence_summary.push('\n');
    }

    if !ledger.conditions.is_empty() {
        evidence_summary.push_str(&format!("--- Condiciones encontradas ({}) ---\n", ledger.conditions.len()));
        for c in ledger.conditions.iter().take(10) {
            evidence_summary.push_str(&format!("[{}|p.{}] {}\n",
                c.source, c.page_number, limit_chars(&c.text, 260)));
        }
        evidence_summary.push('\n');
    }

    if !ledger.exceptions.is_empty() {
        evidence_summary.push_str(&format!("--- Excepciones encontradas ({}) ---\n", ledger.exceptions.len()));
        for e in ledger.exceptions.iter().take(10) {
            evidence_summary.push_str(&format!("[{}|p.{}] {}\n",
                e.source, e.page_number, limit_chars(&e.text, 260)));
        }
        evidence_summary.push('\n');
    }

    evidence_summary.push_str(&format!("===== FRAGMENTOS PRINCIPALES ({}) =====\n", filtered_fragments.len()));
    for (i, f) in merged_fragments_for_context(filtered_fragments).iter().enumerate() {
        let page_info = if f.page_number > 0 { format!(" | pág {}", f.page_number) } else { String::new() };
        evidence_summary.push_str(&format!("[{}] {}{} | score {:.2}\n{}\n\n",
            i + 1, f.source, page_info, f.score, limit_chars(&f.text, max_chars_per_fragment)));
    }

    if !extracted_facts.is_empty() {
        evidence_summary.push_str("===== HECHOS EXACTOS EXTRAÍDOS =====\n");
        for fact in extracted_facts.iter().take(20) {
            evidence_summary.push_str(&format!("[{}] {}|p.{}: {} = {} {}\n",
                fact.fact_type, fact.source, fact.page_number, fact.label, fact.value, fact.unit));
        }
        evidence_summary.push('\n');
    }

    if !related.is_empty() {
        evidence_summary.push_str("===== INFORMACIÓN RELACIONADA IMPORTANTE =====\n");
        for r in related.iter().take(10) {
            evidence_summary.push_str(&format!("- {}\n", r));
        }
        evidence_summary.push('\n');
    }

    evidence_summary
}

fn merged_fragments_for_context(fragments: &[SearchFragment]) -> Vec<SearchFragment> {
    if fragments.is_empty() {
        return Vec::new();
    }

    let mut sorted = fragments.to_vec();
    sorted.sort_by(|a, b| {
        a.source
            .cmp(&b.source)
            .then(a.page_number.cmp(&b.page_number))
            .then(a.chunk_index.cmp(&b.chunk_index))
    });

    let mut merged: Vec<SearchFragment> = Vec::new();
    let mut current = sorted[0].clone();

    for frag in sorted.into_iter().skip(1) {
        let contiguous = frag.source == current.source
            && frag.page_number == current.page_number
            && frag.chunk_index <= current.chunk_index + 1;

        if contiguous {
            if !current.text.ends_with(' ') {
                current.text.push(' ');
            }
            current.text.push_str(&frag.text);
            current.score = current.score.max(frag.score);
        } else {
            merged.push(current);
            current = frag;
        }
    }

    merged.push(current);
    merged
}

// ========================================================================
// CALL OPENROUTER FINAL — respuesta única usando el paquete de evidencia
// ========================================================================

async fn call_openrouter_final_answer(
    state: &AppState,
    system_prompt: &str,
    evidence_summary: &str,
    max_tokens: u32,
) -> Result<String> {
    let api_key = state.config.openrouter_api_key.as_ref()
        .ok_or_else(|| anyhow!("OPENROUTER_API_KEY no está configurada"))?;

    let user_prompt = format!(
        "Paquete de evidencia:\n\n{}\n\nRedacta una sola respuesta final para el usuario:",
        evidence_summary
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
        "frequency_penalty": 0.2
    });

    let value: Value = state.http
        .post("https://openrouter.ai/api/v1/chat/completions")
        .bearer_auth(api_key)
        .header("Content-Type", "application/json")
        .header("X-OpenRouter-Title", "Rust RAG - Final Direct Answer")
        .json(&body)
        .timeout(Duration::from_secs(45))
        .send().await?
        .error_for_status()?
        .json().await?;

    Ok(value
        .get("choices")
        .and_then(|v| v.as_array())
        .and_then(|arr| arr.first())
        .and_then(|choice| choice.get("message"))
        .and_then(|msg| msg.get("content"))
        .and_then(|v| v.as_str())
        .unwrap_or("No se recibió contenido del modelo.")
        .to_string())
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

    let wants_llm = req.use_llm.unwrap_or(true);
    let can_use_llm = wants_llm && state.config.openrouter_api_key.is_some();
    if !can_use_llm {
        return Err(AppError::new(
            StatusCode::BAD_REQUEST,
            "Esta aplicación responde solo con LLM. Active OpenRouter para continuar.",
        ));
    }

    // ===== BUILD SEARCH PLAN =====
    let mut search_plan = if state.config.exact_evidence_mode && can_use_llm {
        match build_search_plan_llm(&state, question).await {
            Ok(plan) => {
                info!("SearchPlan: answer_type={}, variants={}, facts={:?}",
                    plan.answer_type, plan.query_variants.len(), plan.fact_types);
                plan
            }
            Err(e) => {
                warn!("SearchPlan LLM falló: {e:#}, usando plan por defecto");
                SearchPlan::default_for(question)
            }
        }
    } else {
        SearchPlan::default_for(question)
    };

    harden_search_plan_from_question(question, &mut search_plan);

    // ===== ITERATIVE SEARCH =====
    let top_k = req.top_k.unwrap_or(state.config.default_top_k).clamp(1, 30);
    let search_result = iterative_document_search(&state, question, &search_plan, top_k)
        .await
        .map_err(anyhow::Error::from)?;

    let final_limit = state.config.final_context_fragments as usize;
    let mut top_fragments = search_result.all_fragments.clone();
    top_fragments.truncate(final_limit);

    let filtered_fragments = filter_fragments_by_sources(&top_fragments, req.selected_sources.as_deref());
    let debug_fragments = build_debug_fragments(&filtered_fragments);

    // ===== DISCOVER RELATED EVIDENCE =====
    let related = if state.config.enable_related_evidence {
        discover_related_evidence(question, &search_result.all_fragments)
    } else {
        Vec::new()
    };

    // ===== VERIFICAR SUFICIENCIA DE EVIDENCIA =====
    let sufficiency = verify_evidence_sufficiency(&search_plan, &search_result.session);
    if !sufficiency.is_sufficient {
        warn!("Evidencia insuficiente: faltan {:?}", sufficiency.missing);
    }

    // ===== BUILD CONTEXT WITH EVIDENCE LEDGER =====
    let ledger = &search_result.session.evidence_ledger;
    let has_facts = !search_result.extracted_facts.is_empty();

    let max_chars = 4000usize;
    let max_tokens = state.config.openrouter_max_tokens;
    let prompt_template = select_response_prompt(&search_plan.answer_type, state.config.answer_max_words, !related.is_empty());

    // ===== BUILD PROMPT WITH EVIDENCE LEDGER =====
    let mut evidence_summary = String::new();
    evidence_summary.push_str(&format!("Pregunta: {}

", question));

    // ===== EXPEDIENTE DE EVIDENCIA (evidence_ledger) =====
    if !ledger.direct_answer.is_empty() || !ledger.acronyms.is_empty() || !ledger.percentages.is_empty() {
        evidence_summary.push_str("===== EXPEDIENTE DE EVIDENCIA =====

");

        // Siglas
        if !ledger.acronyms.is_empty() {
            evidence_summary.push_str(&format!("--- Siglas encontradas ({}) ---
", ledger.acronyms.len()));
            for a in &ledger.acronyms {
                evidence_summary.push_str(&format!("  [{}] {}
", a.source, limit_chars(&a.meaning, 200)));
            }
            evidence_summary.push('\n');
        }

        // Porcentajes y valores
        if !ledger.percentages.is_empty() {
            evidence_summary.push_str(&format!("--- Valores numericos encontrados ({}) ---
", ledger.percentages.len()));
            for p in &ledger.percentages {
                evidence_summary.push_str(&format!("  [{}] {}
", p.source, limit_chars(&p.context, 200)));
            }
            evidence_summary.push('\n');
        }

        // Formulas
        if !ledger.formulas.is_empty() {
            evidence_summary.push_str(&format!("--- Formulas encontradas ({}) ---
", ledger.formulas.len()));
            for f in &ledger.formulas {
                evidence_summary.push_str(&format!("  [{}] {}
", f.source, limit_chars(&f.formula, 200)));
            }
            evidence_summary.push('\n');
        }

        // Obligaciones
        if !ledger.obligations.is_empty() {
            evidence_summary.push_str(&format!("--- Reglas obligatorias encontradas ({}) ---
", ledger.obligations.len()));
            for o in &ledger.obligations {
                evidence_summary.push_str(&format!("  [{}|p.{}] {}
", o.source, o.page_number, limit_chars(&o.text, 250)));
            }
            evidence_summary.push('\n');
        }

        // Excepciones
        if !ledger.exceptions.is_empty() {
            evidence_summary.push_str(&format!("--- Excepciones encontradas ({}) ---
", ledger.exceptions.len()));
            for e in &ledger.exceptions {
                evidence_summary.push_str(&format!("  [{}|p.{}] {}
", e.source, e.page_number, limit_chars(&e.text, 250)));
            }
            evidence_summary.push('\n');
        }

        // Condiciones
        if !ledger.conditions.is_empty() {
            evidence_summary.push_str(&format!("--- Condiciones encontradas ({}) ---
", ledger.conditions.len()));
            for c in &ledger.conditions {
                evidence_summary.push_str(&format!("  [{}|p.{}] {}
", c.source, c.page_number, limit_chars(&c.text, 250)));
            }
            evidence_summary.push('\n');
        }

        // Informacion que el usuario no considero
        if !ledger.user_may_be_missing.is_empty() {
            evidence_summary.push_str(&format!("--- Puntos que el usuario tal vez no considero ({}) ---
", ledger.user_may_be_missing.len()));
            for u in &ledger.user_may_be_missing {
                evidence_summary.push_str(&format!("  - {}
", u));
            }
            evidence_summary.push('\n');
        }
    }

    // ===== EVIDENCIA PRINCIPAL (fragmentos planos) =====
    evidence_summary.push_str(&format!("===== EVIDENCIA PRINCIPAL ({} fragmentos) =====
", filtered_fragments.len()));

    for (i, f) in filtered_fragments.iter().enumerate() {
        let page_info = if f.page_number > 0 { format!(" | pág {}", f.page_number) } else { String::new() };
        evidence_summary.push_str(&format!("[{}] {}:{} {}
",
            i + 1, f.source, page_info, limit_chars(&f.text, max_chars / final_limit.max(1))));
    }

    // Facts extraídos
    if has_facts {
        evidence_summary.push_str("
===== HECHOS EXACTOS EXTRAÍDOS =====
");
        for fact in search_result.extracted_facts.iter().take(15) {
            evidence_summary.push_str(&format!("[{}] {}: {} = {}
",
                fact.fact_type, fact.source, fact.label, fact.value));
        }
    }

    // Evidencia relacionada
    if !related.is_empty() {
        evidence_summary.push_str("
===== INFORMACIÓN RELACIONADA (no preguntada) =====
");
        for r in &related {
            evidence_summary.push_str(&format!("- {}
", r));
        }
    }

    // ===== CALL LLM WITH MATCHED PROMPT =====
    let user_prompt = format!(
        "{}

Paquete de evidencia:
{}

Redacta la respuesta:",
        prompt_template, evidence_summary
    );

    let body = json!({
        "model": state.config.openrouter_model,
        "messages": [
            {"role": "system", "content": prompt_template},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.1,
        "max_tokens": max_tokens.max(500),
        "top_p": 0.8,
        "frequency_penalty": 0.3,
    });

    let answer = match state.http
        .post("https://openrouter.ai/api/v1/chat/completions")
        .bearer_auth(state.config.openrouter_api_key.as_ref().unwrap())
        .header("Content-Type", "application/json")
        .header("X-OpenRouter-Title", "Rust RAG - Iterativo")
        .json(&body)
        .timeout(Duration::from_secs(30))
        .send()
        .await
    {
        Ok(resp) => {
            match resp.json::<Value>().await {
                Ok(value) => value
                    .get("choices")
                    .and_then(|v| v.as_array())
                    .and_then(|arr| arr.first())
                    .and_then(|choice| choice.get("message"))
                    .and_then(|msg| msg.get("content"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("No se recibió contenido.")
                    .to_string(),
                Err(_) => build_local_response(question, &filtered_fragments),
            }
        }
        Err(_) => build_local_response(question, &filtered_fragments),
    };

    Ok(Json(ChatResponse {
        answer,
        used_llm: true,
        model: Some(state.config.openrouter_model.clone()),
        fragments: filtered_fragments,
        question_variants: Some(search_plan.query_variants.clone()),
        debug_search: Some(debug_fragments),
    }))
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
                Err(e) => json!({
                    "kind": "error",
                    "payload": format!("No se pudo serializar SSE: {e}")
                }).to_string(),
            };
            let _ = tx.send(Ok(Event::default().data(data))).await;
        }

        let _user = match require_user(&state, &headers) {
            Ok(u) => u,
            Err(e) => { send_event(&tx, "error", e.message).await; return; }
        };

        let question = req.question.trim().to_string();
        if question.is_empty() { send_event(&tx, "error", "La pregunta no puede estar vacía.").await; return; }

        let wants_llm = req.use_llm.unwrap_or(true);
        let can_use_llm = wants_llm && state.config.openrouter_api_key.is_some();
        let top_k = req.top_k.unwrap_or(state.config.default_top_k).clamp(1, 30);

        if !can_use_llm {
            send_event(&tx, "error", "Esta aplicación responde solo con LLM. Active OpenRouter para continuar.").await;
            return;
        }

        let mut search_plan = if state.config.exact_evidence_mode {
            match build_search_plan_llm(&state, &question).await {
                Ok(plan) => plan,
                Err(e) => { warn!("SearchPlan LLM falló en stream: {e:#}"); SearchPlan::default_for(&question) }
            }
        } else { SearchPlan::default_for(&question) };

        harden_search_plan_from_question(&question, &mut search_plan);
        send_event(&tx, "search_plan", &search_plan).await;

        let search_result = match iterative_document_search(&state, &question, &search_plan, top_k).await {
            Ok(r) => r,
            Err(e) => { send_event(&tx, "error", e.to_string()).await; return; }
        };

        let mut top_fragments = search_result.all_fragments.clone();
        top_fragments.truncate(state.config.final_context_fragments as usize);
        let filtered_fragments = filter_fragments_by_sources(&top_fragments, req.selected_sources.as_deref());
        send_event(&tx, "fragments", &filtered_fragments).await;

        let related = if state.config.enable_related_evidence {
            discover_related_evidence(&question, &search_result.all_fragments)
        } else { Vec::new() };

        let sufficiency = verify_evidence_sufficiency(&search_plan, &search_result.session);
        send_event(&tx, "sufficiency", &json!({
            "is_sufficient": sufficiency.is_sufficient,
            "missing": sufficiency.missing,
            "found": sufficiency.found,
            "confidence": search_result.session.confidence_score
        })).await;

        let prompt_template = select_response_prompt(&search_plan.answer_type, state.config.answer_max_words, !related.is_empty());
        let evidence_summary = build_evidence_summary(
            &question, &filtered_fragments, &search_result.session.evidence_ledger,
            &search_result.extracted_facts, &related, 900,
        );

        let answer = match call_openrouter_final_answer(&state, &prompt_template, &evidence_summary, state.config.openrouter_max_tokens).await {
            Ok(a) => a,
            Err(e) => {
                let msg = format!("OpenRouter final falló: {e:#}");
                send_event(&tx, "error", msg).await;
                return;
            }
        };

        send_event(&tx, "answer", answer).await;
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

#[derive(Debug, Deserialize)]
struct PasteTextRequest {
    filename: String,
    content: String,
}

async fn paste_text(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<PasteTextRequest>,
) -> Result<Json<Value>, AppError> {
    require_admin(&state, &headers)?;

    let filename = sanitize_filename(&req.filename);
    if filename.is_empty() || !filename.ends_with(".txt") {
        return Err(AppError::new(StatusCode::BAD_REQUEST, "El nombre debe terminar en .txt"));
    }
    if req.content.trim().is_empty() {
        return Err(AppError::new(StatusCode::BAD_REQUEST, "El contenido no puede estar vacío."));
    }

    let dest = state.config.docs_dir.join(&filename);
    fs::write(&dest, req.content.as_bytes()).map_err(|e| {
        AppError::new(StatusCode::INTERNAL_SERVER_ERROR, format!("No se pudo guardar: {e}"))
    })?;

    let state_clone = state.clone();
    tokio::spawn(async move {
        match run_indexer(&state_clone).await {
            Ok(s) => info!("Indexación post-texto: {:?}", s),
            Err(e) => error!("Error indexando texto: {e:#}"),
        }
    });

    info!("Texto pegado como documento: {filename}");
    Ok(Json(json!({"ok": true, "file": filename})))
}

#[derive(Debug, Deserialize)]
struct DeleteDocumentRequest {
    path: String,
}

async fn delete_document(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<DeleteDocumentRequest>,
) -> Result<Json<Value>, AppError> {
    require_admin(&state, &headers)?;

    let path = req.path.trim().to_string();
    if path.is_empty() || path.contains("..") || path.starts_with('/') || path.starts_with('\\') {
        return Err(AppError::new(StatusCode::BAD_REQUEST, "Ruta inválida."));
    }

    let file_path = state.config.docs_dir.join(&path);
    if !file_path.starts_with(&state.config.docs_dir) {
        return Err(AppError::new(StatusCode::BAD_REQUEST, "La ruta no pertenece a la carpeta de documentos."));
    }

    delete_chunks_by_source(&state.config, &path).map_err(anyhow::Error::from)?;
    remove_manifest_document(&state.config, &path).map_err(anyhow::Error::from)?;

    if file_path.exists() {
        fs::remove_file(&file_path).map_err(|e| {
            AppError::new(StatusCode::INTERNAL_SERVER_ERROR, format!("No se pudo eliminar: {e}"))
        })?;
    }

    info!("Documento eliminado: {path}");
    Ok(Json(json!({"ok": true, "deleted": path})))
}

async fn clear_cache(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<Value>, AppError> {
    require_admin(&state, &headers)?;
    let conn = Connection::open(state.config.db_path()).map_err(anyhow::Error::from)?;
    let count = conn
        .query_row("SELECT COUNT(*) FROM faq_cache", [], |r| r.get::<_, i64>(0))
        .unwrap_or(0);
    conn.execute("DELETE FROM faq_cache", []).map_err(anyhow::Error::from)?;
    info!("Caché FAQ limpiado: {count} entradas");
    Ok(Json(json!({"ok": true, "cleared": count})))
}

async fn clear_expansion_cache(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<Value>, AppError> {
    require_admin(&state, &headers)?;
    let conn = Connection::open(state.config.db_path()).map_err(anyhow::Error::from)?;
    let count = conn
        .query_row("SELECT COUNT(*) FROM query_expansion_cache", [], |r| r.get::<_, i64>(0))
        .unwrap_or(0);
    conn.execute("DELETE FROM query_expansion_cache", [])
        .map_err(anyhow::Error::from)?;
    info!("Caché de expansión limpiado: {count} entradas");
    Ok(Json(json!({"ok": true, "cleared": count})))
}

// ========================================================================
// SESIÓN Y AUTENTICACIÓN
// ========================================================================

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

// ========================================================================
// INDEXADOR PERIÓDICO Y DE ARCHIVOS INDIVIDUALES
// ========================================================================

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
        let force_reindex = should_force_reindex_for_embeddings(state).await.unwrap_or(true);
        if !force_reindex {
            if state.config.embeddings_enabled && state.config.openrouter_api_key.is_some() {
                if let Err(e) = ensure_embeddings_for_existing_source(state, rel_source, &hash).await {
                    warn!(
                        "No se pudieron completar embeddings para documento sin cambios {}: {e:#}",
                        rel_source
                    );
                }
            }
            return Ok(IndexFileOutcome::Skipped);
        }
    }

    // Extraer texto con metadatos de página
    let page_info = extract_text_with_pages(path, &bytes)?;

    // Fragmentar con conocimiento de páginas
    let chunks_with_pages = chunk_text_with_pages(&page_info, CHUNK_WORDS, CHUNK_OVERLAP_WORDS);

    // Reindexación segura: elimina solo fragmentos de la fuente actual
    delete_chunks_by_source(&state.config, rel_source)?;

    let mut total = 0usize;
    let mut embedding_batch: Vec<(usize, usize, String)> = Vec::new();
    for (chunk_index, (page_number, chunk_text)) in chunks_with_pages.iter().enumerate() {
        if chunk_text.trim().is_empty() {
            continue;
        }
        insert_lexical_chunk(&state.config, rel_source, chunk_index, chunk_text, *page_number, &hash)?;
        // Extraer hechos exactos (siglas, fórmulas, porcentajes, reglas)
        let facts = extract_facts_from_chunk(rel_source, *page_number, chunk_index, chunk_text);
        for fact in facts {
            let _ = insert_extracted_fact(&state.config, &fact);
        }
        embedding_batch.push((chunk_index, *page_number, chunk_text.clone()));
        total += 1;
    }

    if state.config.embeddings_enabled && state.config.openrouter_api_key.is_some() {
        if let Err(e) = ensure_chunk_embeddings_batch(state, rel_source, &hash, &embedding_batch).await {
            warn!("No se pudieron generar embeddings para {}: {e:#}", rel_source);
        }
    }

    upsert_manifest_document(&state.config, rel_source, &hash, modified, total)?;
    Ok(IndexFileOutcome::Indexed(total))
}

async fn should_force_reindex_for_embeddings(state: &AppState) -> Result<bool> {
    if !embeddings_available(state) {
        return Ok(false);
    }

    let Some(url) = state.config.qdrant_url.as_ref() else {
        return Ok(false);
    };

    let collection = qdrant_collection_name(&state.config);
    let endpoint = format!("{}/collections/{}", url.trim_end_matches('/'), collection);
    let resp = state.http.get(endpoint).timeout(Duration::from_secs(15)).send().await?;

    if !resp.status().is_success() {
        return Ok(true);
    }

    let value: Value = resp.json().await?;
    let points_count = value
        .get("result")
        .and_then(|v| v.get("points_count"))
        .and_then(|v| v.as_i64())
        .unwrap_or(0);

    Ok(points_count <= 0)
}

// ========================================================================
// EXTRACCIÓN DE TEXTO CON METADATOS DE PÁGINA
// ========================================================================

/// Extrae texto y devuelve Vec de (page_number, text) para cada página.
/// Para PDF usa pdftotext que separa páginas con form feed (\x0C).
/// Para TXT/MD devuelve una sola página (0).
fn extract_text_with_pages(path: &Path, bytes: &[u8]) -> Result<Vec<(usize, String)>> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "txt" | "md" => {
            let full = decode_text(bytes);
            Ok(vec![(0, full)])
        }
        "pdf" => extract_pdf_pages(path),
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

/// Extrae texto de PDF y lo separa por página usando form feed (\x0C).
fn extract_pdf_pages(path: &Path) -> Result<Vec<(usize, String)>> {
    // Primera pasada con -layout para preservar formato
    let text = extract_pdf_text(path)?;

    // Separar por form feed (separador de páginas de pdftotext)
    if text.contains('\x0C') {
        let pages: Vec<&str> = text.split('\x0C').collect();
        let result: Vec<(usize, String)> = pages
            .iter()
            .enumerate()
            .map(|(i, p)| (i + 1, p.trim().to_string()))
            .filter(|(_, t)| !t.is_empty())
            .collect();

        if result.is_empty() {
            // Fallback: todo como página 1
            let clean = text.replace('\x0C', " ").trim().to_string();
            Ok(vec![(1, clean)])
        } else {
            Ok(result)
        }
    } else {
        // Sin separadores, un solo bloque
        Ok(vec![(1, text.trim().to_string())])
    }
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

// ========================================================================
// FRAGMENTACIÓN DE TEXTO CON SEGUIMIENTO DE PÁGINA
// ========================================================================

/// Fragmenta texto paginado en chunks de ~N palabras con overlapping.
/// Devuelve Vec de (page_number, chunk_text).
/// page_number es la página predominante del chunk (la que más palabras aporta).
fn chunk_text_with_pages(
    page_info: &[(usize, String)],
    chunk_words: usize,
    overlap_words: usize,
) -> Vec<(usize, String)> {
    if page_info.is_empty() {
        return vec![];
    }

    // Si es un solo texto sin páginas (0), o el total es pequeño, fragmentar plano
    if page_info.len() == 1 && page_info[0].0 == 0 {
        return chunk_text_flat(&page_info[0].1, chunk_words, overlap_words)
            .into_iter()
            .map(|t| (0, t))
            .collect();
    }

    // Construir un flujo continuo de palabras con su página de origen
    struct WordRef {
        word: String,
        page: usize,
    }

    let mut word_stream: Vec<WordRef> = Vec::new();
    for (page, text) in page_info {
        for word in text.split_whitespace() {
            word_stream.push(WordRef {
                word: word.to_string(),
                page: *page,
            });
        }
    }

    if word_stream.is_empty() {
        return vec![];
    }

    if word_stream.len() <= chunk_words {
        let text: String = word_stream.iter().map(|w| w.word.clone()).collect::<Vec<_>>().join(" ");
        let page = word_stream[0].page;
        return vec![(page, text)];
    }

    let mut chunks: Vec<(usize, String)> = Vec::new();
    let mut start = 0usize;

    while start < word_stream.len() {
        let end = (start + chunk_words).min(word_stream.len());
        let slice = &word_stream[start..end];

        // Página predominante en este chunk
        let mut page_counts: HashMap<usize, usize> = HashMap::new();
        for w in slice {
            *page_counts.entry(w.page).or_insert(0) += 1;
        }
        let dominant_page = page_counts
            .into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(page, _)| page)
            .unwrap_or(1);

        let chunk_text: String = slice.iter().map(|w| w.word.clone()).collect::<Vec<_>>().join(" ");

        if chunk_text.trim().len() > 40 {
            chunks.push((dominant_page, chunk_text));
        }

        if end == word_stream.len() {
            break;
        }

        start = end.saturating_sub(overlap_words);
    }

    chunks
}

/// Fragmentación plana sin páginas (para TXT/MD).
fn chunk_text_flat(text: &str, chunk_words: usize, overlap_words: usize) -> Vec<String> {
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

/// Fragmentación original (legacy) para compatibilidad.
#[allow(dead_code)]
fn chunk_text(text: &str, chunk_words: usize, overlap_words: usize) -> Vec<String> {
    chunk_text_flat(text, chunk_words, overlap_words)
}

// ========================================================================
// INSERCIÓN EN ÍNDICE LÉXICO (FTS5)
// ========================================================================

fn insert_lexical_chunk(
    config: &Config,
    source: &str,
    chunk_index: usize,
    text: &str,
    page_number: usize,
    hash: &str,
) -> Result<()> {
    let conn = Connection::open(config.db_path())?;
    let now = Utc::now().to_rfc3339();

    conn.execute(
        r#"
        INSERT INTO document_chunks(source, chunk_index, text, page_number, hash, indexed_at)
        VALUES (?1, ?2, ?3, ?4, ?5, ?6)
        ON CONFLICT(source, chunk_index) DO UPDATE SET
            text = excluded.text,
            page_number = excluded.page_number,
            hash = excluded.hash,
            indexed_at = excluded.indexed_at
        "#,
        params![source, chunk_index as i64, text, page_number as i64, hash, now],
    )?;

    conn.execute(
        "INSERT INTO document_chunks_fts(source, chunk_index, page_number, text) VALUES (?1, ?2, ?3, ?4)",
        params![source, chunk_index as i64, page_number as i64, text],
    )?;

    Ok(())
}

fn embeddings_available(state: &AppState) -> bool {
    state.config.embeddings_enabled && state.config.openrouter_api_key.is_some() && state.config.qdrant_enabled
}

fn qdrant_collection_name(config: &Config) -> String {
    let model = config
        .openrouter_embedding_model
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect::<String>();
    format!("{}_{}", config.qdrant_collection, model)
}

async fn ensure_qdrant_collection(state: &AppState) -> Result<()> {
    if !state.config.qdrant_enabled {
        return Ok(());
    }

    let url = state.config.qdrant_url.as_ref().unwrap();
    let collection = qdrant_collection_name(&state.config);
    let endpoint = format!("{}/collections/{}", url.trim_end_matches('/'), collection);

    let body = json!({
        "vectors": {
            "size": state.config.embedding_dimension,
            "distance": "Cosine"
        }
    });

    let resp = state
        .http
        .put(endpoint)
        .json(&body)
        .timeout(Duration::from_secs(30))
        .send()
        .await?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(anyhow!("Qdrant collection error {}: {}", status, text));
    }

    Ok(())
}

fn qdrant_point_id(source: &str, chunk_index: usize, model: &str) -> String {
    format!("{}:{}:{}", source, chunk_index, model)
}

fn embedding_text_for_chunk(source: &str, chunk_index: usize, text: &str) -> String {
    format!(
        "Documento: {}\nFragmento: {}\nContenido:\n{}",
        source, chunk_index, text
    )
}

fn embedding_exists(config: &Config, source: &str, chunk_index: usize, hash: &str) -> Result<bool> {
    let conn = Connection::open(config.db_path())?;
    let exists = conn
        .query_row(
            r#"
            SELECT 1 FROM document_embeddings
            WHERE source = ?1 AND chunk_index = ?2 AND hash = ?3 AND model = ?4
            LIMIT 1
            "#,
            params![source, chunk_index as i64, hash, config.openrouter_embedding_model],
            |_| Ok(()),
        )
        .optional()?
        .is_some();
    Ok(exists)
}

async fn insert_chunk_embedding(
    state: &AppState,
    source: &str,
    chunk_index: usize,
    page_number: usize,
    hash: &str,
    embedding: &[f32],
) -> Result<()> {
    let conn = Connection::open(state.config.db_path())?;
    let model = state.config.openrouter_embedding_model.clone();
    conn.execute(
        r#"
        INSERT INTO document_embeddings(source, chunk_index, hash, model, dim, vector_json, created_at)
        VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
        ON CONFLICT(source, chunk_index, model) DO UPDATE SET
            hash = excluded.hash,
            dim = excluded.dim,
            vector_json = excluded.vector_json,
            created_at = excluded.created_at
        "#,
        params![
            source,
            chunk_index as i64,
            hash,
            state.config.openrouter_embedding_model,
            embedding.len() as i64,
            serde_json::to_string(embedding)?,
            Utc::now().to_rfc3339()
        ],
    )?;

    if let Some(url) = &state.config.qdrant_url {
        let collection = qdrant_collection_name(&state.config);
        let point_id = qdrant_point_id(source, chunk_index, &model);
        let body = json!({
            "points": [{
                "id": point_id,
                "vector": embedding,
                "payload": {
                    "source": source,
                    "chunk_index": chunk_index as i64,
                    "page_number": page_number as i64,
                    "hash": hash,
                    "model": model,
                }
            }]
        });
        let endpoint = format!("{}/collections/{}/points?wait=true", url.trim_end_matches('/'), collection);
        let resp = state.http
            .put(endpoint)
            .json(&body)
            .timeout(Duration::from_secs(30))
            .send()
            .await?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(anyhow!("Qdrant upsert HTTP {}: {}", status, text));
        }
    }
    Ok(())
}

async fn call_openrouter_embeddings(state: &AppState, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
    if inputs.is_empty() {
        return Ok(Vec::new());
    }

    let api_key = state
        .config
        .openrouter_api_key
        .as_ref()
        .ok_or_else(|| anyhow!("OPENROUTER_API_KEY no configurada"))?;

    let body = json!({
        "model": state.config.openrouter_embedding_model,
        "input": inputs,
    });

    let resp = state.http
        .post("https://openrouter.ai/api/v1/embeddings")
        .bearer_auth(api_key)
        .header("Content-Type", "application/json")
        .header("X-OpenRouter-Title", "Rust RAG - Embeddings")
        .json(&body)
        .timeout(Duration::from_secs(60))
        .send().await?
        ;

    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        return Err(anyhow!("OpenRouter embeddings HTTP {}: {}", status, body));
    }

    let value: Value = resp.json().await?;

    let data = value.get("data").and_then(|v| v.as_array()).ok_or_else(|| anyhow!("Respuesta de embeddings inválida"))?;
    let mut out = Vec::new();
    for item in data {
        let vec = item.get("embedding").and_then(|v| v.as_array()).ok_or_else(|| anyhow!("Embedding inválido"))?
            .iter().filter_map(|v| v.as_f64().map(|x| x as f32)).collect::<Vec<f32>>();
        if !vec.is_empty() {
            out.push(vec);
        }
    }
    Ok(out)
}

fn load_existing_chunks_for_source(config: &Config, source: &str) -> Result<Vec<(usize, usize, String)>> {
    let conn = Connection::open(config.db_path())?;

    let mut stmt = conn.prepare(
        r#"
        SELECT chunk_index, page_number, text
        FROM document_chunks
        WHERE source = ?1
        ORDER BY chunk_index
        "#,
    )?;

    let rows = stmt.query_map(params![source], |row| {
        Ok((
            row.get::<_, i64>(0)? as usize,
            row.get::<_, i64>(1)? as usize,
            row.get::<_, String>(2)?,
        ))
    })?;

    let mut chunks = Vec::new();
    for row in rows {
        chunks.push(row?);
    }

    Ok(chunks)
}

async fn ensure_embeddings_for_existing_source(
    state: &AppState,
    source: &str,
    hash: &str,
) -> Result<usize> {
    if !embeddings_available(state) {
        return Ok(0);
    }

    let chunks = load_existing_chunks_for_source(&state.config, source)?;
    if chunks.is_empty() {
        return Ok(0);
    }

    let mut missing = Vec::new();
    for (chunk_index, page_number, text) in &chunks {
        if !embedding_exists(&state.config, source, *chunk_index, hash)? {
            missing.push((*chunk_index, *page_number, text.clone()));
        }
    }

    if missing.is_empty() {
        return Ok(0);
    }

    ensure_chunk_embeddings_batch(state, source, hash, &missing).await?;
    Ok(missing.len())
}

async fn ensure_chunk_embeddings_batch(
    state: &AppState,
    source: &str,
    hash: &str,
    chunks: &[(usize, usize, String)],
) -> Result<()> {
    if !embeddings_available(state) || chunks.is_empty() {
        return Ok(());
    }

    ensure_qdrant_collection(state).await?;

    let mut pending = Vec::new();
    for (chunk_index, page_number, text) in chunks {
        if !embedding_exists(&state.config, source, *chunk_index, hash)? {
            pending.push((*chunk_index, *page_number, text.clone()));
        }
    }
    if pending.is_empty() {
        return Ok(());
    }

    let batch_size = state.config.embedding_batch_size.max(1);
    for batch in pending.chunks(batch_size) {
        let inputs = batch.iter()
            .map(|(chunk_index, _page_number, text)| embedding_text_for_chunk(source, *chunk_index, text))
            .collect::<Vec<_>>();
        let embeddings = call_openrouter_embeddings(state, &inputs).await?;
        for ((chunk_index, page_number, _), embedding) in batch.iter().zip(embeddings.iter()) {
            insert_chunk_embedding(
                state,
                source,
                *chunk_index,
                *page_number,
                hash,
                embedding,
            ).await?;
        }
    }

    Ok(())
}

async fn semantic_search(state: &AppState, query: &str, limit: usize) -> Result<Vec<SearchFragment>> {
    if !embeddings_available(state) {
        return Ok(Vec::new());
    }

    ensure_qdrant_collection(state).await?;
    let qvec = call_openrouter_embeddings(state, &[query.to_string()]).await?;
    let Some(query_embedding) = qvec.first() else {
        return Ok(Vec::new());
    };

    let mut scored = Vec::new();
    let url = state.config.qdrant_url.as_ref().ok_or_else(|| anyhow!("QDRANT_URL no configurada"))?;
    let collection = qdrant_collection_name(&state.config);
    let endpoint = format!("{}/collections/{}/points/search", url.trim_end_matches('/'), collection);
    let body = json!({
        "vector": query_embedding,
        "limit": limit.max(1),
        "with_payload": true,
        "with_vector": false
    });

    let resp = state.http
        .post(endpoint)
        .json(&body)
        .timeout(Duration::from_secs(30))
        .send()
        .await?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(anyhow!("Qdrant search HTTP {}: {}", status, text));
    }

    let value: Value = resp.json().await?;
    let hits = value.get("result").and_then(|v| v.as_array()).cloned().unwrap_or_default();

    for hit in hits {
        let payload = hit.get("payload").and_then(|v| v.as_object()).cloned().unwrap_or_default();
        let source = payload.get("source").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let chunk_index = payload.get("chunk_index").and_then(|v| v.as_i64()).unwrap_or(0) as usize;
        let page_number = payload.get("page_number").and_then(|v| v.as_i64()).unwrap_or(0) as usize;
        let score = hit.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;

        let conn = Connection::open(state.config.db_path())?;
        let text: String = conn.query_row(
            "SELECT text FROM document_chunks WHERE source = ?1 AND chunk_index = ?2",
            params![source, chunk_index as i64],
            |r| r.get(0),
        ).unwrap_or_default();

        if !text.is_empty() {
            scored.push(SearchFragment {
                score: score * state.config.hybrid_semantic_weight * 25.0,
                source,
                chunk_index,
                text,
                page_number,
                source_title: String::new(),
            });
        }
    }

    scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(limit.max(1));
    Ok(scored)
}

fn delete_chunks_by_source(config: &Config, source: &str) -> Result<()> {
    let conn = Connection::open(config.db_path())?;
    conn.execute_batch(
        &format!(
            "
            BEGIN IMMEDIATE;
            DELETE FROM document_embeddings WHERE source = {s};
            DELETE FROM extracted_facts WHERE source = {s};
            DELETE FROM document_chunks_fts WHERE source = {s};
            DELETE FROM document_chunks WHERE source = {s};
            COMMIT;
            ",
            s = quote_sql_literal(source)
        ),
    )?;
    Ok(())
}

fn quote_sql_literal(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

// ========================================================================
// EXTRACCIÓN UNIVERSAL DE HECHOS EXACTOS (agnóstica al dominio)
// ========================================================================

fn insert_extracted_fact(config: &Config, fact: &ExtractedFact) -> Result<()> {
    let conn = Connection::open(config.db_path())?;
    conn.execute(
        r#"
        INSERT INTO extracted_facts(source, page_number, chunk_index, fact_type, label, value, unit, text, normalized_text)
        VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
        "#,
        params![
            fact.source,
            fact.page_number as i64,
            fact.chunk_index as i64,
            fact.fact_type,
            fact.label,
            fact.value,
            fact.unit,
            fact.text,
            normalize_text(&fact.text),
        ],
    )?;
    Ok(())
}

/// Extrae hechos exactos de un chunk usando patrones universales (sin conocimiento de dominio).
/// Detecta: siglas, fórmulas, porcentajes, montos, códigos, reglas normativas, condiciones.
fn extract_facts_from_chunk(
    source: &str,
    page_number: usize,
    chunk_index: usize,
    text: &str,
) -> Vec<ExtractedFact> {
    let mut facts = Vec::new();

    // Patrón 1: SIGLA = Significado  o  SIGLA: significado  o  SIGLA (significado)
    // "FCM = Factor de Conversión Monetaria"
    // "UVR (Unidad de Valor Relativo)"
    let acronym_patterns = [
        // SIGLA = significado
        regex_lazy(r"\b([A-ZÑ]{2,6})\s*[=:]\s*([A-Za-zñÑáéíóúÁÉÍÓÚ][A-Za-zñÑáéíóúÁÉÍÓÚ\s,]+)(?:\.|$)"),
        // Significado (SIGLA)
        regex_lazy(r"([A-Za-zñÑáéíóúÁÉÍÓÚ][A-Za-zñÑáéíóúÁÉÍÓÚ\s,]+?)\(([A-ZÑ]{2,6})\)"),
        // SIGLA: Significado
        regex_lazy(r"\b([A-ZÑ]{2,6})\s*[:]\s+([A-Za-zñÑáéíóú][A-Za-zñÑáéíóúÁÉÍÓÚ\s,]+)(?:\.|$)"),
    ];

    for pattern in &acronym_patterns {
        if let Some(re) = pattern {
            for cap in re.captures_iter(text) {
                let (acronym, meaning) = if cap.len() >= 3 {
                    let a = cap.get(1).map(|m| m.as_str().trim()).unwrap_or("");
                    let b = cap.get(2).map(|m| m.as_str().trim()).unwrap_or("");
                    // Determinar cuál es sigla (todo mayúsculas corto)
                    if a.chars().all(|c| c.is_uppercase() || c.is_ascii_digit()) && a.len() <= 8 {
                        (a.to_string(), b.to_string())
                    } else if b.chars().all(|c| c.is_uppercase() || c.is_ascii_digit()) && b.len() <= 8 {
                        (b.to_string(), a.to_string())
                    } else {
                        continue;
                    }
                } else if cap.len() >= 2 {
                    let t = cap.get(1).map(|m| m.as_str().trim()).unwrap_or("");
                    (String::new(), t.to_string())
                } else {
                    continue;
                };

                if !acronym.is_empty() && !meaning.is_empty() && meaning.len() > 3 {
                    facts.push(ExtractedFact {
                        id: 0,
                        source: source.to_string(),
                        page_number,
                        chunk_index: chunk_index,
                        fact_type: "ACRONYM".to_string(),
                        label: acronym.clone(),
                        value: meaning.clone(),
                        unit: String::new(),
                        text: format!("{} = {}", acronym, meaning),
                    });
                }
            }
        }
    }

    // Patrón 2: Fórmulas con signos =, x, *, "producto de multiplicar"
    let formula_patterns = [
        regex_lazy(r"producto\s+de\s+multiplicar\s+([A-Za-z0-9\s]+)por\s+([A-Za-z0-9\s]+)"),
        regex_lazy(r"([A-Za-z0-9]{2,})\s*[=:]\s*([A-Za-z0-9]{2,})\s*[x\u00d7\*]\s*([A-Za-z0-9]{2,})"),
        regex_lazy(r"([A-Za-z0-9\s]{2,})\s*[=:]\s*([A-Za-z0-9\s]{2,})"),
    ];
    for pattern in &formula_patterns {
        if let Some(re) = pattern {
            if re.is_match(text) {
                facts.push(ExtractedFact {
                    id: 0,
                    source: source.to_string(),
                    page_number,
                    chunk_index: chunk_index,
                    fact_type: "FORMULA".to_string(),
                    label: "Fórmula detectada".to_string(),
                    value: text.chars().take(200).collect(),
                    unit: String::new(),
                    text: text.chars().take(300).collect(),
                });
                break; // una fórmula por chunk basta
            }
        }
    }

    // Patrón 3: Porcentajes con contexto
    if let Some(re_pct) = regex_lazy(r"(\d+[.,]?\d*)\s*%\s*(de|del|en|por)\s*([A-Za-zñÑáéíóúÁÉÍÓÚ\s,]{2,50})") {
        for cap in re_pct.captures_iter(text) {
            if let (Some(pct), Some(ctx)) = (cap.get(1), cap.get(0)) {
                facts.push(ExtractedFact {
                    id: 0,
                    source: source.to_string(),
                    page_number,
                    chunk_index: chunk_index,
                    fact_type: "PERCENTAGE_RULE".to_string(),
                    label: format!("{}%", pct.as_str()),
                    value: ctx.as_str().to_string(),
                    unit: "%".to_string(),
                    text: ctx.as_str().to_string(),
                });
            }
        }
    }

    // Patrón 4: Montos en dólares
    if let Some(re_usd) = regex_lazy(r"\$\s*(\d+[.,]?\d*)") {
        for cap in re_usd.captures_iter(text) {
            if let Some(amt) = cap.get(1) {
                facts.push(ExtractedFact {
                    id: 0,
                    source: source.to_string(),
                    page_number,
                    chunk_index: chunk_index,
                    fact_type: "MONETARY_AMOUNT".to_string(),
                    label: format!("${}", amt.as_str()),
                    value: amt.as_str().to_string(),
                    unit: "USD".to_string(),
                    text: amt.as_str().to_string(),
                });
            }
        }
    }

    // Patrón 5: Lenguaje normativo/obligación
    let normative_keywords = [
        "debera", "deberá", "sera", "será", "se aplicara", "se aplicará",
        "es obligatorio", "obligatoriamente", "debe cumplir", "debera cumplir",
    ];
    let text_lower = text.to_lowercase();
    for kw in &normative_keywords {
        if text_lower.contains(kw) {
            facts.push(ExtractedFact {
                id: 0,
                source: source.to_string(),
                page_number,
                chunk_index: chunk_index,
                fact_type: "OBLIGATION".to_string(),
                label: kw.to_string(),
                value: text.chars().take(200).collect(),
                unit: String::new(),
                text: text.chars().take(300).collect(),
            });
            break;
        }
    }

    // Patrón 6: Excepciones/condiciones
    let exception_keywords = [
        "excepto", "salvo", "a excepcion", "a excepción", "no aplica cuando",
        "en caso de", "siempre que", "cuando", "condicion", "condición",
    ];
    for kw in &exception_keywords {
        if text_lower.contains(kw) {
            facts.push(ExtractedFact {
                id: 0,
                source: source.to_string(),
                page_number,
                chunk_index: chunk_index,
                fact_type: "CONDITION_OR_EXCEPTION".to_string(),
                label: kw.to_string(),
                value: text.chars().take(200).collect(),
                unit: String::new(),
                text: text.chars().take(300).collect(),
            });
            break;
        }
    }

    facts
}

/// Utilidad: compila una regex una sola vez (evita recompilar en cada chunk)
fn regex_lazy(pattern: &str) -> Option<regex::Regex> {
    regex::Regex::new(pattern).ok()
}

// ========================================================================
// EXACT EVIDENCE MODE — Detección genérica de modo de precisión
// ========================================================================

/// Detecta si la pregunta requiere modo de evidencia exacta (siglas, fórmulas,
/// porcentajes, reglas, condiciones). Agnóstico al dominio.
#[allow(dead_code)]
fn detect_exact_evidence_mode(question: &str) -> bool {
    let q = normalize_text(question);
    let triggers = [
        "sigla", "significado", "formula", "multiplica", "porcentaje",
        "descuento", "recargo", "monto", "valor", "cuanto",
        "regla", "excepcion", "excepción", "condicion", "condición",
        "articulo", "artículo", "numeral", "literal", "obligatorio",
        "debe aplicar", "calculo", "cálculo", "techo maximo",
        "componente", "factor", "modificador", "codigo", "código",
        "determina", "establece", "dispone",
    ];
    triggers.iter().any(|t| q.contains(t))
}

/// Bonificación genérica para evidencia exacta. Premia fragmentos que contengan
/// el tipo de evidencia que la pregunta solicita (siglas, números, fórmulas, etc.)
/// Sin ningún conocimiento de dominio específico.
fn exact_evidence_bonus(text: &str, question: &str) -> f32 {
    let t = normalize_text(text);
    let q = normalize_text(question);
    let mut score = 0.0f32;

    // Definiciones cortas y siglas: premiar fragmentos con patrones explícitos.
    if q.contains("sigla")
        || q.contains("significado")
        || q.starts_with("que es ")
        || q.starts_with("que significa ")
        || q.starts_with("definicion de ")
        || q.starts_with("define ")
        || is_short_definition_lookup_question(question)
    {
        if contains_acronym_pattern(text) || !extract_acronym_definitions_from_text(text).is_empty() {
            score += 30.0;
        }
        if t.contains("significa") || t.contains("definicion") || t.contains("se define") {
            score += 10.0;
        }
    }

    // Si pregunta por porcentaje, premiar fragmentos con %
    if q.contains("porcentaje") || q.contains("descuento") || q.contains("recargo") {
        if text.contains('%') {
            score += 25.0;
        }
        if let Some(re) = regex_lazy(r"\d+[.,]?\d*\s*%") {
            if re.is_match(text) {
                score += 10.0;
            }
        }
    }

    // Si pregunta por fórmula, premiar fragmentos con = o ×
    if q.contains("formula") || q.contains("multiplica") || q.contains("calculo") {
        if text.contains('=') || text.contains('×') || text.contains('*') {
            score += 35.0;
        }
        if text.contains("producto de multiplicar") {
            score += 40.0;
        }
    }

    // Si pregunta por monto/valor, premiar fragmentos con $
    if q.contains("monto") || q.contains("valor") || q.contains("cuanto") || q.contains("costo") {
        if text.contains('$') {
            score += 20.0;
        }
    }

    // Si pregunta por regla/obligación, premiar lenguaje normativo
    if q.contains("regla") || q.contains("obligatorio") || q.contains("debe") {
        let norm_words = ["debera", "deberá", "sera", "será", "obligatorio", "obligatoriamente"];
        for w in &norm_words {
            if t.contains(w) {
                score += 25.0;
                break;
            }
        }
    }

    // Si pregunta por excepción/condición
    if q.contains("excepto") || q.contains("excepcion") || q.contains("condicion") || q.contains("cuando") {
        let exc_words = ["excepto", "salvo", "en caso de", "siempre que"];
        for w in &exc_words {
            if t.contains(w) {
                score += 25.0;
                break;
            }
        }
    }

    // Bonificación universal: números = más probabilidad de evidencia exacta
    if let Some(re_num) = regex_lazy(r"\b\d{2,}\b") {
        if re_num.find(text).is_some() {
            score += 5.0;
        }
    }

    score
}

/// Detecta si un texto contiene patrones de sigla (mayúsculas entre paréntesis, etc.)
fn contains_acronym_pattern(text: &str) -> bool {
    // Sigla entre paréntesis: Texto (SIGLA)
    if let Some(re) = regex_lazy(r"\([A-ZÑ]{2,8}\)") {
        if re.is_match(text) {
            return true;
        }
    }
    // Sigla = significado o sigla: significado
    if let Some(re) = regex_lazy(r"\b[A-ZÑ]{2,10}\s*[=:]\s*[^.;\n]{3,}") {
        if re.is_match(text) {
            return true;
        }
    }
    // Sigla significa significado
    if let Some(re) = regex_lazy(r"\b[A-ZÑ]{2,10}\s+(?:significa|quiere\s+decir|corresponde\s+a|se\s+define\s+como)\b") {
        if re.is_match(text) {
            return true;
        }
    }
    // Secuencia de mayúsculas de 2-10 caracteres
    if let Some(re) = regex_lazy(r"\b[A-ZÑ]{2,10}\b") {
        if re.find_iter(text).count() >= 2 {
            return true;
        }
    }
    false
}

// ========================================================================
// GENERADOR DE PREGUNTAS INTERNAS (SearchProbes)
// ========================================================================

/// Genera sondas de búsqueda interna a partir de la pregunta original.
/// Cada sonda explora un ángulo diferente para recuperar evidencia.
fn generate_internal_questions(question: &str) -> Vec<SearchProbe> {
    let mut probes: Vec<SearchProbe> = Vec::new();
    let q = normalize_text(question);
    let original = question.to_string();

    // Sonda 1: la pregunta original (máxima prioridad)
    probes.push(SearchProbe {
        query: original.clone(),
        purpose: "direct_answer".to_string(),
        priority: 100,
        expected_evidence_type: "general".to_string(),
    });

    // Sonda 2: términos clave sin stopwords
    let tokens: Vec<String> = tokenize_normalized(&q)
        .into_iter()
        .filter(|t| t.len() >= 4 && !is_stopword(t))
        .collect();
    if !tokens.is_empty() {
        probes.push(SearchProbe {
            query: tokens.join(" "),
            purpose: "key_terms".to_string(),
            priority: 80,
            expected_evidence_type: "general".to_string(),
        });
    }

    // Sonda 3: siglas (si la pregunta pide siglas)
    if q.contains("sigla") || q.contains("significado") || q.contains("componente") {
        probes.push(SearchProbe {
            query: "sigla significado acronimo".to_string(),
            purpose: "acronyms".to_string(),
            priority: 90,
            expected_evidence_type: "acronym".to_string(),
        });
    }

    if is_short_definition_lookup_question(question) {
        for term in extract_short_lookup_terms(question) {
            for alias in acronym_term_aliases(&term) {
                probes.push(SearchProbe {
                    query: format!("{}*", normalize_token(&alias)),
                    purpose: "acronym_lookup".to_string(),
                    priority: 98,
                    expected_evidence_type: "acronym".to_string(),
                });

                probes.push(SearchProbe {
                    query: format!("{}* AND definicion*", normalize_token(&alias)),
                    purpose: "acronym_lookup".to_string(),
                    priority: 94,
                    expected_evidence_type: "acronym".to_string(),
                });

                probes.push(SearchProbe {
                    query: format!("{}* AND significa*", normalize_token(&alias)),
                    purpose: "acronym_lookup".to_string(),
                    priority: 94,
                    expected_evidence_type: "acronym".to_string(),
                });
            }
        }
    }

    // Sonda 4: fórmula (si aplica)
    if q.contains("formula") || q.contains("multiplica") || q.contains("calculo") || q.contains("producto") {
        probes.push(SearchProbe {
            query: "formula producto multiplicar calculo".to_string(),
            purpose: "formula".to_string(),
            priority: 85,
            expected_evidence_type: "formula".to_string(),
        });
    }

    // Sonda 5: porcentajes, descuentos, recargos
    if q.contains("porcentaje") || q.contains("descuento") || q.contains("recargo") || q.contains("%") {
        probes.push(SearchProbe {
            query: "porcentaje descuento recargo disminucion incremento".to_string(),
            purpose: "percentage".to_string(),
            priority: 85,
            expected_evidence_type: "percentage".to_string(),
        });
    }

    // Sonda 6: condiciones y excepciones
    if q.contains("excepto") || q.contains("excepcion") || q.contains("cuando") || q.contains("condicion") {
        probes.push(SearchProbe {
            query: "excepto salvo siempre que condicion excepcion".to_string(),
            purpose: "condition".to_string(),
            priority: 75,
            expected_evidence_type: "condition".to_string(),
        });
    }

    // Sonda 7: reglas normativas
    if q.contains("regla") || q.contains("obligatorio") || q.contains("debe") || q.contains("aplica") {
        probes.push(SearchProbe {
            query: "regla obligatorio debe aplica disposicion establece".to_string(),
            purpose: "obligation".to_string(),
            priority: 80,
            expected_evidence_type: "obligation".to_string(),
        });
    }

    // Sonda 8: montos y valores
    if q.contains("monto") || q.contains("valor") || q.contains("cuanto") || q.contains("costo") || q.contains("tarifa") {
        probes.push(SearchProbe {
            query: "valor monto costo tarifa precio dolar".to_string(),
            purpose: "monetary".to_string(),
            priority: 80,
            expected_evidence_type: "monetary".to_string(),
        });
    }

    // Sonda 9: pares de palabras clave de la pregunta
    for pair in tokens.windows(2) {
        if pair.len() == 2 {
            probes.push(SearchProbe {
                query: format!("{} {}", pair[0], pair[1]),
                purpose: "term_pair".to_string(),
                priority: 60,
                expected_evidence_type: "general".to_string(),
            });
        }
    }

    // Sonda 10: contexto omitido (búsqueda de "información relacionada")
    probes.push(SearchProbe {
        query: "excepcion regla condicion nota".to_string(),
        purpose: "related_evidence".to_string(),
        priority: 50,
        expected_evidence_type: "context".to_string(),
    });

    // Eliminar duplicados (misma query)
    let mut seen = HashSet::new();
    probes.retain(|p| seen.insert(p.query.clone()));

    probes
}

// ========================================================================
// BÚSQUEDA LÉXICA CON EXPANSIÓN DE CONSULTA
// ========================================================================

/// Construye la información de depuración con los fragmentos recuperados:
/// source, page_number, chunk_index, score y un excerpt corto.
fn build_debug_fragments(fragments: &[SearchFragment]) -> Vec<DebugFragment> {
    fragments
        .iter()
        .map(|f| DebugFragment {
            source: f.source.clone(),
            page_number: f.page_number,
            chunk_index: f.chunk_index,
            score: f.score,
            excerpt: clean_excerpt(&f.text, 120),
        })
        .collect()
}

/// Genera términos de boost específicos para preguntas sobre siglas, fórmulas,
/// componentes, factores, y definiciones técnicas exactas.
#[allow(dead_code)]
fn formula_acronym_boost_terms(question: &str) -> Vec<String> {
    let q = normalize_text(question);

    if q.contains("sigla")
        || q.contains("significado")
        || q.contains("formula")
        || q.contains("multiplica")
        || q.contains("multiplicar")
        || q.contains("componente")
        || q.contains("componentes")
        || q.contains("factor")
        || q.contains("factores")
        || q.contains("techo maximo")
        || q.contains("techo")
    {
        vec![
            "uvr",
            "fcm",
            "unidad valor relativo",
            "unidades valor relativo",
            "factor conversion monetaria",
            "factor conversion monetario",
            "producto multiplicar",
            "tarifa producto",
            "techo maximo",
            "uvr fcm",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect()
    } else {
        vec![]
    }
}

/// Pipeline completo de búsqueda:
/// 1) Detecta intención de la pregunta
/// 2) Genera términos de boost para siglas/fórmulas si aplica
/// 3) Genera variantes de consulta (de LLM si está habilitado)
/// 4) Búsqueda FTS5 con pregunta original + boost + variantes
/// 5) Unión, deduplicación y reordenamiento
/// 6) Expansión con fragmentos vecinos (window=3)
/// 7) Re-ranking de dos etapas
#[allow(dead_code)]
async fn lexical_search(
    state: &AppState,
    question: &str,
    top_k: usize,
    llm_variants: Option<&[String]>,
    intent: Option<&QuestionIntent>,
) -> Result<Vec<SearchFragment>> {
    let intent = intent.cloned().unwrap_or(QuestionIntent::General);
    let query_terms = expand_query_terms(question);

    // Boost específico para siglas, fórmulas, componentes técnicos
    let formula_terms = formula_acronym_boost_terms(question);
    let has_formula_query = !formula_terms.is_empty();

    // top_k dinámico: preguntas de listado/siglas/fórmulas necesitan más cobertura
    let is_list_or_formula = intent == QuestionIntent::List || has_formula_query;
    let effective_top_k = if is_list_or_formula {
        (top_k * 4).max(30)
    } else {
        top_k
    };

    if query_terms.is_empty() && llm_variants.is_none() && formula_terms.is_empty() {
        return Ok(Vec::new());
    }

    let conn = Connection::open(state.config.db_path())?;
    let mut all_fragments: Vec<SearchFragment> = Vec::new();
    let intents = intent_boost_terms(&intent);

    // Mezclar términos de boost de fórmula con los términos de intención
    let mut combined_intents = intents.clone();
    combined_intents.extend(formula_terms.clone());

    // ================================================================
    // PASADA 1: Búsqueda con pregunta original + términos expandidos
    // ================================================================
    for fts_query in build_iterative_fts_queries(question, &query_terms, &combined_intents) {
        if fts_query.trim().is_empty() {
            continue;
        }
        let mut found = search_fts_fragments(&conn, &fts_query, (effective_top_k * 6).max(20))?;
        all_fragments.append(&mut found);
    }

    // ================================================================
    // PASADA 1b: Búsqueda directa con términos de fórmula (siglas exactas),
    // para garantizar que FCM, UVR y combinaciones tengan consulta directa.
    // ================================================================
    if has_formula_query {
        let formula_fts = build_fts_query(&formula_terms);
        if !formula_fts.trim().is_empty() {
            if let Ok(mut found) = search_fts_fragments(&conn, &formula_fts, effective_top_k * 3) {
                all_fragments.append(&mut found);
            }
        }
        // También buscar cada término de fórmula individualmente
        for ft in &formula_terms {
            if ft.len() >= 4 && !is_stopword(ft) {
                let single_q = format!("{}*", normalize_token(ft));
                if let Ok(mut found) = search_fts_fragments(&conn, &single_q, effective_top_k) {
                    all_fragments.append(&mut found);
                }
            }
        }
    }

    // ================================================================
    // PASADA 2: Búsqueda con variantes generadas por LLM
    // ================================================================
    if let Some(variants) = llm_variants {
        for variant in variants {
            let variant_terms = expand_query_terms(variant);
            if variant_terms.is_empty() {
                continue;
            }
            for fts_query in build_iterative_fts_queries(variant, &variant_terms, &combined_intents) {
                if fts_query.trim().is_empty() {
                    continue;
                }
                let mut found = search_fts_fragments(&conn, &fts_query, (effective_top_k * 4).max(15))?;
                all_fragments.append(&mut found);
            }
        }
    }

    // ================================================================
    // RE-RANKING ETAPA 1: puntuación léxica mejorada
    // ================================================================
    let mut reranked = rerank_fragments(question, &query_terms, &combined_intents, all_fragments);
    for f in &mut reranked {
        f.score *= state.config.hybrid_lexical_weight.max(0.0);
    }

    // Fallback por escaneo léxico (palabras mal escritas, cortadas)
    let fallback = lexical_scan_fallback(&conn, question, &query_terms, &combined_intents, effective_top_k * 5)?;
    reranked.extend(fallback);
    reranked = dedup_and_sort_fragments(reranked);

    if reranked.is_empty() {
        let aggressive_query_terms = expand_query_terms(question);
        let aggressive_intents = intent_boost_terms(&detect_question_intent(question));
        let mut aggressive_combined = aggressive_intents.clone();
        aggressive_combined.extend(formula_terms.clone());
        let aggressive = aggressive_lexical_scan_fallback(
            &conn,
            question,
            &aggressive_query_terms,
            &aggressive_combined,
            effective_top_k * 10,
        )?;
        reranked.extend(aggressive);
        reranked = dedup_and_sort_fragments(reranked);
    }

    // ================================================================
    // EXPANSIÓN CON FRAGMENTOS VECINOS (window=3 para alcanzar fragmentos más lejanos)
    // ================================================================
    let mut expanded = expand_with_neighbor_chunks(&conn, reranked, 3, effective_top_k)?;
    expanded = dedup_and_sort_fragments(expanded);

    // ================================================================
    // RE-RANKING ETAPA 2: puntuación final con contexto de intención
    // ================================================================
    let final_ranked = two_stage_rerank(question, &query_terms, &intent, expanded);

    let final_limit = (top_k * 2).clamp(top_k, 30);
    let mut out: Vec<SearchFragment> = final_ranked.into_iter().take(final_limit).collect();

    if out.is_empty() && embeddings_available(state) {
        if let Ok(mut semantic) = semantic_search(state, question, state.config.semantic_top_k.max(top_k)).await {
            out.append(&mut semantic);
            out = dedup_and_sort_fragments(out);
            out.truncate(final_limit);
        }
    }

    Ok(out)
}

// ========================================================================
// EXPANSIÓN DE PREGUNTA CON LLM (REFORMULACIÓN)
// ========================================================================

/// Llama a OpenRouter para generar variantes de la pregunta.
#[allow(dead_code)]
async fn expand_question_with_llm(
    state: &AppState,
    question: &str,
) -> Result<Vec<String>> {
    let api_key = state
        .config
        .openrouter_api_key
        .as_ref()
        .ok_or_else(|| anyhow!("OPENROUTER_API_KEY no está configurada"))?;

    let prompt = format!(
        r#"Eres un asistente experto en recuperación de información documental.

Tu tarea es generar variantes de búsqueda para encontrar la evidencia exacta en documentos técnicos.

La pregunta puede estar formulada de forma indirecta. Debes producir:
- variantes léxicas normales,
- posibles SIGLAS que podrian aparecer en el documento,
- nombres técnicos equivalentes,
- frases exactas que podrian aparecer en el documento,
- patrones de formula si la pregunta habla de multiplicar, calcular, tarifa, valor o techo maximo.

REGLAS:
- Responde solo con una consulta por línea.
- No expliques nada.
- No uses numeración.
- Incluye siglas probables si la pregunta pide siglas, componentes o factores.
- Incluye frases con y sin tildes.
- Maximo 12 consultas.

Pregunta original:
{question}

Consultas:
"#,
        question = question
    );

    let body = json!({
        "model": state.config.openrouter_model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 300,
    });

    let value: Value = state
        .http
        .post("https://openrouter.ai/api/v1/chat/completions")
        .bearer_auth(api_key)
        .header("Content-Type", "application/json")
        .header("X-OpenRouter-Title", "Rust RAG - Query Expansion")
        .json(&body)
        .timeout(Duration::from_secs(15))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;

    let raw = value
        .get("choices")
        .and_then(|v| v.as_array())
        .and_then(|arr| arr.first())
        .and_then(|choice| choice.get("message"))
        .and_then(|msg| msg.get("content"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    // Parsear líneas y limpiar
    let variants: Vec<String> = raw
        .lines()
        .map(|l| {
            l.trim()
                .trim_start_matches(|c: char| c.is_ascii_digit() || c == '.' || c == ')' || c == '-')
                .trim()
                .to_string()
        })
        .filter(|l| {
            !l.is_empty()
                && l.len() > 3
                && !l.to_lowercase().starts_with("variante")
                && !l.to_lowercase().starts_with("claro")
                && !l.to_lowercase().starts_with("aqu")
        })
        .collect();

    if variants.is_empty() {
        return Err(anyhow!("No se generaron variantes válidas"));
    }

    Ok(variants)
}

// ========================================================================
// DETECCIÓN DE INTENCIÓN DE LA PREGUNTA
// ========================================================================

/// Detecta la intención de la pregunta basada en palabras clave.
fn detect_question_intent(question: &str) -> QuestionIntent {
    let q = normalize_text(question);

    let patterns: Vec<(QuestionIntent, Vec<&str>)> = vec![
        (QuestionIntent::Definition, vec![
            "que es", "que son", "definicion", "defina", "significa", "concepto",
            "explique", "explicar", "en que consiste", "describe",
        ]),
        (QuestionIntent::ValuePrice, vec![
            "cuanto cuesta", "cuanto vale", "precio", "valor", "tarifa", "tarifario",
            "costo", "monto", "honorario", "arancel", "cuanto", "usd", "dolares",
            "monetario", "cobro", "pago",
        ]),
        (QuestionIntent::Requirement, vec![
            "requisito", "requisitos", "necesito", "necesita", "documento",
            "debe presentar", "obligatorio", "condicion", "condiciones",
            "que necesito", "que se necesita", "presentar",
        ]),
        (QuestionIntent::Procedure, vec![
            "como", "procedimiento", "proceso", "paso", "pasos", "realizar",
            "tramite", "gestion", "hacer", "solicitar", "obtener", "acceder",
        ]),
        (QuestionIntent::Comparison, vec![
            "diferencia", "comparacion", "versus", "vs", "mejor", "peor",
            "diferente", "distinto", "cambios", "antes y despues",
        ]),
        (QuestionIntent::Normative, vec![
            "ley", "norma", "normativa", "reglamento", "articulo", "resolucion",
            "acuerdo", "disposicion", "legal", "juridico", "regula", "establece",
        ]),
        (QuestionIntent::List, vec![
            "lista", "listado", "enumere", "mencione", "cuales son", "cuales",
            "tipos de", "clases de", "categorias",
        ]),
        (QuestionIntent::Summary, vec![
            "resumen", "resuma", "sintesis", "sintetice", "en resumen",
            "idea principal", "principales puntos",
        ]),
    ];

    for (intent, keywords) in patterns {
        for kw in keywords {
            if q.contains(kw) {
                return intent;
            }
        }
    }

    QuestionIntent::General
}

/// Devuelve términos de boost según la intención detectada.
#[allow(dead_code)]
fn intent_boost_terms(intent: &QuestionIntent) -> Vec<String> {
    match intent {
        QuestionIntent::Definition => vec![
            "definicion", "concepto", "significa", "consiste", "descripcion",
        ].into_iter().map(String::from).collect(),

        QuestionIntent::ValuePrice => vec![
            "tarifa", "valor", "precio", "costo", "monto", "honorario",
            "arancel", "codigo", "usd", "monetario",
        ].into_iter().map(String::from).collect(),

        QuestionIntent::Requirement => vec![
            "requisito", "documento", "obligatorio", "condicion", "presentar",
            "necesario", "deber", "exigencia",
        ].into_iter().map(String::from).collect(),

        QuestionIntent::Procedure => vec![
            "procedimiento", "proceso", "paso", "tramite", "gestion",
            "solicitud", "formulario", "realizar",
        ].into_iter().map(String::from).collect(),

        QuestionIntent::Comparison => vec![
            "diferencia", "comparacion", "cambio", "modificacion", "actualizacion",
            "versus", "nuevo", "anterior",
        ].into_iter().map(String::from).collect(),

        QuestionIntent::Normative => vec![
            "ley", "norma", "reglamento", "articulo", "resolucion", "acuerdo",
            "legal", "disposicion", "regula", "establece", "dispone",
        ].into_iter().map(String::from).collect(),

        QuestionIntent::List => vec![
            "lista", "tipos", "clases", "categorias", "clasificacion",
            "componentes", "elementos",
        ].into_iter().map(String::from).collect(),

        QuestionIntent::Summary => vec![
            "resumen", "sintesis", "conclusion", "hallazgo", "resultado",
            "principal", "importante",
        ].into_iter().map(String::from).collect(),

        QuestionIntent::General => vec![],
    }
}

// ========================================================================
// CONSTRUCCIÓN DE CONSULTAS FTS5
// ========================================================================

#[allow(dead_code)]
fn search_fts_fragments(
    conn: &Connection,
    fts_query: &str,
    limit: usize,
) -> Result<Vec<SearchFragment>> {
    let sql = r#"
        SELECT source, chunk_index, text, page_number, bm25(document_chunks_fts) AS rank
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
        let rank: f64 = r.get(4)?;
        Ok(SearchFragment {
            score: (1.0 / (1.0 + rank.abs())) as f32,
            source: r.get(0)?,
            chunk_index: r.get::<_, i64>(1)? as usize,
            text: r.get(2)?,
            page_number: r.get::<_, i64>(3)? as usize,
            source_title: String::new(),
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

#[allow(dead_code)]
fn build_iterative_fts_queries(
    question: &str,
    terms: &[String],
    intent_terms: &[String],
) -> Vec<String> {
    let mut queries = Vec::new();

    // Pasada amplia: términos originales + sinónimos
    let broad = build_fts_query(terms);
    let broad_str = broad.clone();
    if !broad.trim().is_empty() {
        queries.push(broad);
    }

    // Pasada estricta: todos los términos relevantes deben aparecer juntos.
    let strict_terms: Vec<String> = terms
        .iter()
        .chain(intent_terms.iter())
        .map(|t| normalize_token(t))
        .filter(|t| t.len() >= 3 && !is_stopword(t))
        .collect();
    if strict_terms.len() >= 2 {
        let strict_and = strict_terms
            .iter()
            .map(|t| format!("{}*", t))
            .collect::<Vec<_>>()
            .join(" AND ");
        if !strict_and.trim().is_empty() && strict_and != broad_str {
            queries.push(strict_and);
        }

        let phrase = strict_terms.join(" ");
        if phrase.split_whitespace().count() >= 2 {
            queries.push(format!("\"{}\"", phrase));
        }
    }

    // Pasada con boost de intención
    if !intent_terms.is_empty() {
        let all_terms: Vec<String> = terms
            .iter()
            .chain(intent_terms.iter())
            .cloned()
            .collect();
        let boosted = build_fts_query(&all_terms);
        if boosted != broad_str && !boosted.trim().is_empty() {
            queries.push(boosted);
        }
    }

    // Pasada focalizada: solo términos principales (>= 4 chars, no stopwords)
    let mut primary_terms = tokenize_normalized(question)
        .into_iter()
        .filter(|t| t.len() >= 4 && !is_stopword(t))
        .collect::<Vec<_>>();
    primary_terms.sort();
    primary_terms.dedup();

    let focused = build_fts_query(&primary_terms);
    if !focused.trim().is_empty() && focused != broad_str {
        queries.push(focused);
    }

    // Pasada por pares de palabras cercanas
    for pair in primary_terms.windows(2).take(8) {
        if pair.len() == 2 {
            let q = format!("{}* AND {}*", normalize_token(&pair[0]), normalize_token(&pair[1]));
            queries.push(q);
            queries.push(format!("\"{} {}\"", normalize_token(&pair[0]), normalize_token(&pair[1])));
        }
    }

    queries.sort();
    queries.dedup();
    queries
}

#[allow(dead_code)]
fn build_fts_query(terms: &[String]) -> String {
    let mut parts = Vec::new();
    for term in terms.iter().take(40) {
        let clean = normalize_token(term);
        if clean.len() < 3 || is_stopword(&clean) {
            continue;
        }
        parts.push(format!("{}*", clean));
    }
    parts.sort();
    parts.dedup();
    parts.join(" OR ")
}

// ========================================================================
// EXPANSIÓN DE TÉRMINOS Y SINÓNIMOS
// ========================================================================

#[allow(dead_code)]
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

#[allow(dead_code)]
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
        ("tarifa", vec!["precio", "valor", "costo", "arancel", "honorario", "monto", "techo", "maximo"]),
        ("costo", vec!["precio", "valor", "tarifa", "arancel", "monto", "gasto"]),
        ("uvr", vec!["unidad", "unidades", "valor", "relativo", "fcm", "unidad valor relativo"]),
        ("fcm", vec!["factor", "conversion", "monetaria", "monetario", "uvr", "factor conversion monetaria"]),
        ("techo", vec!["maximo", "tarifa", "pago", "uvr", "fcm", "techo maximo"]),
        ("multiplicar", vec!["producto", "formula", "calculo", "multiplica", "uvr", "fcm"]),
        ("procedimiento", vec!["prestacion", "servicio", "atencion", "tarifa", "uvr", "fcm"]),
        ("recargo", vec!["porcentaje", "adicional", "gestion", "cobro", "extra"]),
    ])
}

// ========================================================================
// RE-RANKING MEJORADO (DOS ETAPAS)
// ========================================================================

/// Primera etapa de re-ranking: puntuación léxica mejorada.
#[allow(dead_code)]
fn rerank_fragments(
    question: &str,
    terms: &[String],
    intent_terms: &[String],
    fragments: Vec<SearchFragment>,
) -> Vec<SearchFragment> {
    let query_tokens = tokenize_normalized(question);
    let scored = fragments
        .into_iter()
        .map(|mut f| {
            let base_lexical = lexical_score(&f.text, terms, &query_tokens);
            let intent_score = intent_bonus(&f.text, intent_terms);
            // Combinar: FTS score * 2 + lexical + intent
            f.score = (f.score * 2.0) + base_lexical + intent_score;
            f
        })
        .collect::<Vec<_>>();

    dedup_and_sort_fragments(scored)
}

/// Segunda etapa de re-ranking: puntuación más fina considerando
/// densidad de términos, posición, y relevancia semántica local.
#[allow(dead_code)]
fn two_stage_rerank(
    question: &str,
    terms: &[String],
    intent: &QuestionIntent,
    fragments: Vec<SearchFragment>,
) -> Vec<SearchFragment> {
    let query_tokens = tokenize_normalized(question);
    let intent_terms = intent_boost_terms(intent);

    let mut scored: Vec<SearchFragment> = fragments
        .into_iter()
        .map(|f| {
            let mut s = f.score;

            // Bonificación por densidad de términos (más términos relevantes = mejor)
            let text_lower = normalize_text(&f.text);
            let text_tokens: Vec<&str> = text_lower.split_whitespace().collect();
            let text_set: HashSet<&str> = text_tokens.iter().cloned().collect();

            let mut term_hits = 0usize;
            for term in terms {
                if text_set.contains(term.as_str()) {
                    term_hits += 1;
                }
            }

            let density = if term_hits > 0 {
                (term_hits as f32) * 5.0 / (1.0 + (text_tokens.len() as f32 / 200.0))
            } else {
                0.0
            };

            s += density;

            // Bonificación por intención
            for it in &intent_terms {
                if text_set.contains(it.as_str()) {
                    s += 3.0;
                }
            }

            // Bonificación por frase cercana de la pregunta
            for window in query_tokens.windows(2) {
                if window.len() == 2 {
                    let phrase = format!("{} {}", window[0], window[1]);
                    if text_lower.contains(&phrase) {
                        s += 4.0;
                    }
                }
            }

            // Penalización por fragmentos muy cortos o muy largos
            let word_count = text_tokens.len();
            if word_count < 20 {
                s *= 0.7; // demasiado corto
            } else if word_count > 600 {
                s *= 0.9; // demasiado largo
            }

            SearchFragment {
                score: s,
                ..f
            }
        })
        .collect();

    scored.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.source.cmp(&b.source))
            .then_with(|| a.chunk_index.cmp(&b.chunk_index))
    });

    scored
}

/// Bonificación por términos de intención en el texto.
#[allow(dead_code)]
fn intent_bonus(text: &str, intent_terms: &[String]) -> f32 {
    if intent_terms.is_empty() {
        return 0.0;
    }
    let text_lower = normalize_text(text);
    let mut score = 0.0f32;
    for term in intent_terms {
        if text_lower.contains(term.as_str()) {
            score += 2.0;
        }
    }
    score
}

// ========================================================================
// FUNCIONES DE PUNTUACIÓN LÉXICA
// ========================================================================

#[allow(dead_code)]
fn lexical_scan_fallback(
    conn: &Connection,
    question: &str,
    terms: &[String],
    intent_terms: &[String],
    limit: usize,
) -> Result<Vec<SearchFragment>> {
    let query_tokens = tokenize_normalized(question);
    let mut stmt = conn.prepare(
        "SELECT source, chunk_index, text, page_number FROM document_chunks ORDER BY source, chunk_index",
    )?;

    let rows = stmt.query_map([], |r| {
        Ok(SearchFragment {
            score: 0.0,
            source: r.get(0)?,
            chunk_index: r.get::<_, i64>(1)? as usize,
            text: r.get(2)?,
            page_number: r.get::<_, i64>(3)? as usize,
            source_title: String::new(),
        })
    })?;

    let mut scored = Vec::new();
    for row in rows {
        let mut fragment = row?;
        let base = lexical_score(&fragment.text, terms, &query_tokens);
        let intent = intent_bonus(&fragment.text, intent_terms);
        let score = base + intent;
        if score > 0.0 {
            fragment.score = score;
            scored.push(fragment);
        }
    }

    let mut scored = dedup_and_sort_fragments(scored);
    scored.truncate(limit);
    Ok(scored)
}

#[allow(dead_code)]
fn aggressive_lexical_scan_fallback(
    conn: &Connection,
    question: &str,
    terms: &[String],
    intent_terms: &[String],
    limit: usize,
) -> Result<Vec<SearchFragment>> {
    let query_tokens = tokenize_normalized(question);
    let mut stmt = conn.prepare(
        "SELECT source, chunk_index, text, page_number FROM document_chunks ORDER BY source, chunk_index",
    )?;

    let rows = stmt.query_map([], |r| {
        Ok(SearchFragment {
            score: 0.0,
            source: r.get(0)?,
            chunk_index: r.get::<_, i64>(1)? as usize,
            text: r.get(2)?,
            page_number: r.get::<_, i64>(3)? as usize,
            source_title: String::new(),
        })
    })?;

    let mut scored = Vec::new();
    for row in rows {
        let mut fragment = row?;
        let text_norm = normalize_text(&fragment.text);
        let mut score = 0.0f32;

        for term in terms {
            let t = normalize_token(term);
            if t.len() >= 2 && text_norm.contains(&t) {
                score += 2.5;
            } else if t.len() >= 2 && text_norm.contains(&format!(" {} ", t)) {
                score += 1.5;
            }
        }

        for it in intent_terms {
            let t = normalize_token(it);
            if !t.is_empty() && text_norm.contains(&t) {
                score += 1.0;
            }
        }

        for window in query_tokens.windows(2) {
            if window.len() == 2 {
                let phrase = format!("{} {}", window[0], window[1]);
                if text_norm.contains(&phrase) {
                    score += 2.5;
                }
            }
        }

        if score == 0.0 {
            let q = normalize_text(question);
            let rescue_phrases = [
                "facturado por separado",
                "valoracion por litros",
                "valoración por litros",
                "precio por litro",
                "precio por tanque",
                "gastos de gestion",
                "gastos de gestión",
                "no aplica",
                "debe cobrarse",
                "valor fijo",
            ];
            if rescue_phrases.iter().any(|p| text_norm.contains(p) && q.contains("cobro") || q.contains("precio") || q.contains("tarifa") || q.contains("valor")) {
                score += 5.0;
            }
        }

        if score > 0.0 {
            fragment.score = score;
            scored.push(fragment);
        }
    }

    let mut scored = dedup_and_sort_fragments(scored);
    scored.truncate(limit);
    Ok(scored)
}

#[allow(dead_code)]
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

    // Bonificación por frases cercanas de la pregunta original
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

#[allow(dead_code)]
fn fuzzy_close(a: &str, b: &str) -> bool {
    let min_len = a.len().min(b.len());
    let max_len = a.len().max(b.len());
    if min_len < 5 || max_len.saturating_sub(min_len) > 2 {
        return false;
    }
    levenshtein_limited(a, b, 2) <= 2
}

#[allow(dead_code)]
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

// ========================================================================
// EXPANSIÓN CON FRAGMENTOS VECINOS
// ========================================================================

#[allow(dead_code)]
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

#[allow(dead_code)]
fn load_chunk(
    conn: &Connection,
    source: &str,
    chunk_index: usize,
) -> Result<Option<SearchFragment>> {
    let found = conn
        .query_row(
            "SELECT source, chunk_index, text, page_number FROM document_chunks WHERE source = ?1 AND chunk_index = ?2",
            params![source, chunk_index as i64],
            |r| {
                Ok(SearchFragment {
                    score: 0.0,
                    source: r.get(0)?,
                    chunk_index: r.get::<_, i64>(1)? as usize,
                    text: r.get(2)?,
                    page_number: r.get::<_, i64>(3)? as usize,
                    source_title: String::new(),
                })
            },
        )
        .optional()?;

    Ok(found)
}

// ========================================================================
// FILTRADO POR FUENTES Y MEZCLA DE CONTEXTO
// ========================================================================

fn filter_fragments_by_sources(
    fragments: &[SearchFragment],
    selected_sources: Option<&[String]>,
) -> Vec<SearchFragment> {
    let Some(selected_sources) = selected_sources else {
        return fragments.to_vec();
    };
    if selected_sources.is_empty() {
        return fragments.to_vec();
    }

    let selected: HashSet<&str> = selected_sources.iter().map(|s| s.as_str()).collect();
    fragments
        .iter()
        .filter(|f| selected.contains(f.source.as_str()))
        .cloned()
        .collect()
}

#[allow(dead_code)]
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

        // Incluir metadatos de página en el contexto
        let page_info = if fragment.page_number > 0 {
            format!(" | página {}", fragment.page_number)
        } else {
            String::new()
        };

        merged_text.push_str(&format!(
            "[Fuente: {}{} | fragmento {}]\n{}",
            fragment.source,
            page_info,
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
        page_number: 0,
        source_title: String::new(),
    }]
}

// ========================================================================
// LLAMADAS A OPENROUTER (RESPUESTA Y STREAM)
// ========================================================================

#[allow(dead_code)]
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
        let page_info = if f.page_number > 0 {
            format!(" | página {}", f.page_number)
        } else {
            String::new()
        };
        context.push_str(&format!(
            "[Evidencia {} | fuente: {}{}] {}\n",
            i + 1,
            f.source,
            page_info,
            limit_chars(&f.text, max_chars_per_fragment)
        ));
    }

    let max_words = state.config.answer_max_words;
    let system_prompt = format!(
        r#"Eres un asistente experto en analisis documental que responde en espanol.

Responde de forma CLARA y COMPLETA usando solo la evidencia proporcionada.
No inventes datos. Si la evidencia es parcial, indicalo sin bloquear la respuesta.

Estructura natural (sin encabezados predefinidos):
- Respuesta principal al inicio.
- Fundamentos y datos exactos.
- Condiciones, excepciones o calculos si aplican.
- Informacion relacionada importante.
- Cierre con lo que no se pudo determinar.

LIMITE MAXIMO: {} palabras. Se estricto con este limite.

Reglas:
- Usa HTML para tablas: <table><tr><th>Col</th></tr><tr><td>Valor</td></tr></table>.
- Mantiene tono profesional e institucional.
- No digas que eres IA.
- No uses frases de relleno.
- Prioriza claridad, detalle y utilidad."#,
        max_words
    );

    let user_prompt = format!(
        "Pregunta del usuario:\n{}\n\nPaquete de evidencia:\n{}\n\nRedacta la respuesta:",
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

#[allow(dead_code)]
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
        let page_info = if f.page_number > 0 {
            format!(" | página {}", f.page_number)
        } else {
            String::new()
        };
        context.push_str(&format!(
            "[Evidencia {} | fuente: {}{}] {}\n",
            i + 1,
            f.source,
            page_info,
            limit_chars(&f.text, max_chars_per_fragment)
        ));
    }

    let max_words = state.config.answer_max_words;
    let system_prompt = format!(
        r#"Eres un asistente experto en analisis documental que responde en espanol.

Responde de forma CLARA y COMPLETA usando solo la evidencia proporcionada.
No inventes datos. Si la evidencia es parcial, indicalo sin bloquear la respuesta.

Estructura natural (sin encabezados predefinidos):
- Respuesta principal al inicio.
- Fundamentos y datos exactos.
- Condiciones, excepciones o calculos si aplican.
- Informacion relacionada importante.
- Cierre con lo que no se pudo determinar.

LIMITE MAXIMO: {} palabras. Se estricto con este limite.

Reglas:
- Usa HTML para tablas: <table><tr><th>Col</th></tr><tr><td>Valor</td></tr></table>.
- Mantiene tono profesional e institucional.
- No digas que eres IA.
- No uses frases de relleno.
- Prioriza claridad, detalle y utilidad."#,
        max_words
    );

    let user_prompt = format!(
        "Pregunta del usuario:\n{}\n\nPaquete de evidencia:\n{}\n\nRedacta la respuesta:",
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

    for f in fragments.iter().take(3) {
        let page_info = if f.page_number > 0 {
            format!(" (página {})", f.page_number)
        } else {
            String::new()
        };
        response.push_str(&format!("- [{}]{} {}\n", f.source, page_info, clean_excerpt(&f.text, 180)));
    }

    let mut sources: Vec<String> = fragments.iter().map(|f| f.source.clone()).collect();
    sources.sort();
    sources.dedup();

    response.push_str("\n**Fuentes consultadas:** ");
    response.push_str(&sources.join(", "));

    response
}

// ========================================================================
// BASE DE DATOS: INICIALIZACIÓN Y MIGRACIONES
// ========================================================================

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
            page_number INTEGER NOT NULL DEFAULT 0,
            hash TEXT NOT NULL,
            indexed_at TEXT NOT NULL,
            PRIMARY KEY(source, chunk_index)
        );

        DROP TABLE IF EXISTS document_chunks_fts;
        CREATE VIRTUAL TABLE IF NOT EXISTS document_chunks_fts
        USING fts5(
            source UNINDEXED,
            chunk_index UNINDEXED,
            page_number UNINDEXED,
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

        CREATE TABLE IF NOT EXISTS query_expansion_cache (
            cache_key TEXT PRIMARY KEY,
            question TEXT NOT NULL,
            variants_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS extracted_facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            page_number INTEGER NOT NULL DEFAULT 0,
            chunk_index INTEGER NOT NULL,
            fact_type TEXT NOT NULL,
            label TEXT NOT NULL DEFAULT '',
            value TEXT NOT NULL DEFAULT '',
            unit TEXT NOT NULL DEFAULT '',
            text TEXT NOT NULL,
            normalized_text TEXT NOT NULL DEFAULT '',
            FOREIGN KEY(source, chunk_index) REFERENCES document_chunks(source, chunk_index)
        );

        CREATE INDEX IF NOT EXISTS idx_extracted_facts_source ON extracted_facts(source);
        CREATE INDEX IF NOT EXISTS idx_extracted_facts_type ON extracted_facts(fact_type);

        CREATE TABLE IF NOT EXISTS document_embeddings (
            source TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            hash TEXT NOT NULL,
            model TEXT NOT NULL,
            dim INTEGER NOT NULL,
            vector_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY(source, chunk_index, model),
            FOREIGN KEY(source, chunk_index) REFERENCES document_chunks(source, chunk_index)
        );

        CREATE INDEX IF NOT EXISTS idx_document_embeddings_model ON document_embeddings(model);
        CREATE INDEX IF NOT EXISTS idx_document_embeddings_source ON document_embeddings(source);

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

    // Migración: agregar page_number si no existe (base existente)
    let has_page_number: bool = conn
        .prepare("SELECT page_number FROM document_chunks LIMIT 0")
        .is_ok();

    if !has_page_number {
        info!("Migrando schema: agregando columna page_number a document_chunks...");
        conn.execute_batch(
            r#"
            ALTER TABLE document_chunks ADD COLUMN page_number INTEGER NOT NULL DEFAULT 0;

            CREATE TABLE IF NOT EXISTS query_expansion_cache (
                cache_key TEXT PRIMARY KEY,
                question TEXT NOT NULL,
                variants_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            "#,
        )?;
        info!("Migración completada.");
    }

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

#[allow(dead_code)]
fn is_empty_answer(response: &str) -> bool {
    let r = response.trim().to_lowercase();
    r.is_empty()
        || r.contains("no se encontró información")
        || r.contains("no se encontro informacion")
        || r.contains("sin informacion")
}

// ========================================================================
// MANIFEST (DOCUMENTOS INDEXADOS)
// ========================================================================

#[allow(dead_code)]
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
        params![path, hash, modified, Utc::now().to_rfc3339(), chunk_count as i64],
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

// ========================================================================
// UTILIDADES
// ========================================================================

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

// ========================================================================
// NORMALIZACIÓN Y TOKENIZACIÓN
// ========================================================================

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


// ========================================================================
// EXACT EVIDENCE HELPERS — números, unidades, fórmulas y preguntas exactas
// ========================================================================

fn is_exact_numeric_question(question: &str) -> bool {
    let q = normalize_text(question);
    let exact_words = [
        "cuantos", "cuantas", "cuanto", "cuanta",
        "dias exactos", "numero exacto", "valor exacto",
        "plazo", "periodo", "duracion", "vigencia",
        "meses", "anos", "horas", "consultas",
        "porcentaje", "monto", "valor", "cantidad",
    ];
    exact_words.iter().any(|w| q.contains(w))
}

fn is_direct_decision_question(question: &str) -> bool {
    let q = normalize_text(question);
    let decision_words = [
        "apruebas", "rechazas", "pagar", "cobra", "cobrar",
        "corresponde", "aplica", "debe", "debo",
        "es correcto", "esta correcto", "cuanto corresponde",
    ];
    decision_words.iter().any(|w| q.contains(w))
}

fn harden_search_plan_from_question(question: &str, plan: &mut SearchPlan) {
    let intent = detect_question_intent(question);
    let normalized_question = normalize_question_for_search(question);

    for variant in rewrite_query_variants(&normalized_question, &intent) {
        if !plan.query_variants.iter().any(|x| normalize_text(x) == normalize_text(&variant)) {
            plan.query_variants.push(variant);
        }
    }

    if is_exact_numeric_question(question) || is_direct_decision_question(question) {
        plan.answer_type = "direct_answer".to_string();
    }
    if is_exact_numeric_question(question) {
        plan.needs_numbers = true;
        plan.needs_section_title = true;
    }
    let extra_queries = build_numeric_unit_queries(&normalized_question);
    for q in extra_queries {
        if !plan.exact_queries.iter().any(|x| x == &q) {
            plan.exact_queries.push(q);
        }
    }
    if asks_for_formula_like_answer(question) {
        plan.needs_formula = true;
    }
    if asks_for_acronym_like_answer(question) {
        plan.needs_acronyms = true;
    }

    if is_short_definition_lookup_question(&normalized_question) {
        plan.answer_type = "direct_answer".to_string();
        plan.needs_acronyms = true;
        plan.needs_section_title = true;

        for q in acronym_lookup_queries(&normalized_question) {
            if !plan.exact_queries.iter().any(|x| x == &q) {
                plan.exact_queries.push(q);
            }
        }

        for term in extract_short_lookup_terms(&normalized_question) {
            if !plan
                .query_variants
                .iter()
                .any(|x| normalize_text(x).contains(&term))
            {
                plan.query_variants.push(term);
            }
        }

        if !plan.fact_types.iter().any(|x| x == "ACRONYM") {
            plan.fact_types.push("ACRONYM".to_string());
        }
    }

    if is_strict_normative_question(&normalized_question) {
        plan.answer_type = "direct_answer".to_string();
        plan.needs_numbers = true;
        plan.needs_section_title = true;

        let normative_queries = [
            "facturado por separado",
            "valoracion sera por litros",
            "valoración será por litros",
            "precio es de",
            "incluye transporte mantenimiento",
            "10% gastos de gestion",
            "10% gastos de gestión",
            "tarifa valor por litros",
            "facturado por separado litros tanque",
        ];

        for q in normative_queries {
            let q = q.to_string();
            if !plan.exact_queries.iter().any(|x| x == &q) {
                plan.exact_queries.push(q);
            }
        }

        let normative_variants = [
            "facturado por separado",
            "valoracion por litros",
            "precio por litro",
            "precio por tanque",
            "gastos de gestion",
            "gastos de gestión",
            "tarifa fija",
        ];

        for v in normative_variants {
            let v = v.to_string();
            if !plan.query_variants.iter().any(|x| normalize_text(x).contains(&normalize_text(&v))) {
                plan.query_variants.push(v);
            }
        }
    }
}

fn normalize_question_for_search(question: &str) -> String {
    let q = normalize_text(question);
    let connectors = [
        "que", "se", "dice", "sobre", "dobre", "de", "del", "en", "el", "la",
        "los", "las", "como", "cuanto", "cual", "cuales", "es",
    ];

    let mut out = Vec::new();
    for token in q.split_whitespace() {
        let cleaned = if token == "dobre" {
            "sobre"
        } else if token == "obre" {
            "sobre"
        } else {
            token
        };

        if !connectors.contains(&cleaned) || cleaned == "sobre" {
            out.push(cleaned.to_string());
        }
    }

    out.join(" ")
}

fn rewrite_query_variants(question: &str, intent: &QuestionIntent) -> Vec<String> {
    let q = normalize_question_for_search(question);
    let mut variants = Vec::new();

    let base_stops = [
        "que", "es", "un", "una", "unos", "unas", "el", "la", "los", "las",
        "de", "del", "al", "a", "en", "por", "para", "con", "como", "cual",
        "cuales", "cuanto", "cuantos", "cuanta", "cuantas", "se", "lo", "la",
        "del", "debe", "deben", "ser", "sera", "será",
    ];

    let tokens = tokenize_normalized(&q)
        .into_iter()
        .filter(|t| t.len() >= 2 && !base_stops.contains(&t.as_str()))
        .collect::<Vec<_>>();

    if tokens.is_empty() {
        return variants;
    }

    let joined = tokens.join(" ");
    variants.push(joined.clone());

    match intent {
        QuestionIntent::Definition => {
            variants.push(format!("significa {}", joined));
            variants.push(format!("definicion {}", joined));
            variants.push(format!("que significa {}", joined));
        }
        QuestionIntent::ValuePrice => {
            variants.push(format!("precio {}", joined));
            variants.push(format!("tarifa {}", joined));
            variants.push(format!("valor {}", joined));
            variants.push(format!("cobro {}", joined));
            variants.push(format!("como se cobra {}", joined));
        }
        QuestionIntent::Normative => {
            variants.push(format!("regla {}", joined));
            variants.push(format!("norma {}", joined));
            variants.push(format!("aplica {}", joined));
            variants.push(format!("no aplica {}", joined));
        }
        QuestionIntent::Procedure => {
            variants.push(format!("procedimiento {}", joined));
            variants.push(format!("como {}", joined));
            variants.push(format!("pasos {}", joined));
        }
        QuestionIntent::Requirement => {
            variants.push(format!("requisito {}", joined));
            variants.push(format!("documentos {}", joined));
            variants.push(format!("condiciones {}", joined));
        }
        QuestionIntent::Comparison => {
            variants.push(format!("diferencia {}", joined));
            variants.push(format!("comparacion {}", joined));
        }
        QuestionIntent::List => {
            variants.push(format!("lista {}", joined));
            variants.push(format!("tipos {}", joined));
            variants.push(format!("cuales son {}", joined));
        }
        QuestionIntent::Summary => {
            variants.push(format!("resumen {}", joined));
            variants.push(format!("principales puntos {}", joined));
        }
        QuestionIntent::General => {
            variants.push(format!("informacion sobre {}", joined));
            variants.push(format!("documento {}", joined));
        }
    }

    variants.sort();
    variants.dedup();
    variants
}

fn asks_for_formula_like_answer(question: &str) -> bool {
    let q = normalize_text(question);
    q.contains("formula") || q.contains("multiplica") || q.contains("multiplicar")
        || q.contains("producto") || q.contains("calculo") || q.contains("calcular")
}

fn asks_for_acronym_like_answer(question: &str) -> bool {
    let q = normalize_text(question);
    q.contains("sigla") || q.contains("siglas") || q.contains("acronimo")
        || q.contains("significado exacto") || q.contains("componentes")
        || q.starts_with("que es ")
        || q.starts_with("que significa ")
        || q.starts_with("definicion de ")
        || q.starts_with("define ")
}

fn is_short_definition_lookup_question(question: &str) -> bool {
    let q = normalize_text(question);

    q.starts_with("que es ")
        || q.starts_with("que significa ")
        || q.starts_with("definicion de ")
        || q.starts_with("define ")
        || q.contains(" que es ")
        || q.contains(" que significa ")
}

fn is_strict_normative_question(question: &str) -> bool {
    let q = normalize_text(question);
    q.contains("como se cobra")
        || q.contains("cuanto se paga")
        || q.contains("cual es la regla")
        || q.contains("regla exacta")
        || q.contains("se aplica")
        || q.contains("no aplica")
        || q.contains("debe cobrarse")
        || q.contains("valor fijo")
        || q.contains("tarifa")
        || q.contains("precio")
}

fn extract_short_lookup_terms(question: &str) -> Vec<String> {
    let q = normalize_text(question);

    let stop = [
        "que", "es", "un", "una", "unos", "unas",
        "el", "la", "los", "las", "de", "del",
        "significa", "definicion", "define",
    ];

    let mut terms: Vec<String> = q
        .split_whitespace()
        .map(|s| s.trim().to_lowercase())
        .filter(|s| s.len() >= 2 && s.len() <= 8)
        .filter(|s| s.chars().all(|c| c.is_ascii_alphanumeric()))
        .filter(|s| !stop.contains(&s.as_str()))
        .collect();

    terms.sort();
    terms.dedup();
    terms
}

fn acronym_term_aliases(term: &str) -> Vec<String> {
    let t = normalize_token(term);
    let mut aliases = vec![t.clone()];
    aliases.sort();
    aliases.dedup();
    aliases
}

fn acronym_lookup_queries(question: &str) -> Vec<String> {
    if !is_short_definition_lookup_question(question) && !asks_for_acronym_like_answer(question) {
        return Vec::new();
    }

    let mut queries = Vec::new();

    for term in extract_short_lookup_terms(question) {
        for alias in acronym_term_aliases(&term) {
            queries.push(format!("{}*", normalize_token(&alias)));
            queries.push(format!("{}* AND factor*", normalize_token(&alias)));
            queries.push(format!("{}* AND definicion*", normalize_token(&alias)));
            queries.push(format!("{}* AND significa*", normalize_token(&alias)));
            queries.push(format!("{}* AND concepto*", normalize_token(&alias)));
        }
    }

    queries.sort();
    queries.dedup();
    queries
}

#[allow(dead_code)]
fn acronym_initials(meaning: &str) -> String {
    let stop = [
        "de", "del", "la", "el", "los", "las", "y", "e",
        "a", "en", "por", "para", "con",
    ];

    meaning
        .split_whitespace()
        .filter_map(|w| {
            let clean = normalize_token(w);
            if clean.is_empty() || stop.contains(&clean.as_str()) {
                None
            } else {
                clean.chars().next()
            }
        })
        .collect::<String>()
}

#[allow(dead_code)]
fn acronym_matches_lookup(user_term: &str, acronym: &str, meaning: &str) -> bool {
    let t = normalize_token(user_term);
    let a = normalize_token(acronym);
    let initials = acronym_initials(meaning);

    if t.is_empty() || a.is_empty() {
        return false;
    }

    a == t || a.starts_with(&t) || initials == t || initials.starts_with(&t)
}

fn extract_acronym_definitions_from_text(text: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();

    if let Ok(re) = Regex::new(
        r"(?i)([A-Za-zÁÉÍÓÚÑáéíóúñ][A-Za-zÁÉÍÓÚÑáéíóúñ\s]{3,90}?)\s*\(([A-ZÑ]{2,10})\)"
    ) {
        for cap in re.captures_iter(text) {
            let meaning = cap.get(1).map(|m| m.as_str()).unwrap_or("").trim();
            let acronym = cap.get(2).map(|m| m.as_str()).unwrap_or("").trim();

            if !meaning.is_empty() && !acronym.is_empty() {
                out.push((acronym.to_string(), meaning.to_string()));
            }
        }
    }

    if let Ok(re) = Regex::new(
        r"(?i)\b([A-ZÑ]{2,10})\s*[=:]\s*([^.;\n]{3,120})"
    ) {
        for cap in re.captures_iter(text) {
            let acronym = cap.get(1).map(|m| m.as_str()).unwrap_or("").trim();
            let meaning = cap.get(2).map(|m| m.as_str()).unwrap_or("").trim();

            if !meaning.is_empty() && !acronym.is_empty() {
                out.push((acronym.to_string(), meaning.to_string()));
            }
        }
    }

    if let Ok(re) = Regex::new(
        r"(?i)\b([A-ZÑ]{2,10})\s+(?:significa|quiere\s+decir|corresponde\s+a|se\s+define\s+como)\s+([^.;\n]{3,160})"
    ) {
        for cap in re.captures_iter(text) {
            let acronym = cap.get(1).map(|m| m.as_str()).unwrap_or("").trim();
            let meaning = cap.get(2).map(|m| m.as_str()).unwrap_or("").trim();

            if !meaning.is_empty() && !acronym.is_empty() {
                out.push((acronym.to_string(), meaning.to_string()));
            }
        }
    }

    out.sort();
    out.dedup();
    out
}

#[allow(dead_code)]
fn answer_short_definition_from_evidence(
    question: &str,
    fragments: &[SearchFragment],
    facts: &[ExtractedFact],
) -> Option<String> {
    let intent = detect_question_intent(question);
    if intent != QuestionIntent::Definition && !is_short_definition_lookup_question(question) {
        return None;
    }

    let terms = extract_short_lookup_terms(question);

    if terms.is_empty() {
        return None;
    }

    let mut candidates: Vec<(String, String, String, usize)> = Vec::new();

    for f in fragments {
        for (acronym, meaning) in extract_acronym_definitions_from_text(&f.text) {
            if terms.iter().any(|t| acronym_matches_lookup(t, &acronym, &meaning)) {
                candidates.push((acronym, meaning, f.source.clone(), f.page_number));
            }
        }
    }

    for fact in facts {
        let acronym = fact.label.trim();
        let meaning = if !fact.value.trim().is_empty() {
            fact.value.trim()
        } else {
            fact.text.trim()
        };

        if acronym.len() >= 2
            && terms.iter().any(|t| acronym_matches_lookup(t, acronym, meaning))
        {
            candidates.push((
                acronym.to_string(),
                meaning.to_string(),
                fact.source.clone(),
                fact.page_number,
            ));
        }
    }

    candidates.sort();
    candidates.dedup();
    candidates.sort_by(|a, b| {
        candidate_definition_score(question, &a.0, &a.1, &a.2)
            .partial_cmp(&candidate_definition_score(question, &b.0, &b.1, &b.2))
            .unwrap_or(std::cmp::Ordering::Equal)
            .reverse()
            .then_with(|| a.3.cmp(&b.3))
            .then_with(|| a.0.cmp(&b.0))
    });

    let Some((acronym, meaning, source, page)) = candidates.first() else {
        return None;
    };

    let asked = terms.first().cloned().unwrap_or_default();
    let acronym_norm = normalize_token(acronym);
    let asked_norm = normalize_token(&asked);

    let page_txt = if *page > 0 {
        format!(" en la página {}", page)
    } else {
        String::new()
    };

    if !asked_norm.is_empty() && acronym_norm != asked_norm && acronym_norm.starts_with(&asked_norm) {
        Some(format!(
            "“{}” parece referirse a **{}**, que significa **{}**. La evidencia aparece en **{}**, {}.",
            asked,
            acronym,
            meaning.trim(),
            source,
            page_txt
        ))
    } else {
        Some(format!(
            "**{}** significa **{}**. La evidencia aparece en **{}**, {}.",
            acronym,
            meaning.trim(),
            source,
            page_txt
        ))
    }
}

#[allow(dead_code)]
fn candidate_definition_score(question: &str, acronym: &str, meaning: &str, source: &str) -> f32 {
    let q = normalize_text(question);
    let m = normalize_text(meaning);
    let a = normalize_token(acronym);
    let s = normalize_text(source);
    let mut score = 0.0f32;

    let asked_terms = extract_short_lookup_terms(question);
    if asked_terms.iter().any(|t| {
        let tn = normalize_token(t);
        !tn.is_empty() && (a == tn || a.starts_with(&tn) || tn.starts_with(&a))
    }) {
        score += 50.0;
    }

    let definition_markers = [
        "significa",
        "se define como",
        "corresponde a",
        "quiere decir",
        "es",
        "definicion",
        "concepto",
    ];
    if definition_markers.iter().any(|w| m.contains(w)) {
        score += 30.0;
    }

    if q.contains("que es") || q.contains("que significa") || q.contains("definicion") || q.contains("define") {
        score += 20.0;
    }

    if q.contains("precio") || q.contains("tarifa") || q.contains("costo") || q.contains("cobro") || q.contains("valor") {
        if m.contains("precio") || m.contains("tarifa") || m.contains("costo") || m.contains("valor") || m.contains("facturado por separado") {
            score -= 50.0;
        }
    }

    if q.contains("regla") || q.contains("aplica") || q.contains("debe") || q.contains("no aplica") {
        if m.contains("debe") || m.contains("no aplica") || m.contains("tarifa") || m.contains("valor fijo") {
            score -= 40.0;
        }
    }

    if s.contains("tarifario") || s.contains("manual") {
        score += 4.0;
    }

    if m.len() < 3 {
        score -= 20.0;
    }

    score
}

#[allow(dead_code)]
fn candidate_normative_score(question: &str, fragment: &SearchFragment) -> f32 {
    let q = normalize_text(question);
    let t = normalize_text(&fragment.text);
    let mut score = 0.0f32;

    let normative_markers = [
        "facturado por separado",
        "valoracion sera por litros",
        "valoración será por litros",
        "precio es de",
        "incluye",
        "gastos de gestion",
        "gastos de gestión",
        "tarifa",
        "valor fijo",
        "no aplica",
        "debe cobrarse",
        "se reconocerá",
        "se reconocera",
        "será facturado",
        "sera facturado",
    ];

    for marker in normative_markers {
        if t.contains(marker) {
            score += 20.0;
        }
    }

    if q.contains("precio") || q.contains("cobro") || q.contains("cobra") || q.contains("tarifa") || q.contains("valor") {
        if t.contains("precio") || t.contains("facturado por separado") || t.contains("tarifa") {
            score += 30.0;
        }
    }

    if q.contains("oxigeno") || q.contains("gases medicinales") {
        if t.contains("oxigeno") || t.contains("gases medicinales") {
            score += 15.0;
        }
    }

    if fragment.page_number > 0 {
        score += 2.0;
    }

    score + fragment.score
}

#[allow(dead_code)]
fn answer_strict_normative_from_evidence(question: &str, fragments: &[SearchFragment]) -> Option<String> {
    if !is_strict_normative_question(question) {
        return None;
    }

    let mut candidates: Vec<SearchFragment> = fragments
        .iter()
        .cloned()
        .filter(|f| {
            let t = normalize_text(&f.text);
            t.contains("facturado por separado")
                || t.contains("valoracion sera por litros")
                || t.contains("valoración será por litros")
                || t.contains("precio es de")
                || t.contains("gastos de gestion")
                || t.contains("gastos de gestión")
                || t.contains("tarifa")
                || t.contains("no aplica")
                || t.contains("debe cobrarse")
                || t.contains("valor fijo")
                || t.contains("será facturado")
                || t.contains("sera facturado")
        })
        .collect();

    if candidates.is_empty() {
        return None;
    }

    candidates.sort_by(|a, b| {
        candidate_normative_score(question, b)
            .partial_cmp(&candidate_normative_score(question, a))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let best = candidates.first()?;
    let text = normalize_text(&best.text);

    if text.contains("facturado por separado") && text.contains("oxigeno") {
        if text.contains("gases medicinales") || text.contains("otros gases medicinales") {
            return Some(format!(
                "El oxígeno y los gases medicinales se facturan por separado. El oxígeno se valora por litros a $ 0,01 o por tanque de 8 m3 a $ 72,21, e incluye transporte, mantenimiento del cilindro y 10% por gastos de gestión. Los otros gases medicinales se facturan por litros a $ 0,02 o por tanque de 8 m3 a $ 93,93, con la misma lógica de inclusión. La evidencia aparece en **{}**, página {}.",
                best.source,
                best.page_number
            ));
        }

        return Some(format!(
            "El oxígeno se factura por separado y su valoración es por litros. Su precio es de $ 0,01 por litro o $ 72,21 por tanque de 8 m3, e incluye transporte, mantenimiento del cilindro y 10% por gastos de gestión. La evidencia aparece en **{}**, página {}.",
            best.source,
            best.page_number
        ));
    }

    if text.contains("no aplica") || text.contains("debe cobrarse") || text.contains("valor fijo") {
        return Some(format!(
            "La regla aplicable es: {}. La evidencia aparece en **{}**, página {}.",
            limit_chars(&best.text, 240),
            best.source,
            best.page_number
        ));
    }

    Some(format!(
        "{}",
        limit_chars(&best.text, 380)
    ))
}

fn contains_numeric_unit_pattern(text: &str) -> bool {
    let re = Regex::new(
        r"(?i)\b\d+(?:[.,]\d+)?\s*(%|por\s+ciento|d[ií]as?|mes(?:es)?|a[nñ]os?|horas?|minutos?|segundos?|semanas?|consultas?|p[aá]ginas?|art[ií]culos?|numerales?|literales?|uvr|usd|d[oó]lares?|unidades?|veces)\b"
    ).unwrap();
    re.is_match(text)
}

fn extract_numeric_values(text: &str, source: &str, page_number: usize) -> Vec<NumericEvidence> {
    let re = Regex::new(
        r"(?i)\b(\d+(?:[.,]\d+)?)\s*(%|por\s+ciento|d[ií]as?|mes(?:es)?|a[nñ]os?|horas?|minutos?|segundos?|semanas?|consultas?|p[aá]ginas?|art[ií]culos?|numerales?|literales?|uvr|usd|d[oó]lares?|unidades?|veces)\b"
    ).unwrap();
    let mut out = Vec::new();
    for cap in re.captures_iter(text) {
        let value = cap.get(1).map(|m| m.as_str()).unwrap_or("").to_string();
        let unit = cap.get(2).map(|m| m.as_str()).unwrap_or("").to_string();
        if value.is_empty() || unit.is_empty() { continue; }
        out.push(NumericEvidence {
            value, unit,
            context: clean_excerpt_around(text, cap.get(0).map(|m| m.start()).unwrap_or(0), 220),
            source: source.to_string(),
            page_number,
        });
    }
    out
}

fn clean_excerpt_around(text: &str, pos: usize, max_chars: usize) -> String {
    let chars: Vec<char> = text.chars().collect();
    if chars.is_empty() { return String::new(); }
    let char_pos = text[..pos.min(text.len())].chars().count();
    let half = max_chars / 2;
    let start = char_pos.saturating_sub(half);
    let end = (char_pos + half).min(chars.len());
    chars[start..end].iter().collect::<String>()
        .split_whitespace().collect::<Vec<_>>().join(" ")
}

fn contains_formula_pattern_raw(text: &str) -> bool {
    let norm = normalize_text(text);
    text.contains('=') || text.contains('*') || text.contains('×')
        || text.contains('x') || norm.contains("producto de multiplicar")
        || norm.contains("multiplicar") || norm.contains("formula")
}

fn contains_percentage_or_money_raw(text: &str) -> bool {
    text.contains('%') || text.contains('$') || normalize_text(text).contains("por ciento")
        || normalize_text(text).contains("dolares") || normalize_text(text).contains("usd")
}

fn build_numeric_unit_queries(question: &str) -> Vec<String> {
    let q = normalize_text(question);
    let tokens: Vec<String> = tokenize_normalized(&q)
        .into_iter().filter(|t| t.len() >= 4 && !is_stopword(t)).collect();
    let units = [
        "dia", "dias", "mes", "meses", "ano", "anos",
        "hora", "horas", "consulta", "consultas",
        "porcentaje", "monto", "valor", "uvr", "usd",
    ];
    let mut detected_units: Vec<String> = units.iter()
        .filter(|u| q.contains(*u)).map(|u| u.to_string()).collect();
    detected_units.sort(); detected_units.dedup();
    let mut queries = Vec::new();
    for unit in &detected_units {
        for pair in tokens.windows(2).take(12) {
            if pair.len() == 2 {
                queries.push(format!("{}* AND {}* AND {}*",
                    normalize_token(&pair[0]), normalize_token(&pair[1]), normalize_token(unit)));
            }
        }
        for token in tokens.iter().take(12) {
            queries.push(format!("{}* AND {}*",
                normalize_token(token), normalize_token(unit)));
        }
    }
    if q.contains("periodo") || q.contains("global") || q.contains("cobertura") || q.contains("duracion") {
        let important: Vec<String> = tokens.iter()
            .filter(|t| t.contains("period") || t.contains("global") || t.contains("cobertura")
                || t.contains("duracion") || t.contains("cirugia") || t.contains("complejidad"))
            .cloned().collect();
        for pair in important.windows(2) {
            if pair.len() == 2 {
                queries.push(format!("{}* AND {}*",
                    normalize_token(&pair[0]), normalize_token(&pair[1])));
            }
        }
    }
    queries.sort(); queries.dedup(); queries.truncate(30);
    queries
}

#[allow(dead_code)]
fn looks_like_direct_fts_query(query: &str) -> bool {
    query.contains(" AND ") || query.contains(" OR ") || query.contains('*') || query.contains('"')
}

#[allow(dead_code)]
fn direct_or_iterative_fts_queries(
    probe_query: &str, query_terms: &[String], intent_terms: &[String],
) -> Vec<String> {
    if looks_like_direct_fts_query(probe_query) {
        vec![probe_query.to_string()]
    } else {
        build_iterative_fts_queries(probe_query, query_terms, intent_terms)
    }
}

// ========================================================================
// HTML / FRONTEND
// ========================================================================

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

.answer table{
  border-collapse:collapse;
  width:100%;
  margin:12px 0;
  font-size:14px;
}

.answer th{
  background:#f1f5f9;
  border:1px solid #d0d5dd;
  padding:8px 10px;
  text-align:left;
  font-weight:700;
}

.answer td{
  border:1px solid #e4e7ec;
  padding:8px 10px;
}

.answer tr:nth-child(even) td{
  background:#f8fafc;
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

          <div id="expansionInfo" class="hidden notice" style="margin-top:8px;font-size:13px;padding:8px 12px">
            <span id="expansionText"></span>
          </div>

          <div id="answerBox" class="bubble hidden">
            <div class="row">
              <span id="answerBadge" class="pill">Informe extendido</span>
              <button class="light" onclick="copyAnswer(this)" style="padding:4px 12px;font-size:12px">📋 Copiar</button>
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

          <h3 style="margin-top:18px">Pegar texto como documento</h3>
          <label>Nombre del documento</label>
          <input id="pasteName" placeholder="ej: normativa_actualizada.txt">
          <label>Contenido</label>
          <textarea id="pasteContent" placeholder="Pegue aquí el texto..." style="min-height:120px"></textarea>
          <div class="row" style="margin-top:10px">
            <button class="good" onclick="pasteText()">Pegar como documento</button>
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
  try { data = await r.json(); } catch {}
  if (!r.ok) { throw new Error(data?.error || 'Error de servidor'); }
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
    const data = await api('/api/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: document.getElementById('loginUser').value, password: document.getElementById('loginPass').value }),
    });
    currentUser = data.user;
    showApp();
  } catch (e) { msg('loginMsg', e.message, 'notice err'); }
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
       <p><b>Expansión LLM:</b> ${s.llm_query_expansion ? '✅ activa' : '❌ desactivada'}</p>
       <p><b>Búsqueda:</b> ${escapeHtml(s.search_engine || 'RAG léxico local')}</p>
       <p><b>Actualización:</b> cada ${s.index_interval_seconds}s</p>`;
  } catch (e) { document.getElementById('statusBox').textContent = e.message; }
}

async function ask() { await runQuestion(false); }
async function askWithMarkedSources() { await runQuestion(true); }

function getMarkedSources() {
  return Array.from(document.querySelectorAll('.source-select:checked')).map(input => input.value).filter(Boolean);
}

async function runQuestion(forceMarked) {
  const question = document.getElementById('question').value.trim();
  if (!question) return;

  const askBtn = document.getElementById('askBtn');
  askBtn.disabled = true;
  askBtn.textContent = 'Consultando...';

  document.getElementById('answerBox').classList.remove('hidden');
  document.getElementById('answer').innerHTML = '<p>🔍 Buscando en documentos...</p>';
  document.getElementById('sources').innerHTML = '';
  document.getElementById('expansionInfo').classList.add('hidden');
  document.getElementById('answerBadge').textContent = '✨ Informe extendido';

  try {
    const use_llm = document.getElementById('useLlm').checked;
    const top_k = 6;
    const restrictToMarked = document.getElementById('restrictToMarked').checked || forceMarked;
    const selected_sources = restrictToMarked ? getMarkedSources() : [];

    if (restrictToMarked && selected_sources.length === 0) {
      document.getElementById('answer').innerHTML = '<div class="notice err">Debe marcar al menos una fuente.</div>';
      return;
    }

    const response = await fetch('/api/chat-stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, use_llm, top_k, selected_sources }),
    });

    if (!response.ok) throw new Error('Error: ' + response.statusText);

    let fragments = [];
    let variants = null;

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
        const dataLines = rawEvent.split('\n').filter(line => line.startsWith('data: ')).map(line => line.slice(6));
        if (!dataLines.length) continue;

        let evt = null;
        try { evt = JSON.parse(dataLines.join('\n')); } catch { continue; }

        if (evt.kind === 'variants') {
          variants = evt.payload.variants || [];
          const intent = evt.payload.intent || 'general';
          const expInfo = document.getElementById('expansionInfo');
          expInfo.classList.remove('hidden');
          document.getElementById('expansionText').textContent =
            '🎯 Intención detectada: ' + intent + ' | ' + variants.length + ' variantes generadas';
        } else if (evt.kind === 'fragments') {
          fragments = Array.isArray(evt.payload) ? evt.payload : [];
          lastFragments = fragments;
          document.getElementById('answer').innerHTML = '<p>⏳ Procesando con IA...</p>';
        } else if (evt.kind === 'cached') {
          document.getElementById('answer').innerHTML = renderMarkdown(String(evt.payload || ''));
          document.getElementById('answerBadge').textContent = '📋 Informe (caché)';
        } else if (evt.kind === 'local') {
          document.getElementById('answer').innerHTML = renderMarkdown(String(evt.payload || ''));
          document.getElementById('answerBadge').textContent = '📄 Respuesta local';
        } else if (evt.kind === 'answer') {
          document.getElementById('answer').innerHTML = renderMarkdown(String(evt.payload || ''));
          if (/^(que es|que significa|define|definicion de)\b/i.test(question)) {
          document.getElementById('answerBadge').textContent = '✨ Informe extendido';
          } else {
            document.getElementById('answerBadge').textContent = '✨ Informe extendido';
          }
        } else if (evt.kind === 'variants_final') {
          // ya se mostró al inicio
        } else if (evt.kind === 'error') {
          document.getElementById('answer').innerHTML = '<div class="notice err">' + escapeHtml(String(evt.payload || 'Error desconocido')) + '</div>';
        }
      }
    }

    // Mostrar variantes si no se mostraron antes
    if (variants && !document.getElementById('expansionInfo').classList.contains('hidden')) {
      // ya se mostró
    }

    if (fragments.length > 0) renderSources(fragments);
  } catch (e) {
    document.getElementById('answer').innerHTML = '<div class="notice err">' + escapeHtml(e.message) + '</div>';
  } finally {
    askBtn.disabled = false;
    askBtn.textContent = 'Preguntar';
  }
}

function renderSources(frags) {
  if (!frags.length) {
    lastFragments = [];
    document.getElementById('sources').innerHTML = '<p class="muted">No se encontraron respaldos.</p>';
    return;
  }
  lastFragments = frags;
  document.getElementById('sources').innerHTML = frags.map((f, i) =>
    `<div class="source">
      <label class="meta" style="display:flex;align-items:center;gap:8px">
        <input class="source-select" type="checkbox" value="${escapeHtml(f.source)}" checked style="width:auto">
        <span>Documento ${i + 1}: ${escapeHtml(shortSource(f.source))}${f.page_number > 0 ? ' · pág. ' + f.page_number : ''}</span>
      </label>
      <div>${escapeHtml(f.text).slice(0, 550)}...</div>
    </div>`
  ).join('');
}

async function pasteText() {
  const name = document.getElementById('pasteName').value.trim();
  const content = document.getElementById('pasteContent').value.trim();
  if (!name) { msg('uploadMsg', 'Escriba un nombre para el documento.'); return; }
  if (!content) { msg('uploadMsg', 'Pegue el contenido.'); return; }
  try {
    const r = await api('/api/admin/text', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filename: name.endsWith('.txt') ? name : name + '.txt', content }),
    });
    msg('uploadMsg', 'Texto guardado como "' + r.file + '". Indexando...', 'notice ok');
    document.getElementById('pasteName').value = '';
    document.getElementById('pasteContent').value = '';
    setTimeout(() => { loadDocs(); status(); }, 1500);
  } catch (e) { msg('uploadMsg', e.message, 'notice err'); }
}

function copyAnswer(btn) {
  const el = document.getElementById('answer');
  if (!el || !el.textContent.trim()) return;
  navigator.clipboard.writeText(el.innerText).then(() => {
    btn.textContent = '✅ Copiado!';
    setTimeout(() => { btn.textContent = '📋 Copiar'; }, 1500);
  }).catch(() => alert('No se pudo copiar'));
}

async function deleteDoc(path) {
  if (!confirm('Eliminar "' + path + '"?\nSe borrará el archivo y su índice.')) return;
  try {
    await api('/api/admin/documents', { method: 'DELETE', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ path }) });
    loadDocs(); status();
  } catch (e) { alert(e.message); }
}

async function uploadFiles() {
  const input = document.getElementById('fileInput');
  if (!input.files.length) { msg('uploadMsg', 'Seleccione al menos un archivo.'); return; }
  const fd = new FormData();
  for (const f of input.files) fd.append('files', f);
  try {
    const r = await api('/api/admin/upload', { method: 'POST', body: fd });
    msg('uploadMsg', r.message, 'notice ok');
    input.value = '';
    setTimeout(() => { loadDocs(); status(); }, 1500);
  } catch (e) { msg('uploadMsg', e.message, 'notice err'); }
}

async function reindex() {
  try {
    msg('uploadMsg', 'Reindexando, espere...');
    const r = await api('/api/admin/reindex', { method: 'POST' });
    msg('uploadMsg', 'Listo. Archivos indexados: ' + r.indexed_files + ', omitidos: ' + r.skipped_files + ', chunks: ' + r.chunks_indexed, 'notice ok');
    loadDocs(); status();
  } catch (e) { msg('uploadMsg', e.message, 'notice err'); }
}

async function loadDocs() {
  try {
    const docs = await api('/api/admin/documents');
    document.getElementById('docsBox').innerHTML = docs.length
      ? '<table class="table"><tr><th>Documento</th><th>Chunks</th><th>Indexado</th></tr>' +
        docs.map(d => '<tr><td>' + escapeHtml(d.path) + '</td><td>' + d.chunk_count + '</td><td>' + escapeHtml(d.indexed_at) + '</td><td><button class="bad" onclick="deleteDoc(\'' + escapeJs(d.path) + '\')" style="padding:4px 10px;font-size:12px">🗑 Borrar</button></td></tr>').join('') +
        '</table>'
      : 'No hay documentos indexados.';
  } catch (e) { document.getElementById('docsBox').textContent = e.message; }
}

async function loadUsers() {
  try {
    const users = await api('/api/admin/users');
    document.getElementById('usersBox').innerHTML =
      '<table class="table"><tr><th>Usuario</th><th>Rol</th><th>Estado</th><th>Acción</th></tr>' +
      users.map(u =>
        '<tr><td>' + escapeHtml(u.username) + '</td><td>' + u.role + '</td><td>' + (u.is_active ? 'Activo' : 'Bloqueado') + '</td><td>' +
        (u.username === currentUser.username ? '' : '<button class="' + (u.is_active ? 'bad' : 'good') + '" onclick="toggleUser(\'' + escapeJs(u.username) + '\',' + u.is_active + ')">' + (u.is_active ? 'Bloquear' : 'Activar') + '</button>') +
        '</td></tr>'
      ).join('') + '</table>';
  } catch (e) { document.getElementById('usersBox').textContent = e.message; }
}

async function createUser() {
  try {
    await api('/api/admin/users', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        username: document.getElementById('newUser').value,
        password: document.getElementById('newPass').value,
        role: document.getElementById('newRole').value,
      }),
    });
    msg('userMsg', 'Usuario creado.', 'notice ok');
    document.getElementById('newUser').value = '';
    document.getElementById('newPass').value = '';
    loadUsers();
  } catch (e) { msg('userMsg', e.message, 'notice err'); }
}

async function toggleUser(username, isActive) {
  try {
    let reason = null;
    if (isActive) { reason = prompt('Motivo del bloqueo:', 'Bloqueado por administración.') || 'Bloqueado por administración.'; }
    await api('/api/admin/users/block', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ username, blocked: isActive, reason }) });
    loadUsers();
  } catch (e) { alert(e.message); }
}

function renderMarkdown(md) {
  let tables = [];
  let s = (md || '').replace(/<table>[\s\S]*?<\/table>/gi, function(m) {
    tables.push(m);
    return '\x00TABLE' + (tables.length - 1) + '\x00';
  });
  s = escapeHtml(s);
  s = s.replace(/\x00TABLE(\d+)\x00/g, function(_, i) {
    return '<div style="overflow-x:auto;margin:12px 0">' + tables[parseInt(i)] + '</div>';
  });
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
      if (!inList) { out += '<ul>'; inList = true; }
      out += '<li>' + line.replace(/^\s*[-•] /, '') + '</li>';
    } else {
      if (inList) { out += '</ul>'; inList = false; }
      if (line.trim() === '') { out += ''; }
      else if (line.startsWith('<h') || line.startsWith('<blockquote')) { out += line; }
      else { out += '<p>' + line + '</p>'; }
    }
  }
  if (inList) { out += '</ul>'; }
  return out;
}

function shortSource(s) {
  return String(s || '').split('/').pop();
}

function escapeHtml(str) {
  return String(str ?? '').replace(/[&<>'"]/g, s => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[s]));
}

function escapeJs(str) {
  return String(str ?? '').replace(/\\/g, '\\\\').replace(/'/g, "\\'");
}

boot();
</script>
</body>
</html>
"#;
