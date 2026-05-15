#!/bin/bash
# rag-cli.sh — Probador CLI para Rust Local RAG
# Uso:
#   ./rag-cli.sh "pregunta"                    # modo rápido + LLM
#   ./rag-cli.sh -m informe "pregunta"         # modo informe
#   ./rag-cli.sh -n "pregunta"                 # sin LLM (respuesta local)
#   ./rag-cli.sh -s "pregunta"                 # modo streaming (SSE)
#   ./rag-cli.sh -m informe -n "pregunta"      # informe sin LLM

set -euo pipefail

BASE_URL="${RAG_URL:-http://127.0.0.1:8080}"
USER="${RAG_USER:-admin}"
PASS="${RAG_PASS:-admin123}"
MODE="rapido"
NO_LLM=false
STREAM=false
TOPK=""

usage() {
    echo "Uso: $0 [-m rapido|informe] [-n] [-s] [-t N] <pregunta>"
    echo ""
    echo "  -m MODE    Modo: rapido (default) | informe"
    echo "  -n         Sin OpenRouter (respuesta local)"
    echo "  -s         Streaming SSE (tiempo real)"
    echo "  -t N       Top-K manual (default: 3 rápido, 6 informe)"
    echo "  -h         Esta ayuda"
    echo ""
    echo "Variables de entorno:"
    echo "  RAG_URL    URL base (default: http://127.0.0.1:8080)"
    echo "  RAG_USER   Usuario (default: admin)"
    echo "  RAG_PASS   Contraseña (default: admin123)"
    exit 0
}

while getopts "m:nst:h" opt; do
    case $opt in
        m) MODE="$OPTARG" ;;
        n) NO_LLM=true ;;
        s) STREAM=true ;;
        t) TOPK="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done
shift $((OPTIND - 1))

QUESTION="${*:-}"
if [ -z "$QUESTION" ]; then
    read -rp "Pregunta: " QUESTION
fi
if [ -z "$QUESTION" ]; then
    echo "Error: debe proporcionar una pregunta." >&2
    exit 1
fi

# --- Login ---
TMPFILE=$(mktemp)
trap 'rm -f "$TMPFILE"' EXIT

echo "→ Autenticando como $USER..." >&2
LOGIN_RESP=$(curl -sS -c "$TMPFILE" -X POST "$BASE_URL/api/login" \
    -H "Content-Type: application/json" \
    -d "{\"username\":\"$USER\",\"password\":\"$PASS\"}" 2>&1)

if ! echo "$LOGIN_RESP" | grep -q '"ok":true'; then
    echo "Error: login fallido — $LOGIN_RESP" >&2
    exit 1
fi

# --- Calcular top_k ---
if [ -z "$TOPK" ]; then
    if [ "$MODE" = "informe" ]; then
        TOPK=6
    else
        TOPK=3
    fi
fi

USE_LLM="true"
if $NO_LLM; then
    USE_LLM="false"
fi

BODY=$(cat <<EOF
{"question":"$QUESTION","use_llm":$USE_LLM,"mode":"$MODE","top_k":$TOPK}
EOF
)

echo "→ Modo: $MODE | top_k: $TOPK | LLM: $USE_LLM" >&2
echo "" >&2

# --- Llamada a la API ---
if $STREAM; then
    echo "═╦═ RESPUESTA (streaming) ═══════════════════════════════════" >&2
    echo "" >&2
    
    curl -sS -b "$TMPFILE" -X POST "$BASE_URL/api/chat-stream" \
        -H "Content-Type: application/json" \
        -d "$BODY" 2>&1 | while IFS= read -r line; do
        if [[ "$line" == data:* ]]; then
            data="${line#data: }"
            case "$data" in
                __FRAGMENTS__*)
                    echo "📎 Fragmentos recuperados" >&2
                    ;;
                __CACHED__*)
                    echo -e "\n💾 Cache:\n${data#__CACHED__}" 
                    ;;
                __LOCAL__*)
                    echo -e "\n📋 Local:\n${data#__LOCAL__}"
                    ;;
                __ANSWER__*)
                    # Limpiar pantalla parcial, mostrar progresivo
                    text="${data#__ANSWER__}"
                    printf "\r\033[K%s" "$text"
                    ;;
                __ERROR__*)
                    echo -e "\n❌ Error: ${data#__ERROR__}" >&2
                    ;;
            esac
        fi
    done
    echo ""
else
    echo "═╦═ RESPUESTA ═══════════════════════════════════════════════" >&2
    
    RESP=$(curl -sS -b "$TMPFILE" -X POST "$BASE_URL/api/chat" \
        -H "Content-Type: application/json" \
        -d "$BODY" 2>&1)
    
    if echo "$RESP" | grep -q '"error"'; then
        echo "❌ $(echo "$RESP" | grep -o '"error":"[^"]*"' | head -1)" >&2
        exit 1
    fi
    
    # Extraer y mostrar campos clave
    ANSWER=$(echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['answer'])" 2>/dev/null || echo "$RESP")
    USED_LLM=$(echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['used_llm'])" 2>/dev/null || echo "?")
    MODEL=$(echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('model',''))" 2>/dev/null || echo "")
    FRAG_COUNT=$(echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('fragments',[])))" 2>/dev/null || echo "?")
    
    if [ "$USED_LLM" = "True" ]; then
        echo "🤖 LLM: $MODEL | Fragmentos: $FRAG_COUNT"
    elif echo "$ANSWER" | grep -q '\[Cache\]'; then
        echo "💾 Cache | Fragmentos: $FRAG_COUNT"
        ANSWER="${ANSWER//\[Cache\] /}"
    else
        echo "📋 Local | Fragmentos: $FRAG_COUNT"
    fi
    echo ""
    echo "$ANSWER"
fi

echo "" >&2
echo "═╩════════════════════════════════════════════════════════════" >&2
