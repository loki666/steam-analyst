#!/usr/bin/env bash
# Hosted deploy to the Oracle A1 VM, from the dev box (which holds the data and the index).
#
#   SGA_DEPLOY_HOST=ubuntu@<vm ip> SGA_DEPLOY_KEY=~/.ssh/<key> scripts/deploy.sh <step>
#
# Steps (in order for a first deploy; `all` runs them all):
#   bootstrap  install Docker on the VM and open 80/443 in its firewall (once)
#   env        write deploy/.env on the VM from this box's .env (LLM endpoint + key)
#   sync       copy the source, data/processed and the trained models to the VM
#   index      snapshot the local Qdrant collection and restore it on the VM
#   up         build the arm64 image and start qdrant + app + caddy
# Re-deploying code only: `sync` then `up`.
set -euo pipefail

HOST="${SGA_DEPLOY_HOST:?set SGA_DEPLOY_HOST, e.g. ubuntu@1.2.3.4}"
KEY="${SGA_DEPLOY_KEY:-}"
REMOTE_DIR="${SGA_DEPLOY_DIR:-sga}"
COLLECTION="${SGA_QDRANT_COLLECTION:-sga_chunks}"
LOCAL_QDRANT="${QDRANT_URL:-http://localhost:6333}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

SSH_OPTS=(-o ConnectTimeout=15)
[[ -n "$KEY" ]] && SSH_OPTS+=(-i "$KEY")
remote() { ssh "${SSH_OPTS[@]}" "$HOST" "$@"; }
copy() { rsync -az --info=progress2 -e "ssh ${SSH_OPTS[*]}" "$@"; }
log() { printf '\n== %s\n' "$*"; }

bootstrap() {
    log "Docker + firewall on $HOST"
    remote 'bash -s' <<'EOF'
set -euo pipefail
if ! command -v docker >/dev/null; then
    sudo apt-get update -q
    sudo apt-get install -y -q docker.io docker-compose-v2 rsync
    sudo usermod -aG docker "$USER"
fi
# Oracle's Ubuntu images reject everything but 22 in iptables; open 80/443 before the REJECT
for port in 80 443; do
    sudo iptables -C INPUT -p tcp --dport "$port" -m state --state NEW -j ACCEPT 2>/dev/null ||
        sudo iptables -I INPUT 5 -p tcp --dport "$port" -m state --state NEW -j ACCEPT
done
sudo mkdir -p /etc/iptables && sudo sh -c 'iptables-save > /etc/iptables/rules.v4'
docker --version || true
EOF
}

env_file() {
    log "deploy/.env on $HOST (LLM endpoint and key from $ROOT/.env)"
    local site="${SITE_ADDRESS:-$(echo "${HOST#*@}" | tr . -).sslip.io}"
    local llm
    llm=$(grep -E '^(SGA_LLM_BASE_URL|SGA_LLM_API_KEY)=.' "$ROOT/.env" || true)
    if [[ $(grep -c . <<<"$llm") -ne 2 ]]; then
        echo "SGA_LLM_BASE_URL and SGA_LLM_API_KEY must both be set in $ROOT/.env" >&2
        return 1
    fi
    printf '%s\nSITE_ADDRESS=%s\n' "$llm" "$site" | remote "mkdir -p $REMOTE_DIR/deploy && umask 077 && cat > $REMOTE_DIR/deploy/.env"
    echo "site: https://$site"
}

sync() {
    log "source, tables and models -> $HOST:$REMOTE_DIR"
    (cd "$ROOT" && git ls-files -z --cached --others --exclude-standard -- . ":!.claude" |
        copy --from0 --files-from=- --ignore-missing-args ./ "$HOST:$REMOTE_DIR/")
    copy "$ROOT/data/processed/" "$HOST:$REMOTE_DIR/data/processed/"
    # trained models only: the Hugging Face cache (bge-m3) downloads on the VM itself
    copy --exclude hf/ "$ROOT/models/" "$HOST:$REMOTE_DIR/models/"
}

index() {
    log "Qdrant snapshot of $COLLECTION"
    local name
    name=$(curl -fsS -X POST "$LOCAL_QDRANT/collections/$COLLECTION/snapshots?wait=true" |
        python3 -c 'import json,sys; print(json.load(sys.stdin)["result"]["name"])')
    local tmp="$ROOT/deploy/snapshots"
    mkdir -p "$tmp"
    curl -fS -o "$tmp/$name" "$LOCAL_QDRANT/collections/$COLLECTION/snapshots/$name"
    curl -fsS -X DELETE "$LOCAL_QDRANT/collections/$COLLECTION/snapshots/$name" >/dev/null
    copy "$tmp/$name" "$HOST:$REMOTE_DIR/deploy/snapshots/"
    log "restore on $HOST"
    remote "cd $REMOTE_DIR && docker compose -f deploy/compose.yml up -d qdrant --wait &&
        docker run --rm --network sga_default curlimages/curl -fsS -X PUT \
            'http://qdrant:6333/collections/$COLLECTION/snapshots/recover?wait=true' \
            -H 'Content-Type: application/json' \
            -d '{\"location\": \"file:///qdrant/snapshots/$name\"}' &&
        rm -f deploy/snapshots/$name"
    rm -f "$tmp/$name"
}

up() {
    log "build + start on $HOST"
    remote "cd $REMOTE_DIR && docker compose -f deploy/compose.yml up -d --build --wait"
    log "warm-up: load bge-m3 (downloads once into models/hf) so no visitor waits for it"
    remote "docker exec sga-app python -c 'from steam_game_analyst.cli import _embedder; \
        _embedder(\"cpu\").encode([\"warm-up\"]); print(\"bge-m3 ready\")'"
    remote "docker ps --format '{{.Names}}  {{.Status}}'"
}

case "${1:-}" in
bootstrap | sync | index | up) "$1" ;;
env) env_file ;;
all) bootstrap && env_file && sync && index && up ;;
*) sed -n '2,13p' "$0" && exit 1 ;;
esac
