#!/usr/bin/env bash
# ============================================
# DeepTutor — Start Script (Bazzite Native)
# ============================================
# Usage:  ./start.sh          (start both services)
#         ./start.sh backend   (backend only)
#         ./start.sh frontend  (frontend only)
#         ./start.sh stop      (stop both services)
# ============================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
WEB_DIR="$SCRIPT_DIR/web"
NODE22="/home/linuxbrew/.linuxbrew/opt/node@22/bin/node"
BACKEND_PORT="${BACKEND_PORT:-8001}"
FRONTEND_PORT="${FRONTEND_PORT:-3782}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[DeepTutor]${NC} $*"; }
warn()  { echo -e "${YELLOW}[DeepTutor]${NC} $*"; }
error() { echo -e "${RED}[DeepTutor]${NC} $*" >&2; }

check_prereqs() {
    if [[ ! -d "$VENV_DIR" ]]; then
        error "Python venv not found at $VENV_DIR"
        error "Run:  cd $SCRIPT_DIR && python3 -m venv .venv && source .venv/bin/activate && pip install -e '.[server]' && pip install numpy && pip install -r requirements/server.txt"
        exit 1
    fi

    if [[ ! -x "$NODE22" ]]; then
        error "Node.js 22 not found at $NODE22"
        error "Run:  brew install node@22"
        exit 1
    fi

    if [[ ! -d "$WEB_DIR/node_modules" ]]; then
        error "Frontend node_modules not found."
        error "Run:  cd $WEB_DIR && $NODE22/../npm install"
        exit 1
    fi

    # Verify gh CLI has copilot scope
    if ! command -v gh &>/dev/null; then
        error "GitHub CLI (gh) not found. Install it and run: gh auth refresh -s copilot"
        exit 1
    fi
    if ! gh auth status 2>&1 | grep -q copilot; then
        error "gh CLI missing 'copilot' scope. Run: gh auth refresh -s copilot"
        exit 1
    fi
}

fix_swc_permissions() {
    local swc="$WEB_DIR/node_modules/@next/swc-linux-x64-gnu/next-swc.linux-x64-gnu.node"
    if [[ -f "$swc" && ! -x "$swc" ]]; then
        warn "Fixing SWC binary permissions..."
        chmod +x "$swc"
    fi
}

inject_copilot_token() {
    info "Fetching GitHub Copilot token from gh CLI..."
    local token
    token="$(gh auth token 2>/dev/null)" || {
        error "Failed to get gh auth token"
        exit 1
    }
    export LLM_API_KEY="$token"
    info "Copilot token injected into LLM_API_KEY"
}

start_backend() {
    if ss -tlnp 2>/dev/null | grep -q ":${BACKEND_PORT} "; then
        warn "Backend already running on port $BACKEND_PORT"
        return 0
    fi

    info "Starting backend on port $BACKEND_PORT..."
    cd "$SCRIPT_DIR"
    source "$VENV_DIR/bin/activate"
    python -m deeptutor.api.run_server &
    BACKEND_PID=$!
    echo "$BACKEND_PID" > "$SCRIPT_DIR/.backend.pid"

    # Wait for backend to be ready
    for i in $(seq 1 30); do
        if ss -tlnp 2>/dev/null | grep -q ":${BACKEND_PORT} "; then
            info "Backend ready (PID $BACKEND_PID)"
            return 0
        fi
        sleep 1
    done
    error "Backend failed to start within 30s"
    return 1
}

start_frontend() {
    if ss -tlnp 2>/dev/null | grep -q ":${FRONTEND_PORT} "; then
        warn "Frontend already running on port $FRONTEND_PORT"
        return 0
    fi

    fix_swc_permissions

    info "Starting frontend on port $FRONTEND_PORT..."
    cd "$WEB_DIR"
    PATH="/home/linuxbrew/.linuxbrew/opt/node@22/bin:$PATH" \
        node --max-old-space-size=4096 ./node_modules/next/dist/bin/next dev -p "$FRONTEND_PORT" &
    FRONTEND_PID=$!
    echo "$FRONTEND_PID" > "$SCRIPT_DIR/.frontend.pid"

    # Wait for frontend to be ready
    for i in $(seq 1 30); do
        if ss -tlnp 2>/dev/null | grep -q ":${FRONTEND_PORT} "; then
            info "Frontend ready (PID $FRONTEND_PID)"
            return 0
        fi
        sleep 1
    done
    error "Frontend failed to start within 30s"
    return 1
}

stop_services() {
    info "Stopping services..."

    if [[ -f "$SCRIPT_DIR/.backend.pid" ]]; then
        local pid
        pid=$(cat "$SCRIPT_DIR/.backend.pid")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            info "Backend stopped (PID $pid)"
        fi
        rm -f "$SCRIPT_DIR/.backend.pid"
    fi

    # Also kill any lingering processes on the ports
    local pids
    pids=$(ss -tlnp 2>/dev/null | grep ":${BACKEND_PORT} " | grep -oP 'pid=\K[0-9]+' || true)
    for pid in $pids; do
        kill "$pid" 2>/dev/null || true
    done

    if [[ -f "$SCRIPT_DIR/.frontend.pid" ]]; then
        local pid
        pid=$(cat "$SCRIPT_DIR/.frontend.pid")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            info "Frontend stopped (PID $pid)"
        fi
        rm -f "$SCRIPT_DIR/.frontend.pid"
    fi

    pids=$(ss -tlnp 2>/dev/null | grep ":${FRONTEND_PORT} " | grep -oP 'pid=\K[0-9]+' || true)
    for pid in $pids; do
        kill "$pid" 2>/dev/null || true
    done

    info "All services stopped."
}

show_status() {
    echo ""
    if ss -tlnp 2>/dev/null | grep -q ":${BACKEND_PORT} "; then
        info "Backend:  ${GREEN}RUNNING${NC}  → http://localhost:$BACKEND_PORT"
    else
        info "Backend:  ${RED}STOPPED${NC}"
    fi

    if ss -tlnp 2>/dev/null | grep -q ":${FRONTEND_PORT} "; then
        info "Frontend: ${GREEN}RUNNING${NC}  → http://localhost:$FRONTEND_PORT"
    else
        info "Frontend: ${RED}STOPPED${NC}"
    fi
    echo ""
}

# --- Main ---
case "${1:-all}" in
    backend)
        check_prereqs
        inject_copilot_token
        start_backend
        show_status
        info "Press Ctrl+C to stop"
        wait
        ;;
    frontend)
        check_prereqs
        start_frontend
        show_status
        info "Press Ctrl+C to stop"
        wait
        ;;
    stop)
        stop_services
        ;;
    status)
        show_status
        ;;
    all|"")
        check_prereqs
        inject_copilot_token
        start_backend
        start_frontend
        show_status
        info "DeepTutor is running at http://localhost:$FRONTEND_PORT"
        info "Press Ctrl+C to stop both services"
        trap stop_services EXIT INT TERM
        wait
        ;;
    *)
        echo "Usage: $0 [all|backend|frontend|stop|status]"
        exit 1
        ;;
esac
