#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Voxtral Stack — Helper Scripts
# Usage: bash scripts/run.sh <command>
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

COMPOSE="docker compose"
GATEWAY_URL="http://localhost:7860"

cmd=${1:-help}

case "$cmd" in

  # ── Start everything ───────────────────────────────────────────────────────
  up)
    echo "▶ Starting Voxtral stack..."
    $COMPOSE up -d --build
    echo ""
    echo "⏳ Waiting for gateway to be ready (model loading can take 3–5 min)..."
    until curl -sf "$GATEWAY_URL/health" | grep -q '"vllm_backend": "ok"'; do
      printf "."
      sleep 10
    done
    echo ""
    echo "✅ Stack is ready!"
    echo "   Gateway:     $GATEWAY_URL"
    echo "   Swagger UI:  $GATEWAY_URL/docs"
    echo "   vLLM direct: http://localhost:8000"
    ;;

  # ── Stop ──────────────────────────────────────────────────────────────────
  down)
    echo "⏹ Stopping Voxtral stack..."
    $COMPOSE down
    ;;

  # ── View logs ─────────────────────────────────────────────────────────────
  logs)
    service=${2:-}
    if [ -n "$service" ]; then
      $COMPOSE logs -f "$service"
    else
      $COMPOSE logs -f
    fi
    ;;

  # ── Transcribe an audio file ───────────────────────────────────────────────
  transcribe)
    file=${2:-}
    if [ -z "$file" ]; then
      echo "Usage: $0 transcribe <path/to/audio.wav>"
      exit 1
    fi
    echo "📤 Uploading $file for transcription..."
    curl -s -X POST "$GATEWAY_URL/transcribe" \
      -F "file=@$file" \
      | python3 -m json.tool
    ;;

  # ── Health check ──────────────────────────────────────────────────────────
  health)
    echo "🔍 Health check:"
    curl -s "$GATEWAY_URL/health" | python3 -m json.tool
    ;;

  # ── GPU status ────────────────────────────────────────────────────────────
  gpu)
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu \
               --format=csv,noheader,nounits \
    | awk -F',' '{
        printf "GPU:   %s\n", $1
        printf "VRAM:  %s / %s MiB\n", $2, $3
        printf "Util:  %s%%\n", $4
        printf "Temp:  %s°C\n", $5
    }'
    ;;

  # ── Pull updated images ───────────────────────────────────────────────────
  pull)
    echo "⬇ Pulling latest vLLM nightly base image..."
    docker pull vllm/vllm-openai:nightly
    ;;

  # ── Remove model cache volume (forces re-download) ────────────────────────
  purge-models)
    echo "⚠  This will delete downloaded model weights (~8 GB). Continue? [y/N]"
    read -r confirm
    if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
      $COMPOSE down -v
      docker volume rm voxtral_voxtral-models 2>/dev/null || true
      echo "✅ Model cache removed."
    fi
    ;;

  help|*)
    cat <<EOF
Voxtral Stack Helper

Commands:
  up                   Build & start all services
  down                 Stop all services
  logs [service]       Tail logs (service: voxtral-server | voxtral-gateway)
  transcribe <file>    Transcribe an audio file via the gateway REST API
  health               Show health status
  gpu                  Show GPU utilization
  pull                 Pull latest vLLM base image
  purge-models         Delete cached model weights (forces re-download)
EOF
    ;;
esac
