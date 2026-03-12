#!/usr/bin/env bash
set -euo pipefail

# Isolated helper for DB restore operations on Modal volume.
# Safety defaults:
# - Restores from latest backup unless a backup name is provided.
# - Creates an automatic pre-restore backup by default.
# - Refuses empty-table restores unless --restore-allow-empty is set.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/modal_restore_db.sh --list [limit]
  ./scripts/modal_restore_db.sh [backup_name] [extra restore flags]

Examples:
  ./scripts/modal_restore_db.sh --list 20
  ./scripts/modal_restore_db.sh
  ./scripts/modal_restore_db.sh 20260312T172948Z_abort-slow-rebuild
  ./scripts/modal_restore_db.sh 20260312T172948Z_abort-slow-rebuild --restore-skip-pre-backup
EOF
}

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  usage
  exit 0
fi

if [ "${1:-}" = "--list" ]; then
  LIMIT="${2:-20}"
  cd "${ROOT_DIR}"
  ./scripts/modal_capture.sh modal-list-db-backups "${LIMIT}"
  exit 0
fi

BACKUP_NAME=""
if [ "${1:-}" != "" ] && [[ "${1:-}" != --* ]]; then
  BACKUP_NAME="$1"
  shift || true
fi

cd "${ROOT_DIR}"
if [ -n "${BACKUP_NAME}" ]; then
  ./scripts/modal_capture.sh modal-restore-db "${BACKUP_NAME}" "$@"
else
  ./scripts/modal_capture.sh modal-restore-db "$@"
fi

