#!/usr/bin/env bash
set -euo pipefail

# Usage: ./deploy.sh [commit-ish]
# When no commit is supplied, the script deploys the current branch's HEAD.

commit_ref="${1:-HEAD}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if ! REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"; then
  echo "Error: $SCRIPT_DIR is not inside a git repository." >&2
  exit 1
fi

if ! commit="$(git -C "$REPO_ROOT" rev-parse --verify "$commit_ref" 2>/dev/null)"; then
  echo "Error: Unable to resolve commit '$commit_ref'." >&2
  exit 1
fi

DEPLOY_ROOT="/home/Neural-Network-Visualisation/releases"
CURRENT_DIR="$DEPLOY_ROOT/current"
BACKUP_DIR="$DEPLOY_ROOT/backups"
TMP_DIR="$DEPLOY_ROOT/.deploy_tmp"

mkdir -p "$CURRENT_DIR" "$BACKUP_DIR"

# Ensure we start from a clean staging area before checking out the commit.
rm -rf "$TMP_DIR"
mkdir -p "$TMP_DIR"

git -C "$REPO_ROOT" archive "$commit" | tar -C "$TMP_DIR" -xf -

timestamp="$(date +%Y%m%d-%H%M%S)"
backup_path="$BACKUP_DIR/$timestamp"

rsync -a --delete --exclude=".git" "$TMP_DIR/" "$CURRENT_DIR/"
rsync -a --delete "$TMP_DIR/" "$backup_path/"
echo "$commit" > "$backup_path/.commit"

echo "Deployed commit $commit to $CURRENT_DIR"
echo "Backup created at $backup_path"
