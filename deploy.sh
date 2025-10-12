#!/usr/bin/env bash
set -euo pipefail

# Usage: ./deploy.sh <commit-ish>
# The post-receive hook passes the updated commit; you can also call this manually.

commit="${1:-}"
if [[ -z "$commit" ]]; then
  echo "Usage: $0 <commit>" >&2
  exit 1
fi

DEPLOY_ROOT="/home/Neural-Network-Visualisation/releases"
REPO_DIR="$DEPLOY_ROOT/repo.git"
CURRENT_DIR="$DEPLOY_ROOT/current"
BACKUP_DIR="$DEPLOY_ROOT/backups"
TMP_DIR="$DEPLOY_ROOT/.deploy_tmp"

mkdir -p "$CURRENT_DIR" "$BACKUP_DIR"

# Ensure we start from a clean staging area before checking out the commit.
rm -rf "$TMP_DIR"
mkdir -p "$TMP_DIR"

git --git-dir="$REPO_DIR" --work-tree="$TMP_DIR" checkout -f "$commit"
git --git-dir="$REPO_DIR" --work-tree="$TMP_DIR" clean -fd

timestamp="$(date +%Y%m%d-%H%M%S)"
backup_path="$BACKUP_DIR/$timestamp"

rsync -a --delete --exclude=".git" "$TMP_DIR/" "$CURRENT_DIR/"
rsync -a --delete "$TMP_DIR/" "$backup_path/"
echo "$commit" > "$backup_path/.commit"

echo "Deployed commit $commit to $CURRENT_DIR"
echo "Backup created at $backup_path"
