#!/bin/sh
set -eu

# Safe default: build and verify the sanitized copy without touching GitHub.
# Preview:  ./push_fpem_to_github.sh
# Push:     PUSH=true ./push_fpem_to_github.sh

OWNER="${OWNER:-fffffffuuuuuccckkk}"
REPO_NAME="${REPO_NAME:-fpem}"
SOURCE_DIR="${SOURCE_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
WORK_DIR="${WORK_DIR:-/tmp/${REPO_NAME}-sanitized-push}"
BRANCH="${BRANCH:-main}"
REMOTE_URL="git@github.com:${OWNER}/${REPO_NAME}.git"
PUSH="${PUSH:-false}"
SUMMARY_MAX_SIZE="${SUMMARY_MAX_SIZE:-1m}"
MAX_FILE_SIZE_MB="${MAX_FILE_SIZE_MB:-5}"
SSH_TRANSPORT="${SSH_TRANSPORT:-${SOURCE_DIR}/.push-tools/git_ssh_paramiko.py}"

if [ ! -x "${SSH_TRANSPORT}" ] && [ -x /data/OuXiaoyu/STEVE_CODE/.push-tools/git_ssh_paramiko.py ]; then
  SSH_TRANSPORT=/data/OuXiaoyu/STEVE_CODE/.push-tools/git_ssh_paramiko.py
fi

log() {
  printf '[push-fpem] %s\n' "$*"
}

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

git_remote() {
  PYTHONWARNINGS=ignore GIT_SSH="${SSH_TRANSPORT}" GIT_SSH_VARIANT=ssh "$@"
}

repo_exists() {
  git_remote git ls-remote "${REMOTE_URL}" >/dev/null 2>&1
}

validate_paths() {
  if [ ! -d "${SOURCE_DIR}" ]; then
    log "Source directory does not exist: ${SOURCE_DIR}"
    exit 2
  fi
  case "${WORK_DIR}" in
    /tmp/*-sanitized-push) ;;
    *)
      log "Refusing unsafe WORK_DIR; expected /tmp/*-sanitized-push: ${WORK_DIR}"
      exit 2
      ;;
  esac
}

append_publish_gitignore() {
  cat >> "${WORK_DIR}/.gitignore" <<'EOF'

# Sanitized GitHub publishing policy
dataset/
data/
checkpoints/
results/
test_results/
logs/
*.pt
*.pth
*.ckpt
*.bin
*.npy
*.npz
*.pkl
*.pickle
*.h5
*.hdf5
*.onnx
*.tar
*.tar.*
*.zip
*.7z
*.png
*.jpg
*.jpeg
*.gif
*.pdf
*.ipynb
*.out
__pycache__/
*.pyc
.pytest_cache/
.env
.env.*
*.pem
*.key

# Publish only lightweight, human-readable result summaries. The directory
# rules are required because the source repository also ignores results/.
!results/
results/*
!results/**/
!results/**/*.txt
!results/**/*.log
EOF
}

copy_small_summaries() {
  for relative in case_outputs decomp_shift_outputs graph_diagnostics foil_env test_results; do
    if [ -d "${SOURCE_DIR}/${relative}" ]; then
      mkdir -p "${WORK_DIR}/${relative}"
      rsync -a --prune-empty-dirs --max-size="${SUMMARY_MAX_SIZE}" \
        --include='*/' \
        --include='*.json' \
        --include='*.csv' \
        --include='*.txt' \
        --include='*.md' \
        --exclude='*' \
        "${SOURCE_DIR}/${relative}/" "${WORK_DIR}/${relative}/"
    fi
  done

  if [ -d "${SOURCE_DIR}/results" ]; then
    mkdir -p "${WORK_DIR}/results"
    rsync -a --prune-empty-dirs --max-size="${SUMMARY_MAX_SIZE}" \
      --include='*/' \
      --include='*.txt' \
      --include='*.log' \
      --exclude='*' \
      "${SOURCE_DIR}/results/" "${WORK_DIR}/results/"
  fi
}

prepare_clean_copy() {
  log "Preparing sanitized copy: ${WORK_DIR}"
  rm -rf "${WORK_DIR}"
  mkdir -p "${WORK_DIR}"
  rsync -a --delete \
    --exclude='.git/' \
    --exclude='.push-tools/' \
    --exclude='.env' \
    --exclude='.env.*' \
    --exclude='*.pem' \
    --exclude='*.key' \
    --exclude='dataset/' \
    --exclude='data/' \
    --exclude='checkpoints/' \
    --exclude='results/' \
    --exclude='test_results/' \
    --exclude='case_outputs/' \
    --exclude='decomp_shift_outputs/' \
    --exclude='graph_diagnostics/' \
    --exclude='foil_env/' \
    --exclude='logs/' \
    --exclude='*.pt' \
    --exclude='*.pth' \
    --exclude='*.ckpt' \
    --exclude='*.bin' \
    --exclude='*.npy' \
    --exclude='*.npz' \
    --exclude='*.pkl' \
    --exclude='*.pickle' \
    --exclude='*.h5' \
    --exclude='*.hdf5' \
    --exclude='*.onnx' \
    --exclude='*.7z' \
    --exclude='*.zip' \
    --exclude='*.tar' \
    --exclude='*.tar.*' \
    --exclude='*.png' \
    --exclude='*.jpg' \
    --exclude='*.jpeg' \
    --exclude='*.gif' \
    --exclude='*.pdf' \
    --exclude='*.ipynb' \
    --exclude='*.out' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.pytest_cache/' \
    --exclude='.DS_Store' \
    "${SOURCE_DIR}/" "${WORK_DIR}/"

  copy_small_summaries
  append_publish_gitignore
}

verify_clean_copy() {
  log "Checking sanitized copy for blocked artifacts"
  blocked=$(find "${WORK_DIR}" -type f \( \
    -iname '*.pt' -o -iname '*.pth' -o -iname '*.ckpt' -o -iname '*.bin' -o \
    -iname '*.npy' -o -iname '*.npz' -o -iname '*.pkl' -o -iname '*.pickle' -o \
    -iname '*.h5' -o -iname '*.hdf5' -o -iname '*.onnx' -o \
    -iname '*.7z' -o -iname '*.zip' -o -iname '*.tar' -o -iname '*.tar.*' -o \
    -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.gif' -o \
    -iname '*.pdf' -o -iname '*.ipynb' -o -iname '*.out' -o \
    -iname '*.pyc' -o -iname '*.pem' -o -iname '*.key' \
  \) -print)
  if [ -n "${blocked}" ]; then
    log "Blocked artifacts remain:"
    printf '%s\n' "${blocked}"
    exit 2
  fi

  unexpected_results=""
  if [ -d "${WORK_DIR}/results" ]; then
    unexpected_results=$(find "${WORK_DIR}/results" -type f \
      ! -iname '*.txt' ! -iname '*.log' -print)
  fi
  if [ -n "${unexpected_results}" ]; then
    log "Unexpected non-text result files remain:"
    printf '%s\n' "${unexpected_results}"
    exit 2
  fi

  oversized=$(find "${WORK_DIR}" -type f -size "+${MAX_FILE_SIZE_MB}M" -print)
  if [ -n "${oversized}" ]; then
    log "Files larger than ${MAX_FILE_SIZE_MB} MiB remain:"
    printf '%s\n' "${oversized}"
    exit 2
  fi

  for blocked_dir in dataset data checkpoints logs; do
    if [ -e "${WORK_DIR}/${blocked_dir}" ]; then
      log "Blocked directory remains: ${WORK_DIR}/${blocked_dir}"
      exit 2
    fi
  done

  file_count=$(find "${WORK_DIR}" -type f | wc -l | tr -d ' ')
  total_size=$(du -sh "${WORK_DIR}" | awk '{print $1}')
  log "Sanitized copy verified: ${file_count} files, ${total_size} total"
  result_summary_count=$(find "${WORK_DIR}/results" -type f 2>/dev/null | wc -l | tr -d ' ')
  log "Included readable result summaries: ${result_summary_count}"
  log "Largest included files:"
  find "${WORK_DIR}" -type f -printf '%s %P\n' | sort -nr | head -20
}

push_repo() {
  if [ ! -x "${SSH_TRANSPORT}" ]; then
    log "Git SSH transport is missing or not executable: ${SSH_TRANSPORT}"
    exit 2
  fi
  if ! repo_exists; then
    log "GitHub repository is unavailable: ${REMOTE_URL}"
    exit 4
  fi

  remote_head=$(git_remote git ls-remote "${REMOTE_URL}" "refs/heads/${BRANCH}" | awk 'NR == 1 {print $1}')
  cd "${WORK_DIR}"
  rm -rf .git
  git init >/dev/null
  git checkout -b "${BRANCH}" >/dev/null 2>&1 || git branch -M "${BRANCH}"
  git config user.name "${GIT_AUTHOR_NAME:-OuXiaoyu}"
  git config user.email "${GIT_AUTHOR_EMAIL:-ouxiaoyu@example.com}"
  git add .
  git commit -m "${COMMIT_MESSAGE:-Publish sanitized FPem code and summaries}" >/dev/null
  git remote add origin "${REMOTE_URL}"

  log "Pushing sanitized snapshot to ${REMOTE_URL}"
  if [ -n "${remote_head}" ]; then
    # Replace only the exact remote revision inspected above. If another push
    # races us, --force-with-lease rejects this operation instead of overwriting it.
    git_remote git push -u origin "${BRANCH}" \
      "--force-with-lease=refs/heads/${BRANCH}:${remote_head}"
  else
    git_remote git push -u origin "${BRANCH}"
  fi
  log "Done: https://github.com/${OWNER}/${REPO_NAME}"
}

main() {
  if ! have_cmd rsync; then
    log "rsync is required but not found."
    exit 2
  fi
  validate_paths
  prepare_clean_copy
  verify_clean_copy

  if [ "${PUSH}" != "true" ]; then
    cat <<EOF
[push-fpem] Preview only; GitHub was not changed.
[push-fpem] Inspect the sanitized copy at: ${WORK_DIR}
[push-fpem] To publish it, run:
  PUSH=true ${SOURCE_DIR}/push_fpem_to_github.sh
EOF
    exit 0
  fi
  push_repo
}

main "$@"
