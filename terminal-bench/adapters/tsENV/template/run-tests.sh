#!/bin/bash
set -euo pipefail

# Run pytest-based verification that checks the agent's reported answer.

TEST_DIR="${TEST_DIR:-/app/tests}"

choose_target() {
  local dir="$1"
  if [ ! -d "$dir" ]; then
    return 1
  fi
  if find "$dir" -maxdepth 2 -name 'test*.py' -print -quit | grep -q .; then
    printf "%s" "$dir"
    return 0
  fi
  return 1
}

TARGET=""
for candidate in "$TEST_DIR/tests" "$TEST_DIR" "/app/tests"; do
  target_dir="$(choose_target "$candidate" || true)"
  if [ -n "$target_dir" ]; then
    TARGET="$target_dir"
    break
  fi
done

if [ -z "$TARGET" ]; then
  echo "No test files found under $TEST_DIR or /app/tests" >&2
  find "$TEST_DIR" -maxdepth 2 -type f 2>/dev/null >&2 || true
  find /app/tests -maxdepth 2 -type f 2>/dev/null >&2 || true
  exit 1
fi


mapfile -t TEST_FILES < <(find "$TARGET" -maxdepth 2 -type f -name 'test*.py' -print | sort)
if [ "${#TEST_FILES[@]}" -eq 0 ]; then
  echo "No test files matching test*.py under $TARGET" >&2
  find "$TARGET" -maxdepth 2 -type f 2>/dev/null >&2 || true
  exit 1
fi

echo "Running pytest on: ${TEST_FILES[*]}"
# Use python_files=test*.py so bare test.py files are collected.
pytest -o python_files="test*.py" -rA "${TEST_FILES[@]}"
