#!/usr/bin/env bash
#
# Formats this project's own C++ and Slang sources with clang-format using the
# style in .clang-format. Submodules and vendored code under external/ are never
# touched. Run with --check to verify formatting without modifying files.
#
#   ./format.sh           # format in place
#   ./format.sh --check   # exit non-zero if anything is unformatted
#
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v clang-format >/dev/null 2>&1; then
    echo "error: clang-format not found (try: sudo apt install clang-format)" >&2
    exit 1
fi

# Only our own source trees — external/ (submodules + vendored stb) is
# excluded, as is the generated SPIR-V header under nga/src.
roots=(nga platform samples tests)

mapfile -d '' files < <(find "${roots[@]}" -type f \
    \( -name '*.cpp' -o -name '*.h' -o -name '*.hpp' -o -name '*.slang' \) \
    ! -name 'PatchDescriptorsSpv.h' -print0)

if [ "${#files[@]}" -eq 0 ]; then
    echo "no source files found"
    exit 0
fi

if [ "${1:-}" = "--check" ]; then
    echo "Checking ${#files[@]} files with $(clang-format --version)..."
    clang-format --dry-run --Werror "${files[@]}"
    echo "All files are formatted."
else
    echo "Formatting ${#files[@]} files with $(clang-format --version)..."
    clang-format -i "${files[@]}"
    echo "Done."
fi
