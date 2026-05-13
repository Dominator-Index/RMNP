#!/bin/bash
# Verify pipeline dependencies and script availability.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

ok=0; fail=0
check() {
    if [[ -e "$1" ]]; then echo "OK      $2"; ((ok++)); else echo "MISSING $2"; ((fail++)); fi
}

echo "=== Pipeline check: $PROJECT_DIR ==="

check "$PROJECT_DIR/torchrun_main.py"               "torchrun_main.py"
check "$PROJECT_DIR/configs/llama_60m.json"         "configs/llama_60m.json"
check "$PROJECT_DIR/configs/llama_135m.json"        "configs/llama_135m.json"
check "$PROJECT_DIR/configs/llama_350m.json"        "configs/llama_350m.json"
check "$PROJECT_DIR/optimizers/muon_optimizer.py"   "optimizers/muon_optimizer.py"
check "$PROJECT_DIR/optimizers/RMNP_optimizer.py"   "optimizers/RMNP_optimizer.py"

for s in train_universal.sh train_muon_60m.sh train_RMNP_60m.sh train_muon_135m.sh train_RMNP_135m.sh; do
    check "$SCRIPT_DIR/$s" "scripts/$s"
done

python3 -m py_compile "$PROJECT_DIR/torchrun_main.py" 2>/dev/null \
    && echo "OK      torchrun_main.py syntax" \
    || echo "FAIL    torchrun_main.py syntax error"

echo "=== $ok OK, $fail missing ==="
