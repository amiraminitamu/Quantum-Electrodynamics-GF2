#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
output_dir=${1:-$(mktemp -d)}
mkdir -p "$output_dir"
python -m gf2.cli "$repo_root/examples/open_shell_ugf2.inp" --json "$output_dir/ugf2.json"
python -m gf2.cli "$repo_root/examples/open_shell_ucs_gf2.inp" --json "$output_dir/ucs_gf2.json"
python - "$output_dir" <<'PY2'
import json
import sys
from pathlib import Path
for name in ('ugf2', 'ucs_gf2'):
    result = json.loads((Path(sys.argv[1]) / f'{name}.json').read_text())
    summary = result['summary']
    assert summary['converged'] and summary['spin_mode'] == 'unrestricted'
    assert (summary['n_alpha'], summary['n_beta']) == (2, 1)
    print(f"{name}: converged, E={summary['total_gf2']:.10f} Eh, Nα={summary['trace_p_alpha_s']:.6f}, Nβ={summary['trace_p_beta_s']:.6f}")
PY2
