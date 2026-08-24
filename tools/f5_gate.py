"""Gate F5/F6 runner: execute the tournament-format notebook twice in the current interpreter's
environment, compare the printed metrics between runs, and verify the boilerplate hashes.

Run with the pinned environment's interpreter from the repository root:
    venv-py310\\Scripts\\python.exe tools\\f5_gate.py
"""
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
NB = os.path.join("tournament_2025", "btc_accumulation_model.ipynb")
TEMPLATE = os.path.join("tournament_2025", "model_development_template.ipynb")

KEY_LINES = re.compile(
    r"(Final Model Score.*|Exponential-Decay Average SPD Percentile.*|Summary: .*win rate\)|"
    r".*Strategy is ready for submission.*|.*meets performance requirement.*|Data loaded: .*)"
)


def execute_once(tag):
    print(f"--- run {tag}: executing {NB} (this takes a few minutes) ---", flush=True)
    r = subprocess.run(
        [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
         "--inplace", f"--ExecutePreprocessor.timeout=3600", NB],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print("EXECUTION FAILED"); print(r.stderr[-1500:]); sys.exit(1)
    nb = json.load(open(NB, encoding="utf-8"))
    lines = []
    for cell in nb["cells"]:
        for out in cell.get("outputs", []):
            text = "".join(out.get("text", [])) if out.get("output_type") == "stream" else ""
            for line in text.splitlines():
                if KEY_LINES.fullmatch(line.strip()):
                    lines.append(line.strip())
    print("\n".join("  " + l for l in lines), flush=True)
    return lines


def main():
    print("interpreter:", sys.executable)
    print("python:", sys.version.split()[0])
    import numpy, pandas
    print("numpy:", numpy.__version__, "| pandas:", pandas.__version__)
    ok = sys.version_info[:2] == (3, 10) and numpy.__version__ == "1.26.4" and pandas.__version__ == "2.3.1"
    print("pinned environment:", "YES" if ok else "NO (wrong interpreter? run with venv-py310)")

    run1 = execute_once(1)
    run2 = execute_once(2)
    deterministic = run1 == run2 and len(run1) > 0
    ready = any("ready for submission" in l for l in run2)

    sys.path.insert(0, os.path.join(ROOT, "tournament_2025"))
    import io, contextlib
    import hasher
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = hasher.check_cells_identical(TEMPLATE, NB, [1, 3], 3)

    print("\n================ GATE REPORT ================")
    print("F5 pinned-environment execution :", "PASS" if ready else "FAIL")
    print("F6 determinism (run1 == run2)   :", "PASS" if deterministic else "FAIL")
    print("F4 hasher (cells 1 and 3, count):", "PASS" if res["all_identical"] else "FAIL")
    print("=============================================")
    if not (ready and deterministic and res["all_identical"]):
        sys.exit(1)


if __name__ == "__main__":
    main()
