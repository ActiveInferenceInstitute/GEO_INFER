#!/usr/bin/env python
"""Guarded import probe: times `import <module>` in a child interpreter.

Child-side faulthandler dumps a full traceback (which threads, where) if the
import exceeds DUMP_AFTER seconds and exits. Parent enforces a hard
subprocess timeout. Requires the literal stdout line 'IMPORTED <path>' to
count as success.
"""

import subprocess
import sys
import time

venv_python = sys.argv[1]
module = sys.argv[2]
reps = int(sys.argv[3])
label = sys.argv[4] if len(sys.argv) > 4 else ""

CHILD = (
    "import faulthandler; faulthandler.dump_traceback_later(120, exit=True); "
    f"import {module}, sys; "
    "print('IMPORTED', getattr(%s, '__file__', '?'), sys.version.split()[0])" % module
)

times = []
for i in range(1, reps + 1):
    t0 = time.perf_counter()
    try:
        p = subprocess.run(
            [venv_python, "-c", CHILD],
            capture_output=True,
            text=True,
            timeout=300,
        )
        elapsed = time.perf_counter() - t0
        ok = p.returncode == 0 and "IMPORTED" in p.stdout
        print(f"[{label}] rep{i}: {elapsed:.2f}s rc={p.returncode} ok={ok}")
        if not ok:
            print(f"  stdout: {p.stdout[-2000:]!r}")
            print(f"  stderr tail: {p.stderr[-3000:]!r}")
        else:
            times.append(elapsed)
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - t0
        print(f"[{label}] rep{i}: TIMEOUT after {elapsed:.2f}s (parent guard 300s)")
    except Exception as e:
        print(f"[{label}] rep{i}: ERROR {e!r}")

if times:
    times.sort()
    print(
        f"[{label}] min={times[0]:.2f}s median={times[len(times) // 2]:.2f}s max={times[-1]:.2f}s n={len(times)}"
    )
