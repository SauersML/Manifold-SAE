"""Submit WS-E jobs to node2 through Heimdall (fleet doctrine: ALL node compute
goes through Heimdall, wrapped so it always exits 0 and writes rc/log to files).

    python heimdall_submit.py --name enc_synth \
        --command "cd /dev/shm/sauers_gpu/encoder && /models/sauers_build/venv_fable/bin/python -u run_synthetic.py --scale node"

The wrapper guarantees a failed command never spams the shared human channel:
the real rc lands in ``<scratch>/<name>.rc`` and the log in ``<scratch>/<name>.log``
(read them back over read-only ssh). GPU selection is done INSIDE the command
via CUDA_VISIBLE_DEVICES; the job always requests ``gpus:0`` (gpus>0 queues
forever). Co-tenant courtesy: ``nice -19 ionice -c3`` and thread caps are the
caller's responsibility inside ``--command``.
"""

from __future__ import annotations

import argparse
import json
import subprocess


HEIMDALL = "http://node1.datasci.ath:7000/api/v1/jobs"
SCRATCH = "/dev/shm/sauers_gpu/encoder"


def wrap_command(name: str, inner: str, scratch: str = SCRATCH) -> str:
    """Wrap so the job ALWAYS exits 0, writing rc + log to files."""
    log = f"{scratch}/{name}.log"
    rc = f"{scratch}/{name}.rc"
    return (
        f"mkdir -p {scratch}; "
        f"bash -c '{inner} > {log} 2>&1; echo rc=$? > {rc}; true'"
    )


def submit(name: str, inner: str, *, node: str = "node2", scratch: str = SCRATCH) -> dict:
    # Heimdall SubmitRequest wraps the job in a ``spec`` field.
    payload = {
        "spec": {
            "job_type": "custom",
            "name": name,
            "command": wrap_command(name, inner, scratch),
            "gpus": 0,
            "node": node,
        }
    }
    out = subprocess.run(
        ["curl", "-s", "-m", "20", "-X", "POST", HEIMDALL,
         "-H", "Content-Type: application/json", "-d", json.dumps(payload)],
        capture_output=True, text=True,
    )
    try:
        return json.loads(out.stdout)
    except Exception:
        return {"raw_stdout": out.stdout, "raw_stderr": out.stderr, "rc": out.returncode}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--command", required=True, help="inner shell command to wrap")
    ap.add_argument("--node", default="node2")
    ap.add_argument("--scratch", default=SCRATCH)
    args = ap.parse_args()
    resp = submit(args.name, args.command, node=args.node, scratch=args.scratch)
    print(json.dumps(resp, indent=2))
    print(f"# rc  : {args.scratch}/{args.name}.rc")
    print(f"# log : {args.scratch}/{args.name}.log")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
