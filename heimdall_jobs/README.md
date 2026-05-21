# heimdall_jobs

Submit Manifold-SAE experiments to a Heimdall-style job scheduler.

The submitter ships **no cluster-specific defaults**. Every site value (API
URL, node name, working directory, GPU count) is read from one of:

1. CLI flags (`--api`, `--working-dir`, `--node`, `--gpus`, ...).
2. Environment variables (`HEIMDALL_API`, `MSAE_WORKING_DIR`, `MSAE_NODE`,
   `MSAE_GPUS`, `MSAE_DEFAULT_MINUTES`, `MSAE_GIT_URL`).
3. Local JSON config at `~/.config/manifold-sae/heimdall.json` (gitignored).

`HEIMDALL_API` and `MSAE_WORKING_DIR` are required; the others have safe
defaults.

## Setup

Pick one of the configuration sources:

### Option A: environment variables (preferred for one-off use)

```sh
export HEIMDALL_API='http://...'                       # your cluster
export MSAE_WORKING_DIR='/path/on/node'                 # job's working_dir
export MSAE_NODE='node-label'                           # site-specific
export MSAE_GPUS=8                                      # default GPU count
```

### Option B: local config file

```sh
mkdir -p ~/.config/manifold-sae
cat > ~/.config/manifold-sae/heimdall.json <<'JSON'
{
  "api":          "http://...",
  "working_dir":  "/path/on/node",
  "node":         "node-label",
  "gpus":         8,
  "estimated_minutes": 120
}
JSON
```

Never commit this file. The default path is under `~/.config/`, which is
out of repo by construction.

## Usage

```sh
# Default: experiment=llm_sweep, branch=main, dry-run to inspect first.
python heimdall_jobs/submit.py --dry-run

# Submit for real once the dry-run JSON looks right.
python heimdall_jobs/submit.py

# Pick a different experiment or branch.
python heimdall_jobs/submit.py --experiment llm_real --git-ref some-branch

# Override one value from the CLI:
python heimdall_jobs/submit.py --gpus 0 --estimated-minutes 60
```

The submitter prints the JSON spec with `--dry-run` and the (masked) API
target so you can verify the command before hitting the network.

## What the job does

The shell command in the spec:

1. `mkdir -p $WORKDIR/{Manifold-SAE,runs/$RUN_NAME}`
2. clone-or-pull `Manifold-SAE` at `--git-ref`
3. install/refresh `uv` if missing, `uv sync --extra llm`
4. export `MANIFOLD_SAE_OUTPUT_DIR=$WORKDIR/runs/$RUN_NAME` so the
   experiment writes outputs under the working_dir tree, not into the
   freshly-cloned repo
5. `uv run python -u -m experiments.<experiment>`

`MANIFOLD_SAE_OUTPUT_DIR` is read by the experiment drivers as the
override for `Config.output_dir`. Plots, results.json, and checkpoints
all land there and persist across re-runs.

## Concerns

- **API URL never goes into a commit.** Use env vars or the local JSON
  config. The submitter doesn't print the URL in `--dry-run` output (only
  to stderr, with explicit marking).
- **Working-dir paths never go into a commit either.** Treat
  `/home/<user>/...` the same way.
- **VPN-blind operation.** If your cluster requires a VPN to reach, run
  `vpn` / equivalent first. The submitter does not touch network state.
