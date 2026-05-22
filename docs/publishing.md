# Publishing flow — gamfit 0.1.105 → cluster

## Current state

* gam main is at 0.1.105 with the multi-dim Duchon + additive REML API.
* PyPI is still at 0.1.98 (old API).
* Cluster's uv.lock pins `gamfit==0.1.98`.
* manifold_sae main uses the new API and will fail at runtime against gamfit 0.1.98.

## Steps to fully publish

### 1. Build & test wheel locally (already in progress)

```
cd /Users/user/gam
.venv/bin/maturin develop --release
.venv/bin/python -c "from gamfit.torch import duchon_basis, gaussian_reml_fit_additive; print('OK')"
```

### 2. Publish to PyPI

Requires PyPI credentials (`~/.pypirc` or `PYPI_TOKEN` env var).

```
cd /Users/user/gam
# Build wheels for all targets (macos arm64 + linux x86_64 + linux aarch64)
.venv/bin/maturin build --release --target aarch64-apple-darwin
.venv/bin/maturin build --release --target x86_64-unknown-linux-gnu
.venv/bin/maturin build --release --target aarch64-unknown-linux-gnu

# Publish
.venv/bin/maturin publish --release
```

(In practice CI/cibuildwheel is normally what does this — check existing
GitHub Actions in `.github/workflows/`.)

### 3. Update Manifold-SAE to require gamfit 0.1.105

```
cd /Users/user/Manifold-SAE
# bump in pyproject.toml
sed -i.bak 's/gamfit>=0.1.67/gamfit>=0.1.105/' pyproject.toml
rm pyproject.toml.bak
# regenerate uv.lock
uv lock --upgrade-package gamfit
git add pyproject.toml uv.lock
git commit -m "Require gamfit>=0.1.105 (multi-dim Duchon + additive REML)"
git push
```

### 4. Cluster updates next job submission

The heimdall_jobs/submit.py script does `uv sync` at job start, which will
pull the new gamfit wheel.

## Backout

If publication needs to be rolled back:

```
cd /Users/user/gam
# Revert the multi-d Duchon API commits
git revert fc649755 c4a8cb47 3bacda5d 8a0e88fb e04802b4 fbee8ea3
git push
# Re-publish 0.1.106 with the revert
.venv/bin/maturin publish --release
```

## Verification (post-publish)

After cluster sync:
```
cd /Users/user/Manifold-SAE
python3 heimdall_jobs/submit.py --node node2 --experiment llm_sweep \
    --run-name llm_sweep_validate_new_gamfit \
    --env MSAE_MODEL=Qwen/Qwen2.5-0.5B --env MSAE_LAYER=18 \
    --env MSAE_F_VALUES=16,32 --env MSAE_N_TOKENS=20000 \
    --estimated-minutes 30
```

Watch for clean training (no AttributeError / TypeError) and compare alive-atom
counts + EV vs. our prior post-fix Qwen-0.5B L18 numbers.
