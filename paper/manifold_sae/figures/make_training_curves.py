"""Render val-R^2 vs epoch for TopK / L1 / Manifold from the saved JSON.

Reads runs/sae_comparison/comparison.json — no model retraining.
"""
import json, pathlib
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[3]
data = json.loads((ROOT / "runs" / "sae_comparison" / "comparison.json").read_text())

fig, ax = plt.subplots(figsize=(4.6, 3.2))
styles = {"TopK": ("--", "tab:gray"), "L1": ("-.", "tab:blue"), "Manifold": ("-", "tab:red")}
for name in ["TopK", "L1", "Manifold"]:
    h = data[name]["history"]
    xs = [r["epoch"] + 1 for r in h]
    ys = [r["val_r2"] for r in h]
    ls, c = styles[name]
    ax.plot(xs, ys, ls, color=c, label=f"{name} (final {ys[-1]:.3f})", lw=1.8)

ax.set_xlabel("epoch")
ax.set_ylabel("validation $R^2$")
ax.set_title("Cogito-L40, $F=512$, by-color split")
ax.grid(True, alpha=0.3)
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
out = pathlib.Path(__file__).parent / "training_curves.pdf"
fig.savefig(out, bbox_inches="tight")
fig.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=160)
print(f"wrote {out}")
