import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sweep = json.load(open("sweep_results.json"))
margin = json.load(open("margin_results.json"))

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

# Panel 1: double-parse fraction vs p (identifiability transition)
pt = sweep["p_transition"]
ps = [r["p"] for r in pt]
mf = [r["mean_frac"] for r in pt]
ax[0].plot(ps, mf, "o-", color="#c0392b", lw=2, ms=9)
ax[0].axvline(5, ls="--", color="gray", alpha=0.6)
ax[0].set_xlabel("ambient dimension p")
ax[0].set_ylabel("fraction of z with a 2nd parse")
ax[0].set_title("Theorem A: uniqueness transition\n(hypothesis needs p-1 >= sum(d+1)=4, i.e. p>=5)")
ax[0].set_ylim(-0.05, 1.08)
for r in pt:
    ax[0].annotate(f"slack={r['slack']}", (r["p"], r["mean_frac"]),
                   textcoords="offset points", xytext=(4, 8), fontsize=8)

# Panel 2: co-collapse margin (sigma_min) vs separation
cc = margin["cocollapse"]
alpha = np.array([r["collapse_alpha"] for r in cc])
sm = np.array([r["median_sigma_min"] for r in cc])
sep = 1 - alpha
mask = sep > 0
ax[1].loglog(sep[mask], sm[mask], "o-", color="#2471a3", lw=2, ms=8, label="median $\\sigma_{min}(J)$")
# reference line proportional to separation
ax[1].loglog(sep[mask], 0.9 * sep[mask], "k--", alpha=0.5, label="$\\propto$ separation (slope 1)")
ax[1].set_xlabel("atom separation  (1 - collapse $\\alpha$)")
ax[1].set_ylabel("parse Jacobian $\\sigma_{min}$")
ax[1].set_title("Theorem B: identifiability margin\nscales linearly with atom separation")
ax[1].legend(fontsize=8)
ax[1].grid(True, which="both", alpha=0.2)

# Panel 3: conditioning vs curvature
cvc = margin["curvature_conditioning"]
cv = [r["curv"] for r in cvc]
smc = [r["median_sigma_min"] for r in cvc]
p05 = [r["p05_sigma_min"] for r in cvc]
ax[2].semilogx(cv, smc, "o-", color="#1e8449", lw=2, ms=8, label="median")
ax[2].semilogx(cv, p05, "s--", color="#1e8449", alpha=0.5, ms=6, label="5th pctile")
ax[2].set_xlabel("atom curvature (2nd-axis scale)")
ax[2].set_ylabel("parse Jacobian $\\sigma_{min}$")
ax[2].set_title("Curvature conditioning\n(flatter atoms -> worse but bounded away from 0)")
ax[2].legend(fontsize=8)
ax[2].grid(True, which="both", alpha=0.2)

plt.tight_layout()
plt.savefig("thmA_thmB_summary.png", dpi=130)
print("saved thmA_thmB_summary.png")
