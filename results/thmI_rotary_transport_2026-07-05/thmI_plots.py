"""Figures for the Theorem I rotary-transport test. Consumes thmI_v3_results.json."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

res = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "thmI_v3_results.json"))
hops = res["hops"]
names = [h["hop"] for h in hops]
rigid = np.array([h["rigid_circ_rms"] for h in hops]) * 180 / np.pi   # degrees
iso = np.array([h["isometry_defect"] for h in hops])
conf = np.array([h["conformal_departure"] for h in hops])
imp = np.array([h["impurity_mean"] for h in hops])
conc = np.array([h["degree_concentration"] for h in hops])
nullc = np.array([h["null_conc_mean"] for h in hops])
nullsd = np.array([h["null_conc_sd"] for h in hops])
x = np.arange(len(names))
short = [n.replace("->", "-") for n in names]
C0, C1, C2, C3 = "#2b6cb0", "#c05621", "#2f855a", "#9b2c2c"

fig, ax = plt.subplots(2, 2, figsize=(13.5, 9.5))

# (a) THE MECHANISM: phase-shift residual vs O(2)-departure of induced M  (r=0.83)
a = ax[0, 0]
a.scatter(conf, rigid, c=C0, s=70, zorder=3)
for i, n in enumerate(short):
    a.annotate(n, (conf[i], rigid[i]), fontsize=7, alpha=0.75, xytext=(4, 3), textcoords="offset points")
if conf.std() > 0:
    b1 = np.polyfit(conf, rigid, 1); xs = np.linspace(conf.min(), conf.max(), 50)
    a.plot(xs, np.polyval(b1, xs), color=C3, ls="--", lw=1.3)
r_mech = res["corr_rigid_confdep"]
a.set_xlabel(r"conformal departure of $M=A'^{+}WA$:  $||M^{T}M-\lambda I||/\lambda$")
a.set_ylabel(r"phase-shift residual of $h$ vs $\pm\theta+\phi$  (deg)")
a.set_title(f"(a) MECHANISM: h-deviation is driven by M leaving O(2)\nPearson r = {r_mech:.2f}")
a.grid(alpha=0.25)

# (b) P3 test: phase-shift residual vs harmonic impurity  (null result)
b = ax[0, 1]
b.scatter(imp, rigid, c=C2, s=70, zorder=3)
for i, n in enumerate(short):
    b.annotate(n, (imp[i], rigid[i]), fontsize=7, alpha=0.75, xytext=(4, 3), textcoords="offset points")
r_p3 = res["corr_rigid_impurity"]
b.set_xlabel("harmonic impurity of the atom  (energy k>=2 / fundamental)")
b.set_ylabel("phase-shift residual of h  (deg)")
b.set_title(f"(b) P3 test: deviation vs non-ellipticity -- NOT differentiable\nr = {r_p3:.2f} (deviations ~0; impurity ~const)")
b.grid(alpha=0.25)

# (c) the ladder: everything is a phase shift at every hop
c = ax[1, 0]
c.plot(x, rigid, "o-", color=C0, label=r"rigid $\pm\theta+\phi$ residual (deg)")
c.plot(x, conf * 100, "s--", color=C1, label=r"conformal departure $\times 100$")
c.plot(x, iso * 100, "^:", color=C3, label=r"gamfit isometry defect $\times 100$")
c.set_xticks(x); c.set_xticklabels(short, rotation=60, fontsize=7)
c.set_ylabel("deviation from a pure phase shift")
c.set_title("(c) L11->L23 ladder: transport is a phase shift at every hop")
c.legend(fontsize=8); c.grid(alpha=0.25)

# (d) null control
d = ax[1, 1]
d.plot(x, conc, "o-", color=C0, label="real transport")
d.errorbar(x, nullc, yerr=nullsd, fmt="s--", color=C3, capsize=2, label="shuffled-day null")
d.set_xticks(x); d.set_xticklabels(short, rotation=60, fontsize=7)
d.set_ylim(0, 1.05)
d.set_ylabel("degree concentration")
d.set_title("(d) null control: rigidity is a property of the genuine circle")
d.legend(fontsize=8); d.grid(alpha=0.25)

fig.suptitle("Theorem I: linear cross-layer transport of the weekday circle is forced to be a phase shift  h = +/-theta + phi",
             fontsize=13, y=1.005)
fig.tight_layout()
fig.savefig("thmI_figure.png", dpi=140, bbox_inches="tight")
print("wrote thmI_figure.png")
