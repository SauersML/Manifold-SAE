"""Generalize the real-LLM read/steer/gate finding across layers AND concepts.

Tests whether "a single atom is a causal handle, but the FIRING atom isn't it"
is universal or a layer-14/days fluke. Concepts: days (7), months (12) — both
circular per the literature. Sweeps several layers. For each (concept, layer):
circularity, train CircleSAE, then READ (circ-corr atom θ vs true value) and
STEER (move atom θ, patch residual, run model, does predicted next-value advance)
for the best-read atom vs the gate-winner. Emits a compact JSON table.
"""
import json, math
import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-1.5B"
DEV = "cuda"
LAYERS = [6, 10, 14, 18, 22]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
DAY_T = ["Today is {x}", "It is {x}", "The meeting is on {x}", "My favorite day is {x}",
         "See you next {x}", "We met on {x}", "The party is {x}", "School starts {x}",
         "The deadline is {x}", "The game is this {x}", "Her birthday is {x}",
         "The exam is {x}", "The concert is {x}", "The wedding is {x}", "Practice is {x}",
         "The show airs {x}", "Payday is {x}", "The flight leaves {x}", "Class meets {x}",
         "The report is due {x}"]
MON_T = ["It happened in {x}", "We met in {x}", "The event is in {x}", "I was born in {x}",
         "The deadline is in {x}", "School starts in {x}", "Vacation is in {x}",
         "The wedding is in {x}", "Taxes are due in {x}", "The festival is in {x}",
         "Her birthday is in {x}", "The launch is in {x}", "We travel in {x}",
         "The harvest is in {x}", "The exam is in {x}", "The season opens in {x}",
         "The bill is due in {x}", "The reunion is in {x}", "Snow comes in {x}",
         "The fiscal year ends in {x}"]
CONCEPTS = {
    "days": (DAYS, DAY_T, "The day after {x} is"),
    "months": (MONTHS, MON_T, "The month after {x} is"),
}


def circ_corr(a, b):
    a = a - math.atan2(np.sin(a).mean(), np.cos(a).mean())
    b = b - math.atan2(np.sin(b).mean(), np.cos(b).mean())
    n = np.sum(np.sin(a) * np.sin(b)); d = math.sqrt(np.sum(np.sin(a) ** 2) * np.sum(np.sin(b) ** 2))
    return float(abs(n / d)) if d > 0 else 0.0


class CircleSAE(nn.Module):
    def __init__(self, D, F=48, H=3):
        super().__init__()
        self.H, self.M, self.F = H, 2 * H + 1, F
        self.B = nn.Parameter(torch.randn(F, self.M, D) / math.sqrt(self.M * D))
        self.b = nn.Parameter(torch.zeros(D))
        self.gh = nn.Linear(D, F); self.ch = nn.Linear(D, 2 * F)

    def basis(self, th):
        f = [torch.ones_like(th)]
        for h in range(1, self.H + 1):
            f += [torch.cos(h * th), torch.sin(h * th)]
        return torch.stack(f, -1)

    def fwd(self, x):
        pre = self.gh(x); gate = pre * (pre > 0.05).float()
        cs = self.ch(x).reshape(x.shape[0], self.F, 2)
        th = torch.atan2(cs[..., 1], cs[..., 0])
        return gate, th

    def recon(self, gate, th):
        return torch.einsum("nf,nfd->nd", gate, torch.einsum("nfm,fmd->nfd", self.basis(th), self.B)) + self.b

    def incoh(self):
        r = self.B.reshape(self.F * self.M, -1); r = r / r.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        g2 = (r @ r.t()).pow(2); aid = torch.arange(self.F * self.M, device=r.device) // self.M
        c = aid[:, None] != aid[None, :]; return (g2 * c).sum() / c.sum().clamp_min(1)

    def loss(self, x):
        g, th = self.fwd(x); return (self.recon(g, th) - x).pow(2).mean() + 1e-3 * g.abs().mean() + 1e-3 * self.incoh()


def val_pos(tok, text, val):
    return tok(text[:text.index(val) + len(val)], return_tensors="pt").input_ids.shape[1] - 1


@torch.no_grad()
def harvest(model, tok, values, templates):
    perlayer = {L: [] for L in LAYERS}; ys = []
    for t in templates:
        for vi, v in enumerate(values):
            ids = tok(t.format(x=v), return_tensors="pt").input_ids.to(DEV)
            hs = model(ids, output_hidden_states=True).hidden_states
            for L in LAYERS:
                perlayer[L].append(hs[L][0, -1].float().cpu().numpy())
            ys.append(vi)
    return {L: np.array(perlayer[L]) for L in LAYERS}, np.array(ys)


def circularity(X, y, nv):
    cent = np.stack([X[y == v].mean(0) for v in range(nv)]); cent -= cent.mean(0)
    S = np.linalg.svd(cent, compute_uv=False)
    return float((S[:2] ** 2).sum() / (S ** 2).sum())


def diagnose(model, tok, X, y, nv, L, values, after):
    D = X.shape[1]; mu = X.mean(0)
    sae = CircleSAE(D).to(DEV); Xt = torch.tensor(X - mu, dtype=torch.float32, device=DEV)
    opt = torch.optim.Adam(sae.parameters(), lr=3e-3)
    for _ in range(500):
        opt.zero_grad(); sae.loss(Xt).backward(); opt.step()
    sae.eval()
    with torch.no_grad():
        g, th = sae.fwd(Xt)
    th = th.cpu().numpy(); g = g.abs().cpu().numpy(); ang = y * 2 * math.pi / nv
    ccs = [circ_corr(ang, th[:, j]) for j in range(sae.F)]
    bra = int(np.argmax(ccs)); gw = int(g.mean(0).argmax())

    # real-model steering
    vids = [tok(" " + v, add_special_tokens=False).input_ids[0] for v in values]
    prompt = after.format(x=values[0]); pos = val_pos(tok, prompt, values[0])
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEV); B = sae.B.detach()
    vang = np.arange(nv) * 2 * math.pi / nv

    def steer(atom):
        with torch.no_grad():
            hs0 = model(ids, output_hidden_states=True).hidden_states[L][0, pos].float()
            xt = torch.tensor((hs0.cpu().numpy() - mu), dtype=torch.float32, device=DEV).unsqueeze(0)
            gg, th0 = sae.fwd(xt)
            cur0 = torch.einsum("fm,fmd->fd", sae.basis(th0[0]), B)[atom]
        outs, grid = [], np.linspace(-math.pi, math.pi, 12, endpoint=False)
        for tv in grid:
            tvv = torch.tensor([float(tv)], dtype=torch.float32, device=DEV)
            cur1 = torch.einsum("m,md->d", sae.basis(tvv)[0], B[atom])
            delta = (gg[0, atom] * (cur1 - cur0)).half()

            def hook(m, i, o):
                oo = o[0] if isinstance(o, tuple) else o; oo[0, pos] = oo[0, pos] + delta; return o
            h = model.model.layers[L].register_forward_hook(hook)
            with torch.no_grad():
                lg = model(ids).logits[0, -1]
            h.remove()
            p = torch.softmax(lg[vids].float(), 0).cpu().numpy()
            outs.append(math.atan2((p * np.sin(vang)).sum(), (p * np.cos(vang)).sum()))
        return circ_corr(grid, np.array(outs))

    return {"best_read_atom": bra, "best_read": round(ccs[bra], 3),
            "gate_winner_atom": gw, "gate_winner_read": round(ccs[gw], 3),
            "best_read_steer": round(steer(bra), 3), "gate_winner_steer": round(steer(gw), 3)}


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(DEV).eval()
    out = {}
    for cname, (values, templates, after) in CONCEPTS.items():
        Xs, y = harvest(model, tok, values, templates)
        for L in LAYERS:
            try:
                circ = circularity(Xs[L], y, len(values))
                d = diagnose(model, tok, Xs[L], y, len(values), L, values, after)
                out[f"{cname}_L{L}"] = {"circularity": round(circ, 3), **d}
                print(f"{cname} L{L}: circ={circ:.2f} read={d['best_read']}/{d['gate_winner_read']} "
                      f"steer={d['best_read_steer']}/{d['gate_winner_steer']}", flush=True)
            except Exception as e:
                out[f"{cname}_L{L}"] = {"error": str(e)[:120]}
    json.dump(out, open("/home/user/sweep_result.json", "w"), indent=2)
    print("DONE")


if __name__ == "__main__":
    main()
