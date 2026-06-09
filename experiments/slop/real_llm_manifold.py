"""Real-LLM test of the read/steer/gate diagnostic (replaces the toy behavior model).

Concept: days of the week (the papers show this is circular in LLMs). We harvest
mid-layer residual activations for prompts ending in a day, verify the circle via
PCA, train a pure-torch amortized manifold-SAE (Fourier circle atoms + incoherence
+ isometry + JumpReLU gate), then measure per the toy experiments — but with the
REAL model as the steering oracle:
  READ  : circ-corr(atom coordinate θ, true day angle), best atom vs gate-winner.
  STEER : move an atom's θ around the circle, PATCH the residual at the day token,
          run the model, read the output distribution over the 7 day tokens; does
          the predicted day advance with the swept θ? (real behavior, not a toy).
The question: on a real model, is the FIRING atom the read+steer handle, or is the
concept diluted across atoms (as the synthetic suggested)?
"""
import json, math, sys
import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-1.5B"
LAYER = 14
DEV = "cuda"
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
TEMPLATES = [
    "Today is {d}", "It is {d}", "The meeting is on {d}", "My favorite day is {d}",
    "See you next {d}", "It happened last {d}", "We met on {d}", "The party is {d}",
    "I was born on a {d}", "School starts {d}", "The deadline is {d}", "Let's talk {d}",
    "The game is this {d}", "Payday is {d}", "I'll call you {d}", "The flight leaves {d}",
    "Church is on {d}", "The market opens {d}", "Her birthday is {d}", "The exam is {d}",
    "We arrive {d}", "The concert is {d}", "Trash pickup is {d}", "The wedding is {d}",
    "Class meets every {d}", "The store closes early on {d}", "He visits each {d}",
    "The report is due {d}", "Practice is held {d}", "The show airs on {d}",
]


@torch.no_grad()
def harvest(model, tok):
    X, y = [], []
    for ti, t in enumerate(TEMPLATES):
        for di, d in enumerate(DAYS):
            ids = tok(t.format(d=d), return_tensors="pt").input_ids.to(DEV)
            hs = model(ids, output_hidden_states=True).hidden_states[LAYER]  # (1,T,D)
            X.append(hs[0, -1].float().cpu().numpy())  # last token = day's final subtoken
            y.append(di)
    return np.array(X), np.array(y)


def circularity(X, y):
    cent = np.stack([X[y == d].mean(0) for d in range(7)])  # (7,D)
    cent = cent - cent.mean(0)
    U, S, _ = np.linalg.svd(cent, full_matrices=False)
    var2 = float((S[:2] ** 2).sum() / (S ** 2).sum())
    # are consecutive days adjacent on the top-2 PCA ring?
    proj = cent @ np.linalg.svd(cent, full_matrices=False)[2][:2].T  # (7,2)
    ang = np.arctan2(proj[:, 1], proj[:, 0])
    order = np.argsort(ang)
    return var2, order.tolist(), proj.tolist()


class CircleSAE(nn.Module):
    def __init__(self, D, F, H=3, thr=0.05, sw=1e-3, iw=1e-3, isow=1e-2):
        super().__init__()
        self.H, self.M = H, 2 * H + 1
        self.B = nn.Parameter(torch.randn(F, self.M, D) / math.sqrt(self.M * D))
        self.b = nn.Parameter(torch.zeros(D))
        self.gate_head = nn.Linear(D, F)
        self.coord_head = nn.Linear(D, 2 * F)
        self.F, self.thr, self.sw, self.iw, self.isow = F, thr, sw, iw, isow

    def basis(self, th):
        f = [torch.ones_like(th)]
        for h in range(1, self.H + 1):
            f += [torch.cos(h * th), torch.sin(h * th)]
        return torch.stack(f, -1)

    def forward(self, x):
        pre = self.gate_head(x)
        gate = pre * (pre > self.thr).float()
        cs = self.coord_head(x).reshape(x.shape[0], self.F, 2)
        th = torch.atan2(cs[..., 1], cs[..., 0])
        curve = torch.einsum("nfm,fmd->nfd", self.basis(th), self.B)
        xh = torch.einsum("nf,nfd->nd", gate, curve) + self.b
        return xh, gate, th, pre

    def incoh(self):
        r = self.B.reshape(self.F * self.M, -1)
        r = r / r.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        g2 = (r @ r.t()).pow(2)
        aid = torch.arange(self.F * self.M, device=r.device) // self.M
        cross = aid[:, None] != aid[None, :]
        return (g2 * cross).sum() / cross.sum().clamp_min(1)

    def iso(self, G=24):
        th = torch.linspace(-math.pi, math.pi, G + 1, device=self.B.device)[:-1]
        dphi = [torch.zeros_like(th)]
        for h in range(1, self.H + 1):
            dphi += [-h * torch.sin(h * th), h * torch.cos(h * th)]
        dphi = torch.stack(dphi, -1)
        sp = torch.einsum("gm,fmd->fgd", dphi, self.B).norm(dim=-1)
        return (sp / sp.mean(-1, keepdim=True).clamp_min(1e-8)).var(-1).mean()

    def loss(self, x):
        xh, gate, th, pre = self(x)
        rec = (xh - x).pow(2).mean()
        return rec + self.sw * gate.abs().mean() + self.iw * self.incoh() + self.isow * self.iso()


def circ_corr(a, b):
    a = a - math.atan2(np.sin(a).mean(), np.cos(a).mean())
    b = b - math.atan2(np.sin(b).mean(), np.cos(b).mean())
    n = np.sum(np.sin(a) * np.sin(b)); de = math.sqrt(np.sum(np.sin(a) ** 2) * np.sum(np.sin(b) ** 2))
    return float(abs(n / de)) if de > 0 else 0.0


def find_day_pos(tok, text, day):
    pre = tok(text[:text.index(day)], return_tensors="pt").input_ids.shape[1]
    full = tok(text[:text.index(day) + len(day)], return_tensors="pt").input_ids.shape[1]
    return full - 1  # last subtoken of the day word


def main():
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(DEV).eval()
    print("model loaded", flush=True)

    X, y = harvest(model, tok)
    mu = X.mean(0); Xc = (X - mu)
    var2, order, proj = circularity(X, y)
    print(f"[circularity] top-2 PCA var of day-centroids = {var2:.3f}  ring order = {order}", flush=True)

    D = X.shape[1]
    sae = CircleSAE(D, F=48).to(DEV)
    Xt = torch.tensor(Xc, dtype=torch.float32, device=DEV)
    opt = torch.optim.Adam(sae.parameters(), lr=3e-3)
    for ep in range(800):
        opt.zero_grad(); sae.loss(Xt).backward(); opt.step()
    sae.eval()
    with torch.no_grad():
        _, gate, th, _ = sae(Xt)
    th = th.cpu().numpy(); gate = gate.abs().cpu().numpy()
    day_ang = y * 2 * math.pi / 7

    # READ: best atom vs gate-winner
    ccs = [circ_corr(day_ang, th[:, j]) for j in range(sae.F)]
    best_read_atom = int(np.argmax(ccs))
    gate_winner = int(gate.mean(0).argmax())
    print(f"[read] best-aligned atom #{best_read_atom} corr={ccs[best_read_atom]:.3f} | "
          f"gate-winner atom #{gate_winner} corr={ccs[gate_winner]:.3f}", flush=True)

    # STEER (real behavior): patch the day token, sweep an atom's θ, read predicted-day shift.
    day_ids = [tok(" " + d, add_special_tokens=False).input_ids[0] for d in DAYS]
    steer_prompt = "The day after Monday is"
    pos = find_day_pos(tok, steer_prompt, "Monday")
    ids = tok(steer_prompt, return_tensors="pt").input_ids.to(DEV)
    B = sae.B.detach()

    def steer_curve(atom, theta):  # ambient delta to set atom to theta (vs its read at the token)
        with torch.no_grad():
            hs = model(ids, output_hidden_states=True).hidden_states[LAYER][0, pos].float()
            x1 = (hs.cpu().numpy() - mu)
            xt = torch.tensor(x1, device=DEV).unsqueeze(0)
            _, g, th0, _ = sae(xt)
            cur0 = torch.einsum("fm,fmd->fd", sae.basis(th0[0]), B)[atom]
            thv = torch.tensor([theta], device=DEV)
            cur1 = torch.einsum("m,md->d", sae.basis(thv)[0], B[atom])
            return (g[0, atom] * (cur1 - cur0)).half()

    def run_steered(delta):
        h = {}
        def hook(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            o[0, pos] = o[0, pos] + delta
            return out
        hd = model.model.layers[LAYER].register_forward_hook(hook)
        with torch.no_grad():
            lg = model(ids).logits[0, -1]
        hd.remove()
        p = torch.softmax(lg[day_ids].float(), 0).cpu().numpy()
        return float(np.arctan2((p * np.sin(np.arange(7) * 2 * math.pi / 7)).sum(),
                                (p * np.cos(np.arange(7) * 2 * math.pi / 7)).sum()))

    def steer_score(atom):
        grid = np.linspace(-math.pi, math.pi, 14, endpoint=False)
        outs = [run_steered(steer_curve(atom, float(t))) for t in grid]
        return circ_corr(grid, np.array(outs))

    sr_best = steer_score(best_read_atom)
    sr_gate = steer_score(gate_winner)
    print(f"[steer] best-read atom steer={sr_best:.3f} | gate-winner steer={sr_gate:.3f}", flush=True)

    out = {"model": MODEL, "layer": LAYER, "n_act": int(X.shape[0]), "dim": int(D),
           "circularity_top2_var": var2, "ring_order": order,
           "best_read_atom": best_read_atom, "best_read_corr": ccs[best_read_atom],
           "gate_winner_atom": gate_winner, "gate_winner_read_corr": ccs[gate_winner],
           "best_read_steer": sr_best, "gate_winner_steer": sr_gate,
           "day_proj": proj}
    json.dump(out, open("real_llm_result.json", "w"), indent=2)
    print("\nSUMMARY:", json.dumps({k: v for k, v in out.items() if k != "day_proj"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
