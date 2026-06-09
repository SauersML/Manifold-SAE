"""Payoff test: manifold-atom steering vs linear steering on the real model.

The steering papers' core claim: moving ALONG a curved concept manifold steers
coherently, while LINEAR steering cuts a chord off-manifold and produces
incoherent output. Here we test it with an UNSUPERVISED manifold-SAE atom as the
steering handle. Days @ layer 14, Qwen2.5-1.5B. From a Monday-eliciting prompt we
steer the day token toward each other day by three methods and measure:
  TRANSFER  : circ-corr(target day angle, model's predicted-day circular mean)
  COHERENCE : mean max-prob mass on the 7 day tokens (does the output stay a
              confident DAY, or degrade to garbage?)
Methods: (a) manifold = move best-read SAE atom's coordinate to the target angle;
         (b) linear = add (centroid_target - centroid_source) at the residual;
         (c) gate-winner atom (the firing atom).
"""
import json, math
import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-1.5B"; DEV = "cuda"; L = 14
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
T = ["Today is {x}", "It is {x}", "The meeting is on {x}", "My favorite day is {x}",
     "See you next {x}", "We met on {x}", "The party is {x}", "School starts {x}",
     "The deadline is {x}", "The game is this {x}", "Her birthday is {x}", "The exam is {x}",
     "The concert is {x}", "The wedding is {x}", "Practice is {x}", "The show airs {x}",
     "Payday is {x}", "The flight leaves {x}", "Class meets {x}", "The report is due {x}"]


def cc(a, b):
    a = a - math.atan2(np.sin(a).mean(), np.cos(a).mean()); b = b - math.atan2(np.sin(b).mean(), np.cos(b).mean())
    n = np.sum(np.sin(a) * np.sin(b)); d = math.sqrt(np.sum(np.sin(a) ** 2) * np.sum(np.sin(b) ** 2))
    return float(abs(n / d)) if d > 0 else 0.0


def basis(th, H=3):
    f = [torch.ones_like(th)]
    for h in range(1, H + 1):
        f += [torch.cos(h * th), torch.sin(h * th)]
    return torch.stack(f, -1)


class CircleSAE(nn.Module):
    def __init__(s, D, F=48, H=3):
        super().__init__(); s.F, s.H = F, H
        s.B = nn.Parameter(torch.randn(F, 2 * H + 1, D) / math.sqrt((2 * H + 1) * D)); s.b = nn.Parameter(torch.zeros(D))
        s.gh = nn.Linear(D, F); s.ch = nn.Linear(D, 2 * F)

    def fwd(s, x):
        pre = s.gh(x); g = pre * (pre > 0.05).float()
        cs = s.ch(x).reshape(x.shape[0], s.F, 2); return g, torch.atan2(cs[..., 1], cs[..., 0])

    def loss(s, x):
        g, th = s.fwd(x); xh = torch.einsum("nf,nfd->nd", g, torch.einsum("nfm,fmd->nfd", basis(th, s.H), s.B)) + s.b
        return (xh - x).pow(2).mean() + 1e-3 * g.abs().mean()


@torch.no_grad()
def harvest(model, tok):
    X, y = [], []
    for t in T:
        for di, d in enumerate(DAYS):
            ids = tok(t.format(x=d), return_tensors="pt").input_ids.to(DEV)
            X.append(model(ids, output_hidden_states=True).hidden_states[L][0, -1].float().cpu().numpy()); y.append(di)
    return np.array(X), np.array(y)


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(DEV).eval()
    X, y = harvest(model, tok); mu = X.mean(0)
    cents = np.stack([X[y == d].mean(0) for d in range(7)])          # day centroids
    Xt = torch.tensor(X - mu, dtype=torch.float32, device=DEV)
    torch.manual_seed(0); sae = CircleSAE(X.shape[1]).to(DEV)
    opt = torch.optim.Adam(sae.parameters(), lr=3e-3)
    for _ in range(600):
        opt.zero_grad(); sae.loss(Xt).backward(); opt.step()
    sae.eval()
    with torch.no_grad():
        g, th = sae.fwd(Xt)
    th = th.cpu().numpy(); g = g.abs().cpu().numpy(); ang = y * 2 * math.pi / 7
    ccs = [cc(ang, th[:, j]) for j in range(sae.F)]
    bra, gw = int(np.argmax(ccs)), int(g.mean(0).argmax())
    B = sae.B.detach()

    vids = [tok(" " + d, add_special_tokens=False).input_ids[0] for d in DAYS]
    prompt = "The day after Monday is"; pos = tok(prompt[:prompt.index("Monday") + 6], return_tensors="pt").input_ids.shape[1] - 1
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEV); vang = np.arange(7) * 2 * math.pi / 7

    def out_dist(delta):
        def hook(m, i, o):
            oo = o[0] if isinstance(o, tuple) else o; oo[0, pos] = oo[0, pos] + delta; return o
        h = model.model.layers[L].register_forward_hook(hook)
        with torch.no_grad():
            p = torch.softmax(model(ids).logits[0, -1][vids].float(), 0).cpu().numpy()
        h.remove(); return p

    with torch.no_grad():
        hs0 = model(ids, output_hidden_states=True).hidden_states[L][0, pos].float()
        gg, th0 = sae.fwd(torch.tensor((hs0.cpu().numpy() - mu), dtype=torch.float32, device=DEV).unsqueeze(0))

    def atom_delta(atom, target_ang):
        cur0 = torch.einsum("fm,fmd->fd", basis(th0[0], sae.H), B)[atom]
        cur1 = torch.einsum("m,md->d", basis(torch.tensor([float(target_ang)], dtype=torch.float32, device=DEV), sae.H)[0], B[atom])
        return (gg[0, atom] * (cur1 - cur0)).half()

    def lin_delta(target_day):
        v = (cents[target_day] - cents[0])  # Monday(0) -> target, linear chord
        return torch.tensor(v, dtype=torch.float16, device=DEV)

    res = {}
    for name, fn in [("manifold_bestread", lambda d: atom_delta(bra, d * 2 * math.pi / 7)),
                     ("manifold_gatewinner", lambda d: atom_delta(gw, d * 2 * math.pi / 7)),
                     ("linear_centroid", lambda d: lin_delta(d))]:
        peaks, cohs = [], []
        for d in range(7):
            p = out_dist(fn(d))
            peaks.append(math.atan2((p * np.sin(vang)).sum(), (p * np.cos(vang)).sum()))
            cohs.append(float(p.max()))   # how much mass on the top day token
        res[name] = {"transfer": round(cc(vang, np.array(peaks)), 3),
                     "coherence": round(float(np.mean(cohs)), 3)}
        print(name, res[name], flush=True)
    res["best_read_atom"], res["gate_winner_atom"] = bra, gw
    json.dump(res, open("/home/user/steer_result.json", "w"), indent=2); print("DONE")


if __name__ == "__main__":
    main()
