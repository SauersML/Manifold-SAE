"""Does a CONSOLIDATION mechanism make the FIRING atom the read+steer handle, on
a real model? Toy auto_exp_83 said tying barely helped (gate-winner read 0.44->0.50).
Real-model rerun: days @ layer 14, free-head SAE vs tied SAE (gate read off the
atom's own plane), compare gate-winner read & real-model steer.
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


def basis(th, H=1):
    f = [torch.ones_like(th)]
    for h in range(1, H + 1):
        f += [torch.cos(h * th), torch.sin(h * th)]
    return torch.stack(f, -1)


class FreeSAE(nn.Module):
    def __init__(s, D, F=48):
        super().__init__(); s.F = F
        s.B = nn.Parameter(torch.randn(F, 3, D) / math.sqrt(3 * D)); s.b = nn.Parameter(torch.zeros(D))
        s.gh = nn.Linear(D, F); s.ch = nn.Linear(D, 2 * F)

    def fwd(s, x):
        pre = s.gh(x); g = pre * (pre > 0.05).float()
        cs = s.ch(x).reshape(x.shape[0], s.F, 2); th = torch.atan2(cs[..., 1], cs[..., 0]); return g, th

    def loss(s, x):
        g, th = s.fwd(x); xh = torch.einsum("nf,nfd->nd", g, torch.einsum("nfm,fmd->nfd", basis(th), s.B)) + s.b
        return (xh - x).pow(2).mean() + 1e-3 * g.abs().mean()


class TiedSAE(nn.Module):
    """gate + coordinate both read off B_j's own plane (firing atom = reading atom)."""
    def __init__(s, D, F=48, thr=0.5):
        super().__init__(); s.F = F; s.thr = thr
        s.B = nn.Parameter(torch.randn(F, 3, D) / math.sqrt(D)); s.b = nn.Parameter(torch.zeros(D))

    def frame(s):
        c0, v1, v2 = s.B[:, 0], s.B[:, 1], s.B[:, 2]
        e1 = v1 / v1.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        v2p = v2 - (v2 * e1).sum(-1, keepdim=True) * e1
        return c0, e1, v2p / v2p.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    def fwd(s, x):
        c0, e1, e2 = s.frame(); xc = x.unsqueeze(1) - s.b - c0.unsqueeze(0)
        a = (xc * e1).sum(-1); b = (xc * e2).sum(-1); th = torch.atan2(b, a)
        g = torch.sigmoid(8 * (torch.sqrt(a * a + b * b + 1e-9) - s.thr)); return g, th

    def loss(s, x):
        g, th = s.fwd(x); xh = torch.einsum("nf,nfd->nd", g, torch.einsum("nfm,fmd->nfd", basis(th), s.B)) + s.b
        return (xh - x).pow(2).mean() + 1e-3 * g.mean()


@torch.no_grad()
def harvest(model, tok):
    X, y = [], []
    for t in T:
        for di, d in enumerate(DAYS):
            ids = tok(t.format(x=d), return_tensors="pt").input_ids.to(DEV)
            X.append(model(ids, output_hidden_states=True).hidden_states[L][0, -1].float().cpu().numpy()); y.append(di)
    return np.array(X), np.array(y)


def diagnose(model, tok, sae, X, y, mu):
    with torch.no_grad():
        g, th = sae.fwd(torch.tensor(X - mu, dtype=torch.float32, device=DEV))
    th = th.cpu().numpy(); g = g.abs().cpu().numpy(); ang = y * 2 * math.pi / 7
    ccs = [cc(ang, th[:, j]) for j in range(sae.F)]; gw = int(g.mean(0).argmax())
    vids = [tok(" " + d, add_special_tokens=False).input_ids[0] for d in DAYS]
    prompt = "The day after Monday is"; pos = tok(prompt[:prompt.index("Monday") + 6], return_tensors="pt").input_ids.shape[1] - 1
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEV); B = sae.B.detach(); vang = np.arange(7) * 2 * math.pi / 7

    def steer(atom):
        with torch.no_grad():
            hs0 = model(ids, output_hidden_states=True).hidden_states[L][0, pos].float()
            gg, th0 = sae.fwd(torch.tensor((hs0.cpu().numpy() - mu), device=DEV).unsqueeze(0))
            cur0 = torch.einsum("fm,fmd->fd", basis(th0[0]), B)[atom]
        outs, grid = [], np.linspace(-math.pi, math.pi, 12, endpoint=False)
        for tv in grid:
            cur1 = torch.einsum("m,md->d", basis(torch.tensor([tv], device=DEV))[0], B[atom])
            delta = (gg[0, atom] * (cur1 - cur0)).half()
            def hook(m, i, o):
                oo = o[0] if isinstance(o, tuple) else o; oo[0, pos] = oo[0, pos] + delta; return o
            h = model.model.layers[L].register_forward_hook(hook)
            with torch.no_grad():
                p = torch.softmax(model(ids).logits[0, -1][vids].float(), 0).cpu().numpy()
            h.remove(); outs.append(math.atan2((p * np.sin(vang)).sum(), (p * np.cos(vang)).sum()))
        return cc(grid, np.array(outs))
    bra = int(np.argmax(ccs))
    return {"best_read": round(ccs[bra], 3), "best_read_steer": round(steer(bra), 3),
            "gate_winner_read": round(ccs[gw], 3), "gate_winner_steer": round(steer(gw), 3)}


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(DEV).eval()
    X, y = harvest(model, tok); mu = X.mean(0); Xt = torch.tensor(X - mu, dtype=torch.float32, device=DEV)
    out = {}
    for name, Cls in [("free_head", FreeSAE), ("tied", TiedSAE)]:
        torch.manual_seed(0); sae = Cls(X.shape[1]).to(DEV)
        opt = torch.optim.Adam(sae.parameters(), lr=3e-3)
        for _ in range(600):
            opt.zero_grad(); sae.loss(Xt).backward(); opt.step()
        sae.eval(); out[name] = diagnose(model, tok, sae, X, y, mu)
        print(name, out[name], flush=True)
    json.dump(out, open("/home/user/consol_result.json", "w"), indent=2); print("DONE")


if __name__ == "__main__":
    main()
