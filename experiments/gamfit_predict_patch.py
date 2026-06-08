"""Workaround for gamfit predict() KeyError 'model_class' regression (SauersML/gam#867).

Present in gamfit 0.1.179 and 0.1.180: the FFI prediction payload returns only
{"columns": ...} and no longer carries the model_class / family discriminators that
shape_predict_response() dispatches on, so .predict() raises KeyError('model_class').

For standard Gaussian/duchon GAMs the correct shaper is _shape_standard (returns
columns["mean"]). This shim injects model_class="gam"/family="gaussian" when the payload
lacks them, routing to that shaper. Survival/competing-risks payloads (which start with
'{"class":...') are passed through untouched. Import this module before calling .predict().
Remove once #867 is fixed upstream.
"""
import json


def apply():
    try:
        import gamfit._predict_shape as ps
        import gamfit._model as gm
    except Exception:
        return
    if getattr(ps, "_mc_patched", False):
        return
    orig = ps.shape_predict_response

    def patched(raw, *a, **k):
        try:
            if isinstance(raw, str) and not raw.startswith('{"class":"'):
                p = json.loads(raw)
                if "model_class" not in p:
                    p["model_class"] = "gam"
                    p.setdefault("family", "gaussian")
                    raw = json.dumps(p, separators=(",", ":"))
        except Exception:
            pass
        return orig(raw, *a, **k)

    ps.shape_predict_response = patched
    gm.shape_predict_response = patched
    ps._mc_patched = True


apply()
