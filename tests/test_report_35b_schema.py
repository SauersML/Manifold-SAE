from __future__ import annotations

from experiments import report_35b_figures as rpt


def test_fig2_rejects_missing_theta_without_silent_skip(tmp_path):
    compose = {
        "min_effect_ev": 0.005,
        "atoms": [
            {
                "idx": 0,
                "topology": "circle",
                "theta": None,
                "delta_ev": 0.02,
                "delta_ev_source": "heldout_loao",
            }
        ],
    }
    res = rpt.fig2_theta_dev(compose, tmp_path / "fig2.png")
    assert res["status"] == "MISS"
    assert "missing theta" in res["reason"]


def test_null_gate_rejects_missing_mean_theta(tmp_path):
    nc = {
        "salience_floor": 0.005,
        "real_reference": {"n_curved_accepted": 7, "mean_theta": 1.8},
        "gaussian_matched": {"n_curved_accepted": 0, "mean_theta": None},
        "shuffled": {"n_curved_accepted": 1, "mean_theta": 0.1},
        "harmonic_null": {"higher_modes_on_first_harmonic_plus_noise": False},
    }
    compose = {"min_effect_ev": 0.005}
    res = rpt.g0_null_gate(nc, tmp_path / "fig9.png", compose)
    assert res["status"] == "MISS"
    assert "mean_theta" in res["reason"]


def test_schema_gate_requires_seed2_provenance():
    atom = {
        "idx": 0,
        "topology": "circle",
        "theta": 1.2,
        "delta_ev": 0.02,
        "delta_ev_source": "heldout_loao",
    }
    op = {
        "total_actives": 40,
        "heldout_ev": 0.9,
        "linear_only_heldout_ev": 0.86,
        "heldout_subsample_n": 50000,
    }
    births = [{"ev": 0.86, "collapse_events": 0}, {"ev": 0.9, "collapse_events": 0}]
    compose = rpt.artifact_schema.make_compose_per_atom_artifact(
        gamfit_version=rpt.artifact_schema.MIN_GAMFIT_VERSION,
        random_state=0,
        min_effect_ev=0.005,
        operating_point=op,
        atoms=[atom],
        births=births,
    )
    not_seed2 = rpt.artifact_schema.make_compose_per_atom_artifact(
        gamfit_version=rpt.artifact_schema.MIN_GAMFIT_VERSION,
        random_state=0,
        min_effect_ev=0.005,
        operating_point=op,
        atoms=[atom],
        births=births,
    )
    nc = rpt.artifact_schema.make_null_control_artifact(
        salience_floor=0.005,
        real_reference={"n_curved_accepted": 7, "mean_theta": 1.8},
        gaussian_matched={"n_curved_accepted": 0, "mean_theta": 0.1},
        shuffled={"n_curved_accepted": 1, "mean_theta": 0.1},
        harmonic_null={"higher_modes_on_first_harmonic_plus_noise": False},
    )
    gate = rpt._schema_gate(compose, nc, not_seed2)
    assert gate["status"] == "MISS"
    assert any("compose_per_atom_seed2.json" in e and "random_state" in e for e in gate["errors"])


def test_8b_dose_is_supporting_smoke_not_35b_crown(tmp_path):
    dose = {
        "model": "Qwen3-8B",
        "probe_order": list(range(7)),
        "probe_angles": [0.0, 0.9, 1.8, 2.7, 3.6, 4.5, 5.4],
        "ordering_corr": 0.97,
        "predicted_nats": [0.0, 0.5, 1.0],
        "measured_kl": [0.0, 0.55, 1.1],
        "slope": 1.1,
        "r2": 0.95,
    }
    res = rpt.fig78_dose(dose, tmp_path / "order.png", tmp_path / "dose.png")
    assert res["dose_scope"] == "supporting_smoke"
    assert res["A5_status"] == "PENDING"
    assert res["A6_status"] == "PENDING"
    assert res["supporting_A5_status"] == "ACCEPT"
    assert res["supporting_A6_status"] == "ACCEPT"
    assert "not from the 35B/36B" in res["crown_reason"]
