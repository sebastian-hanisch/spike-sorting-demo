"""Jedes Preset zeigt, was sein Name und seine Hilfe behaupten (Bänder mit dem ausgelieferten Code kalibriert, bewusst weit)."""

import pytest

import ss_constants as C
from ss_evaluation import Settings, analyse_for, verdict


def _measure(p):
    params = (p["m"], p["n"], p["rate_scale"], p["similarity"], p["jitter"], p["noise"], p["n_samples"], p["seed"])
    a = analyse_for(params, Settings(p["threshold"], p["feature"], p["n_components"], p["cluster_mode"], p["contrast"], p["init_start"]))
    r = a.result
    return {"verdict": verdict(a)[1], "recall": r.recall, "accuracy": r.accuracy, "f1": r.f1, "precision": r.precision}


def test_every_preset_has_help_and_bands():
    assert set(C.PRESETS) == set(C.PRESET_HELP) == set(C.PRESET_EXPECTED_BANDS)
    assert len(C.PRESETS) == 6


def test_preset_settings_are_within_slider_bounds():
    for p in C.PRESETS.values():
        assert C.N_NEURONS_MIN <= p["m"] <= C.N_NEURONS_MAX and C.N_ELECTRODES_MIN <= p["n"] <= C.N_ELECTRODES_MAX
        assert C.RATE_SCALE_MIN <= p["rate_scale"] <= C.RATE_SCALE_MAX and abs(p["rate_scale"] * 4 - round(p["rate_scale"] * 4)) < 1e-9
        assert C.SIMILARITY_MIN <= p["similarity"] <= C.SIMILARITY_MAX and C.JITTER_MIN <= p["jitter"] <= C.JITTER_MAX and C.NOISE_MIN <= p["noise"] <= C.NOISE_MAX
        assert C.N_SAMPLES_MIN <= p["n_samples"] <= C.N_SAMPLES_MAX and p["n_samples"] % 1000 == 0 and C.THRESHOLD_MIN <= p["threshold"] <= C.THRESHOLD_MAX and abs(p["threshold"] * 2 - round(p["threshold"] * 2)) < 1e-9
        assert p["feature"] in C.FEATURES and C.N_COMPONENTS_MIN <= p["n_components"] <= C.N_COMPONENTS_MAX and p["cluster_mode"] in C.CLUSTER_MODES and p["contrast"] in C.CONTRASTS and p["init_start"] in C.INIT_STARTS


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_preset_stays_inside_its_bands(name):
    measured = _measure(C.PRESETS[name])
    for key, expected in C.PRESET_EXPECTED_BANDS[name].items():
        value = measured[key]
        if isinstance(expected, str):
            assert value == expected, f"{key}: {value}"
        else:
            lo, hi = expected
            assert lo <= value <= hi, f"{key}: {value} nicht in [{lo}, {hi}]"
