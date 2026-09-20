"""Auswertung und die Aussagen der App als Tests: jede Zahl in den Hilfetexten, Tabellen und Presets ist hier über die festen Sweep-Datensätze belegt (Toleranzen bewusst weit).
k-means ist datensatzabhängig - deshalb Mittel und Spannen, nicht einzelne Datensätze."""

import numpy as np
import pytest

import ss_algorithm as alg
import ss_constants as C
import ss_evaluation as ev


def _analyses(seeds=C.SWEEP_SEEDS, settings=ev.Settings(), comparators=False, **kw):
    return [ev.analyse(ev.make_dataset(seed=s, **kw), settings, with_comparators=comparators) for s in seeds]


def _mean(analyses, attr):
    return float(np.mean([getattr(a.result, attr) for a in analyses]))


# --- Zuordnung und Kennzahlen: Handinstanzen -----------------------------------------------------------------------------------------


def test_match_detections_hand_instance():
    truth = np.array([100, 200, 205, 400])
    out = ev.match_detections(np.array([98, 203, 210, 300, 401]), truth)
    assert list(out) == [0, 2, -1, -1, 3]                                                 # 203 liegt näher an 205 als an 200; 210 findet nichts Freies in Reichweite, 300 hat keinen Partner
    assert list(ev.match_detections(np.array([200, 201]), np.array([200]))) == [0, -1]     # jede wahre Spitze nur einmal


def test_truth_spikes_marks_collisions_by_start_distance():
    ds = ev.make_dataset(noise=0.0)
    times, neuron, coll = ev.truth_spikes(ds)
    assert (np.diff(times) >= 0).all() and len(times) == sum(len(t) for t in ds.spike_times) and set(neuron) == set(range(4))
    starts = np.sort(np.concatenate(ds.spike_starts))
    for s in ds.spike_starts[0][:30]:
        near = ((np.abs(starts - s) <= C.COLLISION_WINDOW).sum() > 1)
        idx = np.flatnonzero((neuron == 0))[list(ds.spike_starts[0]).index(s)]
        assert coll[idx] == near


def test_within_scatter_hand_instance_and_evaluate_sorting_on_a_perfect_case():
    F = np.array([[0.0], [2.0], [10.0], [14.0]])
    assert ev.within_scatter(F, np.array([0, 0, 1, 1]), 2) == pytest.approx(2.0 + 8.0)
    ds = ev.make_dataset(m=2, n=4, noise=0.0, rate_scale=0.25)
    r = ev.evaluate_sorting(ds, alg.sort_spikes(ds.X, 2, 4.5, "pca", 3, "known", seed=1))
    assert r.accuracy == 1.0 and r.recall > 0.97 and r.precision == 1.0 and r.f1 > 0.97 and r.confusion.shape == (2, 2) and r.majority_baseline < 0.8


def test_evaluate_sorting_confusion_matrix_and_split_accuracies_are_consistent():
    ds = ev.make_dataset()
    srt = ev.sort_dataset(ds, ev.Settings())
    r = ev.evaluate_sorting(ds, srt)
    assert r.confusion.sum() == round(r.recall * r.n_true) and 0 <= r.accuracy <= 1 and r.accuracy_single >= r.accuracy_collision
    assert abs(r.accuracy * r.confusion.sum() - sum(r.confusion[i, c] for i, c in enumerate(r.cluster_of_neuron) if c >= 0)) < 1e-9
    assert len(r.neuron_of_spike) == len(r.truth_of_spike) == len(r.collision_of_spike) == len(srt.times)


def test_analysis_has_the_expected_structure_and_references_only_when_asked():
    a = ev.analyse_for((4, 4, 2.0, 0.5, 0.2, 0.7, 20000, 7), ev.Settings())
    assert set(a.comparators) == {"ica", "sobi", "sca"} and all(np.isfinite(x) for x in (a.ref_low_rate, a.ref_similar, a.ref_no_jitter, a.ref_clean))
    b = ev.analyse(ev.make_dataset(), ev.Settings(), with_comparators=False)
    assert b.comparators == {} and all(np.isnan(x) for x in (b.ref_low_rate, b.ref_similar, b.ref_no_jitter, b.ref_clean))


def test_oracle_accuracy_is_higher_when_collisions_disturb_the_clustering():
    """Beleg für den Verdict-Text zu kmeans_misled: bei doppelter Feuerrate läge k-means ohne Kollisions-Spikes (Orakel) im Mittel deutlich höher."""
    gains = []
    for seed in C.SWEEP_SEEDS:
        ds = ev.make_dataset(rate_scale=2.0, seed=seed)
        a = ev.analyse(ds, ev.Settings(), with_comparators=False)
        gains.append(ev.oracle_accuracy(ds, a.sorting, a.result) - a.result.accuracy)
    assert np.mean(gains) > 0.05


# --- Verdict-Codes ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("params,kwargs,codes", [
    ((4, 4, 1.0, 1.0, 0.0, 0.05), {}, {"pipeline_ok"}),
    ((4, 1, 1.0, 1.0, 0.0, 0.05), {}, {"kmeans_misled"}),
    ((4, 4, 2.0, 1.0, 0.0, 0.05), {}, {"overlap_high"}),
    ((4, 4, 4.0, 1.0, 0.0, 0.05), {}, {"overlap_high"}),
    ((5, 4, 1.0, 1.0, 0.0, 0.05), {}, {"kmeans_misled"}),
    ((4, 4, 1.0, 1.0, 0.0, 1.0), {"threshold": 6.0}, {"no_detection"}),
    ((4, 4, 1.0, 1.0, 0.0, 0.05), {"threshold": 2.0}, {"false_positives"}),
    ((4, 4, 1.0, 1.0, 0.0, 0.05), {"cluster_mode": "silhouette"}, {"wrong_k"}),
    ((4, 4, 1.0, 1.0, 0.3, 0.05), {}, {"jitter", "kmeans_misled"}),
])
def test_verdict_codes_hold_on_several_datasets(params, kwargs, codes):
    for seed in (7,) + C.SWEEP_SEEDS:
        a = ev.analyse_for(params + (20000, seed), ev.Settings(**kwargs), with_comparators=False)
        assert ev.verdict(a)[1] in codes, (seed, ev.verdict(a)[1])


def test_sweep_seeds_are_separate_from_demo_seeds():
    assert min(C.SWEEP_SEEDS) >= 100_000 > C.DEFAULT_SEED and len(C.SWEEP_SEEDS) >= 5


def test_sweep_feature_k_and_scene_tables_have_the_expected_shape_and_are_deterministic():
    rows = ev.sweep("n_electrodes", values=(2, 4))
    assert rows == ev.sweep("n_electrodes", values=(2, 4)) and [r["x"] for r in rows] == [2, 4] and all(set(r) >= {"f1", "accuracy", "recall", "precision", "collision_share", "ica", "sobi", "sca", "f1_std"} for r in rows)
    thr = ev.sweep("threshold", values=(3.0, 4.5))
    assert np.isnan(thr[0]["ica"]) and thr[0]["recall"] > 0.8
    feat = ev.feature_table(electrodes=(2,))
    assert [r["feature"] for r in feat] == list(C.FEATURES)
    k_rows = ev.k_selection(neurons=(2,))
    assert k_rows[0]["m"] == 2 and len(k_rows[0]["chosen"]) == 5 and 2 in k_rows[0]["curve"]
    scenes = ev.scene_table()
    assert [r["scene"] for r in scenes] == [s for s, _ in ev.SCENES] and all(r["pipeline_min"] <= r["pipeline"] <= r["pipeline_max"] for r in scenes)


# --- Aussagen der App ---------------------------------------------------------------------------------------------------------------------


def test_default_scene_numbers_in_the_preset_help():
    """Beleg für 'Vier Neuronen, vier Elektroden': 92 % der Spikes erkannt, 98 % davon richtig, Spitzen-F1 0.94; ICA, SOBI und SCA 1.0."""
    a = _analyses(comparators=True)
    assert abs(_mean(a, "recall") - 0.926) < 0.03 and abs(_mean(a, "accuracy") - 0.98) < 0.02 and abs(_mean(a, "f1") - 0.945) < 0.02 and _mean(a, "precision") > 0.99
    assert all(x.comparators[n]["f1"] > 0.99 for x in a for n in ("ica", "sobi", "sca"))
    assert np.mean([x.result.missed_collision_share for x in a]) > 0.7                                   # die verpassten Spikes sind fast alle Kollisionen


@pytest.mark.parametrize("m,expected", [(2, 1.0), (3, 0.995), (4, 0.98), (5, 0.80)])
def test_neuron_help_text(m, expected):
    """Beleg für die Hilfe zu den Neuronen (4 Elektroden): Sortiergenauigkeit 1.00 / 0.995 / 0.98 / 0.80."""
    assert abs(_mean(_analyses(m=m), "accuracy") - expected) < 0.04


@pytest.mark.parametrize("n,expected", [(1, 0.51), (2, 0.95), (3, 0.89), (4, 0.95), (8, 0.95)])
def test_electrode_help_text(n, expected):
    """Beleg für die Hilfe zu den Elektroden: Spitzen-F1 0.51 (1), 0.95 (2), 0.89 (3), 0.95 (4-8)."""
    assert abs(_mean(_analyses(n=n), "f1") - expected) < 0.06


def test_signal_separators_fail_with_two_electrodes_where_the_pipeline_works():
    """Beleg für die Hilfe zu den Elektroden: bei 2 Elektroden ICA / SOBI / SCA 0.36 / 0.35 / 0.97; bei einer 0.16 / 0.16 / 0.05."""
    a = _analyses(n=2, comparators=True)
    assert abs(np.mean([x.comparators["ica"]["f1"] for x in a]) - 0.36) < 0.04 and abs(np.mean([x.comparators["sobi"]["f1"] for x in a]) - 0.35) < 0.04
    assert abs(np.mean([x.comparators["sca"]["f1"] for x in a]) - 0.97) < 0.03 and _mean(a, "f1") > 0.9
    b = _analyses(n=1, comparators=True)
    assert abs(np.mean([x.comparators["ica"]["f1"] for x in b]) - 0.16) < 0.03 and abs(np.mean([x.comparators["sca"]["f1"] for x in b]) - 0.05) < 0.04
    assert abs(_mean(b, "accuracy") - 0.65) < 0.05


def test_one_electrode_features_contain_the_information_that_kmeans_misses():
    """Beleg für 'Eine Elektrode': ein Klassifikator mit bekannten Neuronen käme auf 0.94, k-means findet die Aufteilung mit kleinerer Streuung."""
    sup, found, true = [], [], []
    for s in C.SWEEP_SEEDS:
        ds = ev.make_dataset(n=1, seed=s)
        srt = ev.sort_dataset(ds, ev.Settings())
        r = ev.evaluate_sorting(ds, srt)
        ok = r.truth_of_spike >= 0
        F, y = srt.features.values[ok], r.truth_of_spike[ok]
        cent = np.array([F[y == i].mean(axis=0) for i in range(4)])
        sup.append((((F[:, None, :] - cent[None]) ** 2).sum(2).argmin(1) == y).mean())
        found.append(r.scatter_found), true.append(r.scatter_true)
    assert abs(np.mean(sup) - 0.94) < 0.03 and all(f < t for f, t in zip(found, true))


@pytest.mark.parametrize("rate,collision,f1", [(0.25, 0.02, 0.99), (1.0, 0.16, 0.95), (2.0, 0.34, 0.77), (4.0, 0.57, 0.59)])
def test_rate_help_text(rate, collision, f1):
    """Beleg für die Hilfe zur Feuerrate: Kollisionsanteil 2 / 16 / 34 / 57 %, Spitzen-F1 0.99 / 0.95 / 0.77 / 0.59."""
    a = _analyses(rate_scale=rate)
    assert abs(_mean(a, "collision_share") - collision) < 0.04 and abs(_mean(a, "f1") - f1) < 0.06


def test_high_rate_numbers_of_the_preset_and_the_comparators_stay_at_one():
    a = _analyses(rate_scale=2.0, comparators=True)
    assert abs(_mean(a, "recall") - 0.84) < 0.04 and all(x.comparators[n]["f1"] > 0.98 for x in _analyses(rate_scale=4.0, comparators=True) for n in ("ica", "sobi", "sca"))
    assert abs(np.mean([x.ref_low_rate for x in [ev.analyse_for((4, 4, 2.0, 1.0, 0.0, 0.05, 20000, s), ev.Settings(), False) for s in C.SWEEP_SEEDS]]) - 0.99) < 0.02


def test_five_neurons_preset_numbers():
    a = _analyses(m=5, n=4, comparators=True)
    assert abs(_mean(a, "accuracy") - 0.80) < 0.05 and abs(_mean(a, "f1") - 0.75) < 0.07
    assert np.mean([x.comparators["sca"]["f1"] for x in a]) > 0.97 and abs(np.mean([x.comparators["ica"]["f1"] for x in a]) - 0.77) < 0.05


def test_similarity_help_text():
    """Beleg für die Hilfe zur Wellenform-Ähnlichkeit: Sortiergenauigkeit 0.98 (1) gegen 0.89 (0), stark datensatzabhängig; mit einer Elektrode bei Ähnlichkeit 0 nur 0.55."""
    same = _analyses(similarity=0.0)
    accs = [x.result.accuracy for x in same]
    assert abs(np.mean(accs) - 0.89) < 0.04 and min(accs) < 0.85 and max(accs) > 0.95
    assert abs(_mean(_analyses(similarity=0.0, n=1), "accuracy") - 0.55) < 0.06


@pytest.mark.parametrize("jitter,expected", [(0.0, 0.98), (0.1, 0.98), (0.2, 0.95), (0.3, 0.87)])
def test_jitter_help_text(jitter, expected):
    """Beleg für die Hilfe zur Amplitudenschwankung: Sortiergenauigkeit 0.98 / 0.98 / 0.95 / 0.87."""
    assert abs(_mean(_analyses(jitter=jitter), "accuracy") - expected) < 0.03


@pytest.mark.parametrize("noise,f1", [(0.2, 0.94), (0.4, 0.94), (1.0, 0.90)])
def test_noise_help_text(noise, f1):
    """Beleg für die Hilfe zum Rauschen: Spitzen-F1 0.94 / 0.94 / 0.90."""
    assert abs(_mean(_analyses(noise=noise), "f1") - f1) < 0.03


def test_noise_help_text_high_noise_and_comparators():
    """Beleg: bei Rauschen 2.0 Trefferquote 0.34; bei 1.0 ICA / SOBI / SCA 0.43 / 0.38 / 0.86."""
    assert abs(_mean(_analyses(noise=2.0), "recall") - 0.34) < 0.06
    a = _analyses(noise=1.0, comparators=True)
    assert abs(np.mean([x.comparators["ica"]["f1"] for x in a]) - 0.43) < 0.05 and abs(np.mean([x.comparators["sobi"]["f1"] for x in a]) - 0.38) < 0.05 and abs(np.mean([x.comparators["sca"]["f1"] for x in a]) - 0.86) < 0.05


def test_threshold_help_text():
    """Beleg für die Hilfe zur Schwelle: 3-10 alle gleich gut (F1 0.94-0.95); bei 2 nur 49 % Genauigkeit der Detektion; bei Rauschen 1.0 Trefferquote 0.56 (Schwelle 6) gegen 0.86 (4.5)."""
    for thr in (3.0, 4.5, 6.0, 10.0):
        assert 0.92 < _mean(_analyses(settings=ev.Settings(threshold=thr)), "f1") < 0.97
    assert abs(_mean(_analyses(settings=ev.Settings(threshold=2.0)), "precision") - 0.49) < 0.06
    assert abs(_mean(_analyses(noise=1.0, settings=ev.Settings(threshold=6.0)), "recall") - 0.56) < 0.08 and abs(_mean(_analyses(noise=1.0), "recall") - 0.86) < 0.05


def test_feature_help_text():
    """Beleg für die Hilfe zu den Merkmalen: 4 Elektroden PCA 0.98 vor Amplituden 0.96 und Rohwerten 0.95; 1 Elektrode Amplituden 0.78, PCA 0.65, Rohwerte 0.57."""
    rows = {(r["n"], r["feature"]): r["accuracy"] for r in ev.feature_table(electrodes=(1, 4))}
    assert abs(rows[(4, "pca")] - 0.98) < 0.02 and abs(rows[(4, "amplitude")] - 0.96) < 0.03 and abs(rows[(4, "raw")] - 0.95) < 0.04 and rows[(4, "pca")] > rows[(4, "raw")]
    assert abs(rows[(1, "amplitude")] - 0.78) < 0.05 and abs(rows[(1, "pca")] - 0.65) < 0.05 and abs(rows[(1, "raw")] - 0.57) < 0.05 and rows[(1, "amplitude")] > rows[(1, "pca")] > rows[(1, "raw")]


@pytest.mark.parametrize("p,expected", [(1, 0.96), (2, 0.98), (3, 0.98), (5, 0.99), (8, 0.99)])
def test_component_help_text(p, expected):
    """Beleg für die Hilfe zu den PCA-Komponenten: eine 0.96, 2-3 0.98, 5-8 0.99."""
    assert abs(_mean(_analyses(settings=ev.Settings(n_components=p)), "accuracy") - expected) < 0.02


def test_silhouette_always_chooses_too_many_clusters():
    """Beleg für die Hilfe zur Neuronenzahl und die Preset-Hilfe: die Silhouette wählt in jedem Fall zu viele (bei 4 Neuronen im Mittel 6.8); F1 0.93 statt 0.945, Genauigkeit 0.94 statt 0.98."""
    rows = ev.k_selection()
    assert all(k > r["m"] for r in rows for k in r["chosen"]) and all(r["correct"] == 0.0 for r in rows)
    assert abs(np.mean([r for r in rows if r["m"] == 4][0]["chosen"]) - 6.8) < 0.01
    a = _analyses(settings=ev.Settings(cluster_mode="silhouette"))
    assert abs(_mean(a, "f1") - 0.93) < 0.03 and abs(_mean(a, "accuracy") - 0.944) < 0.03


def test_start_hardly_changes_the_result():
    """Beleg für die Hilfe zum Start: die Spitzen-F1 der fünf Starts liegen im Standardfall innerhalb von 0.01, mit einer Elektrode sind sie identisch."""
    values = {}
    for n in (4, 1):
        ds = ev.make_dataset(n=n)
        values[n] = [ev.analyse(ds, ev.Settings(init_start=s), with_comparators=False).result.f1 for s in C.INIT_STARTS]
    assert max(values[4]) - min(values[4]) < 0.01 and max(values[1]) - min(values[1]) < 1e-9
