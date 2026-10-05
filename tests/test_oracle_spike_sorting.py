"""Orakel-Tests der Standardpipeline und ihrer Auswertung (unabhängige Rechenwege): Detektion und Ausschnitte per Schleife, k-means gegen die Streuungs-Definition und Vollaufzählung,
Kollisionen und Zuordnung erkannter Spitzen per Vollsuche, Sortiergüte (Konfusion, Genauigkeit, F1) gegen scipy linear_sum_assignment. PCA und Silhouette prüft test_algorithm gegen scikit-learn."""

import itertools

import numpy as np
import pytest

import ss_algorithm as alg
import ss_constants as C
import ss_evaluation as ev

scipy_opt = pytest.importorskip("scipy.optimize")


def _truth(ds):
    return sorted(((int(t), j, int(s)) for j in range(ds.n_neurons) for t, s in zip(ds.spike_times[j], ds.spike_starts[j])), key=lambda x: x[0])


def _match_by_full_search(det, truth_times, tolerance=C.MATCH_TOLERANCE):
    used = [False] * len(truth_times)
    out = []
    for t in det:
        best, best_d = -1, tolerance + 1
        for c, tt in enumerate(truth_times):
            if not used[c] and abs(int(tt) - int(t)) < best_d:
                best, best_d = c, abs(int(tt) - int(t))
        if best >= 0:
            used[best] = True
        out.append(best)
    return out


def test_detection_signal_detect_and_snippets_match_loops():
    rng = np.random.default_rng(0)
    for _ in range(20):
        n, T = int(rng.integers(1, 4)), int(rng.integers(400, 900))
        X = rng.standard_normal((n, T)) * rng.uniform(0.5, 2, (n, 1)) + rng.uniform(-1, 1, (n, 1))
        for _ in range(int(rng.integers(0, 12))):
            t = int(rng.integers(30, T - 30))
            X[:, t:t + 5] -= rng.uniform(3, 12, (n, 1))
        thr, dead = float(rng.uniform(2, 8)), int(rng.integers(5, 25))
        sigma = np.array([np.median(np.abs(X[j] - np.median(X[j]))) / 0.6745 for j in range(n)])
        sigma = np.maximum(np.maximum(sigma, 0.002 * np.abs(X - np.median(X, axis=1, keepdims=True)).max(axis=1)), 1e-12)
        d_ref = np.array([min((X[j, t] - np.median(X[j])) / sigma[j] for j in range(n)) for t in range(T)])
        times, d = alg.detect(X, thr, dead)
        assert np.allclose(d, d_ref)
        taken = []
        for t in sorted((t for t in range(T) if d_ref[t] < -thr), key=lambda t: (d_ref[t], t)):
            if not any(abs(t - u) <= dead for u in taken):
                taken.append(t)
        assert sorted(taken) == list(times)
        kept, snips = alg.snippets(X, times)
        assert list(kept) == [t for t in times if t - C.SNIPPET_BEFORE >= 0 and t + C.SNIPPET_AFTER <= T]
        for t, s in zip(kept, snips):
            assert np.array_equal(s, X[:, t - C.SNIPPET_BEFORE:t + C.SNIPPET_AFTER])


def test_kmeans_inertia_is_the_within_cluster_scatter_and_finds_the_optimum_on_tiny_instances_most_of_the_time():
    rng = np.random.default_rng(1)
    optimal = 0
    for i in range(40):
        N, k = int(rng.integers(5, 9)), int(rng.integers(2, 4))
        F = rng.standard_normal((N, 2)) * rng.uniform(0.5, 3)
        km = alg.kmeans(F, k, seed=i)
        scatter = sum(((F[km.labels == j] - F[km.labels == j].mean(0)) ** 2).sum() for j in range(k) if (km.labels == j).any())
        assert abs(scatter - km.inertia) < 1e-8 * max(1.0, scatter)
        best = min(sum(((F[np.array(lab) == j] - F[np.array(lab) == j].mean(0)) ** 2).sum() for j in range(k)) for lab in itertools.product(range(k), repeat=N) if len(set(lab)) == k)
        assert km.inertia >= best - 1e-9                                                                 # nie besser als das Optimum
        optimal += km.inertia <= best * (1 + 1e-9) + 1e-12
    assert optimal >= 30                                                                                 # Lloyd ist heuristisch: meist, nicht immer optimal


def test_truth_spikes_and_collisions_match_a_brute_force_definition():
    for seed, rate in ((1, 1.0), (2, 4.0), (3, 2.0)):
        ds = ev.make_dataset(m=5, n=2, rate_scale=rate, n_samples=8000, seed=seed)
        times, neuron, collision = ev.truth_spikes(ds)
        truth = _truth(ds)
        assert list(times) == [t[0] for t in truth] and list(neuron) == [t[1] for t in truth]
        for (t, j, s), c in zip(truth, collision):
            assert c == any((o[1], o[2]) != (j, s) and abs(o[2] - s) <= C.COLLISION_WINDOW for o in truth)


def test_match_detections_equals_a_full_search_also_for_crowded_truth():
    """Regression: früher wurden nur vier Nachbarn der Einfügestelle geprüft; bei mehreren gleichzeitigen wahren Spitzen blieben freie unentdeckt."""
    assert list(ev.match_detections(np.array([60] * 5), np.array([60] * 6))) == [0, 1, 2, 3, 4]
    rng = np.random.default_rng(2)
    for _ in range(40):
        truth = np.sort(rng.integers(0, 300, int(rng.integers(5, 80))))
        det = np.sort(rng.integers(-5, 305, int(rng.integers(1, 80))))
        assert list(ev.match_detections(det, truth)) == _match_by_full_search(det, truth)


def test_evaluate_sorting_matches_an_independent_reference():
    rng = np.random.default_rng(3)
    for i in range(10):
        m = int(rng.integers(2, 6))
        ds = ev.make_dataset(m=m, n=int(rng.integers(1, 5)), rate_scale=float(rng.choice([1, 2, 4])), noise=float(rng.choice([0.05, 0.5])), n_samples=8000, seed=int(rng.integers(0, 10**6)))
        k = int(rng.integers(1, 7))
        srt = alg.sort_spikes(ds.X, k, float(rng.choice([3, 4.5, 6])), str(rng.choice(C.FEATURES)), 3, "known", seed=i)
        r = ev.evaluate_sorting(ds, srt)
        times, labels = srt.times, srt.clustering.labels
        truth = _truth(ds)
        coll = [any((o[1], o[2]) != (j, s) and abs(o[2] - s) <= C.COLLISION_WINDOW for o in truth) for (t, j, s) in truth]
        mt = _match_by_full_search(times, [t[0] for t in truth])
        ok = [c >= 0 for c in mt]
        assert abs(r.recall - sum(ok) / len(truth)) < 1e-12 and abs(r.precision - sum(ok) / max(len(times), 1)) < 1e-12
        conf = np.zeros((m, max(srt.k, 1)))
        for q in range(len(times)):
            if ok[q]:
                conf[truth[mt[q]][1], labels[q]] += 1
        assert np.array_equal(conf, r.confusion)
        if sum(ok) and srt.k > 0:
            rr, cc = scipy_opt.linear_sum_assignment(conf, maximize=True)
            assert abs(r.accuracy - conf[rr, cc].sum() / sum(ok)) < 1e-12
        counts = np.bincount([t[1] for t in truth], minlength=m)
        assert abs(r.majority_baseline - counts.max() / counts.sum()) < 1e-12 and abs(r.collision_share - np.mean(coll)) < 1e-12
        missed = [c for c in range(len(truth)) if c not in set(x for x in mt if x >= 0)]
        if missed:
            assert abs(r.missed_collision_share - np.mean([coll[c] for c in missed])) < 1e-12
        f1s = []
        for j in range(m):
            c = r.cluster_of_neuron[j]
            det_j = sorted(int(t) for t, lab in zip(times, labels) if c >= 0 and lab == c)
            tr = [int(t) for t in ds.spike_times[j]]
            if not det_j:
                f1s.append(0.0)
                continue
            hit = (np.abs(np.array(tr)[:, None] - np.array(det_j)[None, :]) <= ev.DETECT_TOLERANCE).astype(float)
            a, b = scipy_opt.linear_sum_assignment(hit, maximize=True)
            f1s.append(2 * hit[a, b].sum() / (len(det_j) + len(tr)))
        assert abs(np.mean(f1s) - r.f1) < 1e-9
