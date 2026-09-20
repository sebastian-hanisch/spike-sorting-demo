"""Auswertung der Spike-Sorting-Demo: Zuordnung, Spitzen-Treffer (aus ica/sobi/sca-demo übernommen) und die Kennzahlen der Pipeline (Detektion, Sortiergenauigkeit, Kollisionen, Zufallsniveau),
Sweeps, Vergleichs- und Merkmals-Tabellen und das Urteil."""

from dataclasses import dataclass

import numpy as np

import ss_algorithm as alg
import ss_constants as C
import ss_ica as ica
import ss_sca as sca
import ss_scenario as sc
import ss_sobi as sobi


@dataclass(frozen=True)
class Settings:
    threshold: float = C.DEFAULT_THRESHOLD
    feature: str = C.DEFAULT_FEATURE
    n_components: int = C.DEFAULT_N_COMPONENTS
    cluster_mode: str = C.DEFAULT_CLUSTER_MODE
    contrast: str = C.DEFAULT_CONTRAST
    init_start: int = C.DEFAULT_INIT_START


def make_dataset(m=C.DEFAULT_N_NEURONS, n=C.DEFAULT_N_ELECTRODES, rate_scale=C.DEFAULT_RATE_SCALE, similarity=C.DEFAULT_SIMILARITY, jitter=C.DEFAULT_JITTER, noise=C.DEFAULT_NOISE,
                 n_samples=C.DEFAULT_N_SAMPLES, seed=C.DEFAULT_SEED):
    return sc.make_dataset(m, n, rate_scale, similarity, jitter, noise, n_samples, seed)


def n_components(ds):
    """Die Zahl der Quellen wird als bekannt angenommen (Standardannahme von FastICA und SOBI); bei weniger Elektroden als Quellen bleibt nur n."""
    return min(ds.n_electrodes, ds.S.shape[0])


def correlation_matrix(S, estimates):
    """|Korrelation| (k, nc) zwischen wahren Quellen und Schätzungen."""
    a = (S - S.mean(axis=1, keepdims=True)) / S.std(axis=1, keepdims=True)
    b = estimates - estimates.mean(axis=1, keepdims=True)
    sd = b.std(axis=1, keepdims=True)
    b = b / np.where(sd > 0, sd, 1.0)
    return np.abs(a @ b.T) / S.shape[1]


def assign(C_abs):
    """Optimale Zuordnung Quelle -> Schätzung (jede Schätzung höchstens einer Quelle), Summe der |Korrelationen| maximal. Exakt per Bitmasken-DP.
    Rückgabe: Liste je Quelle mit dem Index der Schätzung oder -1 (nicht zugeordnet, nur wenn es weniger Schätzungen als Quellen gibt)."""
    k, nc = C_abs.shape
    best = {}

    def solve(row, used):
        if row == k:
            return 0.0, ()
        key = (row, used)
        if key in best:
            return best[key]
        result = (solve(row + 1, used)[0], (-1,) + solve(row + 1, used)[1])
        for j in range(nc):
            if not used >> j & 1:
                value, rest = solve(row + 1, used | 1 << j)
                if value + C_abs[row, j] > result[0] + 1e-12:
                    result = (value + C_abs[row, j], (j,) + rest)
        best[key] = result
        return result

    return list(solve(0, 0)[1])


def matched(S, estimates):
    """Zuordnung + Vorzeichen-/Skalenkorrektur per Regression: (Zuordnung, |Korrelation| je Quelle, Schätzquellen in Skala und Vorzeichen der wahren Quellen)."""
    cm = correlation_matrix(S, estimates)
    idx = assign(cm)
    corr = np.array([cm[i, j] if j >= 0 else 0.0 for i, j in enumerate(idx)])
    aligned = np.zeros_like(S)
    for i, j in enumerate(idx):
        if j >= 0:
            e = estimates[j] - estimates[j].mean()
            aligned[i] = (e @ (S[i] - S[i].mean()) / (e @ e)) * e + S[i].mean()
    return idx, corr, aligned


def sir_db(corr):
    """Signal-zu-Interferenz in dB aus der Korrelation: rho^2 / (1 - rho^2)."""
    r2 = np.clip(np.asarray(corr) ** 2, 1e-9, 1.0 - 1e-9)
    return 10.0 * np.log10(r2 / (1.0 - r2))


def amari_index(P):
    """Amari-Index einer (nc, k)-Matrix P = Entmischung x Mischung; 0 = perfekt (nur Permutation und Skalierung), 1 = schlechtestmöglich; verallgemeinert auf nicht quadratische P."""
    P = np.abs(P)
    k = P.shape[1]
    rows = (P.sum(axis=1) / P.max(axis=1) - 1.0).sum()
    cols = (P.sum(axis=0) / P.max(axis=0) - 1.0).sum()
    d = max(k, P.shape[0])
    return float((rows + cols) / (2.0 * d * (d - 1))) if d > 1 else 0.0


# --- Spike-Erkennung auf den Schätzquellen ------------------------------------------------------------------------------------
DETECT_MIN_SEPARATION = 15         # Abtastwerte zwischen zwei erkannten Spitzen
DETECT_TOLERANCE = 4               # erlaubte Abweichung zur wahren Spitze
DETECT_SIGMA_FACTOR = 4.0          # Schwelle: 4 robuste Standardabweichungen (MAD) ...
DETECT_DEPTH_FRACTION = 0.3        # ... mindestens aber 30 % der typischen Spitzentiefe (sonst würde ein rauschfreies Signal jeden Rest melden)


def detect_spikes(x):
    """Negative Spitzen von x (Vorzeichen bereits wie beim Neuron): tiefste zuerst, Mindestabstand; Schwelle max(4 sigma_MAD, 0.3 x typische Tiefe)."""
    sigma = np.median(np.abs(x - np.median(x))) / 0.6745
    threshold = -DETECT_SIGMA_FACTOR * sigma
    taken = np.zeros(len(x), bool)
    peaks = []
    for t in np.argsort(x):
        if x[t] > threshold:
            break
        if taken[max(0, t - DETECT_MIN_SEPARATION): t + DETECT_MIN_SEPARATION + 1].any():
            continue
        taken[t] = True
        peaks.append(t)
        if len(peaks) == 10:                                             # typische Tiefe = Median der 10 tiefsten Spitzen
            threshold = min(threshold, DETECT_DEPTH_FRACTION * float(np.median(x[peaks])))
    return np.sort(np.array(peaks, dtype=int))


def spike_f1(detected, true):
    """F1 der erkannten gegen die wahren Spitzenzeiten (Zuordnung je wahrer Spitze zur nächsten, jede erkannte höchstens einmal)."""
    if len(detected) == 0 or len(true) == 0:
        return 0.0
    used = np.zeros(len(detected), bool)
    tp = 0
    for t in true:
        d = np.abs(detected - t)
        j = int(np.argmin(d))
        if d[j] <= DETECT_TOLERANCE and not used[j]:
            used[j] = True
            tp += 1
    if tp == 0:
        return 0.0
    precision, recall = tp / len(detected), tp / len(true)
    return 2 * precision * recall / (precision + recall)



def excess_kurtosis(x):
    x = np.asarray(x, dtype=float)
    x = (x - x.mean(axis=-1, keepdims=True)) / x.std(axis=-1, keepdims=True)
    return (x ** 4).mean(axis=-1) - 3.0








# --- Zuordnung erkannter und wahrer Spikes ---------------------------------------------------------------------------------------------


def truth_spikes(ds):
    """Alle wahren Spikes nach Zeit sortiert: (Zeiten der negativen Spitze, Neuron, Kollision). Kollision = ein anderer Spike beginnt höchstens COLLISION_WINDOW Abtastwerte vor oder nach diesem."""
    times = np.concatenate(ds.spike_times)
    neuron = np.concatenate([np.full(len(t), i) for i, t in enumerate(ds.spike_times)])
    starts = np.concatenate(ds.spike_starts)
    order = np.argsort(times, kind="stable")
    times, neuron, starts = times[order], neuron[order], starts[order]
    all_starts = np.sort(starts)
    left = np.searchsorted(all_starts, starts - C.COLLISION_WINDOW, side="left")
    right = np.searchsorted(all_starts, starts + C.COLLISION_WINDOW, side="right")
    return times, neuron, (right - left) > 1


def match_detections(detected, truth_times, tolerance=C.MATCH_TOLERANCE):
    """Je erkannter Spitze der Index der nächsten noch freien wahren Spitze innerhalb `tolerance`, sonst -1 (jede wahre Spitze höchstens einmal)."""
    out = np.full(len(detected), -1, dtype=int)
    used = np.zeros(len(truth_times), bool)
    for i, t in enumerate(detected):
        j = int(np.searchsorted(truth_times, t))
        best, best_d = -1, tolerance + 1
        for c in (j - 2, j - 1, j, j + 1):
            if 0 <= c < len(truth_times) and not used[c] and abs(int(truth_times[c]) - int(t)) < best_d:
                best, best_d = c, abs(int(truth_times[c]) - int(t))
        if best >= 0 and best_d <= tolerance:
            out[i] = best
            used[best] = True
    return out


def within_scatter(F, labels, k):
    """Summe der quadrierten Abstände zu den Cluster-Mittelpunkten (die Zielgröße von k-means)."""
    total = 0.0
    for j in range(k):
        members = F[labels == j]
        if len(members):
            total += float(((members - members.mean(axis=0)) ** 2).sum())
    return total


@dataclass(frozen=True)
class SortResult:
    recall: float                 # Anteil der wahren Spikes, die erkannt wurden
    precision: float              # Anteil der erkannten Spitzen, die zu einem wahren Spike gehören
    n_detected: int
    n_true: int
    missed_collision_share: float  # Anteil der verpassten Spikes, die Kollisionen sind
    accuracy: float               # richtig sortierte Anteil der erkannten wahren Spikes
    accuracy_single: float
    accuracy_collision: float
    confusion: np.ndarray         # (m, k) Zahl erkannter wahrer Spikes je Neuron und Cluster
    cluster_of_neuron: list       # je Neuron das zugeordnete Cluster (-1 = keins)
    neuron_of_spike: np.ndarray   # je erkanntem Spike das Neuron seines Clusters (-1 = Cluster ohne Neuron)
    truth_of_spike: np.ndarray    # je erkanntem Spike das wahre Neuron (-1 = falsch erkannt)
    collision_of_spike: np.ndarray
    f1: float                     # mittlerer Spitzen-F1 der Neuronen (Cluster-Spikes gegen wahre Spitzenzeiten)
    f1_per_neuron: np.ndarray
    collision_share: float        # Anteil der wahren Spikes, die Kollisionen sind
    majority_baseline: float      # Genauigkeit, wenn jeder Spike dem häufigsten Neuron zugeordnet würde
    scatter_found: float          # Streuung innerhalb der Cluster: gefundene Zuordnung ...
    scatter_true: float           # ... und wahre Zuordnung (auf denselben Merkmalen)


def evaluate_sorting(ds, sorting):
    m = ds.n_neurons
    tt, tn, tc = truth_spikes(ds)
    times = sorting.times
    labels = sorting.clustering.labels
    k = max(sorting.k, 1)
    mi = match_detections(times, tt)
    ok = mi >= 0
    matched_truth = np.full(len(times), -1)
    matched_truth[ok] = tn[mi[ok]]
    collision = np.zeros(len(times), bool)
    collision[ok] = tc[mi[ok]]
    confusion = np.zeros((m, k))
    for lab, y in zip(labels[ok], tn[mi[ok]]):
        confusion[y, lab] += 1
    idx = assign(confusion) if ok.any() and sorting.k > 0 else [-1] * m
    cmap = {c: i for i, c in enumerate(idx) if c >= 0}
    neuron_of_spike = np.array([cmap.get(int(l), -1) for l in labels], dtype=int) if len(labels) else np.zeros(0, dtype=int)
    correct = (neuron_of_spike == matched_truth) & ok
    n_ok = max(int(ok.sum()), 1)
    single, coll = ok & ~collision, ok & collision
    missed = np.ones(len(tt), bool)
    missed[mi[ok]] = False
    f1s = []
    for i in range(m):
        c = idx[i]
        detected_i = times[labels == c] if c >= 0 else np.array([], dtype=int)
        f1s.append(spike_f1(detected_i, ds.spike_times[i]))
    counts = np.bincount(tn, minlength=m)
    F = sorting.features.values
    truth_labels = np.where(matched_truth >= 0, matched_truth, 0)
    usable = len(F) == len(times) and len(F) > 0 and F.shape[1] > 0            # bei (fast) keinen erkannten Spikes gibt es keine Merkmale
    return SortResult(
        recall=float(ok.sum() / max(len(tt), 1)), precision=float(ok.sum() / max(len(times), 1)), n_detected=int(len(times)), n_true=int(len(tt)),
        missed_collision_share=float(tc[missed].mean()) if missed.any() else 0.0,
        accuracy=float(correct.sum() / n_ok), accuracy_single=float(correct[single].sum() / max(single.sum(), 1)) if single.any() else float("nan"),
        accuracy_collision=float(correct[coll].sum() / max(coll.sum(), 1)) if coll.any() else float("nan"),
        confusion=confusion, cluster_of_neuron=list(idx), neuron_of_spike=neuron_of_spike, truth_of_spike=matched_truth, collision_of_spike=collision,
        f1=float(np.mean(f1s)), f1_per_neuron=np.array(f1s), collision_share=float(tc.mean()), majority_baseline=float(counts.max() / max(counts.sum(), 1)),
        scatter_found=within_scatter(F, labels, k) if usable else float("nan"),
        scatter_true=within_scatter(F[ok], truth_labels[ok], m) if usable and ok.any() else float("nan"))


# --- Vergleichsverfahren: Spitzen-F1 mit derselben Definition -----------------------------------------------------------------------


def comparator_f1(ds, settings):
    """Mittlerer Spitzen-F1 der Neuronen für ICA, SOBI und SCA auf denselben Daten (Schätzung -> Zuordnung per Korrelation -> Schwelle -> Treffer), dazu die Neuronen-Korrelation."""
    m = ds.n_neurons
    nc = n_components(ds)
    out = {}
    estimates = {
        "ica": ica.fit_ica(ds.X, nc, settings.contrast, C.DEFAULT_METHOD, settings.init_start).sources,
        "sobi": sobi.fit_sobi(ds.X, nc, C.SOBI_LAGS).sources,
        "sca": sca.fit_sca(ds.X, m, "l1", seed=settings.init_start).sources,
    }
    for name, est in estimates.items():
        idx, corr, aligned = matched(ds.S, est)
        out[name] = {"f1": float(np.mean([spike_f1(detect_spikes(aligned[i]), ds.spike_times[i]) for i in range(m)])), "corr": float(corr[:m].mean())}
    return out


# --- Gesamtanalyse ------------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Analysis:
    ds: sc.Dataset
    settings: Settings
    sorting: alg.Sorting
    result: SortResult
    comparators: dict             # "ica", "sobi", "sca" -> {"f1", "corr"}
    silhouette: float             # Silhouette der gefundenen Zuordnung (0 bei weniger als zwei Clustern)
    ref_low_rate: float           # Pipeline-F1 bei Feuerraten-Faktor 0.25 (nur bei Faktor > 1)
    ref_similar: float            # ... bei Wellenform-Ähnlichkeit 1 (nur bei Ähnlichkeit < 1)
    ref_no_jitter: float          # ... ohne Amplitudenschwankung (nur bei Schwankung > 0)
    ref_clean: float              # ... bei Standard-Rauschen 0.05 (nur bei mehr Rauschen; rauschfreie Daten sind für k-means kein guter Referenzfall)


def sort_dataset(ds, settings):
    return alg.sort_spikes(ds.X, ds.n_neurons, settings.threshold, settings.feature, settings.n_components, settings.cluster_mode, seed=settings.init_start)


def _pipeline_f1(ds, settings):
    return evaluate_sorting(ds, sort_dataset(ds, settings)).f1


def analyse(ds, settings, with_comparators=True, refs=None):
    """`refs` = (rate_scale, similarity, jitter, noise): die Einstellungen, mit denen `ds` erzeugt wurde; nur damit werden die Referenzläufe des Urteils berechnet."""
    sorting = sort_dataset(ds, settings)
    result = evaluate_sorting(ds, sorting)
    comparators = comparator_f1(ds, settings) if with_comparators else {}
    sil = alg.silhouette(sorting.features.values, sorting.clustering.labels, seed=settings.init_start) if sorting.k >= 2 and len(sorting.features.values) > 3 else 0.0
    nan = float("nan")
    ref_low_rate = ref_similar = ref_no_jitter = ref_clean = nan
    if refs is not None:
        rate_scale, similarity, jitter, noise = refs
        base = dict(m=ds.n_neurons, n=ds.n_electrodes, n_samples=ds.S.shape[1], seed=ds.seed)
        cur = dict(rate_scale=rate_scale, similarity=similarity, jitter=jitter, noise=noise)
        if rate_scale > 1:
            ref_low_rate = _pipeline_f1(make_dataset(**base, **{**cur, "rate_scale": 0.25}), settings)
        if similarity < 1:
            ref_similar = _pipeline_f1(make_dataset(**base, **{**cur, "similarity": 1.0}), settings)
        if jitter > 0:
            ref_no_jitter = _pipeline_f1(make_dataset(**base, **{**cur, "jitter": 0.0}), settings)
        if noise > C.DEFAULT_NOISE:
            ref_clean = _pipeline_f1(make_dataset(**base, **{**cur, "noise": C.DEFAULT_NOISE}), settings)
    return Analysis(ds, settings, sorting, result, comparators, sil, ref_low_rate, ref_similar, ref_no_jitter, ref_clean)


def analyse_for(params, settings, with_comparators=True):
    """Analyse mit allen Referenzläufen; `params` = (m, n, rate_scale, similarity, jitter, noise, n_samples, seed)."""
    m, n, rate_scale, similarity, jitter, noise, n_samples, seed = params
    return analyse(make_dataset(m, n, rate_scale, similarity, jitter, noise, n_samples, seed), settings, with_comparators, (rate_scale, similarity, jitter, noise))


def oracle_accuracy(ds, sorting, result):
    """Was k-means erreichen würde, wenn man Kollisions-Spikes vorher aussortieren könnte (Orakel: benutzt die wahre Kollisions-Markierung): Genauigkeit auf den übrigen Spikes."""
    m = ds.n_neurons
    keep = (result.truth_of_spike >= 0) & ~result.collision_of_spike
    if keep.sum() < m + 2 or sorting.features.values.shape[1] == 0:
        return float("nan")
    F = sorting.features.values[keep]
    y = result.truth_of_spike[keep]
    km = alg.kmeans(F, m, seed=1)
    conf = np.zeros((m, m))
    for lab, yy in zip(km.labels, y):
        conf[yy, lab] += 1
    idx = assign(conf)
    cmap = {c: i for i, c in enumerate(idx) if c >= 0}
    return float(np.mean([cmap.get(int(l), -1) == yy for l, yy in zip(km.labels, y)]))


# --- Sweeps und Tabellen ---------------------------------------------------------------------------------------------------------------------

SWEEP_VALUES = {
    "n_electrodes": (1, 2, 3, 4, 6, 8),
    "rate_scale": (0.25, 0.5, 1.0, 2.0, 3.0, 4.0),
    "noise": (0.0, 0.2, 0.4, 0.7, 1.0, 2.0),
    "similarity": (0.0, 0.25, 0.5, 0.75, 1.0),
    "jitter": (0.0, 0.05, 0.1, 0.2, 0.3),
    "threshold": (2.0, 3.0, 4.5, 6.0, 8.0),
    "n_components": (1, 2, 3, 5, 8),
    "n_neurons": (2, 3, 4, 5),
}
SWEEP_LABELS = {"n_electrodes": "Anzahl Elektroden", "rate_scale": "Feuerrate (Faktor)", "noise": "Rauschen (relativ zum Neuronen-Signal)", "similarity": "Wellenform-Ähnlichkeit (0 = alle gleich, 1 = wie gewohnt)",
                "jitter": "Amplitudenschwankung", "threshold": "Schwelle (Rausch-Standardabweichungen)", "n_components": "Zahl der PCA-Komponenten", "n_neurons": "Anzahl Neuronen"}
_DATA_KEYWORD = {"n_electrodes": "n", "rate_scale": "rate_scale", "noise": "noise", "similarity": "similarity", "jitter": "jitter", "n_neurons": "m"}
SWEEP_KEYS = ("f1", "accuracy", "recall", "precision", "collision_share")
COMPARATORS = ("ica", "sobi", "sca")


def _summarise(x, per_seed):
    row = {"x": x}
    for key in per_seed[0]:
        arr = np.array([r[key] for r in per_seed], dtype=float)
        row[key] = float(np.nanmean(arr)) if not np.isnan(arr).all() else float("nan")
        row[key + "_std"] = float(np.nanstd(arr)) if not np.isnan(arr).all() else float("nan")
    return row


def _record(a):
    r = a.result
    out = {"f1": r.f1, "accuracy": r.accuracy, "recall": r.recall, "precision": r.precision, "collision_share": r.collision_share}
    for name in COMPARATORS:
        out[name] = a.comparators[name]["f1"] if a.comparators else float("nan")
    return out


def sweep(parameter, values=None, settings=Settings(), **base):
    """Mittel und Streuung (über die festen Sweep-Datensätze) der Pipeline-Kennzahlen und des Spitzen-F1 von ICA, SOBI und SCA in Abhängigkeit von einem Regler (Schwelle und Komponentenzahl betreffen nur die Pipeline)."""
    values = SWEEP_VALUES[parameter] if values is None else values
    rows = []
    for x in values:
        per_seed = []
        for seed in C.SWEEP_SEEDS:
            if parameter in _DATA_KEYWORD:
                kw = dict(base)
                kw[_DATA_KEYWORD[parameter]] = x
                a = analyse(make_dataset(seed=seed, **kw), settings)
            else:
                s = Settings(**{**settings.__dict__, parameter: x})
                a = analyse(make_dataset(seed=seed, **base), s, with_comparators=False)
            per_seed.append(_record(a))
        rows.append(_summarise(x, per_seed))
    return rows


def feature_table(settings=Settings(), electrodes=(1, 2, 4), **base):
    """Merkmale im Vergleich (PCA, Rohwerte, Spitzenamplituden) für verschiedene Elektrodenzahlen: Sortiergenauigkeit und Spitzen-F1, Mittel über die Sweep-Datensätze."""
    rows = []
    for n in electrodes:
        for feature in C.FEATURES:
            acc, f1 = [], []
            for seed in C.SWEEP_SEEDS:
                a = analyse(make_dataset(seed=seed, **{**base, "n": n}), Settings(**{**settings.__dict__, "feature": feature}), with_comparators=False)
                acc.append(a.result.accuracy), f1.append(a.result.f1)
            rows.append({"n": n, "feature": feature, "accuracy": float(np.mean(acc)), "f1": float(np.mean(f1)), "accuracy_min": float(np.min(acc))})
    return rows


def k_selection(settings=Settings(), neurons=(2, 3, 4, 5), **base):
    """Silhouette-Wahl der Clusterzahl: je wahrer Neuronenzahl die gewählten k über die Sweep-Datensätze, Anteil richtiger Wahlen, mittlere Silhouette je k."""
    rows = []
    for m in neurons:
        chosen, curves = [], []
        for seed in C.SWEEP_SEEDS:
            ds = make_dataset(seed=seed, **{**base, "m": m})
            srt = alg.sort_spikes(ds.X, m, settings.threshold, settings.feature, settings.n_components, "silhouette", seed=settings.init_start)
            chosen.append(srt.k)
            curves.append(srt.silhouette_curve)
        ks = sorted({k for c in curves for k in c})
        rows.append({"m": m, "chosen": chosen, "correct": float(np.mean([k == m for k in chosen])), "curve": {k: float(np.mean([c[k] for c in curves if k in c])) for k in ks}})
    return rows


SCENES = (
    ("Genug Elektroden (4 Neuronen, 6 Elektroden)", dict(m=4, n=6)),
    ("Zwei Elektroden (4 Neuronen)", dict(m=4, n=2)),
    ("Eine Elektrode (4 Neuronen)", dict(m=4, n=1)),
    ("Fünf Neuronen, vier Elektroden", dict(m=5, n=4)),
    ("Hohe Feuerrate (Faktor 2, 4 Neuronen, 4 Elektroden)", dict(m=4, n=4, rate_scale=2.0)),
    ("Ähnliche Wellenformen (Ähnlichkeit 0, 4 Neuronen, 4 Elektroden)", dict(m=4, n=4, similarity=0.0)),
    ("Amplitudenschwankung (0.3, 4 Neuronen, 4 Elektroden)", dict(m=4, n=4, jitter=0.3)),
    ("Starkes Rauschen (Rauschen 2.0, 4 Neuronen, 4 Elektroden)", dict(m=4, n=4, noise=2.0)),
)


def scene_table(settings=Settings(), **base):
    """Pipeline gegen ICA, SOBI und SCA in acht Szenen (Spitzen-F1 der Neuronen und Sortiergenauigkeit der Pipeline), Mittel und Spanne über die Sweep-Datensätze."""
    rows = []
    for label, scene in SCENES:
        kw = dict(base)
        kw.update(scene)
        acc = {"pipeline": [], "accuracy": [], "ica": [], "sobi": [], "sca": []}
        for seed in C.SWEEP_SEEDS:
            a = analyse(make_dataset(seed=seed, **kw), settings)
            acc["pipeline"].append(a.result.f1)
            acc["accuracy"].append(a.result.accuracy)
            for name in COMPARATORS:
                acc[name].append(a.comparators[name]["f1"])
        row = {"scene": label}
        for name, vals in acc.items():
            row[name] = float(np.mean(vals))
            row[name + "_min"], row[name + "_max"] = float(np.min(vals)), float(np.max(vals))
        rows.append(row)
    return rows


# --- Urteil ------------------------------------------------------------------------------------------------------------------------------

VERDICT_DROP = 0.10               # Verlust des Pipeline-F1 gegenüber dem Referenzlauf, ab dem eine Ursache benannt wird
VERDICT_RECALL = 0.7              # Trefferquote darunter, wenn die verpassten Spikes keine Kollisionen sind: Schwelle zu hoch (oder zu viel Rauschen)
VERDICT_MISSED_COLLISIONS = 0.5   # Anteil verpasster Spikes, der Kollisionen sein muss, damit Überlappung (und nicht die Schwelle) als Grund für eine niedrige Trefferquote gilt
VERDICT_PRECISION = 0.9           # Genauigkeit der Detektion darunter: Schwelle zu niedrig
VERDICT_ACCURACY = 0.95           # Sortiergenauigkeit, ab der die Pipeline als gut gilt


def verdict(a):
    """(Art, Code, Kennzahlen). Nur mit großer Marge - kein Urteil nahe an einer Schwelle. Bei mehreren Ursachen gewinnt die mit dem größten Verlust."""
    ds, r = a.ds, a.result
    m = ds.n_neurons
    data = {"f1": r.f1, "accuracy": r.accuracy, "accuracy_single": r.accuracy_single, "accuracy_collision": r.accuracy_collision, "recall": r.recall, "precision": r.precision,
            "collision_share": r.collision_share, "k": a.sorting.k, "m": m, "n": ds.n_electrodes, "scatter_found": r.scatter_found, "scatter_true": r.scatter_true,
            "ref_low_rate": a.ref_low_rate, "ref_similar": a.ref_similar, "ref_no_jitter": a.ref_no_jitter, "ref_clean": a.ref_clean, "silhouette": a.silhouette,
            "majority": r.majority_baseline, **{f"{k}_f1": v["f1"] for k, v in a.comparators.items()}}
    if r.precision < VERDICT_PRECISION:
        return "warning", "false_positives", data
    if r.recall < VERDICT_RECALL and r.missed_collision_share < VERDICT_MISSED_COLLISIONS:
        return "warning", "no_detection", data
    if a.settings.cluster_mode == "silhouette" and a.sorting.k != m:
        return "warning", "wrong_k", data
    drops = {"overlap_high": a.ref_low_rate - r.f1, "similar_waveforms": a.ref_similar - r.f1, "jitter": a.ref_no_jitter - r.f1, "noise": a.ref_clean - r.f1}
    drops = {k: (v if not np.isnan(v) else 0.0) for k, v in drops.items()}
    cause, drop = max(drops.items(), key=lambda kv: kv[1])
    if drop > VERDICT_DROP:
        return "warning", cause, data
    if r.accuracy < VERDICT_ACCURACY:
        return "warning", "kmeans_misled" if r.scatter_found < r.scatter_true else "sorting_poor", data
    return "success", "pipeline_ok", data
