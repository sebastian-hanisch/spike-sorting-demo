"""Defaults, Slider-Grenzen und feste Szenario-Größen der Spike-Sorting-Demo. Das Szenario ist das der ica/sobi/sca-demo (Neuronen, Wellenformen, Feuerraten, Mischung wortgleich);
neu sind die Wellenform-Ähnlichkeit, die Amplitudenschwankung und die Pipeline-Regler."""

# --- Szenario (fest, wortgleich aus ica/sobi/sca-demo) ------------------------------------------------------------------------
SAMPLE_RATE = 10_000                       # Hz
WAVEFORM_LENGTH = 30                       # Abtastwerte je Spike
PEAK_INDEX = 8                             # Lage der negativen Spitze in der Wellenform
OVERSHOOT = 0.4                            # Höhe des positiven Nachschlags
REFRACTORY = 20                            # Abtastwerte (2 ms)
NEURON_SIGMAS = (2.0, 3.0, 2.5, 4.0, 3.5)          # Breite der Spitze je Neuron
SIGMA_CENTER = 3.0                         # Ähnlichkeit 0: alle Neuronen bekommen diese Breite
NEURON_RATES = (20.0, 28.0, 35.0, 24.0, 31.0)      # Feuerrate in Hz (bei Feuerraten-Faktor 1)
NEURON_AMPLITUDES = (1.0, 0.8, 1.2, 0.7, 0.9)      # Spitzenamplitude an der nächsten Elektrode
NEURON_POSITIONS = ((0.10, 0.30), (0.35, 0.20), (0.60, 0.35), (0.85, 0.25), (0.50, 0.55))   # Elektroden liegen bei y = 0, x in [0, 1]
DISTANCE_EPS = 0.05
ACTIVE_THRESHOLD = 0.1                     # ein Neuron gilt an einem Zeitpunkt als aktiv, wenn seine Wellenform dort mehr als 10 % der Spitzenhöhe erreicht

# --- Regler ---------------------------------------------------------------------------------------------------------------------
DEFAULT_N_NEURONS = 4
N_NEURONS_MIN, N_NEURONS_MAX = 2, 5
DEFAULT_N_ELECTRODES = 4
N_ELECTRODES_MIN, N_ELECTRODES_MAX = 1, 8
DEFAULT_RATE_SCALE = 1.0
RATE_SCALE_MIN, RATE_SCALE_MAX = 0.25, 4.0
DEFAULT_SIMILARITY = 1.0
SIMILARITY_MIN, SIMILARITY_MAX = 0.0, 1.0                                # 1 = Wellenformen wie gewohnt, 0 = alle Neuronen haben dieselbe Form
DEFAULT_JITTER = 0.0
JITTER_MIN, JITTER_MAX = 0.0, 0.3                                        # Streuung der Spitzenhöhe je Spike (relativ)
DEFAULT_NOISE = 0.05
NOISE_MIN, NOISE_MAX = 0.0, 1.0
DEFAULT_N_SAMPLES = 20_000
N_SAMPLES_MIN, N_SAMPLES_MAX = 5_000, 40_000
DEFAULT_THRESHOLD = 4.5
THRESHOLD_MIN, THRESHOLD_MAX = 2.0, 10.0                                 # in robusten Rausch-Standardabweichungen
FEATURES = ("pca", "raw", "amplitude")
FEATURE_LABELS = {"pca": "PCA (Hauptkomponenten der Ausschnitte)", "raw": "Rohwerte (alle Abtastwerte)", "amplitude": "Spitzenamplituden (ein Wert je Elektrode)"}
DEFAULT_FEATURE = "pca"
DEFAULT_N_COMPONENTS = 3
N_COMPONENTS_MIN, N_COMPONENTS_MAX = 1, 8
CLUSTER_MODES = ("known", "silhouette")
CLUSTER_MODE_LABELS = {"known": "bekannt (wie eingestellt)", "silhouette": "unbekannt (per Silhouette wählen)"}
DEFAULT_CLUSTER_MODE = "known"
K_MAX = 8                                                                # größte Clusterzahl bei der Silhouette-Wahl
CONTRASTS = ("logcosh", "exp", "cube")
CONTRAST_LABELS = {"logcosh": "log cosh (robust)", "exp": "Gauß-Ableitung (sehr robust)", "cube": "Kurtosis (u³)"}
DEFAULT_CONTRAST = "logcosh"
METHODS = ("symmetric", "deflation")
DEFAULT_METHOD = "symmetric"
INIT_STARTS = (1, 2, 3, 4, 5)
DEFAULT_INIT_START = 1
DEFAULT_SEED = 7

# --- Pipeline (fest) --------------------------------------------------------------------------------------------------------------
SNIPPET_BEFORE = 8                         # Abtastwerte vor dem Minimum
SNIPPET_AFTER = 22                         # ... und ab dem Minimum (Länge 30 = Wellenformlänge)
DEAD_TIME = 15                             # Abtastwerte Mindestabstand zweier erkannter Spitzen
MATCH_TOLERANCE = 6                        # erkannte und wahre Spitze gelten als dieselbe, wenn ihre Zeiten höchstens so weit auseinanderliegen
COLLISION_WINDOW = 12                      # ein wahrer Spike ist "Kollision", wenn ein anderer Spike höchstens so viele Abtastwerte entfernt beginnt (die Spitzen sind 4-8 Abtastwerte breit)
N_RESTARTS = 10                            # Neustarts des k-means
KMEANS_MAX_ITER = 100

# --- Vergleichsverfahren (aus ica/sobi/sca-demo) ---------------------------------------------------------------------------------
RECONSTRUCTIONS = ("l1", "single")
DEFAULT_RECONSTRUCTION = "l1"
MAX_ITER = 200                             # FastICA
TOL = 1e-6
JD_MAX_SWEEPS = 100                        # SOBI: Jacobi-Sweeps
JD_TOL = 1e-8
SOBI_LAGS = tuple(range(2, 21, 2))         # Verzögerungen des SOBI-Vergleichs

# --- Auswertung -----------------------------------------------------------------------------------------------------------------
SWEEP_SEEDS = (100000, 100001, 100002, 100003, 100004)            # feste Datensätze der Sweeps, getrennt vom Demo-Seed


# --- Presets ---------------------------------------------------------------------------------------------------------------------


def _preset(**kw):
    base = dict(m=DEFAULT_N_NEURONS, n=DEFAULT_N_ELECTRODES, rate_scale=DEFAULT_RATE_SCALE, similarity=DEFAULT_SIMILARITY, jitter=DEFAULT_JITTER, noise=DEFAULT_NOISE, n_samples=DEFAULT_N_SAMPLES,
                threshold=DEFAULT_THRESHOLD, feature=DEFAULT_FEATURE, n_components=DEFAULT_N_COMPONENTS, cluster_mode=DEFAULT_CLUSTER_MODE, contrast=DEFAULT_CONTRAST, init_start=DEFAULT_INIT_START,
                seed=DEFAULT_SEED)
    base.update(kw)
    return base


PRESETS = {
    "Vier Neuronen, vier Elektroden": _preset(),
    "Eine Elektrode": _preset(n=1),
    "Hohe Feuerrate": _preset(rate_scale=2.0),
    "Fünf Neuronen": _preset(m=5),
    "Rauschen und hohe Schwelle": _preset(noise=1.0, threshold=6.0),
    "Unbekannte Neuronenzahl": _preset(cluster_mode="silhouette"),
}
PRESET_HELP = {
    "Vier Neuronen, vier Elektroden": "Der Grundfall: 92 % der Spikes werden erkannt (die verpassten sind fast alle Kollisionen), 98 % davon richtig sortiert, Spitzen-F1 0.94. ICA, SOBI und SCA erreichen hier 1.0 - sie trennen "
                                      "die Signale und lösen damit auch überlappende Spikes auf, die die Pipeline verliert.",
    "Eine Elektrode": "Nur eine Elektrode: ICA, SOBI und SCA finden nichts (F1 0.16 / 0.16 / 0.05), die Pipeline arbeitet mit der Wellenform und geht trotzdem - aber mit Spitzen-F1 nur 0.51 und Sortiergenauigkeit 0.65. "
                      "Die Merkmale enthalten die Information (ein Klassifikator mit bekannten Neuronen käme auf 0.94), k-means findet sie nicht: es gibt der falschen Zuordnung die kleinere Streuung.",
    "Hohe Feuerrate": "Doppelte Feuerrate: jeder dritte Spike (34 %) überlappt mit einem anderen. Die Pipeline verpasst sie (Trefferquote 0.84) und die Mischformen stören das Clustering: Spitzen-F1 0.77 gegen 0.99 bei einem Viertel der Feuerrate. "
                      "ICA, SOBI und SCA sind unbeeindruckt (1.0).",
    "Fünf Neuronen": "Fünf statt vier Neuronen: die Wellenformen liegen dichter, k-means teilt große Cluster und verschmilzt kleine: Sortiergenauigkeit 0.80, Spitzen-F1 0.75. SCA erreicht 0.99, ICA 0.77 (vier Elektroden für fünf Neuronen).",
    "Rauschen und hohe Schwelle": "Rauschen von 100 % des Neuronen-Signals und eine Schwelle von 6 Rausch-Standardabweichungen: die Schwelle liegt über vielen Spitzen, die Trefferquote fällt auf 0.56. Bei Schwelle 4.5 wären es 0.86 (F1 0.90). "
                                  "Zu niedrig ist die Schwelle aber auch nicht gut: bei 2 melden mehr als die Hälfte der 'Spitzen' nur Rauschen.",
    "Unbekannte Neuronenzahl": "Die Clusterzahl wird per Silhouette gewählt statt vorgegeben: sie wählt zu viele Cluster (bei vier Neuronen im Mittel 6.8) - große Cluster mit Kollisions-Ausreißern erscheinen als mehrere. "
                               "Die überzähligen Cluster gehören keinem Neuron: Spitzen-F1 0.93 statt 0.945, Sortiergenauigkeit 0.94 statt 0.98 - kleine Zahlenunterschiede, aber die Neuronenzahl ist falsch.",
}
# Bänder (Seed des Presets; Werte mit dem ausgelieferten Code kalibriert, bewusst weit): Trefferquote (recall), Sortiergenauigkeit (accuracy), Spitzen-F1 der Pipeline (f1), Urteil (verdict)
PRESET_EXPECTED_BANDS = {
    "Vier Neuronen, vier Elektroden": {"recall": (0.85, 0.97), "accuracy": (0.94, 1.0), "f1": (0.88, 0.98), "verdict": "pipeline_ok"},
    "Eine Elektrode": {"recall": (0.85, 0.97), "accuracy": (0.45, 0.85), "f1": (0.25, 0.75), "verdict": "kmeans_misled"},
    "Hohe Feuerrate": {"recall": (0.75, 0.9), "f1": (0.55, 0.92), "verdict": "overlap_high"},
    "Fünf Neuronen": {"accuracy": (0.6, 0.94), "f1": (0.55, 0.9), "verdict": "kmeans_misled"},
    "Rauschen und hohe Schwelle": {"recall": (0.35, 0.68), "f1": (0.3, 0.75), "verdict": "no_detection"},
    "Unbekannte Neuronenzahl": {"f1": (0.55, 0.96), "verdict": "wrong_k"},
}
