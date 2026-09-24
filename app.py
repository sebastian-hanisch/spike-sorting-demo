"""Spike-Sorting-Standardpipeline (Schwelle, Ausschnitte, PCA, Clustering) an Mehrelektroden-Signalen - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo EIN Verfahren - die Standardpipeline des Spike-Sortings - und lässt stattdessen das Beispiel wachsen.
Viertes Stück der Quellentrennung-Linie der "Konzepte"-Reihe und Einstieg in den Spike-Sorting-Zweig: die Pipeline trennt nicht Signale, sondern klassifiziert Ereignisse (Spikes) nach ihrer Wellenform.
Was das gegenüber ICA, SOBI und SCA bringt und wo es endet, wird hier gemessen. Siehe README für die Einordnung.

Lauffähig mit: streamlit run app.py
"""

import time

import numpy as np
import streamlit as st

import ss_constants as C
from ss_algorithm import pca_features
from ss_evaluation import (
    SWEEP_LABELS, Settings, analyse_for, feature_table, k_selection, make_dataset, match_detections, oracle_accuracy, scene_table, sweep, truth_spikes, verdict,
)
from ss_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    seed_widget,
    sync_query_params,
)
from ss_visualization import (
    CLUSTER_COLORS,
    build_confusion,
    build_detection,
    build_feature_bars,
    build_features,
    build_layout,
    build_method_bars,
    build_raster,
    build_scenes,
    build_silhouette,
    build_snippets,
    build_sweep,
    build_traces,
    source_color,
    source_labels,
)

st.set_page_config(page_title="Spike-Sorting – Sebastian Hanisch", layout="wide")

STEP_LABELS = {1: "1 · Signal", 2: "2 · Detektion", 3: "3 · Ausschnitte", 4: "4 · Merkmale", 5: "5 · Clustering"}
WINDOW_WIDTH_MS = 60
SWEEP_OPTIONS = {"n_electrodes": "Anzahl Elektroden", "rate_scale": "Feuerrate", "noise": "Rauschen", "similarity": "Wellenform-Ähnlichkeit", "jitter": "Amplitudenschwankung",
                 "threshold": "Schwelle", "n_components": "PCA-Komponenten", "n_neurons": "Anzahl Neuronen"}


@st.cache_data(show_spinner=False)
def _dataset(m, n, rate_scale, similarity, jitter, noise, n_samples, seed):
    return make_dataset(m, n, rate_scale, similarity, jitter, noise, n_samples, seed)


@st.cache_data(show_spinner=False)
def _analysis(data_params, settings):
    return analyse_for(data_params, settings)


@st.cache_data(show_spinner=False)
def _oracle(data_params, settings):
    ds = make_dataset(*data_params)
    a = analyse_for(data_params, settings, with_comparators=False)
    return oracle_accuracy(ds, a.sorting, a.result)


@st.cache_data(show_spinner=False)
def _sweep(parameter, base, settings):
    m, n, rate_scale, similarity, jitter, noise, n_samples = base
    settings = Settings(**{**settings.__dict__, "cluster_mode": "known"})                # Sweeps rechnen mit bekannter Neuronenzahl (die Silhouette-Wahl hat ihren eigenen Abschnitt)
    return sweep(parameter, settings=settings, m=m, n=n, rate_scale=rate_scale, similarity=similarity, jitter=jitter, noise=noise, n_samples=n_samples)


@st.cache_data(show_spinner=False)
def _scenes(base, settings):
    m, n, rate_scale, similarity, jitter, noise, n_samples = base
    settings = Settings(**{**settings.__dict__, "cluster_mode": "known"})
    return scene_table(settings, noise=noise, n_samples=n_samples)


@st.cache_data(show_spinner=False)
def _features_and_k(base, settings):
    m, n, rate_scale, similarity, jitter, noise, n_samples = base
    settings = Settings(**{**settings.__dict__, "cluster_mode": "known"})
    kw = dict(rate_scale=rate_scale, similarity=similarity, jitter=jitter, noise=noise, n_samples=n_samples)
    return feature_table(settings, m=m, **kw), k_selection(settings, n=n, **kw)


st.title("⚡ Spike-Sorting – die Standardpipeline")
st.markdown(
    """
ICA, SOBI und SCA - die ersten drei Stücke der Quellentrennung-Linie - trennen **Signale**: aus den Elektrodenspuren werden die Spuren der einzelnen Neuronen. Das Spike-Sorting geht einen anderen, älteren Weg und
**klassifiziert Ereignisse**: Eine **Schwelle** findet die Spitzen, ein kurzer **Ausschnitt** um jede Spitze wird zu einem **Merkmalsvektor** (meist die ersten **Hauptkomponenten** - die PCA aus der Dimensionsreduktion),
und ein **Clustering** (hier k-means) sortiert die Spikes in Neuronen. Jeder Spike bekommt eine Neuronen-Nummer - mehr nicht. Das braucht **keine Mindestzahl an Elektroden** (auch eine einzige genügt, wenn die Wellenformen verschieden sind),
kennt aber **keine Überlappung**: zwei gleichzeitige Spikes ergeben eine Mischform, die zu keinem Neuron gehört. Die Demo misst, was die Pipeline leistet, wo sie scheitert - und wie sie sich gegenüber den Signaltrennern schlägt.
Wie das Verfahren funktioniert, erklärt der aufgeklappte Abschnitt direkt darunter.
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren vergleichen, zeigt diese Demo - viertes Stück der Quellentrennung-Linie der \"Konzepte\"-Reihe, Einstieg in den Spike-Sorting-Zweig - **ein** Verfahren "
    "an einem wachsenden Beispiel. Das Array ist dasselbe wie in ICA-, SOBI- und SCA-Demo (ohne Laufzeitverzögerung und ohne Hintergrund); neu sind Wellenform-Ähnlichkeit und Amplitudenschwankung als Regler."
)

with st.expander("So funktioniert die Standardpipeline", expanded=True):
    st.markdown(
        """
1. **Detektion.** Aus den Elektrodenspuren wird ein Detektionssignal $d(t) = \\min_j x_j(t)/\\sigma_j$ gebildet: die negativste Auslenkung über alle Elektroden, gemessen in **Rausch-Standardabweichungen**
   ($\\sigma_j$ robust aus dem Median der Abweichungen geschätzt, weil Spitzen selten sind). Unterschreitet $d(t)$ die **Schwelle**, gibt es dort einen Spike; nach jedem Spike gilt eine Totzeit von 1.5 ms.
   Zu hohe Schwelle: Spikes gehen verloren. Zu niedrige: Rauschen wird zum "Spike".
2. **Ausschnitte.** Um das Minimum jedes Spikes wird ein Fenster von 3 ms über **alle** Elektroden geschnitten (30 Abtastwerte je Elektrode) und am Minimum ausgerichtet.
3. **Merkmale.** Die Ausschnitte haben $30 \\cdot n$ Werte. Die **PCA** fasst sie zu wenigen Hauptkomponenten zusammen (Standard 3); Alternativen: alle Rohwerte oder nur die Spitzenamplitude je Elektrode.
4. **Clustering.** **k-means** teilt die Merkmalsvektoren in $k$ Gruppen - eine je Neuron. Die Neuronenzahl $k$ ist hier bekannt oder wird per **Silhouette** geschätzt.
5. **Bewertung** (nur hier möglich, weil die Wahrheit bekannt ist): erkannte Spikes werden den wahren zugeordnet (Zeitfenster ±0.6 ms); Cluster und Neuronen werden optimal einander zugeordnet.

Was die Pipeline **verlangt**: verschiedene Wellenformen (Form oder Amplitudenverhältnis über die Elektroden) und Spikes, die selten überlappen. Was sie **nicht** verlangt: mindestens so viele Elektroden wie Neuronen, ein lineares Mischungsmodell oder
Unabhängigkeit.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_cols = st.columns(len(C.PRESETS))
for i, name in enumerate(C.PRESETS.keys()):
    with preset_cols[i]:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name])

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_neurons = st.slider(
        "Neuronen", *bounds("n_neurons_slider"), key="n_neurons_slider",
        help="Spitzenartige Quellen wie in den anderen Demos. Mehr Neuronen: dichtere Wellenformen und mehr Überlappung. Bei 4 Elektroden: Sortiergenauigkeit 1.00 (2 Neuronen), 0.995 (3), 0.98 (4), 0.80 (5).",
    )
    n_electrodes = st.slider(
        "Elektroden", *bounds("n_electrodes_slider"), key="n_electrodes_slider",
        help="Aufnahmestellen auf einer Zeile. Die Pipeline arbeitet auch mit einer: Spitzen-F1 0.51 (1 Elektrode), 0.95 (2), 0.89 (3), 0.95 (4-8). ICA, SOBI und SCA brauchen mehr: bei 2 Elektroden 0.36 / 0.35 / 0.97.",
    )
    rate_scale = st.slider(
        "Feuerrate (Faktor)", *bounds("rate_slider"), key="rate_slider", step=0.25,
        help="Faktor auf die Feuerraten (20-35 Hz). Mehr Feuern heißt mehr Kollisionen: Anteil der Spikes, die mit einem anderen überlappen, bei Faktor 0.25 / 1 / 2 / 4: 2 % / 16 % / 34 % / 57 %; "
             "Spitzen-F1 der Pipeline 0.99 / 0.95 / 0.77 / 0.59, ICA, SOBI und SCA bleiben bei 1.0.",
    )
    similarity = st.slider(
        "Wellenform-Ähnlichkeit", *bounds("similarity_slider"), key="similarity_slider", step=0.05,
        help="1 = die Neuronen haben verschieden breite Spitzen (wie in den anderen Demos), 0 = alle dieselbe Form; dann unterscheiden sie sich nur noch über ihre Amplitudenverhältnisse an den Elektroden. "
             "Sortiergenauigkeit im Mittel 0.98 (1) gegen 0.89 (0), von Datensatz zu Datensatz stark schwankend (0.73-0.99); mit einer Elektrode bleibt bei Ähnlichkeit 0 nur der Zufall (0.55).",
    )
    jitter = st.slider(
        "Amplitudenschwankung", *bounds("jitter_slider"), key="jitter_slider", step=0.05,
        help="Streuung der Spitzenhöhe von Spike zu Spike (relativ). Bis 0.1 ohne Wirkung, danach verschmieren die Cluster: Sortiergenauigkeit 0.98 (0) / 0.98 (0.1) / 0.95 (0.2) / 0.87 (0.3).",
    )
    noise = st.slider(
        "Rauschen", *bounds("noise_slider"), key="noise_slider", step=0.05,
        help="Sensorrauschen relativ zum Neuronen-Signal. Die Pipeline ist robust, weil sie über 30 Abtastwerte und alle Elektroden mittelt: Spitzen-F1 0.94 (0.2), 0.94 (0.4), 0.90 (1.0); bei 2.0 bricht die Detektion ein (Trefferquote 0.34). "
             "ICA und SOBI leiden mehr (bei 1.0: 0.43 / 0.38), SCA weniger (0.86).",
    )
    n_samples = st.slider(
        "Länge der Aufnahme", *bounds("n_samples_slider"), key="n_samples_slider", step=1000,
        help="Abtastwerte bei 10 kHz. Mehr Länge = mehr Spikes zum Clustern.",
    )
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)

    st.markdown("**Pipeline**")
    threshold = st.slider(
        "Schwelle", *bounds("threshold_slider"), key="threshold_slider", step=0.5,
        help="In Rausch-Standardabweichungen. Bei Rauschen 0.05 ist alles von 3 bis 10 gleich gut (Spitzen-F1 0.94-0.95); bei 2 melden mehr als die Hälfte der 'Spitzen' nur Rauschen (Genauigkeit der Detektion 0.49). "
             "Bei Rauschen 1.0 kostet eine Schwelle von 6 statt 4.5 fast die Hälfte der Spikes (Trefferquote 0.56 gegen 0.86).",
    )
    feature = st.selectbox(
        "Merkmale", C.FEATURES, key="feature_select", format_func=lambda f: C.FEATURE_LABELS[f],
        help="Was aus einem Ausschnitt an k-means geht. Mit 4 Elektroden liegt die PCA vorn (Sortiergenauigkeit 0.98 gegen 0.96 Amplituden und 0.95 Rohwerte); mit einer Elektrode sind Spitzenamplituden besser (0.78 gegen 0.65 PCA und 0.57 Rohwerte).",
    )
    if feature == "pca":
        seed_widget("components_slider")
        n_components = st.slider(
            "PCA-Komponenten", *bounds("components_slider"), key="components_slider",
            help="Wie viele Hauptkomponenten an k-means gehen. Schon eine genügt fast (Sortiergenauigkeit 0.96), 2-3 sind gut (0.98), 5-8 kaum besser (0.99).",
        )
        st.session_state["_components_kept"] = n_components
    else:
        n_components = int(st.session_state.get("_components_kept", C.DEFAULT_N_COMPONENTS))
    cluster_mode = st.selectbox(
        "Neuronenzahl", C.CLUSTER_MODES, key="cluster_mode_select", format_func=lambda c: C.CLUSTER_MODE_LABELS[c],
        help="Bekannt: k-means bekommt die wahre Neuronenzahl. Unbekannt: die Silhouette wählt k unter 2-8 - sie wählt in den Tests immer zu viele (bei 4 Neuronen im Mittel 6.8).",
    )
    init_start = st.selectbox(
        "Start (k-means und ICA)", C.INIT_STARTS, key="init_start_select", format_func=lambda i: f"Start {i}",
        help="Zufallsstart von k-means (10 Neustarts, das mit der kleinsten Streuung zählt) und der ICA. Auf diesen Daten ändert er kaum etwas: die Spitzen-F1 der fünf Starts liegen im Standardfall innerhalb von 0.01, mit einer Elektrode sind sie identisch (die Aufteilung ist dort bei jedem Start gleich schlecht).",
    )
    st.markdown("**ICA (zum Vergleich)**")
    contrast = st.selectbox(
        "Kontrastfunktion", C.CONTRASTS, key="contrast_select", format_func=lambda c: C.CONTRAST_LABELS[c],
        help="Wie die ICA Nicht-Gaußianität misst (siehe ICA-Demo). Für dieses Stück nebensächlich.",
    )

    st.button("🎲 Neue Aufnahme generieren", width="stretch", on_click=randomize_seed, help="Würfelt einen neuen Zufalls-Seed für Spikezeiten und Rauschen.")

sync_query_params({
    "n_neurons_slider": int(n_neurons), "n_electrodes_slider": int(n_electrodes), "rate_slider": float(rate_scale), "similarity_slider": float(similarity), "jitter_slider": float(jitter),
    "noise_slider": noise, "n_samples_slider": int(n_samples), "threshold_slider": float(threshold), "feature_select": feature, "components_slider": int(n_components),
    "cluster_mode_select": cluster_mode, "contrast_select": contrast, "init_start_select": int(init_start), "seed_input": int(seed),
})

data_params = (int(n_neurons), int(n_electrodes), float(rate_scale), float(round(similarity, 2)), float(round(jitter, 2)), float(noise), int(n_samples), int(seed))
settings = Settings(threshold=float(threshold), feature=feature, n_components=int(n_components), cluster_mode=cluster_mode, contrast=contrast, init_start=int(init_start))
with st.spinner("Sortiere die Spikes..."):
    ds = _dataset(*data_params)
    analysis = _analysis(data_params, settings)
sorting, result = analysis.sorting, analysis.result
m_n, n_el = ds.n_neurons, ds.n_electrodes
labels = source_labels(ds)
colors = [source_color(i) for i in range(m_n)]
level, code, vd = verdict(analysis)
data_key = data_params + (settings,)
truth_times, truth_neuron, truth_collision = truth_spikes(ds)
matched_mask = match_detections(sorting.all_times, truth_times) >= 0
has_spikes = len(sorting.times) >= 3

# --- Pipeline in Aktion ---------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Pipeline in Aktion")
if "ss_step" not in st.session_state or st.session_state.get("ss_step_owner") != data_key:
    st.session_state["ss_step"] = 1
    st.session_state["ss_step_owner"] = data_key
duration_ms = ds.S.shape[1] * 1000.0 / C.SAMPLE_RATE
max_start = int(duration_ms - WINDOW_WIDTH_MS)
if st.session_state.get("window_start", 0) > max_start:
    st.session_state["window_start"] = 0
step_col, play_col, win_col = st.columns([4, 2, 3])
with step_col:
    step = st.select_slider("Schritt", options=list(STEP_LABELS), key="ss_step", format_func=lambda s: STEP_LABELS[s])
with play_col:
    auto_play = st.button("▶️ Abspielen", width="stretch")
with win_col:
    window = st.slider(f"Zeitfenster ({WINDOW_WIDTH_MS} ms) ab [ms]", 0, max_start, key="window_start", step=10, help="Welchen Ausschnitt der Aufnahme die Zeitreihen zeigen.")
view_slot = st.empty()

if has_spikes:
    display = sorting.features
    if display.kind == "raw":
        display = pca_features(sorting.snippets, 2)                                   # 120 Rohwerte lassen sich nicht zeichnen: Darstellung über die ersten zwei Hauptachsen
    display_values = display.values
    centers = np.array([display_values[sorting.clustering.labels == j].mean(axis=0) for j in range(sorting.k) if (sorting.clustering.labels == j).any()])
    best_electrode = int(sorting.snippets.min(axis=2).mean(axis=0).argmin())
    axis_names = ("PC 1", "PC 2") if display.kind == "pca" else (("Amplitude Elektrode 1", "Amplitude Elektrode 2") if display.kind == "amplitude" else ("PC 1 (Darstellung)", "PC 2 (Darstellung)"))


def _render(current_step):
    with view_slot.container():
        if current_step == 1:
            c1, c2 = st.columns([3, 2])
            c1.markdown("**Die Elektrodensignale** (▼ = wahre Spitzen der Neuronen)")
            c1.plotly_chart(build_traces([f"E{j + 1}" for j in range(n_el)], list(ds.X), window, WINDOW_WIDTH_MS, normalise=False), width="stretch", key="step_electrodes")
            c2.markdown("**Ort von Neuronen und Elektroden**")
            c2.plotly_chart(build_layout(ds), width="stretch", key="step_layout")
            st.markdown("**Die wahren Neuronen** (unbekannt in der Praxis)")
            st.plotly_chart(build_traces(labels, list(ds.S), window, WINDOW_WIDTH_MS, colors, [ds.spike_times[i] for i in range(m_n)]), width="stretch", key="step_sources")
        elif current_step == 2:
            st.markdown("**Detektionssignal** (grün: erkannte Spitze, orange ×: verpasste wahre Spitze, rot: falsch erkannt)")
            st.plotly_chart(build_detection(sorting.detection, sorting.threshold, sorting.all_times, truth_times, matched_mask, window, WINDOW_WIDTH_MS), width="stretch", key="step_detection")
        elif current_step == 3:
            if has_spikes:
                st.markdown(f"**Alle {len(sorting.times)} Ausschnitte auf Elektrode {best_electrode + 1}** (fein: nach wahrem Neuron gefärbt, grau gepunktet: Kollisionen, rot gepunktet: falsch erkannt; fett: Mittel der nicht überlappenden Spikes je Neuron)")
                st.plotly_chart(build_snippets(sorting.snippets, result.truth_of_spike, result.collision_of_spike, best_electrode), width="stretch", key="step_snippets")
            else:
                st.info("Es wurden (fast) keine Spikes erkannt - die Schwelle liegt zu hoch oder das Rauschen ist zu stark.")
        elif current_step == 4:
            if has_spikes:
                st.markdown("**Merkmalsraum, gefärbt nach dem wahren Neuron** (Kreuze: Kollisions-Spikes, rot: falsch erkannt)")
                st.plotly_chart(build_features(display_values, result.truth_of_spike, result.collision_of_spike, axis_names[0], axis_names[1]), width="stretch", key="step_features")
            else:
                st.info("Ohne erkannte Spikes gibt es keine Merkmale.")
        else:
            if has_spikes:
                c1, c2 = st.columns([3, 2])
                c1.markdown("**Cluster von k-means** (Farbe = Cluster, Rauten = Zentren)")
                c1.plotly_chart(build_features(display_values, sorting.clustering.labels, result.collision_of_spike, axis_names[0], axis_names[1], centers, CLUSTER_COLORS), width="stretch", key="step_clusters")
                c2.markdown("**Verwechslungsmatrix** (Zeilen: wahre Neuronen, Spalten: Cluster)")
                c2.plotly_chart(build_confusion(result.confusion, result.cluster_of_neuron), width="stretch", key="step_confusion")
            else:
                st.info("Ohne erkannte Spikes gibt es nichts zu clustern.")


if auto_play:
    for s in STEP_LABELS:
        _render(s)
        time.sleep(1.2)
    step = 5
else:
    _render(step)

if step == 1:
    st.caption(f"{m_n} Neuronen feuern (insgesamt {result.n_true} Spikes); {n_el} Elektrode(n) messen Mischungen. Die Pipeline sieht nur die Elektrodenspuren - die wahren Neuronen und ihre Spitzen stehen nur zur Bewertung da. "
               f"Anteil der Spikes, die mit einem anderen überlappen (Beginn höchstens 1.2 ms auseinander): {result.collision_share:.0%}.")
elif step == 2:
    st.caption(f"Erkannt: {result.n_detected} Spitzen bei Schwelle {sorting.threshold:g}. Trefferquote {result.recall:.0%} der wahren Spikes, Genauigkeit der Detektion {result.precision:.0%}. "
               f"{'Die verpassten Spikes sind fast alle Kollisionen: zwei überlappende Spikes ergeben nur ein Minimum (Totzeit), der zweite geht verloren. ' if result.missed_collision_share > 0.5 else ''}"
               "Bei Schwelle und Rauschen entscheidet, wie tief die Spitzen unter dem Rauschen liegen.")
elif step == 3:
    st.caption("Jede Linie ist ein Spike auf dieser Elektrode, am Minimum ausgerichtet. Spikes desselben Neurons liegen übereinander (dicke Linie = Mittel); Kollisions-Spikes (grau) weichen ab: ihre Form ist die Summe zweier Neuronen. "
               "Bei Amplitudenschwankung verbreitern sich die Bündel, bei Wellenform-Ähnlichkeit 0 fallen sie zusammen.")
elif step == 4:
    st.caption(f"Jeder Spike ist ein Punkt: {display_values.shape[1] if has_spikes else 0} Merkmal(e), gezeigt werden die ersten zwei. Gute Merkmale zeigen pro Neuron eine kompakte Wolke. "
               f"{('Die PCA-Komponenten erklären ' + ', '.join(f'{v:.0%}' for v in sorting.features.explained[:3]) + ' der Varianz der Ausschnitte. ') if has_spikes and sorting.features.kind == 'pca' else ''}"
               "Kreuze (Kollisionen) liegen zwischen den Wolken oder weit außen.")
else:
    st.caption(f"k-means hat {sorting.k} Cluster gefunden, {'wie vorgegeben' if cluster_mode == 'known' else 'per Silhouette gewählt (wahr: ' + str(m_n) + ')'}. Die Zuordnung zu den Neuronen erfolgt optimal über die Verwechslungsmatrix (Diagonale = richtig). "
               f"Sortiergenauigkeit {result.accuracy:.0%} (nicht überlappende Spikes {result.accuracy_single:.0%}, Kollisionen {result.accuracy_collision:.0%}); ein Klassifikator, der immer das häufigste Neuron nennt, läge bei {result.majority_baseline:.0%}.")

st.markdown("---")

# --- Ergebnis --------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Was die Pipeline gefunden hat - und was ICA, SOBI und SCA auf denselben Daten schaffen")
st.caption(
    "Spitzen-F1: je Neuron die Spikes seines Clusters (bei der Pipeline) bzw. die Spitzen seiner geschätzten Spur (bei ICA, SOBI, SCA) gegen die wahren Spitzenzeiten (Toleranz ±4 Abtastwerte), gemittelt über die Neuronen - "
    "für alle Verfahren dieselbe Definition. Die Pipeline muss zusätzlich jeden Spike der richtigen Neuronen-Nummer zuordnen."
)
best_cmp = max(analysis.comparators.values(), key=lambda v: v["f1"])["f1"] if analysis.comparators else float("nan")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Trefferquote (Detektion)", f"{result.recall:.0%}", delta=f"Genauigkeit {result.precision:.0%}", delta_color="off", help="Anteil der wahren Spikes, die erkannt wurden; darunter der Anteil der erkannten Spitzen, die zu einem wahren Spike gehören.")
m2.metric("Sortiergenauigkeit", f"{result.accuracy:.0%}", delta=f"häufigstes Neuron: {result.majority_baseline:.0%}", delta_color="off", help="Anteil der erkannten wahren Spikes, die dem richtigen Neuron zugeordnet sind; zum Vergleich die Genauigkeit einer Zuordnung aller Spikes zum häufigsten Neuron.")
m3.metric("Spitzen-F1 (Pipeline)", f"{result.f1:.2f}", delta=f"{result.f1 - best_cmp:+.2f} ggü. bester Signaltrenner" if analysis.comparators else None, delta_color="normal", help="Mittlerer Spitzen-F1 der Neuronen; im Delta der Abstand zum besten von ICA, SOBI und SCA.")
m4.metric("Kollisionsanteil", f"{result.collision_share:.0%}", help="Anteil der wahren Spikes, die mit einem anderen Spike überlappen. Sie sind es, die die Pipeline verliert oder falsch sortiert.")

_t = vd
if code == "no_detection":
    st.warning(f"⚠️ Die Detektion verpasst zu viele Spikes: Trefferquote {_t['recall']:.0%}, und es sind nicht die Kollisionen. Bei diesem Rauschen liegen viele Spitzen unter der Schwelle {sorting.threshold:g} - niedrigere Schwelle probieren "
               f"(zu niedrig meldet Rauschen als Spike). Spitzen-F1 {_t['f1']:.2f}; SCA erreicht {_t.get('sca_f1', float('nan')):.2f}.")
elif code == "false_positives":
    st.warning(f"⚠️ Die Schwelle {sorting.threshold:g} ist zu niedrig: nur {_t['precision']:.0%} der erkannten Spitzen gehören zu einem wahren Spike, der Rest ist Rauschen - und Rauschen-Spikes bilden Cluster oder verschmutzen die der Neuronen.")
elif code == "wrong_k":
    st.warning(f"⚠️ Die Silhouette hat {_t['k']} Cluster gewählt, wahr sind {_t['m']}: sie bevorzugt viele kleine, kompakte Cluster (große Neuronen-Cluster mit Kollisions-Ausreißern wirken wie mehrere). Die überzähligen Cluster gehören keinem Neuron - "
               f"Spitzen-F1 {_t['f1']:.2f}, Sortiergenauigkeit der zugeordneten Spikes {_t['accuracy']:.0%}. In der Praxis wird zu viel geteilt und von Hand oder per Regel verschmolzen.")
elif code == "overlap_high":
    st.warning(f"⚠️ {_t['collision_share']:.0%} der Spikes überlappen mit einem anderen: die Pipeline verpasst sie (Trefferquote {_t['recall']:.0%}) und ihre Mischformen verzerren das Clustering. Spitzen-F1 {_t['f1']:.2f} gegen {_t['ref_low_rate']:.2f} bei einem Viertel "
               f"der Feuerrate. ICA, SOBI und SCA lösen Überlappung auf, weil sie die Signale trennen ({_t.get('ica_f1', float('nan')):.2f} / {_t.get('sobi_f1', float('nan')):.2f} / {_t.get('sca_f1', float('nan')):.2f}) - für die Pipeline gibt es den Vorlagenabgleich (nächstes Stück).")
elif code == "similar_waveforms":
    st.warning(f"⚠️ Die Neuronen haben ähnliche Wellenformen: Spitzen-F1 {_t['f1']:.2f} gegen {_t['ref_similar']:.2f} bei den gewohnten verschiedenen Formen. Cluster mit fast gleichen Zentren lassen sich nicht trennen; mehr Elektroden (Amplitudenverhältnisse) helfen.")
elif code == "jitter":
    st.warning(f"⚠️ Die Spitzenhöhe schwankt von Spike zu Spike: die Cluster verschmieren und berühren sich (Spitzen-F1 {_t['f1']:.2f} gegen {_t['ref_no_jitter']:.2f} ohne Schwankung).")
elif code == "noise":
    st.warning(f"⚠️ Das Rauschen kostet Spikes und verschmiert die Cluster: Spitzen-F1 {_t['f1']:.2f} gegen {_t['ref_clean']:.2f} bei Standard-Rauschen.")
elif code == "kmeans_misled":
    oracle = _oracle(data_params, settings)
    if np.isnan(oracle):
        oracle_text = ""
    elif oracle > result.accuracy + 0.05:
        oracle_text = f" Ohne Kollisions-Spikes (Orakel) läge k-means bei {oracle:.0%}: die überlappenden Spikes ziehen die Cluster auseinander."
    else:
        oracle_text = f" Auch ohne Kollisions-Spikes (Orakel) bliebe es bei {oracle:.0%}: das Problem sind die Cluster selbst, nicht die Überlappung."
    st.warning(f"⚠️ Die Sortiergenauigkeit ist nur {_t['accuracy']:.0%}, obwohl die Merkmale die Neuronen trennen könnten: k-means hat eine Aufteilung mit **kleinerer** Streuung innerhalb der Cluster gefunden ({_t['scatter_found']:.0f}) "
               f"als die wahre ({_t['scatter_true']:.0f}). Die Zielfunktion belohnt das Teilen großer und das Verschmelzen kleiner Cluster - das Verfahren optimiert hier das Falsche.{oracle_text}")
elif code == "sorting_poor":
    st.warning(f"⚠️ Sortiergenauigkeit nur {_t['accuracy']:.0%}.")
else:
    st.success(f"✅ Die Pipeline sortiert gut: {result.accuracy:.0%} der erkannten Spikes richtig, Trefferquote {result.recall:.0%}, Spitzen-F1 {result.f1:.2f}. Die fehlenden Prozente sind überwiegend Kollisionen "
               f"({result.collision_share:.0%} der Spikes überlappen); ICA, SOBI und SCA erreichen hier {_t.get('ica_f1', float('nan')):.2f} / {_t.get('sobi_f1', float('nan')):.2f} / {_t.get('sca_f1', float('nan')):.2f}, weil sie Überlappung auflösen.")

t1, t2 = st.columns(2)
with t1:
    st.markdown("**Raster: wahre Spikes (Striche) und ihre Sortierung (Punkte)**")
    st.plotly_chart(build_raster(truth_times, truth_neuron, sorting.times, result.neuron_of_spike, window, WINDOW_WIDTH_MS, m_n), width="stretch", key="res_raster")
with t2:
    st.markdown("**Spitzen-F1: Pipeline gegen ICA, SOBI und SCA**")
    st.plotly_chart(build_method_bars(result, analysis.comparators), width="stretch", key="res_bars")
st.caption("Punkte in der Zeile eines Neurons, ohne dass darüber ein Strich steht, sind falsch zugeordnet; Striche ohne Punkt darunter sind verpasste Spikes. ICA, SOBI und SCA stehen hier nur als Vergleich: sie trennen Signale und sortieren keine Ereignisse; ihr F1 kommt aus der Spitzenerkennung auf der geschätzten Neuronen-Spur.")

st.markdown("---")

# --- Sweeps ----------------------------------------------------------------------------------------------------------------------------

st.subheader("📐 Wie stark hängt das Ergebnis von Elektroden, Feuerrate, Rauschen, Wellenformen und Schwelle ab?")
sweep_options = [p for p in SWEEP_OPTIONS if p != "n_components" or feature == "pca"]
if st.session_state.get("sweep_select") not in sweep_options:
    st.session_state["sweep_select"] = sweep_options[0]
sweep_param = st.selectbox("Welcher Regler soll durchgefahren werden?", sweep_options, format_func=lambda p: SWEEP_OPTIONS[p], key="sweep_select")
current = {"n_electrodes": int(n_electrodes), "rate_scale": float(rate_scale), "noise": float(noise), "similarity": float(similarity), "jitter": float(jitter), "threshold": float(threshold),
           "n_components": int(n_components), "n_neurons": int(n_neurons)}[sweep_param]
with st.spinner("Rechne den Sweep über 5 feste Datensätze..."):
    rows = _sweep(sweep_param, data_params[:7], settings)
st.plotly_chart(build_sweep(rows, SWEEP_LABELS[sweep_param], current=current, with_comparators=sweep_param not in ("threshold", "n_components")), width="stretch", key="sweep_chart")
st.caption("Mittel und Streuung (Band) über 5 feste Sweep-Datensätze (getrennt vom Seed oben); alle anderen Regler wie in der Seitenleiste, aber mit bekannter Neuronenzahl. k-means ist nicht bei jedem Datensatz gleich gut - einzelne Datensätze schlagen stark aus, deshalb sind die Kurven nicht immer glatt. "
           "Schwelle und Komponentenzahl betreffen nur die Pipeline (keine Vergleichslinien).")

st.markdown("---")

# --- Merkmale und Clusterzahl --------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Merkmale und Neuronenzahl")
if st.button("Merkmale und Silhouette-Wahl vergleichen (dauert einige Sekunden)", key="features_start"):
    st.session_state["features_on"] = True
if st.session_state.get("features_on"):
    with st.spinner("Vergleiche 3 Merkmale × 3 Elektrodenzahlen und die Clusterzahl-Wahl über 5 Datensätze..."):
        feat_rows, k_rows = _features_and_k(data_params[:7], settings)
    st.markdown("**Merkmale im Vergleich** (4 Neuronen; Sortiergenauigkeit, Mittel über 5 Datensätze)")
    st.plotly_chart(build_feature_bars(feat_rows), width="stretch", key="feature_bars")
    st.markdown("**Silhouette-Wahl der Clusterzahl** (Elektrodenzahl wie in der Seitenleiste)")
    k_pick = st.selectbox("Wahre Neuronenzahl", [r["m"] for r in k_rows], index=min(2, len(k_rows) - 1), key="k_pick")
    k_row = [r for r in k_rows if r["m"] == k_pick][0]
    st.plotly_chart(build_silhouette(k_row["curve"], k_row["m"], int(max(k_row["curve"], key=k_row["curve"].get))), width="stretch", key="silhouette_chart")
    st.table({"Wahre Neuronenzahl": [r["m"] for r in k_rows], "gewählte k in den 5 Datensätzen": [", ".join(str(k) for k in r["chosen"]) for r in k_rows], "Anteil richtig": [f"{r['correct']:.0%}" for r in k_rows]})
    st.caption("Balken: mittlere Silhouette je Clusterzahl (grün: wahre Zahl, rot: gewählte). Die Silhouette belohnt kompakte Cluster - und teilt deshalb große Neuronen-Cluster mit Kollisions-Ausreißern in mehrere.")

st.markdown("---")

# --- Szenen ----------------------------------------------------------------------------------------------------------------------------

st.subheader("🧩 Wer sortiert was: acht Szenen im Vergleich")
if st.button("Acht Szenen vergleichen (dauert einige Sekunden)", key="scenes_start"):
    st.session_state["scenes_on"] = True
if st.session_state.get("scenes_on"):
    with st.spinner("Vergleiche 8 Szenen × 5 Datensätze × 4 Verfahren..."):
        scene_rows = _scenes(data_params[:7], settings)
    st.plotly_chart(build_scenes(scene_rows), width="stretch", key="scenes_chart")
    st.table({
        "Szene": [r["scene"] for r in scene_rows],
        "Pipeline F1": [f"{r['pipeline']:.2f} ({r['pipeline_min']:.2f}-{r['pipeline_max']:.2f})" for r in scene_rows],
        "Sortiergenauigkeit": [f"{r['accuracy']:.2f}" for r in scene_rows],
        "ICA": [f"{r['ica']:.2f}" for r in scene_rows],
        "SOBI": [f"{r['sobi']:.2f}" for r in scene_rows],
        "SCA": [f"{r['sca']:.2f}" for r in scene_rows],
    })
    st.caption("Jede Szene legt Neuronen- und Elektrodenzahl und ihre Besonderheit fest; Rauschen (sofern die Szene es nicht setzt), Länge und Pipeline-Einstellungen wie in der Seitenleiste, mit bekannter Neuronenzahl. Mittel und Spanne über 5 feste Datensätze.")

st.markdown("---")

# --- Grenzen -----------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden - und wer danach kommt")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Spikes überlappen selten** | Überlappende Spikes werden verpasst (Totzeit) oder als Mischform falsch sortiert (Preset "Hohe Feuerrate"). | **Vorlagenabgleich** (nächstes Stück des Zweigs): zerlegt die Aufnahme in Neuronen-Vorlagen und löst Überlappung auf; **ICA/SOBI/SCA**, wo das Mischungsmodell gilt |
| **verschiedene Wellenformen** | Ähnliche Formen verschmelzen zu einem Cluster; mit einer Elektrode bleibt oft nur die Amplitude. | Mehr Elektroden (Amplitudenverhältnisse), andere Merkmale |
| **k-means passt zu den Clustern** | Ungleich große oder gestreute Cluster und Ausreißer: k-means bevorzugt eine falsche Aufteilung (Preset "Eine Elektrode"). | Andere Clusterverfahren (Mischmodelle, dichtebasiert) - siehe die Clustering-Linie |
| **Neuronenzahl bekannt** | Silhouette wählt zu viele Cluster (Preset "Unbekannte Neuronenzahl"). | Übersplitten und Verschmelzen nach Regeln oder von Hand |
| **Schwelle passt zum Rauschen** | Zu hoch: Spikes gehen verloren; zu niedrig: Rauschen wird zum Spike (Preset "Rauschen und hohe Schwelle"). | Rauschangepasste Schwellen, Filter (hier nicht enthalten) |
| **Wellenform pro Neuron konstant** | Amplitudenschwankung und Drift verschmieren die Cluster. | Drift-Korrektur, Vorlagen mit Amplitudenmodell |
| **Wellenform trägt die Information** | Ein anderer Ansatz nutzt statt der Form die **Zeitverzögerungen** desselben Spikes über mehrere Elektroden (Verzögerungsgraph, Nachbarschaftsmengen, Clique-Überdeckungen) - in der Dissertation des Autors (Universität Rostock, 2017) untersucht. | **Verzögerungsgraph** (späteres Stück des Zweigs; dieses Stück misst nur die Standardpipeline) |
"""
)
st.caption("Die genannten Verfahren sind die nächsten Stücke der Quellentrennung-Linie; hier steht nur, welche Annahme sie jeweils lockern. Ein Leistungsvergleich mit dem Verzögerungsgraph-Ansatz wird in dieser Demo nicht behauptet.")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Modell.** Elektrodensignal $x(t) = \sum_i \sum_{\tau_{i,l}} a_i\, w_i(t - \tau_{i,l}) + \varepsilon(t)$: Neuron $i$ feuert zu den Zeiten $\tau_{i,l}$ mit der festen Wellenform $w_i$, gewichtet mit seiner Mischspalte $a_i \in \mathbb{R}^n$.
Gesucht: die Zuordnung jedes Spikes zu seinem Neuron.

**Detektion.** $\hat\sigma_j = \operatorname{median}|x_j - \tilde x_j| / 0.6745$ (robust), $d(t) = \min_j (x_j(t) - \tilde x_j)/\hat\sigma_j$. Spikes = Minima von $d$ unter $-\theta$ (Schwelle), tiefste zuerst, Mindestabstand 15 Abtastwerte (Totzeit).

**Ausschnitte.** $s_l = x(\,t_l - 8 : t_l + 22\,) \in \mathbb{R}^{n \times 30}$, flachgezogen zu $30\,n$ Werten.

**PCA.** $S = U \Sigma V^\top$ der zentrierten Ausschnittsmatrix; Merkmale $f_l = (s_l - \bar s) V_{1:p}$; Varianzanteil der $j$-ten Komponente $\sigma_j^2 / \sum \sigma^2$.

**k-means.** Minimiere $\sum_l \lVert f_l - \mu_{z_l} \rVert^2$ per Lloyd-Iteration, k-means++-Start, 10 Neustarts. **Silhouette:** $s_l = (b_l - a_l)/\max(a_l, b_l)$ mit mittlerem Abstand $a_l$ im eigenen und $b_l$ zum nächsten fremden Cluster
(bei mehr als 1200 Spikes auf einer festen Stichprobe); gewählt wird $k \in \{2, \dots, 8\}$ mit größter mittlerer Silhouette.

**Bewertung.** Erkannte und wahre Spitzen werden zugeordnet, wenn ihre Zeiten höchstens 6 Abtastwerte auseinanderliegen (jede wahre höchstens einmal). Cluster $\leftrightarrow$ Neuron: optimale Zuordnung auf der Verwechslungsmatrix (Bitmasken-DP).
**Sortiergenauigkeit** = richtig zugeordnete / erkannte wahre Spikes. **Kollision:** ein anderer Spike beginnt höchstens 12 Abtastwerte vor oder nach diesem. **Spitzen-F1** je Neuron aus den Zeiten seiner Cluster-Spikes (Toleranz ±4), gemittelt.

**Grenzen.** (1) Überlappende Spikes sind kein Cluster (die Mischform $w_i + w_j$). (2) k-means minimiert die Streuung, nicht den Fehler: bei ungleichen Clustern und Ausreißern hat die falsche Aufteilung oft die kleinere Streuung.
(3) Die Silhouette bevorzugt viele kompakte Cluster. (4) Die Neuronenzahl wird als bekannt angenommen (oder schlecht geschätzt). (5) Wellenformen werden als konstant angenommen.

Implementiert in `ss_algorithm.py` (Detektion, Ausschnitte, PCA, k-means, Silhouette), `ss_ica.py`, `ss_sobi.py` und `ss_sca.py` (Vergleichsverfahren, wortgleich aus den Vorgänger-Demos), `ss_scenario.py` (Mehrelektroden-Generator),
`ss_evaluation.py` (Zuordnung, Kennzahlen, Sweeps, Tabellen, Urteil).
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
