"""Plotly-Visualisierungen der Spike-Sorting-Demo: Elektrodenlayout, Signalspuren, Detektion, Ausschnitte, Merkmalsraum, Verwechslungsmatrix, Raster der sortierten Spikes, Kennzahlen-Balken,
Sweeps, Merkmals-Vergleich, Silhouette und Szenen-Vergleich. Alle Figuren laufen durch `lock_axes` (Touch-Scrolling-Konvention des Portfolios)."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import ss_constants as C
from ss_scenario import electrode_positions

BLUE, ORANGE, GREEN, RED, GRAY, PURPLE = "#1f77b4", "#d68a2e", "#2ca02c", "#d62728", "#8a8f98", "#8e5fbf"
NEURON_COLORS = ("#1f77b4", "#d68a2e", "#2ca02c", "#8e5fbf", "#c2185b")
CLUSTER_COLORS = NEURON_COLORS + ("#00838f", "#6d4c41", "#455a64", "#9e9d24")
METHOD_COLORS = {"pipeline": BLUE, "ica": ORANGE, "sobi": PURPLE, "sca": GREEN}
METHOD_NAMES = {"pipeline": "Pipeline (Schwelle + PCA + k-means)", "ica": "ICA", "sobi": "SOBI", "sca": "SCA"}


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def source_color(i):
    return NEURON_COLORS[i % len(NEURON_COLORS)]


def source_labels(ds):
    return [f"Neuron {i + 1}" for i in range(ds.n_neurons)]


def build_layout(ds):
    """Elektroden (Quadrate auf y = 0) und Neuronen (Kreise, Größe = Spitzenamplitude)."""
    pos = electrode_positions(ds.n_electrodes)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pos[:, 0], y=pos[:, 1], mode="markers+text", text=[f"E{j + 1}" for j in range(ds.n_electrodes)], textposition="bottom center", name="Elektroden",
                             marker=dict(symbol="square", size=14, color=GRAY), hoverinfo="skip"))
    for i in range(ds.n_neurons):
        x, y = C.NEURON_POSITIONS[i]
        fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers+text", text=[f"N{i + 1}"], textposition="top center", name=f"Neuron {i + 1}", hoverinfo="skip",
                                 marker=dict(size=10 + 14 * C.NEURON_AMPLITUDES[i], color=source_color(i), opacity=0.85)))
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(range=[-0.1, 1.1], title="Ort (willkürliche Einheit)", zeroline=False),
                      yaxis=dict(range=[-0.15, 0.7], title="Abstand", zeroline=False), showlegend=False)
    return lock_axes(fig)


def build_traces(labels, arrays, t0, width, colors=None, spike_times=None, height=None, normalise=True):
    """Gestapelte Spuren eines Zeitfensters [t0, t0 + width) in ms (Abtastrate 10 kHz); optional Markierungen der Spitzen je Zeile."""
    fs = C.SAMPLE_RATE
    lo, hi = int(t0 * fs / 1000), int((t0 + width) * fs / 1000)
    fig = go.Figure()
    n = len(arrays)
    scale_all = max(float(np.abs(a).max()) for a in arrays) if not normalise else None
    for r, (label, y) in enumerate(zip(labels, arrays)):
        seg = y[lo:hi]
        scale = float(np.abs(y).max()) if normalise else scale_all
        offset = (n - 1 - r) * 1.3
        color = colors[r] if colors else BLUE
        fig.add_trace(go.Scatter(x=np.arange(lo, hi) * 1000.0 / fs, y=offset + seg / max(scale, 1e-12), mode="lines", line=dict(color=color, width=1.2), name=label, hoverinfo="skip"))
        if spike_times is not None and r < len(spike_times):
            marks = [t for t in spike_times[r] if lo <= t < hi]
            if marks:
                fig.add_trace(go.Scatter(x=np.array(marks) * 1000.0 / fs, y=[offset + 0.75] * len(marks), mode="markers", marker=dict(symbol="triangle-down", size=7, color=color), hoverinfo="skip", showlegend=False))
    fig.update_layout(height=height or max(180, 42 * n + 60), margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                      xaxis=dict(title="Zeit [ms]"), yaxis=dict(tickmode="array", tickvals=[(n - 1 - r) * 1.3 for r in range(n)], ticktext=list(labels), zeroline=False))
    return lock_axes(fig)


def build_detection(d, threshold, detected, truth_times, matched_mask, t0, width):
    """Detektionssignal d(t) (negativste Auslenkung über alle Elektroden in Rausch-Standardabweichungen) mit Schwelle: grün = erkannte wahre Spitze, rot = falsch erkannt, orange x = verpasste wahre Spitze."""
    fs = C.SAMPLE_RATE
    lo, hi = int(t0 * fs / 1000), int((t0 + width) * fs / 1000)
    x = np.arange(lo, hi) * 1000.0 / fs
    fig = go.Figure(go.Scatter(x=x, y=d[lo:hi], mode="lines", line=dict(color=GRAY, width=1.2), hoverinfo="skip", showlegend=False))
    fig.add_hline(y=-threshold, line=dict(color=RED, dash="dash", width=1.5), annotation_text=f"Schwelle −{threshold:g}", annotation_position="top left")
    inside = (detected >= lo) & (detected < hi)
    good, bad = inside & matched_mask, inside & ~matched_mask
    for sel, color, name in ((good, GREEN, "erkannt"), (bad, RED, "falsch erkannt")):
        if sel.any():
            fig.add_trace(go.Scatter(x=detected[sel] * 1000.0 / fs, y=d[detected[sel]], mode="markers", marker=dict(color=color, size=8), name=name, hoverinfo="skip"))
    found = np.zeros(len(truth_times), bool)
    for t in detected[matched_mask]:
        found[np.argmin(np.abs(truth_times - t))] = True
    missed = (truth_times >= lo) & (truth_times < hi) & ~found
    if missed.any():
        fig.add_trace(go.Scatter(x=truth_times[missed] * 1000.0 / fs, y=d[truth_times[missed]], mode="markers", marker=dict(color=ORANGE, size=9, symbol="x"), name="verpasst", hoverinfo="skip"))
    floor = max(float(d[lo:hi].min()), -(4.0 * threshold + 20.0)) - 3.0                    # bei geringem Rauschen ragen die Spitzen weit unter die Schwelle: unten abschneiden, damit die Schwelle sichtbar bleibt
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="Zeit [ms]"), yaxis=dict(title="d(t) [Rausch-σ]", range=[floor, 8]), legend=dict(orientation="h", y=-0.25))
    return lock_axes(fig)


def build_snippets(snips, truth, collision, electrode, max_lines=250):
    """Alle Ausschnitte einer Elektrode übereinander (fein, nach wahrem Neuron gefärbt; Kollisionen grau gestrichelt) und der Mittelwert je Neuron (fett)."""
    L = snips.shape[2]
    t = (np.arange(L) - C.SNIPPET_BEFORE) * 1000.0 / C.SAMPLE_RATE
    fig = go.Figure()
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(snips))[:max_lines]
    for i in idx:
        if truth[i] < 0:
            color, dash, op = RED, "dot", 0.35
        elif collision[i]:
            color, dash, op = GRAY, "dot", 0.35
        else:
            color, dash, op = source_color(int(truth[i])), "solid", 0.2
        fig.add_trace(go.Scatter(x=t, y=snips[i, electrode], mode="lines", line=dict(color=color, width=1, dash=dash), opacity=op, hoverinfo="skip", showlegend=False))
    for j in sorted(set(int(v) for v in truth if v >= 0)):
        sel = (truth == j) & ~collision
        if sel.any():
            fig.add_trace(go.Scatter(x=t, y=snips[sel, electrode].mean(axis=0), mode="lines", line=dict(color=source_color(j), width=3.5), name=f"Neuron {j + 1}", hoverinfo="skip"))
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="Zeit relativ zum Minimum [ms]"), yaxis=dict(title=f"Elektrode {electrode + 1}"), legend=dict(orientation="h", y=-0.2))
    return lock_axes(fig)


def build_features(F, color_index, collision, title_x="Merkmal 1", title_y="Merkmal 2", centers=None, palette=None):
    """Merkmalsraum (erste zwei Merkmale; bei nur einem Merkmal ein Streifendiagramm): Punkt-Farbe = `color_index` (wahres Neuron oder Cluster; -1 = falsch erkannt, rot); Kollisions-Spikes als Kreuze; optional Cluster-Zentren."""
    palette = palette or NEURON_COLORS
    one_d = F.shape[1] < 2
    x = F[:, 0]
    y = np.random.default_rng(0).uniform(-1, 1, len(F)) if one_d else F[:, 1]
    fig = go.Figure()
    for j in sorted(set(int(v) for v in color_index)):
        for is_coll in (False, True):
            sel = (color_index == j) & (collision == is_coll)
            if sel.any():
                color = RED if j < 0 else palette[j % len(palette)]
                fig.add_trace(go.Scattergl(x=x[sel], y=y[sel], mode="markers", marker=dict(size=6 if is_coll else 5, color=color, symbol="x" if is_coll else "circle", opacity=0.7), hoverinfo="skip", showlegend=False))
    if centers is not None and len(centers):
        cy = np.zeros(len(centers)) if one_d else centers[:, 1]
        fig.add_trace(go.Scatter(x=centers[:, 0], y=cy, mode="markers", marker=dict(size=13, color="black", symbol="diamond-open", line=dict(width=2)), hoverinfo="skip", showlegend=False))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title=title_x), yaxis=dict(title="(zufällig gestreut)" if one_d else title_y, showticklabels=not one_d), showlegend=False)
    return lock_axes(fig)


def build_confusion(confusion, cluster_of_neuron):
    """Verwechslungsmatrix: Zeilen = wahre Neuronen, Spalten = gefundene Cluster (nach der optimalen Zuordnung sortiert, nicht zugeordnete Cluster hinten); die Diagonale ist die richtige Sortierung."""
    m, k = confusion.shape
    order = [c for c in cluster_of_neuron if c >= 0] + [c for c in range(k) if c not in cluster_of_neuron]
    Z = confusion[:, order]
    labels = [f"Cluster {c + 1}" for c in order]
    fig = go.Figure(go.Heatmap(z=Z, x=labels, y=[f"Neuron {i + 1}" for i in range(m)], colorscale="Blues", text=Z.astype(int), texttemplate="%{text}", showscale=False, hoverinfo="skip"))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=60 + 42 * m, margin=dict(l=10, r=10, t=10, b=10))
    return lock_axes(fig)


def build_raster(truth_times, truth_neuron, detected, neuron_of_spike, t0, width, n_neurons):
    """Spikes im Zeitfenster: je Neuron eine Zeile mit den wahren Spikes (oben, Strich) und den ihm zugeordneten erkannten Spikes (unten, Punkt); falsch zugeordnete Spikes liegen in der Zeile des falschen Neurons."""
    fs = C.SAMPLE_RATE
    lo, hi = int(t0 * fs / 1000), int((t0 + width) * fs / 1000)
    fig = go.Figure()
    for i in range(n_neurons):
        base = (n_neurons - 1 - i) * 1.0
        tsel = truth_times[(truth_neuron == i) & (truth_times >= lo) & (truth_times < hi)]
        dsel = detected[(neuron_of_spike == i) & (detected >= lo) & (detected < hi)]
        if len(tsel):
            fig.add_trace(go.Scatter(x=tsel * 1000.0 / fs, y=np.full(len(tsel), base + 0.25), mode="markers", marker=dict(symbol="line-ns-open", size=14, color=source_color(i), line=dict(width=2)), hoverinfo="skip", showlegend=False))
        if len(dsel):
            fig.add_trace(go.Scatter(x=dsel * 1000.0 / fs, y=np.full(len(dsel), base - 0.05), mode="markers", marker=dict(symbol="circle", size=8, color=source_color(i)), hoverinfo="skip", showlegend=False))
    fig.update_layout(height=60 + 60 * n_neurons, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="Zeit [ms]", range=[lo * 1000.0 / fs, hi * 1000.0 / fs]),
                      yaxis=dict(tickmode="array", tickvals=[(n_neurons - 1 - i) * 1.0 + 0.1 for i in range(n_neurons)], ticktext=[f"Neuron {i + 1}" for i in range(n_neurons)], zeroline=False))
    return lock_axes(fig)


def build_method_bars(result, comparators):
    """Spitzen-F1 der Pipeline und (falls berechnet) von ICA, SOBI und SCA; dazu die Sortiergenauigkeit der Pipeline."""
    names = ["pipeline"] + [n for n in ("ica", "sobi", "sca") if n in comparators]
    f1 = [result.f1] + [comparators[n]["f1"] for n in names[1:]]
    fig = go.Figure(go.Bar(x=[METHOD_NAMES[n].split(" (")[0] for n in names], y=f1, marker_color=[METHOD_COLORS[n] for n in names], text=[f"{v:.2f}" for v in f1], textposition="outside", hoverinfo="skip"))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(title="Spitzen-F1 der Neuronen", range=[0, 1.15]), showlegend=False)
    return lock_axes(fig)


def build_sweep(rows, xlabel, current=None, log=False, with_comparators=True):
    """Links: Spitzen-F1 der Pipeline (mit Streuung) und der Vergleichsverfahren, Sortiergenauigkeit gestrichelt; rechts: Trefferquote, Genauigkeit der Detektion und Kollisionsanteil."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Spitzen-F1 und Sortiergenauigkeit", "Detektion und Kollisionen"), horizontal_spacing=0.12)
    xs = [r["x"] for r in rows]
    y, sd = np.array([r["f1"] for r in rows]), np.array([r["f1_std"] for r in rows])
    fig.add_trace(go.Scatter(x=xs + xs[::-1], y=list(y + sd) + list(y - sd)[::-1], fill="toself", fillcolor=BLUE, opacity=0.15, line=dict(width=0), hoverinfo="skip", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y=y, mode="lines+markers", name="Pipeline F1", line=dict(color=BLUE, width=2), hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y=[r["accuracy"] for r in rows], mode="lines+markers", name="Pipeline Genauigkeit", line=dict(color=BLUE, width=1.5, dash="dash"), hoverinfo="skip"), row=1, col=1)
    if with_comparators:
        for name in ("ica", "sobi", "sca"):
            fig.add_trace(go.Scatter(x=xs, y=[r[name] for r in rows], mode="lines+markers", name=METHOD_NAMES[name], line=dict(color=METHOD_COLORS[name], width=1.8), hoverinfo="skip"), row=1, col=1)
    for key, name, color, dash in (("recall", "Trefferquote", GREEN, "solid"), ("precision", "Genauigkeit der Detektion", RED, "solid"), ("collision_share", "Kollisionsanteil", GRAY, "dot")):
        fig.add_trace(go.Scatter(x=xs, y=[r[key] for r in rows], mode="lines+markers", name=name, line=dict(color=color, width=2, dash=dash), hoverinfo="skip"), row=1, col=2)
    fig.update_xaxes(title=xlabel, type="log" if log else "linear")
    fig.update_yaxes(range=[0, 1.05])
    if current is not None:
        for col in (1, 2):
            fig.add_vline(x=current, line=dict(color=RED, dash="dash"), row=1, col=col)
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.3))
    return lock_axes(fig)


def build_feature_bars(rows):
    """Sortiergenauigkeit je Merkmal und Elektrodenzahl (Mittel über die Sweep-Datensätze)."""
    fig = go.Figure()
    for feature, color in zip(C.FEATURES, (BLUE, ORANGE, GREEN)):
        sel = [r for r in rows if r["feature"] == feature]
        fig.add_trace(go.Bar(x=[f"{r['n']} Elektrode(n)" for r in sel], y=[r["accuracy"] for r in sel], name=C.FEATURE_LABELS[feature].split(" (")[0], marker_color=color,
                             text=[f"{r['accuracy']:.2f}" for r in sel], textposition="outside", hoverinfo="skip"))
    fig.update_layout(height=320, barmode="group", margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(title="Sortiergenauigkeit", range=[0, 1.15]), legend=dict(orientation="h", y=-0.2))
    return lock_axes(fig)


def build_silhouette(curve, true_k, chosen_k):
    """Mittlere Silhouette je Clusterzahl k; grün = wahre Neuronenzahl, rot = gewählte."""
    ks = sorted(curve)
    colors = [GREEN if k == true_k else (RED if k == chosen_k else BLUE) for k in ks]
    fig = go.Figure(go.Bar(x=[str(k) for k in ks], y=[curve[k] for k in ks], marker_color=colors, text=[f"{curve[k]:.2f}" for k in ks], textposition="outside", hoverinfo="skip"))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="Clusterzahl k"), yaxis=dict(title="mittlere Silhouette", range=[0, 1.1]), showlegend=False)
    return lock_axes(fig)


def build_scenes(rows):
    """Pipeline gegen ICA, SOBI und SCA: Spitzen-F1 je Szene; Balken = Mittel, Fehlerbalken = Spanne über die Sweep-Datensätze."""
    labels = [r["scene"].replace(" (", "<br>(") for r in rows]
    fig = go.Figure()
    for name in ("pipeline", "ica", "sobi", "sca"):
        y = [r[name] for r in rows]
        fig.add_trace(go.Bar(x=labels, y=y, name=METHOD_NAMES[name].split(" (")[0], marker_color=METHOD_COLORS[name], text=[f"{v:.2f}" for v in y], textposition="outside", hoverinfo="skip",
                             error_y=dict(type="data", symmetric=False, array=[r[name + "_max"] - r[name] for r in rows], arrayminus=[r[name] - r[name + "_min"] for r in rows])))
    fig.update_layout(height=440, barmode="group", margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(title="Spitzen-F1 der Neuronen", range=[0, 1.2]), legend=dict(orientation="h", y=-0.5))
    return lock_axes(fig)
