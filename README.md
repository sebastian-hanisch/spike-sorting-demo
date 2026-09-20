# Spike-Sorting – die Standardpipeline – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-spike-sorting-demo.streamlit.app/)**

Viertes Stück der **Quellentrennung-Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning" und **Einstieg in den Spike-Sorting-Zweig**:
anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo **ein** Verfahren – die **Standardpipeline des Spike-Sortings** (Schwelle → Ausschnitte → PCA-Merkmale → k-means) – an einem wachsenden Beispiel,
mit **ICA**, **SOBI** und **SCA** als Vergleich. Vehikel: dasselbe **Mehrelektroden-Array** wie in [ica-demo](../ica-demo), [sobi-demo](../sobi-demo) und [sca-demo](../sca-demo) (dort übernommen, per Test gegen eingefrorene Werte geprüft), ohne Laufzeitverzögerung und ohne Hintergrund;
neu sind die Regler **Wellenform-Ähnlichkeit** und **Amplitudenschwankung** sowie die Pipeline-Regler (Schwelle, Merkmale, PCA-Komponenten, bekannte oder per Silhouette gewählte Neuronenzahl).

**Einordnung in die Reihe (die Kanten des Graphen):** ICA, SOBI und SCA trennen **Signale**; die Pipeline **klassifiziert Ereignisse**: jeder gefundene Spike bekommt eine Neuronen-Nummer. Das braucht keine Mindestzahl an Elektroden (sie arbeitet über die Wellenform) und kennt keine Überlappung.
Die Pipeline ist im Scoping die **Konvergenz von pca-demo und Clustering-Linie** (PCA als Merkmalsbildung, k-means als Clustering) und der Ausgangspunkt der weiteren Stücke des Zweigs (Vorlagenabgleich für überlappende Spikes; Verzögerungsgraph als anderer Ansatz).
```
ica-demo → sobi-demo → sca-demo
pca-demo + Clustering-Linie → spike-sorting-demo (Standardpipeline)
                              → Vorlagenabgleich (löst Überlappung auf)      [nicht gebaut]
                              → Verzögerungsgraph / Clique-Überdeckung      [nicht gebaut]
```

| Frage | Ergebnis (4 Neuronen, 4 Elektroden, Rauschen 0.05, 20000 Abtastwerte; Mittel über 5 feste Datensätze, Seeds 100000–100004) |
|---|---|
| Grundfall | ✅ **92 %** der wahren Spikes erkannt (Genauigkeit der Detektion 100 %), **98 %** davon richtig sortiert, Spitzen-F1 **0.94**. Die verpassten Spikes sind fast alle **Kollisionen** (Totzeit). ICA, SOBI und SCA erreichen hier 1.0 – sie lösen Überlappung auf |
| Elektroden | ✅ Spitzen-F1 0.51 (1), **0.95 (2)**, 0.89 (3), 0.95 (4–8); ICA / SOBI / SCA bei 2 Elektroden **0.36 / 0.35 / 0.97**, bei einer 0.16 / 0.16 / 0.05 |
| Eine Elektrode | ⚠️ Sortiergenauigkeit **0.65**: die Merkmale enthalten die Information (ein Klassifikator mit bekannten Neuronen: **0.94**), k-means findet sie nicht – die falsche Aufteilung hat in allen Datensätzen die **kleinere Streuung** als die wahre |
| Feuerrate | ⚠️ Kollisionsanteil 2 / 16 / 34 / 57 % bei Faktor 0.25 / 1 / 2 / 4; Spitzen-F1 der Pipeline **0.99 / 0.95 / 0.77 / 0.59**, ICA/SOBI/SCA bleiben bei 1.0. Ohne Kollisions-Spikes läge k-means bei doppelter Feuerrate im Mittel > 5 Punkte höher (Orakel) |
| Fünf Neuronen (4 Elektroden) | ⚠️ Sortiergenauigkeit 0.80, Spitzen-F1 0.75; SCA 0.99, ICA 0.77 |
| Merkmale (4 Elektroden) | PCA 0.98 vor Spitzenamplituden 0.96 und Rohwerten 0.95; **mit einer Elektrode** dreht sich die Reihenfolge: Amplituden **0.78**, PCA 0.65, Rohwerte 0.57 |
| PCA-Komponenten | eine 0.96, zwei bis drei 0.98, fünf bis acht 0.99 (Sortiergenauigkeit) |
| Rauschen | ✅ robust, weil über 30 Abtastwerte und alle Elektroden gemittelt wird: Spitzen-F1 0.94 (0.2), 0.94 (0.4), 0.90 (1.0); bei 2.0 bricht die Detektion ein (Trefferquote 0.34). ICA/SOBI bei 1.0: 0.43 / 0.38, SCA 0.86 |
| Schwelle | Rauschen 0.05: 3 bis 10 gleich gut (F1 0.94–0.95); **2 zu niedrig** (Genauigkeit der Detektion 0.49 – die Hälfte der "Spitzen" ist Rauschen); bei Rauschen 1.0 kostet Schwelle **6** statt 4.5 fast die Hälfte der Spikes (Trefferquote 0.56 gegen 0.86) |
| Wellenform-Ähnlichkeit | ⚠️ Sortiergenauigkeit 0.98 (verschiedene Breiten) gegen 0.89 (alle gleich), von Datensatz zu Datensatz stark schwankend; mit einer Elektrode bleibt bei Ähnlichkeit 0 nur 0.55 |
| Amplitudenschwankung | 0.98 (0) / 0.98 (0.1) / 0.95 (0.2) / **0.87 (0.3)** |
| Unbekannte Neuronenzahl | ❌ die **Silhouette wählt in allen 20 Fällen zu viele Cluster** (bei 4 Neuronen im Mittel 6.8, bei 5 immer das Maximum 8); Spitzen-F1 0.93 statt 0.945, Sortiergenauigkeit 0.94 statt 0.98 |
| Rechenzeit | ✅ 0.1 s für die Analyse im Grundfall; 0.7 s mit Silhouette-Wahl und vierfacher Feuerrate bei T = 40000 |

## Was die Demo zeigt

1. **Pipeline in Aktion** (Schritt-Slider + Abspielen, Zeitfenster-Regler): **Signal** (Elektrodenspuren, wahre Neuronen, Ort) → **Detektion** (d(t) mit Schwelle; grün erkannt, orange × verpasst, rot falsch erkannt) → **Ausschnitte** (alle Spikes einer Elektrode übereinander, nach wahrem Neuron gefärbt, Kollisionen grau, Mittel je Neuron) →
   **Merkmale** (Merkmalsraum nach wahrem Neuron gefärbt, Kollisionen als Kreuze) → **Clustering** (Cluster, Zentren, Verwechslungsmatrix).
2. **Was die Pipeline gefunden hat – und was ICA, SOBI und SCA auf denselben Daten schaffen:** Trefferquote, Sortiergenauigkeit (gegen die Genauigkeit "immer das häufigste Neuron"), Spitzen-F1 (für alle vier Verfahren dieselbe Definition), Kollisionsanteil; Raster der wahren und sortierten Spikes; Urteil
   (Codes: Schwelle zu niedrig → Schwelle zu hoch → falsche Clusterzahl → Überlappung / ähnliche Wellenformen / Amplitudenschwankung / Rauschen (größter Verlust gegen einen Referenzlauf) → **k-means optimiert das Falsche** (mit dem Orakel "ohne Kollisions-Spikes") → schlecht sortiert → gut).
3. **📐 Sweeps** über Elektroden, Feuerrate, Rauschen, Wellenform-Ähnlichkeit, Amplitudenschwankung, Schwelle, PCA-Komponenten und Neuronenzahl (feste Datensätze ab 100000, Streuung, aktueller Wert markiert; rechts Trefferquote, Genauigkeit der Detektion und Kollisionsanteil; Sweeps rechnen mit bekannter Neuronenzahl).
4. **🔬 Merkmale und Neuronenzahl:** Merkmale × Elektrodenzahl; Silhouette-Kurve und die gewählten k in fünf Datensätzen (Experiment auf Abruf).
5. **🧩 Wer sortiert was:** acht Szenen für Pipeline, ICA, SOBI und SCA mit Spanne über die Datensätze (Experiment auf Abruf).
6. **🚧 Grenzen:** Tabelle "Annahme – was passiert – wer setzt an" (Vorlagenabgleich, Clustering-Linie, Verzögerungsgraph …).

Regler: Neuronen (2–5), Elektroden (1–8), Feuerrate (×0.25–×4), Wellenform-Ähnlichkeit (0–1), Amplitudenschwankung (0–0.3), Rauschen, Länge, Schwelle (2–10 Rausch-σ), Merkmale (PCA / Rohwerte / Spitzenamplituden), PCA-Komponenten (nur bei PCA sichtbar, Wert bleibt erhalten), Neuronenzahl (bekannt / Silhouette), Start, ICA-Kontrast.

## Messwerte der Presets (Seed 7; sie prüfen sich mit weiten Bändern selbst)

| Preset | Trefferquote | Sortiergenauigkeit | Spitzen-F1 | Urteil |
|---|---|---|---|---|
| Vier Neuronen, vier Elektroden | 0.90 | 0.975 | 0.938 | Pipeline sortiert gut |
| Eine Elektrode | 0.89 | 0.591 | 0.383 | k-means optimiert das Falsche |
| Hohe Feuerrate (×2) | 0.83 | 0.729 | 0.653 | Überlappung hoch |
| Fünf Neuronen | 0.88 | 0.698 | 0.662 | k-means optimiert das Falsche |
| Rauschen und hohe Schwelle (1.0, Schwelle 6) | 0.51 | 0.878 | 0.507 | Schwelle zu hoch |
| Unbekannte Neuronenzahl (Silhouette wählt 8) | 0.90 | 0.917 | 0.902 | falsche Clusterzahl |

## Modell und Verfahren

- **Szenario** (`ss_scenario.py`): wie in ica/sobi/sca-demo (10 kHz; Neuronen mit biphasischer Wellenform und Poisson-artigem Feuern mit 2 ms Refraktärzeit, Mischung ∝ 1/(d² + ε), Rauschen relativ zum Neuronen-Signal, eigener Zufallsstrom je Quelle); neu: **Wellenform-Ähnlichkeit** (Breite σ = 3 + Ähnlichkeit · (σᵢ − 3),
  0 = alle Neuronen gleich) und **Amplitudenschwankung** (Spitzenhöhe × (1 + s · N(0,1)), eigener Zufallsstrom, Spikezeiten unverändert). Bei Ähnlichkeit 1 und Schwankung 0 sind die Daten identisch zu denen der Vorgänger.
- **Pipeline** (`ss_algorithm.py`, numpy von Grund auf): **Detektion** d(t) = min_j (x_j − Median_j)/σ_j mit σ_j = MAD/0.6745 (robust; Untergrenze 0.2 % der größten Amplitude für rauschfreie Daten), Schwelle, tiefste Minima zuerst, Totzeit 15 Abtastwerte; **Ausschnitte** 8 vor bis 22 nach dem Minimum über alle Elektroden;
  **Merkmale** PCA per SVD (Vorzeichen: größter Eintrag positiv; gegen scikit-learn geprüft), Rohwerte oder Spitzenamplitude je Elektrode; **k-means** (Lloyd, k-means++, 10 Neustarts, kleinste Streuung; gegen scikit-learn geprüft) und **Silhouette** (exakt gegen scikit-learn geprüft; bei über 1200 Spikes auf fester Stichprobe).
- **Vergleich:** ICA, SOBI und SCA wortgleich aus den Vorgänger-Demos (`ss_ica.py`, `ss_sobi.py`, `ss_sca.py`, gegen eingefrorene Werte geprüft): Spitzen-F1 auf der geschätzten Neuronen-Spur (Zuordnung per Korrelation → Schwelle → Treffer, Toleranz ±4) – dieselbe Definition wie bei der Pipeline.
- **Auswertung** (`ss_evaluation.py`): Zuordnung erkannter ↔ wahrer Spitzen (±6 Abtastwerte, jede wahre höchstens einmal), Kollisionen (ein anderer Spike beginnt höchstens 12 Abtastwerte vor oder nach diesem), Cluster ↔ Neuron per optimaler Zuordnung (Bitmasken-DP), Sortiergenauigkeit gesamt / einzeln / Kollision,
  Streuung innerhalb der Cluster (gefundene gegen wahre Zuordnung), Orakel-Genauigkeit ohne Kollisions-Spikes, Sweeps, Merkmals-, Silhouette- und Szenen-Tabellen, Urteil mit Referenzläufen (ein Viertel der Feuerrate, Ähnlichkeit 1, keine Schwankung, Standard-Rauschen).

## Was nicht funktioniert hat / Grenzen

- **k-means ist der Schwachpunkt, nicht die Merkmale.** Ein Klassifikator mit bekannten Neuronen erreicht mit denselben Merkmalen bei einer Elektrode 0.94 (k-means: 0.65), bei fünf Neuronen 0.94 (k-means: 0.80). Die Zielfunktion belohnt das Teilen großer und das Verschmelzen kleiner Cluster: die gefundene Aufteilung hat eine **kleinere** Streuung als die wahre
  (im Test in allen fünf Datensätzen bei einer Elektrode). Mehr Neustarts helfen meist nicht (50 statt 10 ändern in den meisten Datensätzen nichts); die Kollisions-Ausreißer sind bei hoher Feuerrate ein Teil der Ursache (Orakel ohne Kollisionen: 0.98 statt 0.87 bei doppelter Feuerrate), bei einer Elektrode nicht (0.66 statt 0.65).
- **Ein Mischmodell (GMM) war schlechter, nicht besser:** ein erster Versuch mit EM-Mischmodell (volle Kovarianzen, Start aus k-means) erreichte im Grundfall nur 0.82 statt 0.98 – die sehr kompakten Spike-Cluster machen die Kovarianzen fast singulär, und Ausreißer bekommen eigene Komponenten. Es wurde deshalb verworfen; die Grenzen-Tabelle verweist stattdessen auf die Clustering-Linie.
- **Die Silhouette schätzt die Neuronenzahl immer zu hoch** (20 von 20 Fällen), weil große Neuronen-Cluster mit Kollisions-Ausreißern in mehrere kompakte zerfallen. Der Wert selbst (F1 0.93) täuscht darüber hinweg – deshalb ist die falsche Clusterzahl ein eigener Urteilscode.
- **Die Kollisionsdefinition musste enger werden:** mit ±25 Abtastwerten galten 35 % der Spikes als Kollision (zu weit; die Spitzen sind 4–8 Abtastwerte breit), mit ±12 sind es 16 % – und nur diese verzerren wirklich.
- **Die Rauschschätzung hatte eine zu hohe Untergrenze:** 2 % der größten Amplitude lag über dem tatsächlichen Rauschen, die Schwelle lag dadurch faktisch bei etwa 14 statt 4.5 Rausch-Standardabweichungen und ließ sich kaum "zu niedrig" stellen; jetzt 0.2 %.
- **Kein Vorlagenabgleich, keine Filter, keine Drift:** die Wellenform jedes Neurons ist konstant (bis auf die einstellbare Amplitudenschwankung), Spikes werden nur bei ihrem Minimum ausgerichtet, überlappende Spikes werden nicht aufgelöst (das ist der Grund für das nächste Stück des Zweigs).
- **Die Neuronenzahl ist bekannt** (außer im Silhouette-Modus); Sweeps und Szenen rechnen immer mit bekannter Neuronenzahl.
- **Der Verzögerungsgraph-Ansatz** (die Dissertation des Autors, Universität Rostock 2017: statt der Wellenform die Zeitverzögerungen desselben Spikes über mehrere Elektroden) wird nur genannt; **ein Leistungsvergleich mit ihm ist nicht Teil dieser Demo.**
- **Synthetische Daten:** feste Spitzenform je Neuron, exakt lineare Mischung, weißes Gauß'sches Rauschen, Elektroden auf einer Zeile.

## Verifikation

- Rauschschätzung (Gauß-Rauschen, seltene Spitzen, Untergrenze, ein Elektrodenkanal nur aus Nullen); Detektion (geplante Spitzen, Totzeit, negativste Elektrode, positive Ausschläge ignoriert); Ausschnitte (am Minimum ausgerichtet, Randspikes entfallen); **PCA gegen `sklearn.decomposition.PCA`** (Werte bis aufs Vorzeichen, Varianzanteile);
  **k-means gegen `sklearn.cluster.KMeans`** (gleiche Streuung auf gut getrennten Daten); **Silhouette exakt gegen `sklearn.metrics.silhouette_score`**; Clusterzahl-Wahl auf sauberen Daten richtig.
- Zuordnung und Kollisionen mit Handinstanzen; Streuung innerhalb der Cluster; Sortierung eines fast idealen Falls (Genauigkeit 1.0); Verwechslungsmatrix und Genauigkeitsaufteilung konsistent; Orakel-Genauigkeit; keine Abstürze ohne erkannte Spikes.
- Szenario bit-genau gegen eingefrorene Werte aus den Vorgängern; ICA-, SOBI- und SCA-Kopien gegen eingefrorene Werte; Ähnlichkeit 0 = identische Formen; Schwankung ändert die Höhe, nicht die Zeiten; Spikezeiten zeigen auf das Minimum für alle Ähnlichkeiten.
- **Alle Zahlen der App-Texte sind als Tests hinterlegt** (Neuronen, Elektroden, Feuerrate und Kollisionsanteil, Ähnlichkeit, Schwankung, Rauschen, Schwelle, Merkmale, Komponenten, Silhouette, Start, Preset-Hilfen; jeweils Mittel über die festen Sweep-Datensätze mit weiten Toleranzen); Verdict-Codes über sechs Datensätze;
  alle 6 Presets in Bändern; AppTest-Rauchtests (Default, jedes Preset, jeder Schritt auch ohne erkannte Spikes, Randgrößen, ausgeblendete Komponentenzahl behält ihren Wert, Sweep-Optionen, Experimente auf Abruf), Achsensperre und explizite Schlüssel aller Figuren.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App: Schritte, Ergebnis, 📐 Sweeps, 🔬 Merkmale und Neuronenzahl, 🧩 Szenen, 🚧 Grenzen, Mathe |
| `ss_algorithm.py` | Rauschschätzung, Detektion, Ausschnitte, PCA, k-means, Silhouette, `sort_spikes` |
| `ss_ica.py`, `ss_sobi.py`, `ss_sca.py` | Vergleichsverfahren (wortgleich aus ica/sobi/sca-demo) |
| `ss_scenario.py`, `ss_constants.py` | Mehrelektroden-Generator mit Ähnlichkeit und Schwankung; Konstanten, Presets |
| `ss_evaluation.py` | Zuordnung, Kollisionen, Sortiergüte, Vergleich, Sweeps, Tabellen, Urteil |
| `ss_presets.py`, `ss_visualization.py` | Permalink/Presets, Plotly-Figuren (achsengesperrt) |
| `tests/` | Algorithmus (Kreuzprüfungen gegen scikit-learn), Szenario, Aussagen der App, Presets, AppTest |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

Teil des [Operations-Research-Demo-Portfolios](https://sebastianhanisch.net/demos.html) von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
Interesse an einer maßgeschneiderten Lösung? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html).
