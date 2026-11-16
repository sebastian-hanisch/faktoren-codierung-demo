"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go

import fcd_constants as C

ART_COLORS = ("#2a6fb0", "#c77700", "#2e7d4f", "#7a3fb0")


def _c(x, nd=2):
    """Zahl mit Dezimalkomma für Beschriftungen im Diagramm (ein Wert, der auf null rundet, ohne Vorzeichen)."""
    r = round(float(x), nd)
    s = f"{abs(r):.{nd}f}".replace(".", ",")
    return ("−" + s) if r < 0 else s


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360, legend=True):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)", showlegend=legend)
    if legend:
        fig.update_layout(legend=dict(orientation="h", y=-0.22))
    return lock_axes(fig)


def build_fit(h, fitted_cells):
    """Die angepassten Zellmittel (Linien je Art) und die Beobachtungen (Punkte); die Linien sind in jeder Codierung dieselben."""
    fig = go.Figure()
    xs = list(C.WE_LABELS)
    for i, art in enumerate(C.HAND_ARTEN):
        sel = h.a == i
        fig.add_trace(go.Scatter(x=[C.WE_LABELS[j] for j in h.b[sel]], y=h.y[sel], mode="markers", marker=dict(color=ART_COLORS[i], opacity=0.45, size=8), showlegend=False,
                                 hovertemplate=f"{art}: %{{y:.2f}}<extra></extra>"))
        fig.add_trace(go.Scatter(x=xs, y=list(fitted_cells[i]), mode="lines+markers", line=dict(color=ART_COLORS[i], width=3), marker=dict(size=11, symbol="diamond"), name=art,
                                 hovertemplate=f"{art}: %{{y:.3f}}<extra></extra>"))
    fig.update_yaxes(title_text="Ausgang (Tage)")
    fig.update_xaxes(categoryorder="array", categoryarray=xs)
    return _base(fig)


def build_coef_bars(summary):
    """Achsenabschnitt und Wochenende-Koeffizient je Codierung: dieselbe Anpassung, andere Zahlen."""
    names = [r["Codierung"] for r in summary]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=[r["Achsenabschnitt"] for r in summary], name="Achsenabschnitt", marker_color=C.COLOR_BAR, text=[_c(r["Achsenabschnitt"]) for r in summary], textposition="outside"))
    fig.add_trace(go.Bar(x=names, y=[r["Wochenende"] for r in summary], name="Koeffizient Wochenende", marker_color=C.COLOR_LINE, text=[_c(r["Wochenende"]) for r in summary], textposition="outside"))
    fig.update_yaxes(title_text="Koeffizient (Tage)", zeroline=True)
    fig.update_layout(barmode="group")
    return _base(fig, 330)


def build_f_bars(cmp):
    """F-Wert des Tests „Haupteffekt Wochenende“ je Codierung; farbig, wenn p < 0,05."""
    rows = [r for r in cmp["rows"] if r["name"] != "Helmert" and "/ Wochenende" not in r["name"]]
    fig = go.Figure(go.Bar(x=[r["name"] for r in rows], y=[0.0 if np.isnan(r["F_B"]) else r["F_B"] for r in rows],
                           marker_color=[C.COLOR_GOOD if (not np.isnan(r["p_B"]) and r["p_B"] < C.ALPHA) else C.COLOR_GREY for r in rows],
                           text=["F = " + _c(r["F_B"]) for r in rows], textposition="outside", hovertemplate="%{x}: F = %{y:.3f}<extra></extra>"))
    fig.update_yaxes(title_text="F-Wert Haupteffekt Wochenende")
    return _base(fig, 330, legend=False)


def build_rates(names, rates, valid, exact=None, title="Häufigkeit, mit der der Test verwirft (%)"):
    """Verwerfungshäufigkeit je Codierung mit Fehlerbalken (2 Standardfehler der Wiederholungen) und, wenn vorhanden, der exakten Teststärke (Rauten)."""
    ys = [100 * r for r in rates]
    err = [200 * float(np.sqrt(max(r * (1 - r), 0.0) / valid)) for r in rates]
    fig = go.Figure(go.Bar(x=names, y=ys, name="vorgerechnet (Wiederholungen)", marker_color=C.COLOR_BAR, error_y=dict(type="data", array=err, visible=True),
                           text=[_c(y, 1) + " %" for y in ys], textposition="outside", hovertemplate="%{x}: %{y:.1f} %<extra></extra>"))
    if exact is not None:
        fig.add_trace(go.Scatter(x=names, y=[100 * e for e in exact], mode="markers", marker=dict(symbol="diamond", size=12, color=C.COLOR_LINE, line=dict(color="white", width=1)),
                                 name="exakte Teststärke (nichtzentrale F)", hovertemplate="%{x}: %{y:.1f} %<extra></extra>"))
    fig.add_hline(y=100 * C.ALPHA, line_dash="dot", line_color="gray")
    fig.update_yaxes(title_text=title, range=[0, 112])
    return _base(fig, 380)


def build_dummy(res):
    """Achsenabschnitt der Lösungswege der Dummy-Falle (verschiedene Koeffizienten, gleiche Anpassung)."""
    names = [k for k, v in res.solutions.items() if v is not None]
    ys = [float(res.solutions[k][0]) for k in names]
    fig = go.Figure(go.Bar(x=names, y=ys, marker_color=C.COLOR_BAR, text=[_c(y) for y in ys], textposition="outside", hovertemplate="%{x}: %{y:.3f}<extra></extra>"))
    fig.update_yaxes(title_text="Achsenabschnitt (log Tage)")
    return _base(fig, 320, legend=False)


def build_kunde(frozen, ref):
    """Mittlere Zahl „signifikanter“ Koeffizienten (von 11) je Referenzkunde, Kunden nach ihrem Effekt sortiert; die gewählte Referenz ist hervorgehoben."""
    eff = np.array(frozen["effects"])
    order = np.argsort(eff)
    labels = [f"K{j} ({'+' if eff[j] >= 0 else ''}{_c(eff[j])})" for j in order]
    ys = [frozen["sig_mean"][j] for j in order]
    fig = go.Figure(go.Bar(x=labels, y=ys, marker_color=[C.COLOR_LINE if j == ref else C.COLOR_BAR for j in order], text=[_c(y, 1) for y in ys], textposition="outside",
                           hovertemplate="%{x}: %{y:.2f} von 11<extra></extra>"))
    fig.update_yaxes(title_text="Koeffizienten mit |t| > t(0,975) (von 11)", range=[0, 11.5])
    fig.update_xaxes(title_text="Referenzkunde (Effekt in log Tagen)")
    return _base(fig, 340, legend=False)
