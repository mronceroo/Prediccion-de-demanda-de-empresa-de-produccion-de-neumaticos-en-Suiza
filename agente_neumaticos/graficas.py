# ============================================================
# graficas.py — Generación de gráficas con Plotly
# ============================================================

import plotly.graph_objects as go
import plotly.express as px
import json
from datos import ENSEMBLE, COORDENADAS, ITERACIONES_MODELADO, FEATURE_IMPORTANCE

COLORES_MODELO = {
    "XGB_Bayes": "#F59E0B",
    "XGB_v2":    "#10B981",
    "SARIMA":    "#3B82F6",
    "SARIMAX":   "#06B6D4",
    "Media":     "#8B5CF6",
    "Ridge":     "#EF4444",
}

LAYOUT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, -apple-system, sans-serif", size=12, color="#1a1a1a"),
    margin=dict(l=20, r=20, t=50, b=20),
    height=420,
)


def grafica_mape_canton():
    """Barras horizontales MAPE por cantón coloreadas por modelo."""
    datos = sorted(ENSEMBLE.items(), key=lambda x: x[1]["mape"])
    cantones = [d[0] for d in datos]
    mapes    = [d[1]["mape"] for d in datos]
    modelos  = [d[1]["modelo"] for d in datos]
    colores  = [COLORES_MODELO.get(m, "#888") for m in modelos]

    fig = go.Figure()
    for modelo in COLORES_MODELO:
        idx = [i for i, m in enumerate(modelos) if m == modelo]
        if not idx:
            continue
        fig.add_trace(go.Bar(
            y=[cantones[i] for i in idx],
            x=[mapes[i] for i in idx],
            name=modelo,
            orientation="h",
            marker_color=COLORES_MODELO[modelo],
            hovertemplate="<b>%{y}</b><br>MAPE: %{x:.1f}%<extra>" + modelo + "</extra>",
        ))

    fig.add_vline(x=50, line_dash="dash", line_color="#EF4444",
                  annotation_text="Umbral 50%", annotation_position="top right")

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="MAPE walk-forward por cantón — Ensemble definitivo", font_size=14),
        xaxis_title="MAPE (%)",
        barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(tickfont_size=11),
    )
    return json.loads(fig.to_json())


def grafica_predicciones_canton():
    """Barras de predicción a 12 meses por cantón."""
    datos = sorted(ENSEMBLE.items(), key=lambda x: -x[1]["pred_12m"])
    cantones = [d[0] for d in datos]
    preds    = [d[1]["pred_12m"] for d in datos]
    modelos  = [d[1]["modelo"] for d in datos]
    colores  = [COLORES_MODELO.get(m, "#888") for m in modelos]

    fig = go.Figure(go.Bar(
        x=cantones,
        y=preds,
        marker_color=colores,
        hovertemplate="<b>%{x}</b><br>Predicción: %{y:,.0f} uds<extra></extra>",
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Predicción de demanda 12 meses (mar 2026 – feb 2027)", font_size=14),
        yaxis_title="Unidades",
        xaxis_tickangle=-45,
        xaxis_tickfont_size=10,
    )
    return json.loads(fig.to_json())


def grafica_mapa_predicciones():
    """Mapa de burbujas con predicciones por cantón."""
    cantones, lats, lons, preds, modelos, mapes = [], [], [], [], [], []
    for canton, info in ENSEMBLE.items():
        coord = COORDENADAS.get(canton)
        if not coord:
            continue
        cantones.append(canton)
        lats.append(coord["lat"])
        lons.append(coord["lon"])
        preds.append(info["pred_12m"])
        modelos.append(info["modelo"])
        mapes.append(info["mape"])

    fig = go.Figure(go.Scattergeo(
        lat=lats, lon=lons,
        text=cantones,
        customdata=list(zip(preds, modelos, mapes)),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Predicción 12m: %{customdata[0]:,.0f} uds<br>"
            "Modelo: %{customdata[1]}<br>"
            "MAPE: %{customdata[2]:.1f}%"
            "<extra></extra>"
        ),
        mode="markers",
        marker=dict(
            size=[max(6, p / 1500) for p in preds],
            color=preds,
            colorscale="YlOrRd",
            showscale=True,
            colorbar=dict(title="Uds predichas"),
            sizemode="diameter",
            opacity=0.85,
            line=dict(width=0.5, color="#fff"),
        ),
    ))
    fig.update_geos(
        scope="europe",
        center=dict(lat=46.8, lon=8.2),
        lataxis_range=[45.8, 47.9],
        lonaxis_range=[5.9, 10.6],
        showland=True, landcolor="#f0f0eb",
        showocean=True, oceancolor="#e8f4f8",
        showlakes=True, lakecolor="#e8f4f8",
        showrivers=True, rivercolor="#cce5f0",
        showcountries=True, countrycolor="#ccc",
        showframe=False,
    )
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Mapa de predicción de demanda — Suiza 2026-2027", font_size=14),
        geo=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return json.loads(fig.to_json())


def grafica_mapa_mape():
    """Mapa de Suiza coloreado por MAPE del ensemble."""
    cantones, lats, lons, mapes, modelos = [], [], [], [], []
    for canton, info in ENSEMBLE.items():
        coord = COORDENADAS.get(canton)
        if not coord:
            continue
        cantones.append(canton)
        lats.append(coord["lat"])
        lons.append(coord["lon"])
        mapes.append(min(info["mape"], 150))  # cap para escala
        modelos.append(info["modelo"])

    fig = go.Figure(go.Scattergeo(
        lat=lats, lon=lons,
        text=cantones,
        customdata=list(zip([ENSEMBLE[c]["mape"] for c in cantones], modelos,
                            [ENSEMBLE[c]["pred_12m"] for c in cantones])),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "MAPE: %{customdata[0]:.1f}%<br>"
            "Modelo: %{customdata[1]}<br>"
            "Predicción: %{customdata[2]:,.0f} uds"
            "<extra></extra>"
        ),
        mode="markers",
        marker=dict(
            size=18,
            color=mapes,
            colorscale="RdYlGn_r",
            showscale=True,
            colorbar=dict(title="MAPE (%)"),
            opacity=0.9,
            line=dict(width=0.5, color="#fff"),
        ),
    ))
    fig.update_geos(
        scope="europe",
        center=dict(lat=46.8, lon=8.2),
        lataxis_range=[45.8, 47.9],
        lonaxis_range=[5.9, 10.6],
        showland=True, landcolor="#f0f0eb",
        showocean=True, oceancolor="#e8f4f8",
        showlakes=True, lakecolor="#e8f4f8",
        showcountries=True, countrycolor="#ccc",
        showframe=False,
    )
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Precisión del modelo por cantón (verde = mejor)", font_size=14),
        geo=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return json.loads(fig.to_json())


def grafica_evolucion_modelos():
    """Línea de evolución del MAPE a lo largo de las iteraciones."""
    versiones = [d["version"] for d in ITERACIONES_MODELADO]
    mapes     = [d["mape"] for d in ITERACIONES_MODELADO]
    descs     = [d["desc"] for d in ITERACIONES_MODELADO]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(versiones))),
        y=mapes,
        mode="lines+markers",
        line=dict(color="#3B82F6", width=2),
        marker=dict(size=9, color="#3B82F6"),
        customdata=list(zip(versiones, descs)),
        hovertemplate="<b>%{customdata[0]}</b><br>MAPE: %{y:.1f}%<br>%{customdata[1]}<extra></extra>",
    ))
    fig.add_hline(y=50, line_dash="dash", line_color="#EF4444",
                  annotation_text="Umbral 50%")

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Evolución del MAPE a lo largo del proceso de modelado", font_size=14),
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(len(versiones))),
            ticktext=[v.replace(" ", "<br>") for v in versiones],
            tickfont_size=10,
        ),
        yaxis_title="MAPE (%)",
    )
    return json.loads(fig.to_json())


def grafica_feature_importance():
    """Barras horizontales de importancia de features."""
    features = [d["feature"] for d in sorted(FEATURE_IMPORTANCE, key=lambda x: x["importancia"])]
    valores  = [d["importancia"] for d in sorted(FEATURE_IMPORTANCE, key=lambda x: x["importancia"])]
    tipos    = [d["tipo"] for d in sorted(FEATURE_IMPORTANCE, key=lambda x: x["importancia"])]

    colores_tipo = {
        "Retardo histórico": "#3B82F6",
        "Ventana móvil":     "#06B6D4",
        "Climática":         "#10B981",
        "Macroeconómica":    "#F59E0B",
        "Parque vehicular":  "#8B5CF6",
        "Temporal cíclica":  "#EF4444",
    }

    fig = go.Figure(go.Bar(
        y=features,
        x=valores,
        orientation="h",
        marker_color=[colores_tipo.get(t, "#888") for t in tipos],
        customdata=tipos,
        hovertemplate="<b>%{y}</b><br>Importancia: %{x:.1f}%<br>Tipo: %{customdata}<extra></extra>",
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Importancia de variables — XGBoost v2", font_size=14),
        xaxis_title="Importancia (%)",
        yaxis_tickfont_size=11,
    )
    return json.loads(fig.to_json())


def grafica_distribucion_modelos():
    """Pie chart con distribución de modelos en el ensemble."""
    from collections import Counter
    conteo = Counter(v["modelo"] for v in ENSEMBLE.values())

    fig = go.Figure(go.Pie(
        labels=list(conteo.keys()),
        values=list(conteo.values()),
        marker=dict(colors=[COLORES_MODELO.get(m, "#888") for m in conteo.keys()]),
        hole=0.4,
        hovertemplate="<b>%{label}</b><br>Cantones: %{value}<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Distribución de modelos en el ensemble definitivo", font_size=14),
    )
    return json.loads(fig.to_json())


GRAFICAS_DISPONIBLES = {
    "mape_canton":          grafica_mape_canton,
    "predicciones_canton":  grafica_predicciones_canton,
    "mapa_predicciones":    grafica_mapa_predicciones,
    "mapa_mape":            grafica_mapa_mape,
    "evolucion_modelos":    grafica_evolucion_modelos,
    "feature_importance":   grafica_feature_importance,
    "distribucion_modelos": grafica_distribucion_modelos,
}

DESCRIPCION_GRAFICAS = {
    "mape_canton":          "MAPE walk-forward por cantón (barras horizontales, coloreadas por modelo)",
    "predicciones_canton":  "Predicciones a 12 meses por cantón (barras verticales)",
    "mapa_predicciones":    "Mapa de Suiza con burbujas proporcionales a las predicciones",
    "mapa_mape":            "Mapa de Suiza coloreado por MAPE (verde = mejor precisión)",
    "evolucion_modelos":    "Evolución del MAPE a lo largo de las iteraciones del modelado",
    "feature_importance":   "Importancia de variables en XGBoost v2",
    "distribucion_modelos": "Distribución de modelos en el ensemble (gráfico de anillo)",
}


def generar_grafica(nombre: str) -> dict | None:
    fn = GRAFICAS_DISPONIBLES.get(nombre)
    if fn:
        return fn()
    return None
