# ============================================================
# datos.py — Contexto y datos del proyecto TFG
# Análisis y Predicción de Demanda de Neumáticos — Suiza
# ============================================================

# ---- Ensemble definitivo ----
ENSEMBLE = {
    "Aargau":                 {"modelo": "XGB_Bayes", "mape": 15.0, "std": 3.3,   "pred_12m": 23582, "grupo": "grande"},
    "Ticino":                 {"modelo": "XGB_Bayes", "mape": 16.0, "std": 6.5,   "pred_12m": 23660, "grupo": "grande"},
    "Vaud":                   {"modelo": "SARIMAX",   "mape": 20.6, "std": 6.8,   "pred_12m": 30632, "grupo": "grande"},
    "SanktGallen":            {"modelo": "XGB_v2",    "mape": 23.3, "std": 8.7,   "pred_12m": 10561, "grupo": "grande"},
    "Zürich":                 {"modelo": "XGB_Bayes", "mape": 23.3, "std": 3.4,   "pred_12m": 49346, "grupo": "grande"},
    "Fribourg":               {"modelo": "XGB_v2",    "mape": 24.7, "std": 1.0,   "pred_12m": 13438, "grupo": "grande"},
    "Genève":                 {"modelo": "SARIMA",    "mape": 24.8, "std": 4.7,   "pred_12m": 5360,  "grupo": "grande"},
    "Basel-Stadt":            {"modelo": "XGB_v2",    "mape": 25.1, "std": 3.5,   "pred_12m": 5642,  "grupo": "pequeño"},
    "Bern":                   {"modelo": "XGB_Bayes", "mape": 25.2, "std": 8.6,   "pred_12m": 26983, "grupo": "grande"},
    "Basel-Landschaft":       {"modelo": "XGB_v2",    "mape": 29.1, "std": 6.2,   "pred_12m": 7133,  "grupo": "pequeño"},
    "Neuchâtel":              {"modelo": "XGB_v2",    "mape": 29.8, "std": 11.3,  "pred_12m": 4359,  "grupo": "pequeño"},
    "Valais":                 {"modelo": "XGB_Bayes", "mape": 29.9, "std": 5.6,   "pred_12m": 11668, "grupo": "grande"},
    "Schwyz":                 {"modelo": "XGB_v2",    "mape": 30.4, "std": 1.8,   "pred_12m": 6578,  "grupo": "pequeño"},
    "Lucerne":                {"modelo": "XGB_v2",    "mape": 31.7, "std": 15.0,  "pred_12m": 9509,  "grupo": "grande"},
    "Solothurn":              {"modelo": "XGB_Bayes", "mape": 31.9, "std": 4.5,   "pred_12m": 64782, "grupo": "grande"},
    "Thurgau":                {"modelo": "XGB_v2",    "mape": 34.8, "std": 12.5,  "pred_12m": 5880,  "grupo": "pequeño"},
    "Graubünden":             {"modelo": "XGB_Bayes", "mape": 37.1, "std": 9.3,   "pred_12m": 9434,  "grupo": "grande"},
    "Glarus":                 {"modelo": "Media",     "mape": 41.6, "std": 10.5,  "pred_12m": 733,   "grupo": "pequeño"},
    "Jura":                   {"modelo": "XGB_v2",    "mape": 45.2, "std": 11.1,  "pred_12m": 2583,  "grupo": "pequeño"},
    "Obwalden":               {"modelo": "Media",     "mape": 54.7, "std": 1.9,   "pred_12m": 566,   "grupo": "pequeño"},
    "Zug":                    {"modelo": "XGB_Bayes", "mape": 59.7, "std": 44.7,  "pred_12m": 3719,  "grupo": "grande"},
    "Nidwalden":              {"modelo": "XGB_v2",    "mape": 78.9, "std": 49.7,  "pred_12m": 2476,  "grupo": "pequeño"},
    "Schaffhausen":           {"modelo": "XGB_v2",    "mape": 104.1,"std": 51.1,  "pred_12m": 589,   "grupo": "pequeño"},
    "AppenzellInnerrhoden":   {"modelo": "Ridge",     "mape": 125.7,"std": 79.4,  "pred_12m": 48,    "grupo": "pequeño"},
    "Uri":                    {"modelo": "XGB_v2",    "mape": 252.1,"std": 247.1, "pred_12m": 553,   "grupo": "pequeño"},
    "AppenzellAusserrhoden":  {"modelo": "Media",     "mape": 332.3,"std": 367.9, "pred_12m": 496,   "grupo": "pequeño"},
}

TOTAL_SUIZA = 320309
MAPE_GLOBAL = 59.9
MAPE_GRANDES = 28.8
MAPE_PEQUENOS = 91.1
MAPE_SIN_PROBLEMATICOS = 40.1

# ---- Coordenadas de centroides cantonales ----
COORDENADAS = {
    "Aargau":                 {"lat": 47.41, "lon": 8.16},
    "AppenzellAusserrhoden":  {"lat": 47.37, "lon": 9.37},
    "AppenzellInnerrhoden":   {"lat": 47.32, "lon": 9.41},
    "Basel-Landschaft":       {"lat": 47.44, "lon": 7.76},
    "Basel-Stadt":            {"lat": 47.56, "lon": 7.59},
    "Bern":                   {"lat": 46.96, "lon": 7.45},
    "Fribourg":               {"lat": 46.80, "lon": 7.16},
    "Genève":                 {"lat": 46.21, "lon": 6.14},
    "Glarus":                 {"lat": 47.04, "lon": 9.07},
    "Graubünden":             {"lat": 46.66, "lon": 9.58},
    "Jura":                   {"lat": 47.36, "lon": 7.20},
    "Lucerne":                {"lat": 47.08, "lon": 8.11},
    "Neuchâtel":              {"lat": 47.00, "lon": 6.93},
    "Nidwalden":              {"lat": 46.93, "lon": 8.38},
    "Obwalden":               {"lat": 46.88, "lon": 8.25},
    "SanktGallen":            {"lat": 47.42, "lon": 9.37},
    "Schaffhausen":           {"lat": 47.70, "lon": 8.63},
    "Schwyz":                 {"lat": 47.02, "lon": 8.65},
    "Solothurn":              {"lat": 47.21, "lon": 7.53},
    "Thurgau":                {"lat": 47.55, "lon": 9.00},
    "Ticino":                 {"lat": 46.19, "lon": 8.80},
    "Uri":                    {"lat": 46.79, "lon": 8.63},
    "Valais":                 {"lat": 46.20, "lon": 7.54},
    "Vaud":                   {"lat": 46.57, "lon": 6.52},
    "Zug":                    {"lat": 47.17, "lon": 8.52},
    "Zürich":                 {"lat": 47.38, "lon": 8.65},
}

# ---- Iteraciones del modelado ----
ITERACIONES_MODELADO = [
    {"version": "Baseline Naive",       "mape": 66.4, "desc": "Predice el valor del mes anterior"},
    {"version": "Baseline Media",       "mape": 50.9, "desc": "Media histórica por cantón y mes"},
    {"version": "XGBoost global",       "mape": 92.5, "desc": "Falla por heterogeneidad cantonal"},
    {"version": "XGBoost v1",           "mape": 38.8, "desc": "Estratificado, features base"},
    {"version": "Grid Search",          "mape": 39.0, "desc": "54 combinaciones, sin mejora"},
    {"version": "SARIMA",               "mape": 42.2, "desc": "Estadístico por cantón"},
    {"version": "XGBoost v2",           "mape": 32.6, "desc": "Features macro + carpark"},
    {"version": "XGB_Bayes (Optuna)",   "mape": 29.3, "desc": "100 iteraciones bayesianas"},
    {"version": "Ensemble WF final",    "mape": 59.9, "desc": "Walk-forward 3 cortes, 26 cantones"},
]

# ---- Feature importance XGBoost v2 ----
FEATURE_IMPORTANCE = [
    {"feature": "sell_out_lag12", "importancia": 33.3, "tipo": "Retardo histórico"},
    {"feature": "sell_out_roll6", "importancia": 14.4, "tipo": "Ventana móvil"},
    {"feature": "sell_out_lag1",  "importancia": 12.5, "tipo": "Retardo histórico"},
    {"feature": "sell_out_lag6",  "importancia": 5.9,  "tipo": "Retardo histórico"},
    {"feature": "temp_lag2",      "importancia": 5.6,  "tipo": "Climática"},
    {"feature": "inflacion_yoy",  "importancia": 3.5,  "tipo": "Macroeconómica"},
    {"feature": "total_vehiculos","importancia": 2.6,  "tipo": "Parque vehicular"},
    {"feature": "mes_sin",        "importancia": 2.5,  "tipo": "Temporal cíclica"},
    {"feature": "cpi",            "importancia": 1.9,  "tipo": "Macroeconómica"},
    {"feature": "pct_ev",         "importancia": 1.2,  "tipo": "Parque vehicular"},
]

# ---- Contexto del sistema para Claude ----
SYSTEM_PROMPT = """Eres un asistente experto en análisis de demanda de neumáticos en el mercado suizo, desarrollado para el TFG de Ingeniería en Sistemas Inteligentes en la UIE (Universidad Intercontinental de la Empresa), en el marco de prácticas en SDG Group.

Tienes acceso completo a todos los datos y resultados del proyecto. Responde siempre en castellano, de forma clara, estructurada y orientada al negocio.

CRITERIO DE FIABILIDAD (MAPE walk-forward, media de 3 cortes temporales):
- MAPE < 25%: MUY FIABLE — predicciones de alta confianza para planificación operativa
- MAPE 25-50%: ACEPTABLE — útil para planificación con margen de error moderado  
- MAPE 50-100%: DÉBIL — referencia orientativa, no usar para stock
- MAPE > 100%: IMPREDECIBLE — demanda esencialmente aleatoria, no modelable

MÉTRICAS GLOBALES:
- Total Suiza predicho (mar 2026 - feb 2027): 320.309 unidades
- MAPE medio global (26 cantones): 59.9%
- MAPE medio cantones grandes (13): 28.8%
- MAPE medio cantones pequeños (13): 91.1%
- MAPE excluyendo Uri y AppenzellAusserrhoden: 40.1% (más representativo: <0.4% del volumen)
- Cantones bajo umbral 50%: 20 de 26 (77%)

ENSEMBLE DEFINITIVO (cantón: modelo, MAPE, std, predicción 12m):
Aargau: XGB_Bayes, 15.0%, std=3.3%, 23.582 uds — MUY FIABLE
Ticino: XGB_Bayes, 16.0%, std=6.5%, 23.660 uds — MUY FIABLE
Vaud: SARIMAX, 20.6%, std=6.8%, 30.632 uds — MUY FIABLE
SanktGallen: XGB_v2, 23.3%, std=8.7%, 10.561 uds — FIABLE
Zürich: XGB_Bayes, 23.3%, std=3.4%, 49.346 uds — FIABLE y MUY ESTABLE
Fribourg: XGB_v2, 24.7%, std=1.0%, 13.438 uds — FIABLE y MUY ESTABLE (std más baja)
Genève: SARIMA, 24.8%, std=4.7%, 5.360 uds — FIABLE
Basel-Stadt: XGB_v2, 25.1%, std=3.5%, 5.642 uds — ACEPTABLE
Bern: XGB_Bayes, 25.2%, std=8.6%, 26.983 uds — ACEPTABLE
Basel-Landschaft: XGB_v2, 29.1%, std=6.2%, 7.133 uds — ACEPTABLE
Neuchâtel: XGB_v2, 29.8%, std=11.3%, 4.359 uds — ACEPTABLE
Valais: XGB_Bayes, 29.9%, std=5.6%, 11.668 uds — ACEPTABLE
Schwyz: XGB_v2, 30.4%, std=1.8%, 6.578 uds — ACEPTABLE y ESTABLE
Lucerne: XGB_v2, 31.7%, std=15.0%, 9.509 uds — ACEPTABLE pero variable
Solothurn: XGB_Bayes, 31.9%, std=4.5%, 64.782 uds — ACEPTABLE (outlier logístico Fibag CP 4624)
Thurgau: XGB_v2, 34.8%, std=12.5%, 5.880 uds — ACEPTABLE
Graubünden: XGB_Bayes, 37.1%, std=9.3%, 9.434 uds — ACEPTABLE
Glarus: Media, 41.6%, std=10.5%, 733 uds — LIMITADO
Jura: XGB_v2, 45.2%, std=11.1%, 2.583 uds — LIMITADO
Obwalden: Media, 54.7%, std=1.9%, 566 uds — DÉBIL
Zug: XGB_Bayes, 59.7%, std=44.7%, 3.719 uds — DÉBIL
Nidwalden: XGB_v2, 78.9%, std=49.7%, 2.476 uds — MUY DÉBIL
Schaffhausen: XGB_v2, 104.1%, std=51.1%, 589 uds — IMPREDECIBLE
AppenzellInnerrhoden: Ridge, 125.7%, std=79.4%, 48 uds — IMPREDECIBLE (~3 uds/mes)
Uri: XGB_v2, 252.1%, std=247.1%, 553 uds — IMPREDECIBLE (~48 uds/mes, 1-2 clientes)
AppenzellAusserrhoden: Media, 332.3%, std=367.9%, 496 uds — IMPREDECIBLE (~58 uds/mes)

DISTRIBUCIÓN DE MODELOS:
XGB_v2: 12 cantones (46%) — XGBoost + variables macroeconómicas (CPI, inflación, carpark)
XGB_Bayes: 8 cantones (31%) — XGBoost + Optuna bayesiana (100 iteraciones, MAPE=29.3% vs 39.0% GridSearch)
Media histórica: 3 cantones (12%) — volumen insuficiente
SARIMAX: 1 cantón (Vaud)
SARIMA: 1 cantón (Genève)
Ridge: 1 cantón (AppenzellInnerrhoden)

PROCESO DE MODELADO (iteraciones):
Naive: 66.4% → Media: 50.9% → XGBoost global: 92.5% (falla) → Estratificación →
XGBoost v1: 38.8% → Grid Search: 39.0% (sin mejora) → SARIMA: 42.2% →
XGBoost v2 (+macro): 32.6% → SARIMAX: solo mejora Vaud →
Bayes Optuna: 29.3% → Walk-forward 3 cortes → Ensemble final: 59.9% global / 28.8% grandes

TENDENCIAS DEL MERCADO SUIZO:
- Estacionalidad bimodal: picos mar-abr (verano) y sep-oct (invierno), muy estable interanualmente
- Correlación temp-Winter: r=0.535 con lag 2 meses (p<0.001)
- Correlación temp-Summer: r=-0.446 con lag 2 meses (p<0.001)
- Premiumización: Premium 28% (2018) → 80% (2026), sorpasso sobre Quality en 2019
- SUVización: RIM16 37%→22%; RIM18 10%→21% (2019-2026)
- Precio medio: 80 CHF (2023) → 95 CHF (2026), +18%
- Electrificación: 2.2% del parque en 2026
- Outlier logístico: CP 4624 Härkingen (Solothurn) = 17.2% del sell-out total
- Oportunidades: Genève (42.7% penetración), SanktGallen (54.6%), Schaffhausen (29.3%)

POR QUÉ URI Y APPENZELLAUSSERRHODEN SON IMPREDECIBLES:
Uri (~48 uds/mes): depende de 1-2 clientes; un pedido extraordinario puede triplicar el valor habitual. std=247pp significa error extremadamente variable entre períodos.
AppenzellAusserrhoden (~58 uds/mes, MAPE=332%, std=368%): el error medio supera el valor predicho. Demanda esencialmente estocástica. Ambos representan <0.4% del volumen total de Suiza.

GRÁFICAS DISPONIBLES (puedes recomendarlas cuando sean relevantes):
- mape_canton: MAPE por cantón con colores por modelo (barras horizontales)
- predicciones_canton: predicciones a 12 meses por cantón (barras)
- mapa_predicciones: mapa de Suiza con burbujas proporcionales a predicciones
- mapa_mape: mapa de Suiza con colores según MAPE
- evolucion_modelos: evolución del MAPE a lo largo de las iteraciones del modelado
- feature_importance: importancia de variables en XGBoost v2
- distribucion_modelos: distribución de modelos en el ensemble (pie chart)

Cuando el usuario haga una pregunta sobre un cantón específico, sobre predicciones o sobre fiabilidad, SIEMPRE termina tu respuesta recomendando la gráfica más relevante con este formato exacto:
[GRÁFICA_RECOMENDADA: nombre_grafica]

Ejemplo: Si preguntan por fiabilidad → [GRÁFICA_RECOMENDADA: mape_canton]
Si preguntan por predicciones → [GRÁFICA_RECOMENDADA: predicciones_canton]
Si preguntan por un cantón geográficamente → [GRÁFICA_RECOMENDADA: mapa_predicciones]
Si preguntan por el proceso de modelado → [GRÁFICA_RECOMENDADA: evolucion_modelos]
Si preguntan por variables importantes → [GRÁFICA_RECOMENDADA: feature_importance]
"""
