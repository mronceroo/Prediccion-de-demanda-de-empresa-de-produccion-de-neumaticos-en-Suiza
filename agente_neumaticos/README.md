# Agente de Análisis — Demanda de Neumáticos en Suiza
TFG · SDG Group · 2026

## Estructura
```
agente_neumaticos/
├── app.py          # Backend Flask + llamadas API Claude
├── datos.py        # Contexto y datos del proyecto
├── graficas.py     # Gráficas con Plotly
├── templates/
│   └── index.html  # Interfaz web
└── README.md
```

## Instalación
```bash
conda activate tfg_goodyear
pip install flask plotly anthropic
```

## Configuración
Edita `app.py` y sustituye `TU_API_KEY_AQUI` por tu API Key de Anthropic:
```python
API_KEY = "sk-ant-..."
```

O usa variable de entorno:
```bash
set ANTHROPIC_API_KEY=sk-ant-...   # Windows
export ANTHROPIC_API_KEY=sk-ant-... # Linux/Mac
```

## Ejecución
```bash
conda activate tfg_goodyear
python app.py
```
Accede en: http://localhost:5000

## Gráficas disponibles
| Nombre              | Descripción |
|---------------------|-------------|
| mape_canton         | MAPE por cantón (barras horizontales, coloreadas por modelo) |
| predicciones_canton | Predicciones 12 meses por cantón (barras) |
| mapa_predicciones   | Mapa burbujas proporcionales a predicciones |
| mapa_mape           | Mapa coloreado por MAPE (verde = mejor) |
| evolucion_modelos   | Evolución MAPE a lo largo de iteraciones |
| feature_importance  | Importancia variables XGBoost v2 |
| distribucion_modelos| Distribución modelos en ensemble (anillo) |

## Uso
1. Escribe una pregunta en el chat (ej. "¿Es fiable la predicción de Uri?")
2. El agente responde e indica la gráfica más relevante
3. La gráfica se carga automáticamente en el panel derecho
4. También puedes seleccionar cualquier gráfica manualmente con los botones superiores
