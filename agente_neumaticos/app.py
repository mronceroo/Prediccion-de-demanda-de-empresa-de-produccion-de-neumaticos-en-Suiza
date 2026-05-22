# ============================================================
# app.py — Backend Flask para el agente de análisis
# Uso: python app.py
# Acceso: http://localhost:5000
# ============================================================

import re
import os
from flask import Flask, render_template, request, jsonify
from anthropic import Anthropic
from datos import SYSTEM_PROMPT, ENSEMBLE, ITERACIONES_MODELADO, FEATURE_IMPORTANCE
from graficas import generar_grafica, DESCRIPCION_GRAFICAS

app = Flask(__name__)

# ---- API Key — configura aquí o usa variable de entorno ----
API_KEY = os.environ.get("")

# Historial de conversación por sesión (simple, en memoria)
historial = []


def extraer_grafica_recomendada(texto: str):
    """Extrae el nombre de la gráfica recomendada del texto de la respuesta."""
    match = re.search(r'\[GRÁFICA_RECOMENDADA:\s*(\w+)\]', texto)
    if match:
        nombre = match.group(1)
        texto_limpio = re.sub(r'\[GRÁFICA_RECOMENDADA:\s*\w+\]', '', texto).strip()
        return nombre, texto_limpio
    return None, texto


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    global historial
    data = request.get_json()
    mensaje = data.get("mensaje", "").strip()
    grafica_solicitada = data.get("grafica", None)

    # Si el usuario solicita explícitamente una gráfica
    if grafica_solicitada:
        fig_data = generar_grafica(grafica_solicitada)
        if fig_data:
            return jsonify({
                "respuesta": f"Aquí tienes la gráfica: **{DESCRIPCION_GRAFICAS.get(grafica_solicitada, grafica_solicitada)}**",
                "grafica": fig_data,
                "nombre_grafica": grafica_solicitada,
            })
        return jsonify({"respuesta": "No se encontró esa gráfica.", "grafica": None})

    if not mensaje:
        return jsonify({"error": "Mensaje vacío"}), 400

    # Añadir mensaje al historial
    historial.append({"role": "user", "content": mensaje})

    try:
        client = Anthropic(api_key=API_KEY)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            messages=historial,
        )
        respuesta_texto = response.content[0].text
        historial.append({"role": "assistant", "content": respuesta_texto})

        # Extraer gráfica recomendada si la hay
        nombre_grafica, respuesta_limpia = extraer_grafica_recomendada(respuesta_texto)
        fig_data = None
        if nombre_grafica:
            fig_data = generar_grafica(nombre_grafica)

        return jsonify({
            "respuesta": respuesta_limpia,
            "grafica": fig_data,
            "nombre_grafica": nombre_grafica,
        })

    except Exception as e:
        historial.pop()  # revertir si falla
        return jsonify({"error": str(e)}), 500


@app.route("/grafica/<nombre>", methods=["GET"])
def grafica_endpoint(nombre):
    """Endpoint directo para generar cualquier gráfica por nombre."""
    fig_data = generar_grafica(nombre)
    if fig_data:
        return jsonify({
            "grafica": fig_data,
            "nombre": nombre,
            "descripcion": DESCRIPCION_GRAFICAS.get(nombre, ""),
        })
    return jsonify({"error": f"Gráfica '{nombre}' no encontrada"}), 404


@app.route("/graficas", methods=["GET"])
def listar_graficas():
    """Lista todas las gráficas disponibles."""
    return jsonify(DESCRIPCION_GRAFICAS)


@app.route("/reset", methods=["POST"])
def reset():
    """Reinicia el historial de conversación."""
    global historial
    historial = []
    return jsonify({"ok": True})


if __name__ == "__main__":
    print("=" * 55)
    print("  Agente de Análisis — Demanda de Neumáticos Suiza")
    print("=" * 55)
    print(f"  API Key: {'✓ configurada' if API_KEY != 'TU_API_KEY_AQUI' else '✗ CONFIGURA API_KEY en app.py'}")
    print(f"  Cantones en ensemble: {len(ENSEMBLE)}")
    print(f"  Gráficas disponibles: {len(DESCRIPCION_GRAFICAS)}")
    print("  Accede en: http://localhost:5000")
    print("=" * 55)
    app.run(debug=True, port=5000)
