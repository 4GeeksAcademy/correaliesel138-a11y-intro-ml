from flask import Flask, request, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

# Cargamos el modelo y el scaler
model  = pickle.load(open('knn_wine_model.sav', 'rb'))
scaler = pickle.load(open('knn_wine_scaler.sav', 'rb'))

# Las 3 clases de calidad
CALIDAD = {
    0: ("Baja calidad",   "🔴", "danger"),
    1: ("Calidad media",  "🟡", "warning"),
    2: ("Alta calidad",   "🟢", "success"),
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Recogemos los 11 valores del formulario
        features = [float(request.form[f]) for f in [
            'fixed_acidity', 'volatile_acidity', 'citric_acid',
            'residual_sugar', 'chlorides', 'free_sulfur_dioxide',
            'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol'
        ]]

        # Escalamos igual que cuando se entrenó el modelo
        features_scaled = scaler.transform([features])

        # Predecimos
        pred  = model.predict(features_scaled)[0]
        proba = model.predict_proba(features_scaled)[0]

        label, emoji, color = CALIDAD[pred]
        confianza = round(max(proba) * 100, 1)

        return render_template('index.html',
            show_result=True,
            resultado=label,
            emoji=emoji,
            color=color,
            confianza=confianza,
            clase=int(pred)
        )
    except Exception as e:
        return render_template('index.html',
            show_result=True,
            resultado=f"Error: {e}",
            emoji="❌", color="danger",
            confianza=0, clase=-1
        )

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
