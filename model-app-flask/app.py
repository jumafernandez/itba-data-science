from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Cargar el modelo desde la carpeta models
modelo = joblib.load("models/modelo_iris_mejor.joblib")

columnas = [
    'sepal length (cm)',
    'sepal width (cm)',
    'petal length (cm)',
    'petal width (cm)'
]
clases = ['setosa', 'versicolor', 'virginica']

@app.route("/", methods=["GET", "POST"])
def index():
    resultado = None
    valores = {}
    if request.method == "POST":
        try:
            valores = {col: float(request.form[col]) for col in columnas}
            df = pd.DataFrame([valores.values()], columns=columnas)
            pred = modelo.predict(df)[0]
            resultado = clases[pred]
        except Exception as e:
            resultado = f"Error: {str(e)}"
    return render_template("index.html", columnas=columnas, resultado=resultado, valores=valores)

if __name__ == "__main__":
    app.run(debug=True)
