
import streamlit as st
import pandas as pd
import joblib
import gdown

# URL de Google Drive (archivo público)
url = "https://drive.google.com/uc?id=121oCWhwDZKd4bYEBU6C0-Jw-9xWv-w4Q"
output = "modelo_iris_mejor.joblib"
gdown.download(url, output, quiet=False)

# Cargar el modelo entrenado
modelo = joblib.load('modelo_iris_mejor.joblib')

# Nombres de variables y clases
columnas = [
    'sepal length (cm)',
    'sepal width (cm)',
    'petal length (cm)',
    'petal width (cm)'
]
clases = ['setosa', 'versicolor', 'virginica']

st.title("Predicción de especie de Iris 🌸")
st.write("Ingresá las características de la flor para predecir la especie:")

# Inputs para cada característica
valores = []
for col in columnas:
    valor = st.number_input(f"{col}", value=5.0 if "length" in col else 3.0)
    valores.append(valor)

if st.button("Predecir especie"):
    ejemplo = pd.DataFrame([valores], columns=columnas)
    prediccion = modelo.predict(ejemplo)[0]
    st.success(f"La especie predicha es: **{clases[prediccion]}**")
