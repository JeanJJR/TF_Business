import streamlit as st
import pandas as pd
import pickle
import joblib

# =========================
# Configuración
# =========================
st.set_page_config(page_title="Sistema de Predicción Cardíaca", layout="centered")
st.title("Sistema de Predicción de Riesgo de Ataque Cardíaco")

# =========================
# Cargar modelo y escalador
# =========================
modelo = pickle.load(open("modelo_final_xgboost3.sav", "rb"))
scaler = joblib.load("scaler_robust_cardiaco3.sav")

# =========================
# LISTA EXACTA DE 49 COLUMNAS (del scaler.feature_names_in_)
# =========================
orden_columnas = [
    "diabetes", "historial_familiar", "fuma", "obesidad", "consumo_alcohol",
    "problemas_previos_cardiacos", "uso_medicamentos",
    "edad", "colesterol", "presion_arterial", "frecuencia_cardiaca",
    "horas_ejercicio", "nivel_estres", "horas_dormidas", "bmi",
    "grasas_extras", "actividad_fisica_dias_semana", "horas_sueño",
    "genero_Female", "genero_Male",
    "dieta_Average", "dieta_Healthy", "dieta_Unhealthy",
    "pais_Argentina", "pais_Australia", "pais_Brazil", "pais_Canada", "pais_China",
    "pais_Colombia", "pais_Germany", "pais_India", "pais_Italy", "pais_Japan",
    "pais_New Zealand", "pais_Nigeria", "pais_South Africa", "pais_South Korea",
    "pais_Spain", "pais_Thailand", "pais_United Kingdom", "pais_United States",
    "continente_Africa", "continente_Asia", "continente_Australia",
    "continente_Europe", "continente_North America", "continente_South America",
    "hemisferio_Northern Hemisphere", "hemisferio_Southern Hemisphere"
]

# Validación: debe ser 49
assert len(orden_columnas) == 49, f"Error: se esperan 49 columnas, hay {len(orden_columnas)}"

# =========================
# Entradas
# =========================
st.sidebar.header("Ingrese sus datos")

# Factores de riesgo
diabetes = st.sidebar.radio("Diabetes", ["No", "Sí"])
historial_familiar = st.sidebar.radio("Historial familiar", ["No", "Sí"])
fuma = st.sidebar.radio("Fuma", ["No", "Sí"])
obesidad = st.sidebar.radio("Obesidad", ["No", "Sí"])
consumo_alcohol = st.sidebar.radio("Consumo de alcohol", ["No", "Sí"])
problemas_previos_cardiacos = st.sidebar.radio("Problemas cardíacos previos", ["No", "Sí"])
uso_medicamentos = st.sidebar.radio("Uso de medicamentos", ["No", "Sí"])
 
# Demográficos
genero = st.sidebar.selectbox("Género", ["Female", "Male"])
dieta = st.sidebar.selectbox("Dieta", ["Average", "Healthy", "Unhealthy"])
pais = st.sidebar.selectbox("País", [
    "Argentina", "Australia", "Brazil", "Canada", "China", "Colombia",
    "Germany", "India", "Italy", "Japan", "New Zealand", "Nigeria",
    "South Africa", "South Korea", "Spain", "Thailand",
    "United Kingdom", "United States"
])
continente = st.sidebar.selectbox("Continente", [
    "Africa", "Asia", "Australia", "Europe", "North America", "South America"
])
hemisferio = st.sidebar.radio("Hemisferio", ["Northern Hemisphere", "Southern Hemisphere"])

# Numéricas
edad = st.sidebar.number_input("Edad", 0, 120, 45)
colesterol = st.sidebar.number_input("Colesterol", 100, 400, 200)
presion = st.sidebar.number_input("Presión arterial", 80, 200, 120)
frecuencia = st.sidebar.number_input("Frecuencia cardíaca", 40, 200, 75)
horas_ejercicio = st.sidebar.slider("Horas ejercicio/semana", 0.0, 20.0, 3.0)
nivel_estres = st.sidebar.slider("Nivel de estrés (1-10)", 1, 10, 5)
horas_dormidas = st.sidebar.slider("Horas de sueño/noche", 0.0, 12.0, 7.0)
bmi = st.sidebar.number_input("BMI", 10.0, 50.0, 25.0)
grasas_extras = st.sidebar.number_input("Grasas extra (g/día)", 0.0, 100.0, 20.0)
actividad_fisica = st.sidebar.slider("Días actividad/semana", 0, 7, 3)
horas_sueño = st.sidebar.slider("Horas sueño efectivas", 0.0, 12.0, 7.0)

# =========================
# Construir datos
# =========================
data = {col: 0 for col in orden_columnas}

# Factores de riesgo
data["diabetes"] = 1 if diabetes == "Sí" else 0
data["historial_familiar"] = 1 if historial_familiar == "Sí" else 0
data["fuma"] = 1 if fuma == "Sí" else 0
data["obesidad"] = 1 if obesidad == "Sí" else 0
data["consumo_alcohol"] = 1 if consumo_alcohol == "Sí" else 0
data["problemas_previos_cardiacos"] = 1 if problemas_previos_cardiacos == "Sí" else 0
data["uso_medicamentos"] = 1 if uso_medicamentos == "Sí" else 0

# Género
if genero == "Female":
    data["genero_Female"] = 1
else:
    data["genero_Male"] = 1

# Dieta
if dieta == "Average":
    data["dieta_Average"] = 1
elif dieta == "Healthy":
    data["dieta_Healthy"] = 1
else:
    data["dieta_Unhealthy"] = 1

# País, continente, hemisferio
data[f"pais_{pais}"] = 1
data[f"continente_{continente}"] = 1
data[f"hemisferio_{hemisferio}"] = 1

# Numéricas
data["edad"] = edad
data["colesterol"] = colesterol
data["presion_arterial"] = presion
data["frecuencia_cardiaca"] = frecuencia
data["horas_ejercicio"] = horas_ejercicio
data["nivel_estres"] = nivel_estres
data["horas_dormidas"] = horas_dormidas
data["bmi"] = bmi
data["grasas_extras"] = grasas_extras
data["actividad_fisica_dias_semana"] = actividad_fisica
data["horas_sueño"] = horas_sueño


# =========================
# Escalado y predicción
# =========================
df = pd.DataFrame([data], columns=orden_columnas)
# Mostrar los datos ingresados antes de escalar
st.subheader("Datos ingresados")
st.write(df)
df_scaled = scaler.transform(df)

if st.button("Realizar predicción"):
    pred = modelo.predict(df_scaled)
    prob = modelo.predict_proba(df_scaled)
    st.subheader("Resultado")
    if pred[0] == 1:
        st.error(" ALERTA: Riesgo de ataque cardíaco")
    else:
        st.success(" Estado NORMAL del paciente")
    st.subheader("Probabilidad")
    st.write(f"Probabilidad de riesgo: {prob[0][1]:.2%}")