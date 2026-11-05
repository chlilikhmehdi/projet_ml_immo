import streamlit as st
import joblib
import pandas as pd

# Chargement du modèle
model = joblib.load("model.pkl")

st.title("🏠 Estimation Immobilière")
st.write("Renseignez les caractéristiques du logement.")

# Saisie des caractéristiques
med_inc = st.number_input("Revenu médian", min_value=0.0, value=8.3)
house_age = st.number_input("Âge du logement", min_value=0.0, value=41.0)
ave_rooms = st.number_input("Nombre moyen de pièces", min_value=0.0, value=6.98)
ave_bedrms = st.number_input("Nombre moyen de chambres", min_value=0.0, value=1.02)
population = st.number_input("Population", min_value=0.0, value=322.0)
ave_occup = st.number_input("Nombre moyen d'occupants", min_value=0.0, value=2.55)
latitude = st.number_input("Latitude", value=37.88)
longitude = st.number_input("Longitude", value=-122.23)

if st.button("Estimer le prix"):
    features = pd.DataFrame([{
        "MedInc": med_inc,
        "HouseAge": house_age,
        "AveRooms": ave_rooms,
        "AveBedrms": ave_bedrms,
        "Population": population,
        "AveOccup": ave_occup,
        "Latitude": latitude,
        "Longitude": longitude
    }])

    prediction = model.predict(features)[0]

    st.success(f"Prix estimé : {prediction * 100000:.2f} $")