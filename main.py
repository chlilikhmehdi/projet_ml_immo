
import streamlit as st
import joblib
import pandas as pd
from huggingface_hub import hf_hub_download

# ----------------------------------
# Configuration de l'application
# ----------------------------------

st.set_page_config(
    page_title="Estimation Immobilière",
    page_icon="🏠",
    layout="centered"
)

# ----------------------------------
# Chargement du modèle depuis Hugging Face
# ----------------------------------

@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="mehdichlilikh/projet_ml_immo",
        filename="model.pkl"
    )

    return joblib.load(model_path)


try:
    model = load_model()
except Exception as e:
    st.error("Impossible de charger le modèle Machine Learningw.")
    st.exception(e)
    st.stop()


# ----------------------------------
# Interface utilisateur
# ----------------------------------

st.title("🏠 Estimation Immobilière")

st.write(
    "Renseignez les caractéristiques du logement "
    "pour obtenir une estimation de son prix."
)

st.divider()

# ----------------------------------
# Saisie des caractéristiques
# ----------------------------------

st.subheader("📋 Caractéristiques du logement")

med_inc = st.number_input(
    "Revenu médian",
    min_value=0.0,
    value=8.3
)

house_age = st.number_input(
    "Âge du logement",
    min_value=0.0,
    value=41.0
)

ave_rooms = st.number_input(
    "Nombre moyen de pièces",
    min_value=0.0,
    value=6.98
)

ave_bedrms = st.number_input(
    "Nombre moyen de chambres",
    min_value=0.0,
    value=1.02
)

population = st.number_input(
    "Population",
    min_value=0.0,
    value=322.0
)

ave_occup = st.number_input(
    "Nombre moyen d'occupants",
    min_value=0.0,
    value=2.55
)

latitude = st.number_input(
    "Latitude",
    value=37.88
)

longitude = st.number_input(
    "Longitude",
    value=-122.23
)

# ----------------------------------
# Prédiction
# ----------------------------------

st.divider()

if st.button("🏡 Estimer le prix", use_container_width=True):

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

    try:
        prediction = model.predict(features)[0]

        estimated_price = prediction * 100000

        st.success("Estimation réalisée avec succès !")

        st.metric(
            label="💰 Prix estimé du logement",
            value=f"${estimated_price:,.2f}"
        )

        st.dataframe(
            features,
            use_container_width=True,
            hide_index=True
        )

    except Exception as e:
        st.error(f"Erreur pendant la prédiction : {e}")

# ----------------------------------
# Pied de page
# ----------------------------------

st.divider()

st.caption(
    "Application de Machine Learning - Estimation Immobilière"
)
