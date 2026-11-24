import streamlit as st

from app_src.config import MILESTONES


def render_home():
    st.title("CardioPatch · démonstrateur des quatre jalons")
    st.write(
        "Choisissez un jalon, observez une fenêtre ECG de validation, puis lancez "
        "le modèle correspondant. Chaque jalon montre une façon différente de "
        "représenter le même type de signal."
    )

    st.warning(
        "Démonstration pédagogique sur des signaux simulés. Les prédictions ne "
        "sont pas des diagnostics et ne doivent pas guider une décision médicale."
    )

    st.subheader("Parcours dans l’application")
    for name, details in MILESTONES.items():
        with st.container(border=True):
            st.markdown(f"### {name}")
            st.write(details["input_description"])
            st.caption(f"Source de démonstration : {details['source']} · {details['metrics']}")

    st.info(
        "Les exemples disponibles viennent des partitions de validation. "
        "Le jeu de test n’est pas chargé dans l’application. Les résultats "
        "historiques affichés servent de repères et ne sont pas recalculés ici."
    )
