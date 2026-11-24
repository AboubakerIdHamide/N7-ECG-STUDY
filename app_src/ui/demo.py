import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from app_src.config import FS, MILESTONES, RHYTHMS
from app_src.data import load_demo_examples
from app_src.inference.predict import predict_window


def _sample_fields(source):
    if source == "Holter A":
        return "holter_raw", "holter_labels", "holter_age", "holter_sex", "holter_validation_indices"
    return "patch_raw", "patch_labels", None, None, "patch_validation_indices"


def render_demo():
    st.title("Faire une prédiction")
    jalon = st.selectbox("Modèle du jalon", list(MILESTONES))
    details = MILESTONES[jalon]
    examples = load_demo_examples()
    raw_key, label_key, age_key, sex_key, index_key = _sample_fields(details["source"])

    column_signal, column_details = st.columns([2.2, 1])
    with column_details:
        sample_position = st.selectbox(
            "Exemple de validation",
            range(len(examples[raw_key])),
            format_func=lambda i: f"Exemple {i + 1}",
        )
        window = examples[raw_key][sample_position]
        true_class = int(examples[label_key][sample_position])
        age = float(examples[age_key][sample_position]) if age_key else 50.0
        sex_code = float(examples[sex_key][sample_position]) if sex_key else 0.0
        st.caption(f"Fenêtre de validation #{int(examples[index_key][sample_position])} · {details['source']}")
        if jalon.startswith("Jalon 1"):
            st.caption(f"Métadonnées utilisées par le MLP : âge {age:.0f} ans, code sexe {sex_code:g}.")
        st.write(details["input_description"])

    with column_signal:
        time = [i / FS for i in range(len(window))]
        fig, ax = plt.subplots(figsize=(10, 3.1))
        ax.plot(time, window, color="#15857d", linewidth=0.8)
        ax.set(xlim=(0, 10), xlabel="Temps (secondes)", ylabel="Amplitude du signal (mV)")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    if st.button("Prédire le rythme", type="primary", width="stretch"):
        with st.spinner("Le modèle traite la fenêtre…"):
            result = predict_window(jalon, window, age=age, sex_code=sex_code)
        st.session_state["cardiopatch_result"] = (jalon, sample_position, result)

    saved = st.session_state.get("cardiopatch_result")
    if not saved or saved[:2] != (jalon, sample_position):
        st.caption("Lancez une prédiction pour afficher le résultat.")
        return

    _, _, result = saved
    pred = result["predicted_class"]
    st.divider()
    st.subheader("Résultat du modèle")
    result_col, chart_col = st.columns([1, 2])
    with result_col:
        st.metric("Rythme prédit", RHYTHMS[pred])
        st.metric("Score du modèle", f"{result['scores'][pred] * 100:.1f} %")
        if result["threshold"] is not None:
            status = "Alerte FA" if result["alert"] else "Pas d’alerte FA"
            st.metric(status, f"seuil {result['threshold']:.5f}")
    with chart_col:
        fig_scores, ax_scores = plt.subplots(figsize=(8, 3.2))
        class_names = [RHYTHMS[i] for i in range(5)]
        ax_scores.barh(class_names, result["scores"] * 100, color="#15857d")
        ax_scores.invert_yaxis()
        ax_scores.set(xlim=(0, 100), xlabel="Score du modèle (%)")
        ax_scores.grid(axis="x", alpha=0.2)
        fig_scores.tight_layout()
        st.pyplot(fig_scores, width="stretch")
        plt.close(fig_scores)
        st.caption("Scores produits par le modèle ; ils ne sont pas garantis comme probabilités calibrées.")

    if result["extra"].get("features"):
        with st.expander("Voir les 11 descripteurs remis au MLP (Jalon 1)"):
            st.dataframe(
                pd.DataFrame(result["extra"]["features"], columns=["Descripteur", "Valeur"]),
                hide_index=True,
                width="stretch",
            )
        st.info(
            "Le notebook ne contenait pas le fichier de poids du MLP. Le poids fourni "
            "pour la démo a été reconstruit avec le même entraînement documenté. Son "
            "F1 macro mesuré sur validation est 0,967, contre 0,970 dans le notebook."
        )
    if result["extra"].get("beats") is not None:
        with st.expander("Voir les battements transformés en lignes RR / QRS (Jalon 4)"):
            beats = result["extra"]["beats"]
            if len(beats):
                st.dataframe(
                    pd.DataFrame(beats, columns=["Intervalle RR (s)", "Amplitude QRS (mV)"]),
                    hide_index=True,
                    width="stretch",
                )
            else:
                st.info("Aucun intervalle RR n’a été détecté dans cette fenêtre par le détecteur reconstruit.")

    with st.expander("Révéler la vraie étiquette de cet exemple"):
        st.write(f"Étiquette de référence dans le fichier de validation : **{RHYTHMS[true_class]}**.")
        st.caption("Cette comparaison illustre un exemple, elle ne remplace pas les métriques calculées sur l’ensemble de validation.")
