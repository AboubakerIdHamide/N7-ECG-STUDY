import streamlit as st

from app_src.ui.demo import render_demo
from app_src.ui.home import render_home


def main():
    st.set_page_config(
        page_title="CardioPatch · démonstrateur",
        page_icon="💓",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.sidebar.title("CardioPatch")
    page = st.sidebar.radio("Navigation", ["Présentation", "Démonstration"])
    st.sidebar.caption("Étude de cas 3 · quatre jalons")
    if page == "Présentation":
        render_home()
    else:
        render_demo()


if __name__ == "__main__":
    main()
