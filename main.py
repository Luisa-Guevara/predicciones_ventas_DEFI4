import streamlit as st

# Configuración de la página
st.set_page_config(
    page_title="Sistema de Predicción de Ventas",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Definir las páginas
all_pages = [
    st.Page("pages/EDA.py", title="Análisis Exploratorio", default=True),
    st.Page("pages/predicciones.py", title="Predicciones"),
]

# Navegación
pg = st.navigation(all_pages)
pg.run()
