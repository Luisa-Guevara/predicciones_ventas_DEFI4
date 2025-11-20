import streamlit as st
import hydralit_components as hc

# Configuración de la página
st.set_page_config(
    page_title="Sistema de Estimación de Ventas",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Estilos CSS para ajustar el ancho y padding
max_width_str = f"max-width: {90}%;"
st.markdown(f"""
    <style>
    .appview-container .main .block-container{{{max_width_str}}}
    </style>
    """,
            unsafe_allow_html=True,
            )

st.markdown("""
    <style>
    .block-container {
        padding-top: 0rem;
        padding-bottom: 0rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Definir opciones del menú
eda_tab = 'Análisis General'
pred_tab = 'Estimaciones'

option_data = [
    {'icon': "", 'label': eda_tab},
    {'icon': "", 'label': pred_tab}
]

# Tema del menú
theme = {
    'menu_background': '#F1F1F1',
    'txc_inactive': '#999999',
    'txc_active': '#FFFFFF',
    'option_active': '#00bf63'
}

# Crear el menú de navegación horizontal
chosen_tab = hc.option_bar(
    option_definition=option_data,
    title='',
    key='PrimaryOptionx',
    override_theme=theme,
    horizontal_orientation=True
)

# Cargar la página según la selección
if chosen_tab == eda_tab:
    exec(open("pages/EDA.py", encoding="utf-8").read())

elif chosen_tab == pred_tab:
    exec(open("pages/predicciones.py", encoding="utf-8").read())

    
