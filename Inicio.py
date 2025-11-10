import streamlit as st
import pandas as pd
from PIL import Image

# Configuración de la página
st.set_page_config(
    page_title="Sistema de Predicción de Ventas",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header principal
st.markdown('<h1 class="main-header">🏪 Sistema de Predicción de Ventas</h1>', unsafe_allow_html=True)
st.markdown('<h3 style="text-align: center; color: #666;">Análisis y Predicción para Cadena de Retail</h3>', unsafe_allow_html=True)

st.markdown("---")

# Introducción
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<h2 class="sub-header">📋 Descripción del Proyecto</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    Este sistema de **Machine Learning** ha sido desarrollado para predecir las ventas 
    en el mes 24 de tiendas retail, utilizando información:
    
    - 📍 **Geográfica**: Ubicación y densidad poblacional
    - 👥 **Sociodemográfica**: Nivel socioeconómico y características de vivienda
    - 🏬 **Competencia**: Presencia de comercios cercanos
    - 🚶 **Tráfico**: Flujo peatonal y vehicular
    """)
    
    st.markdown('<h2 class="sub-header">🎯 Objetivo</h2>', unsafe_allow_html=True)
    st.info("""
    Identificar los factores clave que impulsan las ventas y predecir el 
    comportamiento de nuevas tiendas antes de su apertura.
    """)

with col2:
    st.markdown('<h2 class="sub-header">📊 Datos del Sistema</h2>', unsafe_allow_html=True)
    
    # Cargar datos para mostrar métricas
    try:
        df = pd.read_csv('data/Tiendas_100.csv')
        
        st.metric("Total de Tiendas", f"{len(df):,}")
        st.metric("Variables Analizadas", f"{len(df.columns)-1}")
        st.metric("Promedio de Ventas", f"${df['ventas_m24'].mean():,.0f}")
        
    except FileNotFoundError:
        st.warning("⚠️ Datos no encontrados. Por favor, verifica la carpeta 'data'.")

st.markdown("---")

# Características del sistema
st.markdown('<h2 class="sub-header">✨ Características del Sistema</h2>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 📊 Análisis Exploratorio
    - Visualizaciones interactivas
    - Estadísticas descriptivas
    - Correlaciones entre variables
    - Distribuciones geográficas
    """)

with col2:
    st.markdown("""
    ### 🤖 Modelos de ML
    - Random Forest
    - XGBoost optimizado
    - Validación cruzada
    - Métricas de performance
    """)

with col3:
    st.markdown("""
    ### 🗺️ Predicciones Espaciales
    - Mapas interactivos
    - Predicciones por ubicación
    - Análisis de zonas
    - Recomendaciones
    """)

st.markdown("---")

# Métricas del modelo
st.markdown('<h2 class="sub-header">📈 Performance del Modelo</h2>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("R² Score", "0.85", "↑ Excelente")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("RMSE", "$748", "↓ Bajo error")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("MAE", "$460", "↓ Preciso")
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("MAPE", "13.38%", "✓ Confiable")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# Instrucciones de uso
st.markdown('<h2 class="sub-header">🚀 Cómo Usar el Sistema</h2>', unsafe_allow_html=True)

with st.expander("📖 Ver Instrucciones Detalladas"):
    st.markdown("""
    ### Página 1: 📊 EDA y Análisis
    1. **Estadísticas Generales**: Visualiza las métricas clave del dataset
    2. **Distribuciones**: Analiza la distribución de ventas y otras variables
    3. **Correlaciones**: Identifica relaciones entre variables
    4. **Análisis Geográfico**: Explora la distribución espacial de las tiendas
    
    ### Página 2: 🗺️ Predicciones Geográficas
    1. **Mapa Interactivo**: Visualiza tiendas existentes y predicciones
    2. **Predicción Individual**: Ingresa datos para predecir ventas de una nueva tienda
    3. **Análisis de Zona**: Evalúa el potencial de diferentes ubicaciones
    4. **Recomendaciones**: Obtén insights basados en el modelo
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>💡 <strong>Desarrollado con</strong> Streamlit, XGBoost y ❤️</p>
    <p>📧 Para soporte técnico, contacta al equipo de Data Science</p>
</div>
""", unsafe_allow_html=True)

# Sidebar con información adicional
with st.sidebar:
    
    st.markdown("### 📊 Variables Principales")
    st.markdown("""
    - Población en radio 100m, 300m, 500m
    - Número de comercios cercanos
    - Nivel socioeconómico
    - Tráfico peatonal y vehicular
    - Competencia en la zona
    """)
    
    st.markdown("### ⚙️ Configuración")
    if st.button("🔄 Recargar Datos"):
        st.rerun()