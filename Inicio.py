import streamlit as st
import pandas as pd
from PIL import Image

# Configuración de la página
st.set_page_config(
    page_title="Sistema de Predicción de Ventas",
    page_icon="📊",
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
st.markdown('<h1 class="main-header">Sistema de Predicción de Ventas</h1>', unsafe_allow_html=True)
st.markdown('<h3 style="text-align: center; color: #666;">Análisis y Predicción para Cadena de Retail</h3>', unsafe_allow_html=True)
st.markdown('<h5 style="text-align: center; color: #666;">Juan David Bocanegra, María José Castillo y Luisa Guevara</h5>', unsafe_allow_html=True)

st.markdown("---")

# Introducción
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<h2 class="sub-header">Descripción del Proyecto</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    Este modelo de **Machine Learning** ha sido desarrollado para predecir las ventas 
    en el mes 24 de tiendas retail, utilizando información:
    
    - **Geográfica**: Ubicación y densidad poblacional
    - **Sociodemográfica**: Nivel socioeconómico y características de vivienda
    - **Competencia**: Presencia de comercios cercanos
    - **Tráfico**: Flujo peatonal y vehicular
    """)
    
    st.markdown('<h2 class="sub-header">Objetivo</h2>', unsafe_allow_html=True)
    st.info("""
    Identificar los factores clave que impulsan las ventas y predecir el 
    comportamiento de nuevas tiendas antes de su apertura.
    """)

with col2:
    st.markdown('<h2 class="sub-header">Datos Importantes</h2>', unsafe_allow_html=True)
    
    # Cargar datos para mostrar métricas
    try:
        df = pd.read_csv('data/Tiendas_100.csv')
        
        st.metric("Total de Tiendas", f"{len(df):,}")
        st.metric("Variables Analizadas", f"{len(df.columns)-1}")
        st.metric("Promedio de Ventas", f"${df['ventas_m24'].mean():,.0f}")
        
    except FileNotFoundError:
        st.warning("⚠️ Datos no encontrados. Por favor, verifica la carpeta 'data'.")

st.markdown("---")

st.markdown('<h2 class="sub-header">Diccionario de Datos</h2>', unsafe_allow_html=True)

try:
    df_tiendas = pd.read_csv('data/Tiendas_100.csv')
    df_ventas = pd.read_csv('data/Ventas_funcioanles.csv')

    # Mostrar vista previa
    st.subheader("Tiendas_100.csv (Top 5 registros)")
    st.dataframe(df_tiendas.head(5), use_container_width=True)

    st.subheader("Ventas_funcionales.csv (Top 5 registros)")
    st.dataframe(df_ventas.head(5), use_container_width=True)

    # Diccionario de datos (personalízalo si quieres)
    st.markdown("### Variables Principales - Tiendas_100.csv")
    dict_tiendas = {
        "Tienda": "Nombre o identificador de la tienda",
        "lat": "Latitud geográfica",
        "lon": "Longitud geográfica",
        "store_cat": "Categoría de la tienda",
        "ventas_m24": "Ventas en el mes 24",
        "pop_100m / 300m / 500m": "Población en distintos radios",
        "commerces": "Número de comercios cercanos",
        "foot_traffic": "Tráfico peatonal promedio",
        "car_traffic": "Tráfico vehicular promedio",
        "socio_level": "Nivel socioeconómico del área",
        "competencia": "Número de tiendas competidoras"
    }
    st.table(pd.DataFrame(list(dict_tiendas.items()), columns=["Variable", "Descripción"]))

    st.markdown("### Variables Principales - Ventas_funcioanles.csv")
    dict_ventas = {
        "Tienda": "Nombre o identificador de la tienda",
        "mes": "Número del mes analizado",
        "ventas": "Monto total de ventas en ese mes",
        "clientes": "Número de clientes atendidos",
        "promedio_ticket": "Valor promedio del ticket de venta"
    }
    st.table(pd.DataFrame(list(dict_ventas.items()), columns=["Variable", "Descripción"]))

except FileNotFoundError as e:
    st.error(f"⚠️ Error al cargar los datos: {e}")



# Instrucciones de uso

st.markdown('<h2 class="sub-header">¿Cómo Usar el Sistema?</h2>', unsafe_allow_html=True)

with st.expander("Ver Instrucciones Detalladas"):
    st.markdown("""
    ### Página 1: EDA y Análisis
    1. **Estadísticas Generales**: Visualiza las métricas clave del dataset
    2. **Distribuciones**: Analiza la distribución de ventas y otras variables
    3. **Correlaciones**: Identifica relaciones entre variables
    4. **Análisis Geográfico**: Explora la distribución espacial de las tiendas
    
    ### Página 2: Predicciones Geográficas
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