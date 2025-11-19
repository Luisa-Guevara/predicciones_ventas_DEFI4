import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_extras.metric_cards import style_metric_cards

st.markdown("""
    <style>
    /* Centrar y expandir tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        justify-content: center;
        width: 100%;
    }
    
    /* Hacer que cada tab ocupe más espacio */
    .stTabs [data-baseweb="tab"] {
        flex-grow: 1;
        text-align: center;
        padding: 1rem 2rem;
        font-size: 1.1rem;
        font-weight: 500;
    }
    
    /* Opcional: Borde inferior para el tab activo */
    .stTabs [aria-selected="true"] {
        border-bottom: 3px solid #00bf63;
    }
    
    /* Contenedor de tabs con ancho completo */
    .stTabs {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

st.set_page_config(page_title="General", page_icon="", layout="wide")


# st.title("Estadísticas Generales")
# st.markdown("Explora las características de las tiendas y sus ventas")
# st.markdown("---")


@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/Tiendas_100.csv')
        return df
    except FileNotFoundError:
        st.error(" Error: No se encontró el archivo de datos")
        return None


df = load_data()


if df is not None:
    
    tab1, tab2, tab3 = st.tabs([
        "Resumen General",
        "Distribuciones",
        "Correlaciones",
    ])

    # TAB 1: RESUMEN GENERAL
    with tab1:

        # Filtro de selección múltiple
        categorias_disponibles = list(df['store_cat'].unique())
        categorias_seleccionadas = st.multiselect(
            "Filtrar por Tipo de Tienda:",
            options=categorias_disponibles,
            default=categorias_disponibles,
            help="Selecciona una o varias categorías. Por defecto muestra todas."
        )

        # Filtrar dataframe según selección
        if categorias_seleccionadas:
            df_filtrado = df[df['store_cat'].isin(categorias_seleccionadas)]
        else:
            df_filtrado = df  # Si no hay selección, mostrar todo

        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Tiendas", f"{len(df_filtrado):,}")

        with col2:
            st.metric("Venta Mínima",
                      f"${df_filtrado['ventas_m24'].min():,.0f}")

        with col3:
            st.metric("Venta Promedio",
                      f"${df_filtrado['ventas_m24'].mean():,.0f}")

        with col4:
            st.metric("Venta Máxima",
                      f"${df_filtrado['ventas_m24'].max():,.0f}")

        # Aplicar estilo a las tarjetas
        style_metric_cards(
            background_color='rgba(255, 255, 255, 0.05)',
            border_left_color="#00bf63",
            border_color="#e0e0e0",
            box_shadow="0 4px 6px rgba(0,191,99,0.2)"
        )

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            store_counts = df_filtrado['store_cat'].value_counts()
            fig_pie = px.pie(
                values=store_counts.values,
                names=store_counts.index,
                title="Distribución por Tipo de Tienda"
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            socio_counts = df_filtrado['socio_level'].value_counts()
            fig_bar = px.bar(
                x=socio_counts.index,
                y=socio_counts.values,
                title="Distribución de Socio Level",
                labels={'x': 'Nivel Socioeconómico', 'y': 'Frecuencia'}
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    # TAB 2: DISTRIBUCIONES
    with tab2:
        st.header("Análisis de Distribuciones")

        col1, col2 = st.columns(2)

        with col1:
            # Distribución de ventas
            fig_hist = px.histogram(
                df,
                x='ventas_m24',
                nbins=30,
                title='Distribución de Ventas (Mes 24)',
                labels={'ventas_m24': 'Ventas'},
                color_discrete_sequence=['#667eea']
            )
            fig_hist.update_layout(showlegend=False)
            st.plotly_chart(fig_hist, use_container_width=True)

            # Box plot de ventas por tipo de tienda
            # fig_box = px.box(
            #     df,
            #     x='store_cat',
            #     y='ventas_m24',
            #     title='Ventas por Tipo de Tienda',
            #     color='store_cat'
            # )
            # st.plotly_chart(fig_box, use_container_width=True)

        with col2:
            # Distribución de población
            fig_pop = px.histogram(
                df,
                x='pop_100m',
                nbins=30,
                title='Distribución de Población (100m)',
                labels={'pop_100m': 'Población'},
                color_discrete_sequence=['#764ba2']
            )
            fig_pop.update_layout(showlegend=False)
            st.plotly_chart(fig_pop, use_container_width=True)

        # Scatter: población vs ventas
        fig_scatter = px.scatter(
            df,
            x='pop_100m',
            y='ventas_m24',
            color='store_cat',
            size='foot_traffic',
            hover_data=['Tienda'],
            title='Población vs Ventas (tamaño = tráfico peatonal)'
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # TAB 3: CORRELACIONES
    with tab3:
        st.header("Análisis de Correlaciones")

        st.subheader("Insights Clave")
        col1, col2 = st.columns(2)

        with col1:
            st.success(
                "**viviendas_100m** tiene la mayor correlación con ventas (0.923).")
            st.info(
                "Variables poblacionales cercanas son los predictores más fuertes.")

        with col2:
            st.info(
                "Zonas con más tráfico peatonal y oficinas también presentan mejores ventas.")
            st.warning(
                "Competencia cercana y gasolineras no muestran correlaciones relevantes.")
        
        st.markdown("---")
        
        # Seleccionar variables numéricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols.remove('ventas_m24')  # Remover target

        # Calcular correlaciones con ventas
        correlations = df[numeric_cols + ['ventas_m24']
                          ].corr()['ventas_m24'].sort_values(ascending=False)
        correlations = correlations.drop('ventas_m24')

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Top Correlaciones con Ventas")
            st.dataframe(correlations.head(10), use_container_width=True)

        with col2:
            # Heatmap de correlaciones
            fig_corr = px.imshow(
                df[numeric_cols + ['ventas_m24']].corr(),
                title='Matriz de Correlación',
                color_continuous_scale='RdBu_r',
                aspect='auto'
            )
            st.plotly_chart(fig_corr, use_container_width=True)
    
    st.markdown("---")

else:
    st.error(
        "No se pudieron cargar los datos. Verifica que el archivo esté en la carpeta 'data'.")
