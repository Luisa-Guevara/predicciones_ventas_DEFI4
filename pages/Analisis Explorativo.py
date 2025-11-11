import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

# Configuración
st.set_page_config(page_title="EDA y Análisis", page_icon="📊", layout="wide")

# CSS personalizado
st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stat-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    </style>
""", unsafe_allow_html=True)

# Título
st.title("Análisis Exploratorio de Datos")
st.markdown("Explora las características de las tiendas y sus ventas")
st.markdown("---")

# Cargar datos
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/Tiendas_100.csv')
        return df
    except FileNotFoundError:
        st.error("❌ Error: No se encontró el archivo de datos")
        return None

df = load_data()

if df is not None:
    # Tabs principales
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Resumen General", 
        "📊 Distribuciones", 
        "🔗 Correlaciones",
        "🗺️ Análisis Geográfico",
        "🏪 Análisis por Tipo"
    ])
    
    # TAB 1: RESUMEN GENERAL
    with tab1:
        st.header("📈 Estadísticas Generales")
        
        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Tiendas", f"{len(df):,}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Venta Promedio", f"${df['ventas_m24'].mean():,.0f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Venta Máxima", f"${df['ventas_m24'].max():,.0f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Venta Mínima", f"${df['ventas_m24'].min():,.0f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Estadísticas descriptivas
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📋 Estadísticas Descriptivas - Variables Numéricas")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            st.dataframe(df[numeric_cols].describe().T, use_container_width=True)
        
        with col2:
            st.subheader("🏷️ Variables Categóricas")
            st.write("**Tipos de Tienda:**")
            store_counts = df['store_cat'].value_counts()
            fig_pie = px.pie(
                values=store_counts.values,
                names=store_counts.index,
                title="Distribución por Tipo de Tienda"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    
    # TAB 2: DISTRIBUCIONES
    with tab2:
        st.header("📊 Análisis de Distribuciones")
        
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
            fig_box = px.box(
                df,
                x='store_cat',
                y='ventas_m24',
                title='Ventas por Tipo de Tienda',
                color='store_cat'
            )
            st.plotly_chart(fig_box, use_container_width=True)
        
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
        
        # Distribuciones múltiples
        st.subheader("📈 Comparación de Variables")
        
        variables = st.multiselect(
            "Selecciona variables para comparar:",
            ['pop_100m', 'pop_300m', 'pop_500m', 'commerces', 'foot_traffic', 
             'car_traffic', 'socio_level', 'competencia'],
            default=['pop_100m', 'commerces', 'foot_traffic']
        )
        
        if variables:
            fig_multi = make_subplots(
                rows=1, cols=len(variables),
                subplot_titles=variables
            )
            
            for i, var in enumerate(variables, 1):
                fig_multi.add_trace(
                    go.Histogram(x=df[var], name=var),
                    row=1, col=i
                )
            
            fig_multi.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_multi, use_container_width=True)
    
    # TAB 3: CORRELACIONES
    with tab3:
        st.header("🔗 Análisis de Correlaciones")
        
        # Seleccionar variables numéricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols.remove('ventas_m24')  # Remover target
        
        # Calcular correlaciones con ventas
        correlations = df[numeric_cols + ['ventas_m24']].corr()['ventas_m24'].sort_values(ascending=False)
        correlations = correlations.drop('ventas_m24')
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Top Correlaciones con Ventas")
            st.dataframe(correlations.head(10), use_container_width=True)
            
            st.markdown("---")
            
            st.subheader("🎯 Insights Clave")
            top_feature = correlations.index[0]
            st.success(f"✅ **{top_feature}** tiene la mayor correlación: {correlations.iloc[0]:.3f}")
            st.info(f"💡 Variables poblacionales son predictores fuertes")
        
        with col2:
            # Heatmap de correlaciones
            fig_corr = px.imshow(
                df[numeric_cols + ['ventas_m24']].corr(),
                title='Matriz de Correlación',
                color_continuous_scale='RdBu_r',
                aspect='auto'
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        
        # Scatter matrix de top variables
        st.subheader("🔍 Relaciones entre Top Variables")
        top_vars = correlations.head(5).index.tolist() + ['ventas_m24']
        fig_scatter_matrix = px.scatter_matrix(
            df[top_vars],
            dimensions=top_vars,
            color='ventas_m24',
            title="Matriz de Dispersión - Top 5 Variables"
        )
        fig_scatter_matrix.update_traces(diagonal_visible=False)
        st.plotly_chart(fig_scatter_matrix, use_container_width=True)
    
    # TAB 4: ANÁLISIS GEOGRÁFICO
    with tab4:
        st.header("🗺️ Distribución Geográfica")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Scatter geográfico
            fig_geo = px.scatter(
                df,
                x='lon',
                y='lat',
                color='ventas_m24',
                size='ventas_m24',
                hover_data=['Tienda', 'store_cat'],
                title='Ubicación de Tiendas',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_geo, use_container_width=True)
        
        with col2:
            # Density plot
            fig_density = px.density_contour(
                df,
                x='lon',
                y='lat',
                z='ventas_m24',
                title='Mapa de Densidad de Ventas'
            )
            fig_density.update_traces(contours_coloring="fill", contours_showlabels=True)
            st.plotly_chart(fig_density, use_container_width=True)
        
        # Estadísticas por zona
        st.subheader("📍 Estadísticas por Ubicación")
        
        # Crear cuadrantes
        df['zona_lat'] = pd.cut(df['lat'], bins=3, labels=['Sur', 'Centro', 'Norte'])
        df['zona_lon'] = pd.cut(df['lon'], bins=3, labels=['Oeste', 'Centro', 'Este'])
        df['zona'] = df['zona_lat'].astype(str) + '-' + df['zona_lon'].astype(str)
        
        zona_stats = df.groupby('zona').agg({
            'ventas_m24': ['mean', 'count'],
            'pop_100m': 'mean',
            'competencia': 'mean'
        }).round(0)
        
        st.dataframe(zona_stats, use_container_width=True)
    
    # TAB 5: ANÁLISIS POR TIPO
    with tab5:
        st.header("🏪 Análisis por Tipo de Tienda")
        
        tipo_tienda = st.selectbox(
            "Selecciona tipo de tienda:",
            df['store_cat'].unique()
        )
        
        df_tipo = df[df['store_cat'] == tipo_tienda]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Número de Tiendas", len(df_tipo))
        with col2:
            st.metric("Venta Promedio", f"${df_tipo['ventas_m24'].mean():,.0f}")
        with col3:
            st.metric("Venta Mediana", f"${df_tipo['ventas_m24'].median():,.0f}")
        
        # Comparación con otros tipos
        st.subheader("📊 Comparación con Otros Tipos")
        
        comparison = df.groupby('store_cat').agg({
            'ventas_m24': ['mean', 'median', 'std'],
            'pop_100m': 'mean',
            'competencia': 'mean'
        }).round(2)
        
        st.dataframe(comparison, use_container_width=True)
        
        # Gráfico de comparación
        fig_comp = px.box(
            df,
            x='store_cat',
            y='ventas_m24',
            title='Distribución de Ventas por Tipo de Tienda',
            color='store_cat'
        )
        st.plotly_chart(fig_comp, use_container_width=True)

else:
    st.error("No se pudieron cargar los datos. Verifica que el archivo esté en la carpeta 'data'.")