import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import folium_static
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import joblib
import os
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Configuración
st.set_page_config(page_title="Predicciones Geográficas", page_icon="🗺️", layout="wide")

# CSS personalizado
st.markdown("""
    <style>
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    .input-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .stAlert {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Título
st.title("🗺️ Predicciones Geográficas de Ventas")
st.markdown("Predice ventas para nuevas ubicaciones y visualiza el mapa de oportunidades")
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

# Entrenar o cargar modelo
@st.cache_resource
def train_model(df):
    # Preparar datos
    feature_cols = [
        'lat', 'lon', 'store_cat', 'pop_100m', 'pop_300m', 'pop_500m',
        'commerces', 'gas_stations', 'malls', 'foot_traffic', 'car_traffic',
        'socio_level', 'viviendas_100m', 'oficinas_100m', 'viviendas_pobreza',
        'competencia', 'tiendas_peq'
    ]
    
    X = df[feature_cols].copy()
    y = df["ventas_m24"].copy()
    
    # Separar train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=28
    )
    
    # Preprocesamiento
    cat_features = ["store_cat"]
    num_features = [c for c in feature_cols if c not in cat_features]
    
    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
            ("num", "passthrough", num_features)
        ]
    )
    
    # Modelo Random Forest
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )
    
    # Pipeline
    pipe = Pipeline(steps=[
        ("prep", preprocess),
        ("model", rf)
    ])
    
    # Entrenar
    pipe.fit(X_train, y_train)
    
    # Guardar modelo
    os.makedirs('models', exist_ok=True)
    joblib.dump(pipe, 'models/sales_model.pkl')
    
    return pipe, X_test, y_test

df = load_data()

if df is not None:
    # Entrenar modelo
    with st.spinner("🔄 Entrenando modelo..."):
        model, X_test, y_test = train_model(df)
    

        # --- Evaluación del modelo ---
    st.subheader("📈 Evaluación del Modelo de Predicción")

    # Predicciones en el set de prueba
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("R² Score", f"{r2:.3f}")
    with col2:
        st.metric("MAE", f"${mae:,.0f}")
    with col3:
        st.metric("RMSE", f"${rmse:,.0f}")

    st.markdown("---")

    # Importancia de variables
    st.subheader("🔍 Importancia de Variables en el Modelo")

    try:
        # Recuperar nombres de features tras one-hot encoding
        feature_names = model.named_steps["prep"].get_feature_names_out()
        importances = model.named_steps["model"].feature_importances_

        importance_df = pd.DataFrame({
            "Variable": feature_names,
            "Importancia": importances
        }).sort_values(by="Importancia", ascending=False)

        fig_imp = px.bar(
            importance_df.head(15),
            x="Importancia",
            y="Variable",
            orientation="h",
            title="Principales Variables Predictoras",
            color="Importancia",
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig_imp, use_container_width=True)
    except Exception as e:
        st.warning(f"No se pudo calcular la importancia de variables: {e}")

    st.markdown("---")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🗺️ Mapa Interactivo",
        "🎯 Predicción Individual",
        "📊 Análisis de Zona",
        "💡 Recomendaciones"
    ])
    
    # TAB 1: MAPA INTERACTIVO
    with tab1:
        st.header("🗺️ Mapa de Tiendas y Predicciones")
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.subheader("⚙️ Filtros")
            
            # Filtros
            tipo_tienda_filter = st.multiselect(
                "Tipo de Tienda:",
                df['store_cat'].unique(),
                default=df['store_cat'].unique()
            )
            
            venta_min, venta_max = st.slider(
                "Rango de Ventas:",
                float(df['ventas_m24'].min()),
                float(df['ventas_m24'].max()),
                (float(df['ventas_m24'].min()), float(df['ventas_m24'].max()))
            )
            
            mostrar_heatmap = st.checkbox("Mostrar mapa de calor", value=False)
        
        with col1:
            # Filtrar datos
            df_filtered = df[
                (df['store_cat'].isin(tipo_tienda_filter)) &
                (df['ventas_m24'] >= venta_min) &
                (df['ventas_m24'] <= venta_max)
            ]
            
            # Crear mapa base
            center_lat = df_filtered['lat'].mean()
            center_lon = df_filtered['lon'].mean()
            
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=12,
                tiles='OpenStreetMap'
            )
            
            # Agregar marcadores
            for idx, row in df_filtered.iterrows():
                # Color según ventas
                if row['ventas_m24'] > df['ventas_m24'].quantile(0.75):
                    color = 'green'
                    icon = 'star'
                elif row['ventas_m24'] > df['ventas_m24'].median():
                    color = 'blue'
                    icon = 'shopping-cart'
                else:
                    color = 'orange'
                    icon = 'shopping-cart'
                
                folium.Marker(
                    location=[row['lat'], row['lon']],
                    popup=f"""
                    <div style='width: 200px'>
                        <b>{row['Tienda']}</b><br>
                        <b>Tipo:</b> {row['store_cat']}<br>
                        <b>Ventas:</b> ${row['ventas_m24']:,.0f}<br>
                        <b>Población 100m:</b> {row['pop_100m']:.0f}<br>
                        <b>Comercios:</b> {row['commerces']}
                    </div>
                    """,
                    icon=folium.Icon(color=color, icon=icon, prefix='fa'),
                    tooltip=f"{row['Tienda']}: ${row['ventas_m24']:,.0f}"
                ).add_to(m)
            
            # Mapa de calor si está activado
            if mostrar_heatmap:
                from folium.plugins import HeatMap
                heat_data = [[row['lat'], row['lon'], row['ventas_m24']] 
                            for idx, row in df_filtered.iterrows()]
                HeatMap(heat_data).add_to(m)
            
            # Agregar leyenda
            legend_html = '''
            <div style="position: fixed; 
                        bottom: 50px; right: 50px; width: 200px; height: 120px; 
                        background-color: white; border:2px solid grey; z-index:9999; 
                        font-size:14px; padding: 10px">
                <p><i class="fa fa-star" style="color:green"></i> Ventas Altas (>75%)</p>
                <p><i class="fa fa-shopping-cart" style="color:blue"></i> Ventas Medias</p>
                <p><i class="fa fa-shopping-cart" style="color:orange"></i> Ventas Bajas</p>
            </div>
            '''
            m.get_root().html.add_child(folium.Element(legend_html))
            
            folium_static(m, width=800, height=600)
        
        # Estadísticas del área filtrada
        st.markdown("---")
        st.subheader("📊 Estadísticas del Área Seleccionada")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Tiendas", len(df_filtered))
        with col2:
            st.metric("Venta Promedio", f"${df_filtered['ventas_m24'].mean():,.0f}")
        with col3:
            st.metric("Venta Total", f"${df_filtered['ventas_m24'].sum():,.0f}")
        with col4:
            st.metric("Densidad Promedio", f"{df_filtered['pop_100m'].mean():.0f}")
    
    # TAB 2: PREDICCIÓN INDIVIDUAL
    with tab2:
        st.header("🎯 Predicción de Ventas para Nueva Tienda")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="input-section">', unsafe_allow_html=True)
            st.subheader("📍 Ubicación y Características")
            
            # Inputs geográficos
            col_a, col_b = st.columns(2)
            with col_a:
                lat_pred = st.number_input(
                    "Latitud",
                    value=float(df['lat'].mean()),
                    format="%.6f",
                    help="Coordenada de latitud de la nueva tienda"
                )
            with col_b:
                lon_pred = st.number_input(
                    "Longitud",
                    value=float(df['lon'].mean()),
                    format="%.6f",
                    help="Coordenada de longitud de la nueva tienda"
                )
            
            store_cat_pred = st.selectbox(
                "Tipo de Tienda",
                df['store_cat'].unique(),
                help="Categoría de la tienda"
            )
            
            st.markdown("---")
            st.subheader("👥 Datos Demográficos")
            
            col_c, col_d = st.columns(2)
            with col_c:
                pop_100m = st.number_input(
                    "Población 100m",
                    value=float(df['pop_100m'].mean()),
                    help="Población en radio de 100 metros"
                )
                pop_300m = st.number_input(
                    "Población 300m",
                    value=float(df['pop_300m'].mean()),
                    help="Población en radio de 300 metros"
                )
            with col_d:
                pop_500m = st.number_input(
                    "Población 500m",
                    value=float(df['pop_500m'].mean()),
                    help="Población en radio de 500 metros"
                )
                socio_level = st.slider(
                    "Nivel Socioeconómico",
                    1, 6, 3,
                    help="Nivel socioeconómico del área (1=bajo, 6=alto)"
                )
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="input-section">', unsafe_allow_html=True)
            st.subheader("🏬 Entorno Comercial")
            
            col_e, col_f = st.columns(2)
            with col_e:
                commerces = st.number_input(
                    "Comercios Cercanos",
                    value=int(df['commerces'].mean()),
                    help="Número de comercios en el área"
                )
                gas_stations = st.number_input(
                    "Estaciones de Gasolina",
                    value=int(df['gas_stations'].mean()),
                    help="Estaciones de servicio cercanas"
                )
                malls = st.selectbox(
                    "Centros Comerciales",
                    [0, 1],
                    help="Presencia de centros comerciales (0=No, 1=Sí)"
                )
            with col_f:
                foot_traffic = st.number_input(
                    "Tráfico Peatonal",
                    value=float(df['foot_traffic'].mean()),
                    help="Flujo peatonal estimado"
                )
                car_traffic = st.number_input(
                    "Tráfico Vehicular",
                    value=float(df['car_traffic'].mean()),
                    help="Flujo vehicular estimado"
                )
                competencia = st.number_input(
                    "Competencia",
                    value=int(df['competencia'].mean()),
                    help="Número de competidores cercanos"
                )
            
            st.markdown("---")
            st.subheader("🏘️ Características del Área")
            
            col_g, col_h = st.columns(2)
            with col_g:
                viviendas_100m = st.number_input(
                    "Viviendas 100m",
                    value=int(df['viviendas_100m'].mean())
                )
                oficinas_100m = st.number_input(
                    "Oficinas 100m",
                    value=int(df['oficinas_100m'].mean())
                )
            with col_h:
                viviendas_pobreza = st.number_input(
                    "Viviendas en Pobreza",
                    value=int(df['viviendas_pobreza'].mean())
                )
                tiendas_peq = st.number_input(
                    "Tiendas Pequeñas",
                    value=int(df['tiendas_peq'].mean())
                )
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Botón de predicción
        st.markdown("---")
        if st.button("🚀 Realizar Predicción", type="primary", use_container_width=True):
            # Crear dataframe con inputs
            new_data = pd.DataFrame({
                'lat': [lat_pred],
                'lon': [lon_pred],
                'store_cat': [store_cat_pred],
                'pop_100m': [pop_100m],
                'pop_300m': [pop_300m],
                'pop_500m': [pop_500m],
                'commerces': [commerces],
                'gas_stations': [gas_stations],
                'malls': [malls],
                'foot_traffic': [foot_traffic],
                'car_traffic': [car_traffic],
                'socio_level': [socio_level],
                'viviendas_100m': [viviendas_100m],
                'oficinas_100m': [oficinas_100m],
                'viviendas_pobreza': [viviendas_pobreza],
                'competencia': [competencia],
                'tiendas_peq': [tiendas_peq]
            })
            
            # Predecir
            prediction = model.predict(new_data)[0]
            
            # Mostrar resultado
            st.markdown("---")
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            st.markdown("## 💰 Predicción de Ventas")
            st.markdown(f"### ${prediction:,.2f}")
            st.markdown("Ventas estimadas para el mes 24")
            st.markdown('</div>', unsafe_allow_html=True)
            
                        # Opción para exportar predicción
            st.download_button(
                label="💾 Descargar Resultados",
                data=new_data.assign(prediccion_ventas=prediction).to_csv(index=False),
                file_name="prediccion_nueva_tienda.csv",
                mime="text/csv"
            )


            # Análisis comparativo
            col1, col2, col3 = st.columns(3)
            
            percentile = (df['ventas_m24'] < prediction).mean() * 100
            
            with col1:
                st.metric(
                    "Comparación con Promedio",
                    f"{((prediction / df['ventas_m24'].mean() - 1) * 100):.1f}%",
                    delta=f"${(prediction - df['ventas_m24'].mean()):,.0f}"
                )
            with col2:
                st.metric(
                    "Percentil",
                    f"{percentile:.0f}%",
                    help="Porcentaje de tiendas con ventas menores"
                )
            with col3:
                if prediction > df['ventas_m24'].quantile(0.75):
                    clasificacion = "🟢 Excelente"
                elif prediction > df['ventas_m24'].median():
                    clasificacion = "🟡 Buena"
                else:
                    clasificacion = "🟠 Regular"
                st.metric("Clasificación", clasificacion)
            
            # Recomendaciones
            st.markdown("---")
            st.subheader("💡 Análisis y Recomendaciones")
            
            if prediction > df['ventas_m24'].quantile(0.75):
                st.success("""
                ✅ **Ubicación Altamente Prometedora**
                - Las ventas proyectadas superan el 75% de las tiendas existentes
                - Excelente potencial de retorno de inversión
                - Se recomienda proceder con la apertura
                """)
            elif prediction > df['ventas_m24'].median():
                st.info("""
                ℹ️ **Ubicación con Buen Potencial**
                - Ventas por encima del promedio
                - Ubicación viable con potencial de crecimiento
                - Considerar optimizaciones en marketing local
                """)
            else:
                st.warning("""
                ⚠️ **Ubicación Requiere Análisis Adicional**
                - Ventas proyectadas bajo el promedio
                - Se recomienda evaluar factores adicionales
                - Considerar estrategias de diferenciación
                """)
            
            # Factores clave
            st.subheader("🔍 Factores Clave para Esta Ubicación")
            
            factors = pd.DataFrame({
                'Factor': ['Población', 'Tráfico', 'Competencia', 'Comercios'],
                'Valor': [
                    pop_100m / df['pop_100m'].mean(),
                    foot_traffic / df['foot_traffic'].mean(),
                    1 - (competencia / df['competencia'].mean()),
                    commerces / df['commerces'].mean()
                ]
            })
            
            fig_factors = px.bar(
                factors,
                x='Factor',
                y='Valor',
                title='Índice de Factores vs Promedio (1.0 = promedio)',
                color='Valor',
                color_continuous_scale='RdYlGn'
            )
            fig_factors.add_hline(y=1.0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_factors, use_container_width=True)
    
    # TAB 3: ANÁLISIS DE ZONA
    with tab3:
        st.header("📊 Análisis de Potencial por Zona")
        
        # Crear grid de predicciones
        st.subheader("🗺️ Mapa de Calor de Potencial de Ventas")
        
        with st.expander("⚙️ Configurar Análisis de Zona"):
            col1, col2 = st.columns(2)
            
            with col1:
                lat_min = st.number_input(
                    "Latitud Mínima",
                    value=float(df['lat'].min()),
                    format="%.6f"
                )
                lat_max = st.number_input(
                    "Latitud Máxima",
                    value=float(df['lat'].max()),
                    format="%.6f"
                )
            
            with col2:
                lon_min = st.number_input(
                    "Longitud Mínima",
                    value=float(df['lon'].min()),
                    format="%.6f"
                )
                lon_max = st.number_input(
                    "Longitud Máxima",
                    value=float(df['lon'].max()),
                    format="%.6f"
                )
            
            grid_size = st.slider("Resolución del Grid", 5, 20, 10)
            tipo_analisis = st.selectbox(
                "Tipo de Tienda para Análisis",
                df['store_cat'].unique()
            )
        
        if st.button("🔍 Generar Análisis", type="primary"):
            with st.spinner("Generando mapa de potencial..."):
                # Crear grid
                lats = np.linspace(lat_min, lat_max, grid_size)
                lons = np.linspace(lon_min, lon_max, grid_size)
                
                predictions_grid = []
                
                for lat in lats:
                    for lon in lons:
                        # Crear datos promedio para cada punto
                        grid_data = pd.DataFrame({
                            'lat': [lat],
                            'lon': [lon],
                            'store_cat': [tipo_analisis],
                            'pop_100m': [df['pop_100m'].mean()],
                            'pop_300m': [df['pop_300m'].mean()],
                            'pop_500m': [df['pop_500m'].mean()],
                            'commerces': [df['commerces'].mean()],
                            'gas_stations': [df['gas_stations'].mean()],
                            'malls': [df['malls'].mode()[0]],
                            'foot_traffic': [df['foot_traffic'].mean()],
                            'car_traffic': [df['car_traffic'].mean()],
                            'socio_level': [df['socio_level'].mode()[0]],
                            'viviendas_100m': [df['viviendas_100m'].mean()],
                            'oficinas_100m': [df['oficinas_100m'].mean()],
                            'viviendas_pobreza': [df['viviendas_pobreza'].mean()],
                            'competencia': [df['competencia'].mean()],
                            'tiendas_peq': [df['tiendas_peq'].mean()]
                        })
                        
                        pred = model.predict(grid_data)[0]
                        predictions_grid.append([lat, lon, pred])
                
                # Crear DataFrame con predicciones
                pred_df = pd.DataFrame(predictions_grid, columns=['lat', 'lon', 'ventas_pred'])
                
                # Visualizar mapa de calor
                fig_heatmap = px.density_contour(
                    pred_df,
                    x='lon',
                    y='lat',
                    z='ventas_pred',
                    title=f'Mapa de Potencial de Ventas - {tipo_analisis}',
                    labels={'ventas_pred': 'Ventas Predichas'}
                )
                fig_heatmap.update_traces(contours_coloring="fill", contours_showlabels=True)
                
                # Agregar tiendas existentes
                fig_heatmap.add_trace(
                    go.Scatter(
                        x=df['lon'],
                        y=df['lat'],
                        mode='markers',
                        marker=dict(size=8, color='red', symbol='star'),
                        name='Tiendas Existentes'
                    )
                )
                
                st.plotly_chart(fig_heatmap, use_container_width=True)
                
                # Identificar mejores ubicaciones
                st.subheader("🎯 Top 5 Mejores Ubicaciones Potenciales")
                
                top_locations = pred_df.nlargest(5, 'ventas_pred')
                
                for idx, row in top_locations.iterrows():
                    col1, col2, col3 = st.columns([2, 2, 1])
                    with col1:
                        st.write(f"**Ubicación {idx+1}**")
                        st.write(f"Lat: {row['lat']:.6f}, Lon: {row['lon']:.6f}")
                    with col2:
                        st.metric("Ventas Estimadas", f"${row['ventas_pred']:,.0f}")
                    with col3:
                        st.button("📍", key=f"loc_{idx}", help="Ver en mapa")
                    st.markdown("---")
    
    # TAB 4: RECOMENDACIONES
    with tab4:
        st.header("💡 Recomendaciones y Mejores Prácticas")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Factores de Éxito")
            
            # Análisis de tiendas exitosas
            top_stores = df.nlargest(10, 'ventas_m24')
            
            st.markdown("""
            ### Características de Tiendas Top Performers:
            """)
            
            success_metrics = {
                'Población Promedio 100m': top_stores['pop_100m'].mean(),
                'Tráfico Peatonal Promedio': top_stores['foot_traffic'].mean(),
                'Comercios Cercanos': top_stores['commerces'].mean(),
                'Nivel Socioeconómico': top_stores['socio_level'].mean()
            }
            
            for metric, value in success_metrics.items():
                avg_value = df[metric.split()[0].lower() + '_' + metric.split()[1].lower() if len(metric.split()) > 1 else metric.lower().replace(' ', '_')].mean() if 'Promedio' not in metric else df[metric.replace(' Promedio', '').lower().replace(' ', '_')].mean()
                diff = ((value / avg_value - 1) * 100) if avg_value != 0 else 0
                st.metric(
                    metric,
                    f"{value:.1f}",
                    f"{diff:+.1f}% vs promedio"
                )
            
            st.markdown("---")
            
            st.success("""
            ### ✅ Recomendaciones Clave:
            
            1. **Ubicación Estratégica**
               - Priorizar áreas con alta densidad poblacional
               - Buscar zonas con buen tráfico peatonal
            
            2. **Análisis de Competencia**
               - Evaluar saturación del mercado
               - Identificar oportunidades de diferenciación
            
            3. **Tipo de Tienda**
               - Adaptar formato según demografía
               - Considerar poder adquisitivo del área
            """)
        
        with col2:
            st.subheader("⚠️ Factores de Riesgo")
            
            # Análisis de tiendas con bajo desempeño
            bottom_stores = df.nsmallest(10, 'ventas_m24')
            
            st.markdown("""
            ### Características a Evitar:
            """)
            
            risk_metrics = {
                'Alta Competencia': bottom_stores['competencia'].mean(),
                'Bajo Tráfico': bottom_stores['foot_traffic'].mean(),
                'Pocos Comercios': bottom_stores['commerces'].mean()
            }
            
            for metric, value in risk_metrics.items():
                st.metric(metric, f"{value:.1f}", delta_color="inverse")
            
            st.markdown("---")
            
            st.warning("""
            ### ⚠️ Señales de Alerta:
            
            1. **Saturación del Mercado**
               - Más de 30 competidores en radio de 500m
               - Múltiples tiendas del mismo formato
            
            2. **Baja Densidad Poblacional**
               - Menos de 200 personas en radio de 100m
               - Área predominantemente industrial
            
            3. **Accesibilidad Limitada**
               - Bajo tráfico peatonal (<100)
               - Difícil acceso vehicular
            """)
        
        st.markdown("---")
        
        # Simulador de escenarios
        st.subheader("🎲 Simulador de Escenarios")
        
        st.info("""
        Utiliza el simulador para entender cómo diferentes factores impactan las ventas:
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            scenario_pop = st.slider(
                "Población 100m",
                float(df['pop_100m'].min()),
                float(df['pop_100m'].max()),
                float(df['pop_100m'].mean())
            )
        
        with col2:
            scenario_traffic = st.slider(
                "Tráfico Peatonal",
                float(df['foot_traffic'].min()),
                float(df['foot_traffic'].max()),
                float(df['foot_traffic'].mean())
            )
        
        with col3:
            scenario_comp = st.slider(
                "Competencia",
                int(df['competencia'].min()),
                int(df['competencia'].max()),
                int(df['competencia'].mean())
            )
        
        # Calcular predicción del escenario
        scenario_data = pd.DataFrame({
            'lat': [df['lat'].mean()],
            'lon': [df['lon'].mean()],
            'store_cat': [df['store_cat'].mode()[0]],
            'pop_100m': [scenario_pop],
            'pop_300m': [df['pop_300m'].mean()],
            'pop_500m': [df['pop_500m'].mean()],
            'commerces': [df['commerces'].mean()],
            'gas_stations': [df['gas_stations'].mean()],
            'malls': [df['malls'].mode()[0]],
            'foot_traffic': [scenario_traffic],
            'car_traffic': [df['car_traffic'].mean()],
            'socio_level': [df['socio_level'].mode()[0]],
            'viviendas_100m': [df['viviendas_100m'].mean()],
            'oficinas_100m': [df['oficinas_100m'].mean()],
            'viviendas_pobreza': [df['viviendas_pobreza'].mean()],
            'competencia': [scenario_comp],
            'tiendas_peq': [df['tiendas_peq'].mean()]
        })
        
        scenario_pred = model.predict(scenario_data)[0]