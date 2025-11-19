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

# Clusters
from utils.functional_analysis import (
    load_functional_sales,
    compute_clusters_automatic,
    assign_cluster_by_sales,
    estimate_sales_curve,
    CLUSTER_COLORS
)

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

st.set_page_config(page_title="Estimaciones",
                   page_icon="", layout="wide")


st.title("Predicciones Geográficas de Ventas")
st.markdown(
    "Predice ventas para nuevas ubicaciones y visualiza el mapa de oportunidades")
st.markdown("---")


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
    with st.spinner("Entrenando modelo..."):
        model, X_test, y_test = train_model(df)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Descripción Modelo",
        "Mapa Interactivo",
        "Predicción Individual",
        # "Análisis de Zona",
        "Recomendaciones"
    ])

    # TAB 1: DESCRIPCIÓN
    with tab1:
       # --- Evaluación del modelo ---
        st.subheader("Evaluación del Modelo de Predicción")

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
        # --- Tarjeta aclaratoria sobre la precisión ---
        st.markdown("""
        <div style="
            background-color: #fff3cd;
            border-left: 6px solid #ffcc00;
            padding: 1.2rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            color: #856404;
        ">
            <h4 style="margin-top:0;">Importante</h4>
            <p>
            Este modelo tiene un nivel de precisión <b>moderado (R² ≈ 0.68)</b>, 
            lo que significa que no puede predecir las ventas con exactitud del 100%.<br><br>
            Se recomienda usar estas predicciones como una <b>guía analítica</b> 
            para apoyar la toma de decisiones, complementándolas siempre con 
            <b>criterio experto y conocimiento del contexto comercial</b>.
            </p>
        </div>
        """, unsafe_allow_html=True)

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
            # Convertir Variable en categoría ordenada para mantener el orden en el gráfico
            importance_df["Variable"] = pd.Categorical(
                importance_df["Variable"],
                categories=importance_df["Variable"],
                ordered=True)

            fig_imp = px.bar(
                importance_df.head(15),
                x="Importancia",
                y="Variable",
                orientation="h",
                title="Principales Variables Predictoras",
                color="Importancia",
                color_continuous_scale="Blues"
            )

            fig_imp.update_yaxes(categoryorder="total ascending")
            st.plotly_chart(fig_imp, use_container_width=True)
        except Exception as e:
            st.warning(f"No se pudo calcular la importancia de variables: {e}")

    # TAB 2: MAPA INTERACTIVO
    with tab2:
        st.header("Mapa de Tiendas y Predicciones")

        col1, col2 = st.columns([2, 1])

        with col2:
            st.subheader("Filtros")

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
            st.subheader("Estadísticas del Área Seleccionada")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Tiendas", len(df_filtered))
            with col2:
                st.metric("Venta Promedio",
                          f"${df_filtered['ventas_m24'].mean():,.0f}")
            with col3:
                st.metric("Venta Total",
                          f"${df_filtered['ventas_m24'].sum():,.0f}")
            with col4:
                st.metric("Densidad Promedio",
                          f"{df_filtered['pop_100m'].mean():.0f}")

    # TAB 3: PREDICCIÓN INDIVIDUAL
    with tab3:
        st.header("Predicción de Ventas para Nueva Tienda")
        st.markdown(
            "En esta sección puedes definir las características de una nueva ubicación "
            "para estimar sus **ventas potenciales** con base en el modelo entrenado.")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown('<div class="input-section">', unsafe_allow_html=True)
            st.subheader("Ubicación y Características")

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
            st.subheader("Características del Área")

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
            st.markdown('<div class="prediction-card">',
                        unsafe_allow_html=True)
            st.markdown("## Predicción de Ventas Mes 24")
            st.markdown(f"### ${prediction:,.2f}")
            st.markdown("Ventas estimadas para el mes 24")
            st.markdown('</div>', unsafe_allow_html=True)

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

            # ===== ANÁLISIS FUNCIONAL DE VENTAS (AUTOMÁTICO) =====
            st.markdown("---")
            st.markdown("## Proyección de Ventas Mensuales (Meses 1-24)")

            # Cargar datos funcionales
            with st.spinner("Calculando proyección funcional con clustering automático..."):
                df_func, periodos, ventas = load_functional_sales()

                if df_func is not None:
                    # Calcular clusters AUTOMÁTICAMENTE
                    labels, fd_eval, eval_points, Z, coef, k_opt, sil_scores = compute_clusters_automatic(
                        ventas, periodos
                    )

                    st.success(
                        f"Clustering automático completado: **K = {k_opt} clusters** (Silhouette Score: {sil_scores[k_opt]:.3f})")

                    # Asignar cluster AUTOMÁTICAMENTE
                    cluster_asignado = assign_cluster_by_sales(
                        prediction, fd_eval, labels, k_opt)

                    st.info(
                        f"Nueva tienda asignada automáticamente al **Cluster {cluster_asignado}** (mejor ajuste según ventas)")

                    # Estimar curva ajustada al mes 24
                    estimacion, mean_cluster, curvas_cluster = estimate_sales_curve(
                        fd_eval, labels, cluster_asignado, prediction
                    )

                    # Crear DataFrame de estimación
                    df_estimacion = pd.DataFrame({
                        "Mes": eval_points,
                        "Ventas_Estimadas": np.ravel(estimacion)
                    })

                    # ===== GRÁFICA PRINCIPAL =====
                    fig_main = go.Figure()

                    # Curvas del cluster (fondo)
                    for idx, y in enumerate(curvas_cluster):
                        fig_main.add_trace(go.Scatter(
                            x=eval_points,
                            y=y.ravel(),  # CORRECCIÓN: aplanar array
                            mode='lines',
                            line=dict(color=CLUSTER_COLORS.get(
                                cluster_asignado, '#999'), width=0.5),
                            opacity=0.15,
                            showlegend=(idx == 0),
                            name=f"Tiendas Cluster {cluster_asignado}",
                            hoverinfo='skip',
                            legendgroup='cluster'
                        ))

                    # Media del cluster
                    fig_main.add_trace(go.Scatter(
                        x=eval_points,
                        y=mean_cluster.ravel(),  # CORRECCIÓN: aplanar array
                        mode='lines',
                        line=dict(color=CLUSTER_COLORS.get(
                            cluster_asignado, '#999'), width=3, dash='dash'),
                        name=f"Media Cluster {cluster_asignado}"
                    ))

                    # Curva estimada para nueva tienda
                    fig_main.add_trace(go.Scatter(
                        x=eval_points,
                        y=np.ravel(estimacion),
                        mode='lines+markers',
                        line=dict(color='red', width=3),
                        marker=dict(size=4),
                        name="Nueva Tienda (estimada)"
                    ))

                    # Punto del mes 24 (dato real de predicción)
                    fig_main.add_trace(go.Scatter(
                        x=[24],
                        y=[prediction],
                        mode='markers',
                        marker=dict(color='black', size=12, symbol='star'),
                        name="Predicción Mes 24"
                    ))

                    fig_main.update_layout(
                        title=f"Proyección de Ventas - Nueva Tienda (Cluster {cluster_asignado} de {k_opt})",
                        xaxis_title="Mes",
                        yaxis_title="Ventas ($)",
                        hovermode='x unified',
                        height=500,
                        template='plotly_white',
                        showlegend=True
                    )

                    st.plotly_chart(fig_main, use_container_width=True)

                    # Tabla de estimaciones
                    col_tabla1, col_tabla2 = st.columns(2)
                    with col_tabla1:
                        st.markdown("**📅 Meses 1-12**")
                        st.dataframe(
                            df_estimacion.head(12).style.format(
                                {"Ventas_Estimadas": "${:,.2f}"}),
                            use_container_width=True,
                            hide_index=True
                        )
                    with col_tabla2:
                        st.markdown("**📅 Meses 13-24**")
                        st.dataframe(
                            df_estimacion.tail(12).style.format(
                                {"Ventas_Estimadas": "${:,.2f}"}),
                            use_container_width=True,
                            hide_index=True
                        )

                    # ===== BOTÓN EXPANDIBLE PARA EXPLORAR CLUSTER =====
                    with st.expander("🔍 **Explorar más sobre el Cluster y Análisis**", expanded=False):
                        st.markdown(
                            f"### Análisis Detallado - Cluster {cluster_asignado}")

                        # Métricas del cluster
                        col_m1, col_m2, col_m3 = st.columns(3)
                        with col_m1:
                            n_tiendas_cluster = int(
                                (labels == cluster_asignado).sum())
                            st.metric("Tiendas en Cluster", n_tiendas_cluster)
                        with col_m2:
                            venta_prom_cluster = float(mean_cluster[-1])
                            st.metric("Venta Promedio Mes 24",
                                      f"${venta_prom_cluster:,.2f}")
                        with col_m3:
                            desv_cluster = float(curvas_cluster[:, -1].std())
                            st.metric("Desviación Estándar",
                                      f"${desv_cluster:,.2f}")

                        # ===== PANEL DE CLUSTERS =====
                        st.markdown("---")
                        st.markdown(
                            "#### Comparación entre Todos los Clusters")

                        # Gráfica comparativa de todos los clusters
                        fig_clusters = go.Figure()

                        for k in range(1, k_opt + 1):
                            idx_k = (labels == k)
                            curvas_k = fd_eval[idx_k]
                            mean_k = curvas_k.mean(axis=0)
                            n_tiendas_k = int(idx_k.sum())

                            # Todas las curvas del cluster
                            for idx, y in enumerate(curvas_k):
                                fig_clusters.add_trace(go.Scatter(
                                    x=eval_points,
                                    y=y.ravel(),  # CORRECCIÓN
                                    mode='lines',
                                    line=dict(color=CLUSTER_COLORS.get(
                                        k, '#999'), width=0.5),
                                    opacity=0.15,
                                    showlegend=False,
                                    hoverinfo='skip',
                                    legendgroup=f'cluster{k}'
                                ))

                            # Media del cluster
                            fig_clusters.add_trace(go.Scatter(
                                x=eval_points,
                                y=mean_k.ravel(),  # CORRECCIÓN
                                mode='lines',
                                line=dict(color=CLUSTER_COLORS.get(
                                    k, '#999'), width=4),
                                name=f"Cluster {k} (n={n_tiendas_k})",
                                legendgroup=f'cluster{k}'
                            ))

                        fig_clusters.update_layout(
                            title=f"Comparación de {k_opt} Clusters de Comportamiento",
                            xaxis_title="Mes",
                            yaxis_title="Ventas ($)",
                            hovermode='x unified',
                            height=500,
                            template='plotly_white'
                        )

                        st.plotly_chart(fig_clusters, use_container_width=True)

                        # Tabla resumen de clusters
                        st.markdown("#### Resumen de Clusters")
                        cluster_summary = []
                        for k in range(1, k_opt + 1):
                            idx_k = (labels == k)
                            curvas_k = fd_eval[idx_k]
                            mean_k24 = float(curvas_k[:, -1].mean())
                            std_k24 = float(curvas_k[:, -1].std())
                            cluster_summary.append({
                                'Cluster': k,
                                'N° Tiendas': int(idx_k.sum()),
                                'Venta Media M24': f"${mean_k24:,.2f}",
                                'Desv. Estándar': f"${std_k24:,.2f}"
                            })

                        df_summary = pd.DataFrame(cluster_summary)
                        st.dataframe(
                            df_summary, use_container_width=True, hide_index=True)

                        # ===== DENDROGRAMA =====
                        st.markdown("---")
                        st.markdown("#### 🌳 Dendrograma Jerárquico")

                        from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram

                        fig_dend = go.Figure()
                        dend_data = scipy_dendrogram(
                            Z, no_plot=True, truncate_mode='lastp', p=30)

                        icoord = np.array(dend_data['icoord'])
                        dcoord = np.array(dend_data['dcoord'])

                        for i in range(len(icoord)):
                            fig_dend.add_trace(go.Scatter(
                                x=icoord[i],
                                y=dcoord[i],
                                mode='lines',
                                line=dict(color='#333', width=1.5),
                                showlegend=False,
                                hoverinfo='skip'
                            ))

                        fig_dend.update_layout(
                            title="Dendrograma de Clustering Jerárquico (Ward)",
                            xaxis_title="Índice de Tienda",
                            yaxis_title="Distancia",
                            height=400,
                            template='plotly_white'
                        )

                        st.plotly_chart(fig_dend, use_container_width=True)

                        # ===== SILHOUETTE SCORES =====
                        st.markdown("---")
                        st.markdown(
                            "#### 📏 Evaluación de Clustering (Silhouette Score)")

                        df_sil = pd.DataFrame({
                            'K': list(sil_scores.keys()),
                            'Silhouette_Score': list(sil_scores.values())
                        })

                        fig_sil = px.line(
                            df_sil,
                            x='K',
                            y='Silhouette_Score',
                            markers=True,
                            title="Silhouette Score por Número de Clusters"
                        )
                        fig_sil.add_vline(
                            x=k_opt,
                            line_dash="dash",
                            line_color="red",
                            annotation_text=f"K={k_opt} (óptimo)"
                        )
                        fig_sil.update_traces(
                            marker=dict(size=8), line=dict(width=2))

                        st.plotly_chart(fig_sil, use_container_width=True)

                        st.success(f"""
**Interpretación:** El análisis automático determinó que **K={k_opt}** es el número 
                        óptimo de clusters con un Silhouette Score de **{sil_scores[k_opt]:.3f}**. 
                        Esto indica una buena separación entre grupos de comportamiento de ventas.
                        """)
                else:
                    st.error(
                        "❌ No se pudieron cargar los datos de ventas funcionales. Verifica que el archivo exista en `data/ventas_funcionales.csv`")

            # Recomendaciones
            st.markdown("---")
            st.subheader("Análisis y Recomendaciones")

            if prediction > df['ventas_m24'].quantile(0.75):
                st.success("""
**Ubicación Altamente Prometedora**
                - Las ventas proyectadas superan el 75% de las tiendas existentes
                - Excelente potencial de retorno de inversión
                - Se recomienda proceder con la apertura
                """)
            elif prediction > df['ventas_m24'].median():
                st.info("""
**Ubicación con Buen Potencial**
                - Ventas por encima del promedio
                - Ubicación viable con potencial de crecimiento
                - Considerar optimizaciones en marketing local
                """)
            else:
                st.warning("""
**Ubicación Requiere Análisis Adicional**
                - Ventas proyectadas bajo el promedio
                - Se recomienda evaluar factores adicionales
                - Considerar estrategias de diferenciación
                """)

        # # TAB 4: ANÁLISIS DE ZONA
        # with tab4:
        #     st.header("Análisis de Potencial por Zona")

        #     # Crear grid de predicciones
        #     st.subheader("Mapa de Calor de Potencial de Ventas")

        #     with st.expander("Configurar Análisis de Zona"):
        #         col1, col2 = st.columns(2)

        #         with col1:
        #             lat_min = st.number_input(
        #                 "Latitud Mínima",
        #                 value=float(df['lat'].min()),
        #                 format="%.6f"
        #             )
        #             lat_max = st.number_input(
        #                 "Latitud Máxima",
        #                 value=float(df['lat'].max()),
        #                 format="%.6f"
        #             )

        #         with col2:
        #             lon_min = st.number_input(
        #                 "Longitud Mínima",
        #                 value=float(df['lon'].min()),
        #                 format="%.6f"
        #             )
        #             lon_max = st.number_input(
        #                 "Longitud Máxima",
        #                 value=float(df['lon'].max()),
        #                 format="%.6f"
        #             )

        #         grid_size = st.slider("Resolución del Grid", 5, 20, 10)
        #         tipo_analisis = st.selectbox(
        #             "Tipo de Tienda para Análisis",
        #             df['store_cat'].unique()
        #         )

        #     if st.button("🔍 Generar Análisis", type="primary"):
        #         with st.spinner("Generando mapa de potencial..."):
        #             # Crear grid
        #             lats = np.linspace(lat_min, lat_max, grid_size)
        #             lons = np.linspace(lon_min, lon_max, grid_size)

        #             predictions_grid = []

        #             for lat in lats:
        #                 for lon in lons:
        #                     # Crear datos promedio para cada punto
        #                     grid_data = pd.DataFrame({
        #                         'lat': [lat],
        #                         'lon': [lon],
        #                         'store_cat': [tipo_analisis],
        #                         'pop_100m': [df['pop_100m'].mean()],
        #                         'pop_300m': [df['pop_300m'].mean()],
        #                         'pop_500m': [df['pop_500m'].mean()],
        #                         'commerces': [df['commerces'].mean()],
        #                         'gas_stations': [df['gas_stations'].mean()],
        #                         'malls': [df['malls'].mode()[0]],
        #                         'foot_traffic': [df['foot_traffic'].mean()],
        #                         'car_traffic': [df['car_traffic'].mean()],
        #                         'socio_level': [df['socio_level'].mode()[0]],
        #                         'viviendas_100m': [df['viviendas_100m'].mean()],
        #                         'oficinas_100m': [df['oficinas_100m'].mean()],
        #                         'viviendas_pobreza': [df['viviendas_pobreza'].mean()],
        #                         'competencia': [df['competencia'].mean()],
        #                         'tiendas_peq': [df['tiendas_peq'].mean()]
        #                     })

        #                     pred = model.predict(grid_data)[0]
        #                     predictions_grid.append([lat, lon, pred])

        #             # Crear DataFrame con predicciones
        #             pred_df = pd.DataFrame(predictions_grid, columns=['lat', 'lon', 'ventas_pred'])

        #             # Visualizar mapa de calor
        #             fig_heatmap = px.density_contour(
        #                 pred_df,
        #                 x='lon',
        #                 y='lat',
        #                 z='ventas_pred',
        #                 title=f'Mapa de Potencial de Ventas - {tipo_analisis}',
        #                 labels={'ventas_pred': 'Ventas Predichas'}
        #             )
        #             fig_heatmap.update_traces(contours_coloring="fill", contours_showlabels=True)

        #             # Agregar tiendas existentes
        #             fig_heatmap.add_trace(
        #                 go.Scatter(
        #                     x=df['lon'],
        #                     y=df['lat'],
        #                     mode='markers',
        #                     marker=dict(size=8, color='red', symbol='star'),
        #                     name='Tiendas Existentes'
        #                 )
        #             )

        #             st.plotly_chart(fig_heatmap, use_container_width=True)

        #             # Identificar mejores ubicaciones
        #             st.subheader("Top 5 Mejores Ubicaciones Potenciales")

        #             top_locations = pred_df.nlargest(5, 'ventas_pred')

        #             for idx, row in top_locations.iterrows():
        #                 col1, col2, col3 = st.columns([2, 2, 1])
        #                 with col1:
        #                     st.write(f"**Ubicación {idx+1}**")
        #                     st.write(f"Lat: {row['lat']:.6f}, Lon: {row['lon']:.6f}")
        #                 with col2:
        #                     st.metric("Ventas Estimadas", f"${row['ventas_pred']:,.0f}")
        #                 with col3:
        #                     st.button("📍", key=f"loc_{idx}", help="Ver en mapa")
        #                 st.markdown("---")

    # TAB : RECOMENDACIONES
    with tab4:
        st.header("Recomendaciones y Mejores Prácticas")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Factores de Éxito")

            # Análisis de tiendas exitosas
            top_stores = df.nlargest(10, 'ventas_m24')

            st.markdown("""
            ### Características de Tiendas Top Performers:
            """)

            success_metrics = {
                "Población 100m": ("pop_100m", top_stores["pop_100m"].mean()),
                "Tráfico Peatonal": ("foot_traffic", top_stores["foot_traffic"].mean()),
                "Comercios Cercanos": ("commerces", top_stores["commerces"].mean()),
                "Nivel Socioeconómico": ("socio_level", top_stores["socio_level"].mean())
            }

            for label, (col, value) in success_metrics.items():
                avg_value = df[col].mean()
                diff = ((value / avg_value - 1) * 100) if avg_value != 0 else 0
                st.metric(
                    label,
                    f"{value:,.1f}",
                    f"{diff:+.1f}% vs promedio"
                )

            st.markdown("---")

            st.success("""
            ### Recomendaciones Clave:
            
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
            st.subheader("Factores de Riesgo")

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
            ### Señales de Alerta:
            
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
