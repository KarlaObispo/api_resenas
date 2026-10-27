import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

CSV_PATH = DATA_DIR / "vendedores_segmentacion.csv"
SILHOUETTE_PATH = DATA_DIR / "silueta.json"

# Pon False para quitar el filtro de estado y su pestaña
MOSTRAR_FILTRO_ESTADO = True

SEGMENT_NAMES = {
    0: "Alta demanda y amplia cobertura",
    1: "Bajo volumen y entregas puntuales",
    2: "Ticket alto con retrasos",
}
ORDEN_SEGMENTOS = list(SEGMENT_NAMES.values())
COLORES = dict(zip(ORDEN_SEGMENTOS, ["#4C78A8", "#F58518", "#54A24B"]))
COLORES_CON_TOTAL = {**COLORES, "Total": "#BDBDBD"}

UF = {"AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA", "MT", "MS", "MG",
      "PA", "PB", "PR", "PE", "PI", "RJ", "RN", "RS", "RO", "RR", "SC", "SP", "SE", "TO"}

# Variables originales usadas para entrenar K-Means
MODEL_FEATURES = [
    "cantidad_pedidos", "monto_mediana", "flete_mediana", "pedidos_retrasados",
    "dias_entrega_mediana", "dias_retraso_mediana", "categorias_diferentes",
    "tasa_pedidos_retrasados", "flete_relativo", "tasa_retraso_dias",
    "estados_cliente_diferentes",
]

NUMERIC_COLUMNS = [
    "cantidad_pedidos", "n_items_vendidos", "monto_mediana", "flete_mediana",
    "pedidos_retrasados_mediana", "pedidos_adelantados_mediana",
    "pedidos_mismo_estado_mediana", "dias_entrega_mediana", "pedidos_retrasados",
    "pedidos_adelantados", "pedidos_mismo_estado", "dias_retraso_mediana",
    "estados_cliente_diferentes", "categorias_diferentes", "pedidos_categoria_principal",
    "tasa_pedidos_retrasados", "tasa_pedidos_adelantados", "tasa_pedidos_mismo_estado",
    "flete_relativo", "tasa_retraso_dias", "pc1", "pc2",
]

REQUIRED_COLUMNS = ["seller_id", "segmento", "pc1", "pc2",
                    "estado_seller", "categoria_mas_pedidos"]


@st.cache_data
def load_data(csv_mtime, silueta_mtime):
    # Los dos argumentos solo sirven para que la caché se renueve si cambian los archivos
    df = pd.read_csv(CSV_PATH)
    with open(SILHOUETTE_PATH, "r", encoding="utf-8") as file:
        silhouette = json.load(file)

    df["segmento"] = pd.to_numeric(df["segmento"], errors="coerce").astype("Int64")
    df["segmento_nombre"] = df["segmento"].map(SEGMENT_NAMES)

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Estado: solo siglas válidas de dos letras; lo demás queda como dato faltante
    bruto = df["estado_seller"].astype("string").str.strip().str.upper()
    n_invalidos = int((bruto.notna() & ~bruto.isin(UF)).sum())
    df["estado_seller"] = bruto.where(bruto.isin(UF))

    df["categoria_mas_pedidos"] = (
        df["categoria_mas_pedidos"].astype("string").str.strip().str.lower()
    )
    return df, silhouette, n_invalidos


def tabla_distribucion(df, col, n=8):
    """% de los vendedores de cada segmento por valor (los n más frecuentes + otros)."""
    s = df[col].astype("string").fillna("sin dato").astype(str)
    top = s.value_counts().head(n).index
    g = s.where(s.isin(top), "otros")

    pct = pd.crosstab(g, df["segmento_nombre"], normalize="columns").mul(100)
    pct = pct[[c for c in ORDEN_SEGMENTOS if c in pct.columns]].astype("float64")
    pct["Total"] = g.value_counts(normalize=True).mul(100).reindex(pct.index).astype("float64")
    pct.index.name = "Valor"
    return pct.sort_values("Total", ascending=False).round(1)


def pagina_segmentos():
    if not CSV_PATH.exists():
        st.error(f"No se encontró el archivo: {CSV_PATH}\n\n"
                 "Verifica que esté dentro de la carpeta data.")
        return
    if not SILHOUETTE_PATH.exists():
        st.error(f"No se encontró el archivo: {SILHOUETTE_PATH}\n\n"
                 "Verifica que silueta.json esté dentro de data.")
        return

    df, silhouette, n_estados_invalidos = load_data(
        CSV_PATH.stat().st_mtime, SILHOUETTE_PATH.stat().st_mtime)

    faltan = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if faltan:
        st.error(f"Faltan columnas requeridas: {faltan}")
        return

    # ------------------------------ Encabezado ------------------------------
    st.title("Segmentación de vendedores")
    st.caption("K-Means identifica tres perfiles de vendedores según su volumen de "
               "pedidos, montos, entregas y cobertura. Explora los perfiles y filtra "
               "los vendedores según sus características.")

    # ------------------------- Botones de segmento -------------------------
    segment_options = sorted(df["segmento"].dropna().unique().tolist())
    segment_buttons = [
        {"label": SEGMENT_NAMES.get(int(s), f"Segmento {s}"), "value": str(int(s))}
        for s in segment_options
    ]

    if "selected_segment" not in st.session_state:
        st.session_state.selected_segment = "todos"

    _, center_area, _ = st.columns([1, 5, 1])
    with center_area:
        button_cols = st.columns(len(segment_buttons) + 1)
        with button_cols[0]:
            if st.button("Todos", key="segment_all", use_container_width=True,
                         type="primary" if st.session_state.selected_segment == "todos"
                         else "secondary"):
                st.session_state.selected_segment = "todos"
                st.rerun()
        for col, item in zip(button_cols[1:], segment_buttons):
            with col:
                selected = st.session_state.selected_segment == item["value"]
                if st.button(item["label"], key=f"segment_{item['value']}",
                             type="primary" if selected else "secondary",
                             use_container_width=True, help=item["label"]):
                    st.session_state.selected_segment = item["value"]
                    st.rerun()

    # ------------------------------ Filtros ------------------------------
    with st.expander("🔎 Buscar por seller_id", expanded=False):
        seller_search = st.text_input("Identificador del vendedor",
                                      placeholder="Escribe el seller_id",
                                      key="filter_seller_id").strip()

    estado_sel = "Todos"
    if MOSTRAR_FILTRO_ESTADO:
        with st.expander("📍 Filtrar por estado del vendedor", expanded=False):
            estados = sorted(df["estado_seller"].dropna().unique().tolist())
            opciones_estado = (["Todos"] + estados
                               + (["Sin dato"] if df["estado_seller"].isna().any() else []))
            estado_sel = st.selectbox("Estado", opciones_estado, key="filter_state")
            if n_estados_invalidos:
                st.caption(f"{n_estados_invalidos} vendedores traían en este campo valores que "
                           "no son siglas de estado y aparecen como «Sin dato».")

    with st.expander("🏷️ Filtrar por categoría principal", expanded=False):
        categorias = sorted(df["categoria_mas_pedidos"].dropna().unique().tolist())
        opciones_cat = (["Todas"] + categorias
                        + (["Sin dato"] if df["categoria_mas_pedidos"].isna().any() else []))
        categoria_sel = st.selectbox("Categoría con más pedidos", opciones_cat,
                                     key="filter_category")

    # --------------------------- Aplicar filtros ---------------------------
    filtered = df.copy()

    if st.session_state.selected_segment != "todos":
        filtered = filtered[filtered["segmento"] == int(st.session_state.selected_segment)]

    if seller_search:
        filtered = filtered[filtered["seller_id"].astype(str).str.contains(
            seller_search, case=False, na=False, regex=False)]

    if estado_sel == "Sin dato":
        filtered = filtered[filtered["estado_seller"].isna()]
    elif estado_sel != "Todos":
        filtered = filtered[(filtered["estado_seller"] == estado_sel).fillna(False)]

    if categoria_sel == "Sin dato":
        filtered = filtered[filtered["categoria_mas_pedidos"].isna()]
    elif categoria_sel != "Todas":
        filtered = filtered[(filtered["categoria_mas_pedidos"] == categoria_sel).fillna(False)]

    if filtered.empty:
        st.warning("No se encontraron vendedores con los criterios seleccionados.")
        return

    # ------------------------------ Resumen ------------------------------
    st.subheader("Resumen general")
    total_sellers = df["seller_id"].nunique()
    filtered_sellers = filtered["seller_id"].nunique()

    c1, c2, c3 = st.columns(3)
    c1.metric("Vendedores totales", f"{total_sellers:,}")
    c2.metric("Vendedores filtrados", f"{filtered_sellers:,}")
    c3.metric("Segmentos identificados", df["segmento"].nunique())

    st.markdown("### Distribución por segmento")
    distribution = (filtered.groupby(["segmento", "segmento_nombre"], dropna=False)["seller_id"]
                    .nunique().reset_index(name="vendedores"))
    distribution["porcentaje"] = distribution["vendedores"] / filtered_sellers * 100

    for col, (_, row) in zip(st.columns(max(len(distribution), 1)), distribution.iterrows()):
        with col:
            st.metric(label=str(row["segmento_nombre"]),
                      value=f'{int(row["vendedores"]):,}',
                      delta=f'{row["porcentaje"]:.1f}% de los filtrados',
                      delta_color="off")

    # ------------------------------ PCA (interactivo) ------------------------------
    st.subheader("Mapa de segmentos: PCA")
    hover = {
        "segmento": True, "estado_seller": True, "categoria_mas_pedidos": True,
        "cantidad_pedidos": ":.0f", "monto_mediana": ":.2f", "flete_mediana": ":.2f",
        "pc1": False, "pc2": False,
    }
    fig = px.scatter(
        filtered, x="pc1", y="pc2", color="segmento_nombre",
        color_discrete_map=COLORES,
        category_orders={"segmento_nombre": ORDEN_SEGMENTOS},
        hover_name="seller_id",
        hover_data={k: v for k, v in hover.items() if k in filtered.columns},
        labels={"pc1": "Componente principal 1", "pc2": "Componente principal 2",
                "segmento_nombre": "Segmento"},
        opacity=0.8,
    )
    fig.update_traces(marker=dict(size=7))
    fig.update_layout(legend_title_text="Segmento", height=600)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Cada punto es un vendedor. Los ejes resumen las variables usadas para "
               "construir los segmentos y no representan unidades comerciales directas. "
               "Puedes acercar, mover y apagar segmentos desde la leyenda.")

    # --------------------- Perfil: medianas por segmento ---------------------
    st.subheader("Perfil de los segmentos")
    disponibles = [c for c in MODEL_FEATURES if c in filtered.columns]
    if disponibles:
        perfil = filtered.groupby("segmento_nombre")[disponibles].median().round(2)
        perfil = perfil.loc[[n for n in ORDEN_SEGMENTOS if n in perfil.index]].T
        st.markdown("Medianas de las variables usadas para entrenar K-Means, calculadas sobre "
                    "los vendedores que cumplen los filtros.")
        st.dataframe(perfil, use_container_width=True)
    else:
        st.warning("No se encontraron las variables del modelo en el CSV.")

    # ---------------- Estados y categorías por segmento ----------------
    st.subheader("Estados y categorías por segmento")
    st.caption("Calculado con todos los vendedores, sin los filtros de arriba. Cada barra es el "
               "porcentaje de los vendedores de ese segmento que tienen ese estado o categoría; "
               "la barra gris es el total. Diferencias de pocos puntos son normales dado el "
               "tamaño de cada segmento.")

    secciones = [("categoria_mas_pedidos", "Categoría principal")]
    if MOSTRAR_FILTRO_ESTADO:
        secciones.insert(0, ("estado_seller", "Estado del vendedor"))

    tabs = st.tabs([titulo for _, titulo in secciones])
    for tab, (col, titulo) in zip(tabs, secciones):
        with tab:
            n = st.slider("Valores más frecuentes a mostrar", 4, 15, 8, key=f"n_{col}")
            pct = tabla_distribucion(df, col, n)

            largo = pct.reset_index().melt(id_vars="Valor", var_name="Segmento",
                                           value_name="Porcentaje")
            fig = px.bar(largo, x="Valor", y="Porcentaje", color="Segmento", barmode="group",
                         color_discrete_map=COLORES_CON_TOTAL,
                         category_orders={"Valor": list(pct.index),
                                          "Segmento": ORDEN_SEGMENTOS + ["Total"]})
            fig.update_traces(hovertemplate="%{x}<br>%{y:.1f}%<extra>%{fullData.name}</extra>")
            fig.update_layout(xaxis_title=titulo, yaxis_title="% de los vendedores del grupo",
                              legend_title_text="", height=450)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(pct, use_container_width=True)

    # ------------------------------ Silueta ------------------------------
    with st.expander("📈 Evaluación de silueta: elección de k"):
        if silhouette:
            sil = pd.DataFrame([{"k": int(k), "silueta": float(v)}
                                for k, v in silhouette.items()]).sort_values("k")
            fig = px.line(sil, x="k", y="silueta", markers=True,
                          labels={"k": "Número de clusters (k)",
                                  "silueta": "Coeficiente de silueta"})
            fig.add_vline(x=3, line_dash="dash", line_color="gray")
            fig.update_xaxes(dtick=1)
            fig.update_yaxes(range=[0, max(0.4, sil["silueta"].max() * 1.15)])
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

            mejor = sil.loc[sil["silueta"].idxmax()]
            texto = (f"La silueta más alta es {mejor['silueta']:.3f} (k = {int(mejor['k'])}); "
                     "la línea punteada marca k = 3, el valor elegido.")
            if mejor["silueta"] < 0.25:
                texto += (" Todos los valores quedan por debajo de 0.25, lo que indica una "
                          "estructura débil: los segmentos son una partición útil de un "
                          "continuo, no grupos claramente separados.")
            st.caption(texto)
        else:
            st.info("No hay resultados de silueta disponibles.")

    # ------------------------- Detalle de vendedores -------------------------
    st.subheader("Detalle de vendedores")
    detail_columns = [
        "seller_id", "estado_seller", "segmento", "segmento_nombre", "cantidad_pedidos",
        "n_items_vendidos", "monto_mediana", "flete_mediana", "pedidos_retrasados_mediana",
        "pedidos_adelantados_mediana", "pedidos_mismo_estado_mediana", "dias_entrega_mediana",
        "pedidos_retrasados", "pedidos_adelantados", "pedidos_mismo_estado",
        "dias_retraso_mediana", "estados_cliente_diferentes", "categorias_diferentes",
        "categoria_mas_pedidos", "pedidos_categoria_principal", "tasa_pedidos_retrasados",
        "tasa_pedidos_adelantados", "tasa_pedidos_mismo_estado", "flete_relativo",
        "tasa_retraso_dias",
    ]
    detail_columns = [c for c in detail_columns if c in filtered.columns]
    detalle = filtered[detail_columns]
    if "cantidad_pedidos" in detalle.columns:
        detalle = detalle.sort_values("cantidad_pedidos", ascending=False)

    st.dataframe(detalle, hide_index=True, use_container_width=True)
    st.download_button("Descargar vendedores filtrados (CSV)",
                       data=detalle.to_csv(index=False).encode("utf-8"),
                       file_name="vendedores_filtrados.csv", mime="text/csv")