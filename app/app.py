import streamlit as st
from segmentacion import pagina_segmentos
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

API_URL_DEFECTO = "https://api-resenas.onrender.com"
UMBRAL_RIESGO_ALTO = 0.364      # API main

UF = {"AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA", "MT", "MS", "MG",
      "PA", "PB", "PR", "PE", "PI", "RJ", "RN", "RS", "RO", "RR", "SC", "SP", "SE", "TO"}

CAMPOS = ["order_id", "order_purchase_timestamp", "order_approved_at",
          "order_delivered_carrier_date", "order_delivered_customer_date",
          "order_estimated_delivery_date", "estado_cliente", "n_items", "n_vendedores",
          "precio_total", "flete_total", "peso_total_g", "fotos_promedio", "categoria",
          "estado_vendedor", "tipo_pago", "cuotas_max"]
REQUERIDOS = ["order_id", "order_purchase_timestamp", "order_delivered_customer_date",
              "order_estimated_delivery_date", "n_items", "n_vendedores",
              "precio_total", "flete_total"]
FECHAS = [c for c in CAMPOS if c.startswith("order_") and c != "order_id"]
ENTEROS = ["n_items", "n_vendedores", "cuotas_max"]
DECIMALES = ["precio_total", "flete_total", "peso_total_g", "fotos_promedio"]

BASE = {
    "order_id": "ejemplo-base",
    "order_purchase_timestamp": "2018-05-01 10:00:00",
    "order_approved_at": "2018-05-01 10:30:00",
    "order_delivered_carrier_date": "2018-05-03 09:00:00",
    "order_delivered_customer_date": "2018-05-10 12:00:00",
    "order_estimated_delivery_date": "2018-05-15 00:00:00",
    "estado_cliente": "SP", "n_items": 1, "n_vendedores": 1,
    "precio_total": 100.0, "flete_total": 20.0, "peso_total_g": 1000.0,
    "fotos_promedio": 2.0, "categoria": "cama_mesa_banho", "estado_vendedor": "SP",
    "tipo_pago": "credit_card", "cuotas_max": 1,
}
EJEMPLOS_SINTETICOS = {
    "Ejemplo: entrega a tiempo": {**BASE, "order_id": "ejemplo-a-tiempo"},
    "Ejemplo: retraso +10 días": {**BASE, "order_id": "ejemplo-retraso",
                                  "order_delivered_customer_date": "2018-05-25 12:00:00"},
    "Ejemplo: dos vendedores": {**BASE, "order_id": "ejemplo-dos-vendedores",
                                "n_vendedores": 2, "n_items": 3},
}


def cargar_ejemplos():
    ejemplos = {}
    ruta = Path(__file__).parent / "ejemplos.json"
    if ruta.exists():
        try:
            for reg in json.loads(ruta.read_text(encoding="utf-8")):
                etiqueta = f"{str(reg.get('order_id', ''))[:8]}… · {reg.get('etiqueta', 'pedido real')}"
                ejemplos[etiqueta] = {c: reg.get(c) for c in CAMPOS}
        except (json.JSONDecodeError, AttributeError):
            pass
    ejemplos.update(EJEMPLOS_SINTETICOS)
    ejemplos["Pedido en blanco (llenar a mano)"] = {c: None for c in CAMPOS}
    return ejemplos


EJEMPLOS = cargar_ejemplos()


def construir_pedido(valores):
    """Convierte los valores de la tabla en un pedido válido para la API."""
    pedido, errores, avisos = {}, [], []
    for campo in CAMPOS:
        v = valores.get(campo)
        v = "" if v is None or (not isinstance(v, str) and pd.isna(v)) else str(v).strip()
        if v == "" or v.lower() in ("none", "nan", "nat", "null"):
            pedido[campo] = None
            if campo in REQUERIDOS:
                errores.append(f"Falta el campo obligatorio: {campo}")
            continue
        try:
            if campo in FECHAS:
                pedido[campo] = datetime.fromisoformat(v).strftime("%Y-%m-%dT%H:%M:%S")
            elif campo in ENTEROS:
                x = float(v)
                if x != int(x):
                    raise ValueError
                pedido[campo] = int(x)
            elif campo in DECIMALES:
                pedido[campo] = float(v)
            else:
                pedido[campo] = v
        except ValueError:
            pedido[campo] = None
            errores.append(f"Valor no válido en {campo}: '{v}' "
                           f"(fechas: AAAA-MM-DD HH:MM:SS; enteros sin decimales)")

    for campo in ("estado_cliente", "estado_vendedor"):
        if pedido.get(campo):
            pedido[campo] = pedido[campo].upper()
            if pedido[campo] not in UF:
                avisos.append(f"{campo}: '{pedido[campo]}' no es una sigla válida; "
                              f"el modelo lo tratará como dato faltante.")
    if pedido.get("order_purchase_timestamp") and pedido.get("order_delivered_customer_date") \
            and pedido["order_delivered_customer_date"] < pedido["order_purchase_timestamp"]:
        avisos.append("La fecha de entrega es anterior a la de compra. Revisa las fechas.")
    return pedido, errores, avisos


def consultar(pedidos, api_url, lote=500):
    """Envía los pedidos a la API en grupos y devuelve la lista de resultados."""
    resultados = []
    barra = st.progress(0.0)
    for i in range(0, len(pedidos), lote):
        r = requests.post(f"{api_url}/riesgo", json=pedidos[i:i + lote], timeout=180)
        if not r.ok:
            raise RuntimeError(f"La API respondió {r.status_code}: {r.text[:500]}")
        resultados += r.json()
        barra.progress(min((i + lote) / len(pedidos), 1.0))
    barra.empty()
    return resultados


# ----------------------------- Páginas -----------------------------
def pagina_pedido():
    api_url = st.session_state["api_url"].rstrip("/")
    st.title("Riesgo de mala reseña")
    st.caption("Elige un pedido de ejemplo, cambia lo que quieras y pregunta a la API.")

    etiqueta = st.selectbox("Pedido de ejemplo", list(EJEMPLOS))
    ej = EJEMPLOS[etiqueta]
    tabla = pd.DataFrame({
        "campo": CAMPOS,
        "valor": ["" if ej.get(c) is None else str(ej.get(c)) for c in CAMPOS],
    })
    editada = st.data_editor(tabla, hide_index=True, disabled=["campo"],
                             height=460, key=f"editor_{etiqueta}")

    if st.button("Calcular riesgo", type="primary"):
        pedido, errores, avisos = construir_pedido(dict(zip(editada["campo"], editada["valor"])))
        if errores:
            for e in errores:
                st.error(e)
            return
        for a in avisos:
            st.warning(a)
        try:
            with st.spinner("Consultando el modelo (la primera vez puede tardar hasta un minuto)..."):
                res = consultar([pedido], api_url)[0]
        except (requests.exceptions.RequestException, RuntimeError) as e:
            st.error(f"No se pudo consultar la API: {e}")
            return

        c1, c2 = st.columns(2)
        c1.metric("Probabilidad de 1–2 estrellas", f"{res['probabilidad']:.1%}")
        c2.metric("Umbral de riesgo alto (10% más riesgoso)", f"{UMBRAL_RIESGO_ALTO:.1%}")
        if res["riesgo_alto"]:
            st.error("Riesgo alto: contactar al cliente")
        else:
            st.success("Riesgo no alto")
        with st.expander("Respuesta completa de la API"):
            st.json(res)


def pagina_lotes():
    api_url = st.session_state["api_url"].rstrip("/")
    st.title("Predicción por lotes")
    st.write("Sube un archivo JSON con una lista de pedidos y descarga las predicciones. "
             "Cada pedido necesita los mismos campos que el formulario.")
    plantilla = json.dumps([construir_pedido(BASE)[0]], indent=2, ensure_ascii=False)
    st.download_button("Descargar plantilla JSON", plantilla,
                       file_name="plantilla_pedidos.json", mime="application/json")

    archivo = st.file_uploader("Archivo JSON", type="json")
    if archivo is None:
        return
    try:
        pedidos = json.load(archivo)
    except json.JSONDecodeError as e:
        st.error(f"El archivo no es un JSON válido: {e}")
        return
    if isinstance(pedidos, dict):
        pedidos = [pedidos]
    if not isinstance(pedidos, list) or not pedidos or not all(isinstance(p, dict) for p in pedidos):
        st.error("El JSON debe ser una lista de pedidos (objetos con los campos del formulario).")
        return

    st.write(f"**{len(pedidos)}** pedidos leídos.")
    problemas = [{"fila": i, "order_id": p.get("order_id"),
                  "faltan": ", ".join(c for c in REQUERIDOS if p.get(c) in (None, ""))}
                 for i, p in enumerate(pedidos)
                 if any(p.get(c) in (None, "") for c in REQUERIDOS)]
    if problemas:
        st.error(f"{len(problemas)} pedidos no tienen todos los campos obligatorios "
                 f"(incluido order_id). Corrígelos y vuelve a subir el archivo.")
        st.dataframe(pd.DataFrame(problemas))
        return
    ids = [p["order_id"] for p in pedidos]
    if len(ids) != len(set(ids)):
        st.warning(f"Hay {len(ids) - len(set(ids))} order_id repetidos.")

    if st.button("Predecir", type="primary"):
        try:
            with st.spinner("Consultando el modelo..."):
                res = consultar([{c: p.get(c) for c in CAMPOS} for p in pedidos], api_url)
            st.session_state["lote"] = {"archivo": archivo.name, "df": pd.DataFrame(res)}
        except (requests.exceptions.RequestException, RuntimeError) as e:
            st.error(f"No se pudo consultar la API: {e}")
            return

    lote = st.session_state.get("lote")
    if lote and lote["archivo"] == archivo.name:
        df = lote["df"]
        n_alto = int(df["riesgo_alto"].sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Pedidos", len(df))
        c2.metric("Riesgo alto", f"{n_alto} ({n_alto / len(df):.1%})")
        c3.metric("Probabilidad media", f"{df['probabilidad'].mean():.1%}")
        st.dataframe(df.sort_values("probabilidad", ascending=False), hide_index=True)
        d1, d2 = st.columns(2)
        d1.download_button("Descargar CSV", df.to_csv(index=False).encode("utf-8"),
                           file_name="predicciones.csv", mime="text/csv")
        d2.download_button("Descargar JSON", df.to_json(orient="records", force_ascii=False, indent=2),
                           file_name="predicciones.json", mime="application/json")


def pagina_ficha():
    st.title("📋 Ficha del modelo")
    st.markdown("""
### 🎯 Qué predice
La probabilidad de que un pedido reciba una reseña de 1 o 2 estrellas, calculada **al momento de la
entrega** (no se usa nada de la reseña). Este modelo no es un pronóstico y no puede utilizarse antes
de la entrega.

### 🤖 Modelo
XGBoost dentro de un pipeline de scikit-learn, con las transformaciones de `preparacion.py`. El modelo
final se entrenó con todos los pedidos disponibles.

### 📅 Validación por fecha
Se entrenó con los pedidos anteriores al 28 de febrero de 2018 (56,579) y se validó con los siguientes,
hasta mayo de 2018 (20,724 pedidos; 14.8% de malas reseñas). AUC en validación: 0.794.

### 📏 Métrica
Precisión en el 10% de pedidos de mayor riesgo.
""")
    # Cifras de la validación
    ficha = pd.DataFrame({
        "Periodo": ["Validación completa", "Marzo 2018", "Abril 2018", "Mayo 2018"],
        "Azar": [0.148, 0.212, 0.118, 0.108],
        "Regla del retraso": [0.634, 0.824, 0.326, 0.335],
        "Modelo": [0.680, 0.856, 0.471, 0.428],
    }).set_index("Periodo")
    st.table(ficha.style.format("{:.3f}"))

    st.markdown("""
### 🔑 Variables que más pesan
Según la pérdida de AUC al desordenar cada variable, las que más pesan son:
- 📆 **Diferencia entre la fecha estimada y la real de entrega** (los días de retraso): pérdida
  promedio de 0.016 de AUC en los meses con menores tasas de mala reseña.
- 🛒 **Cantidad de artículos del pedido:** pérdida de 0.043.
- 🚚 **Tiempos de entrega:** pérdida de 0.017.

Al observar que la regla del retraso pierde puntos de precisión en los meses con menos retrasos, se
infiere que los días de retraso son clave para una mala reseña. Esto tiene sentido: el cliente califica
peor cuando no se cumple la fecha de entrega, y su disgusto puede ser mayor a medida que aumentan los
días de retraso.

Por otra parte, la tasa de mala reseña aumenta con la cantidad de artículos por pedido: pasa de 12.0%
con un artículo a 25.9% con dos, 30.8% con tres y 35.1% con cuatro o más. Esto está relacionado con la
cantidad de vendedores: cuantos más vendedores intervienen en un pedido, más alta puede ser la tasa de
retrasos y de mala reseña. Con un vendedor es de 13.1%, con dos sube a 48.1% (903 pedidos) y con tres
o más llega a 65.9% (solo 44 pedidos).

**Conclusión:** una explicación plausible es que los pedidos con más artículos y vendedores tienen más
oportunidades de llegar después de la fecha estimada, lo que amplía sus días de entrega total. Los
datos muestran la asociación, pero no prueban el motivo.

### ⚠️ Límites
- 📆 La fecha de entrega al cliente es obligatoria: el modelo predice al momento de la entrega.
- 📂 Los datos para la predicción por lotes deben subirse en formato JSON (valida el formato de las
  fechas antes de subirlos).
- 🎚️ El corte de «riesgo alto» (37.3%) viene de la validación y es aproximado.
- ⭐ Por sí solas, las estrellas no explican la inconformidad del cliente, por lo que no deberían
  usarse como único factor para evaluar el desempeño de un producto o de un vendedor.
- ❌ La predicción puede fallar si no se completan los campos obligatorios.

### ❓ ¿A quién puede afectar un error?
- 🏪 **Vendedores:** pueden perder comisiones o incluso asumir el costo de una mala predicción.
- 📦 **Desempeño del producto:** la predicción no considera nada de las reseñas, por lo que se
  desconoce la razón por la que un pedido se identifica como de riesgo. La causa queda a interpretación
  de quien reciba la predicción.
- 📅 **Calidad de la predicción:** el modelo desconoce los días feriados, que alargan los días de
  entrega total. También pueden fallar las fechas estimadas cuando las genera un sistema automático
  con sus propios errores.
- 🙍 **Consumidores:** una mala predicción puede etiquetarlos como clientes de alto riesgo o
  problemáticos.
- 🏢 **Empresas:** quienes confían en el modelo y concentran recursos en retener clientes pueden
  gastarlos en quienes no los necesitaban y perder a quienes sí requieren algún tratamiento.
""")


# ----------------------------- Navegación -----------------------------
st.set_page_config(page_title="Riesgo de mala reseña", page_icon="📦", layout="wide")

paginas = [
    st.Page(pagina_pedido, title="Riesgo de un pedido", icon="⚠️", default=True),
    st.Page(pagina_lotes, title="Predicción por lotes", icon="📂"),
    st.Page(pagina_segmentos, title="Segmentación de vendedores", icon="🧭"),
    st.Page(pagina_ficha, title="Ficha del modelo", icon="📄"),
]
pg = st.navigation(paginas)

st.sidebar.divider()
st.sidebar.text_input("URL de la API", value=API_URL_DEFECTO, key="api_url")
if st.sidebar.button("Probar conexión"):
    try:
        with st.sidebar:
            with st.spinner("Despertando la API..."):
                r = requests.get(st.session_state["api_url"].rstrip("/"), timeout=120)
        if r.ok:
            st.sidebar.success("API activa")
        else:
            st.sidebar.error(f"La API respondió {r.status_code}")
    except requests.exceptions.RequestException as e:
        st.sidebar.error(f"Sin conexión: {e}")

pg.run()