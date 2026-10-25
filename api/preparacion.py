#%%writefile -a preparacion.py

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

def validar_fechas_pedido(df, medianas=None):
    """
    1) Convierte las fechas a datetime64[ns] (lo no convertible queda NaT).
    2) Calcula la mediana (en horas) entre fechas consecutivas, solo con pares válidos.
    3) Imputa ÚNICAMENTE los nulls: fecha previa + mediana.
       Si esa fecha supera a la fecha siguiente (válida), usa la fecha previa.
    Devuelve el mismo DataFrame, con las mismas columnas.
    medianas: dict {columna: horas}. Si es None, se calculan con df.
    """
    d = df.copy()
    orden = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
    ]
    col_estimada = "order_estimated_delivery_date"

    # 1. Tipos
    for c in orden + [col_estimada]:
        s = pd.to_datetime(d[c], errors="coerce")
        if s.dt.tz is not None:
            s = s.dt.tz_localize(None)
        d[c] = s.astype("datetime64[ns]")

    # 2. Medianas en horas (solo pares con ambas fechas y en orden)
    if medianas is None:
        medianas = {}
        for ant, post in zip(orden[:-1], orden[1:]):
            ok = d[ant].notna() & d[post].notna() & (d[post] >= d[ant])
            horas = (d.loc[ok, post] - d.loc[ok, ant]).dt.total_seconds() / 3600
            medianas[post] = horas.median()

    # 3. Imputación en cascada, solo nulls
    original = d[orden].copy()          # fecha siguiente tal como venía

    for i in range(1, len(orden)):
        ant, col = orden[i - 1], orden[i]
        prev = d[ant]                   # ya con las imputaciones anteriores

        imputable = d[col].isna() & prev.notna()
        candidata = prev + pd.to_timedelta(medianas[col], unit="h")

        if i + 1 < len(orden):
            sig = original[orden[i + 1]]
            excede = sig.notna() & (sig >= prev) & (candidata > sig)
            candidata = candidata.where(~excede, prev)

        d.loc[imputable, col] = candidata[imputable]

    d.attrs["medianas"] = medianas      # no agrega columnas; sirve para guardarlas
    return d

def convertir_tipos(X):
    """
    Convierte los tipos de datos de las columnas de interés y devuelve
    el DataFrame completo (las demás columnas quedan intactas).
    """
    # ---------- Configuración ----------
    numericas   = ['precio_total', 'flete_total', 'fotos_promedio']                 # float64
    enteras     = ['n_items', 'n_vendedores', 'cuotas_max']          # Int64
    categoricas = ['order_id', 'estado_cliente', 'categoria',
                   'estado_vendedor', 'tipo_pago']                                  # string
    mayus = ['estado_cliente', 'estado_vendedor']
    minus = ['tipo_pago', 'categoria']

    # ---------- Validaciones previas ----------
    todas = [*numericas, *enteras, *categoricas]

    repetidas = sorted({c for c in todas if todas.count(c) > 1})
    if repetidas:
        raise ValueError(f"Columnas en más de un grupo: {repetidas}")

    faltantes = [c for c in todas if c not in X.columns]
    if faltantes:
        raise KeyError(f"Columnas que no existen en el DataFrame: {faltantes}")

    # ---------- Conversión ----------
    def a_numero(serie):
        # Si ya es numérica no se pasa por texto (evita dañar valores como 1e-05)
        if pd.api.types.is_numeric_dtype(serie):
            return serie.astype('float64')
        limpio = (serie.astype('string').str.strip()
                       .str.replace(',', '.', regex=False)
                       .str.replace(r'[^0-9.\-]', '', regex=True)
                       .replace('', pd.NA))
        return pd.to_numeric(limpio, errors='coerce').astype('float64')

    copia = X.copy()

    for c in numericas:
        copia[c] = a_numero(copia[c])

    for c in enteras:
        serie = a_numero(copia[c])
        copia[c] = serie.where(serie % 1 == 0).astype('Int64')

    for c in categoricas:
        serie = (copia[c].astype('string').str.strip()
                         .str.replace(r'\s+', ' ', regex=True)
                         .replace('', pd.NA))
        if c in mayus:
            serie = serie.str.upper()
        elif c in minus:
            serie = serie.str.lower()
        copia[c] = serie

    return copia

def validar_rangos(X):
    """
    Deja en null los valores que incumplen reglas de negocio.
    Devuelve el DataFrame completo. Debe ir después de convertir_tipos.
    """
    uf_validas = {"AC","AL","AP","AM","BA","CE","DF","ES","GO","MA","MT","MS","MG",
                  "PA","PB","PR","PE","PI","RJ","RN","RS","RO","RR","SC","SP","SE","TO"}

    copia = X.copy()

    for c in ['precio_total', 'flete_total', 'cuotas_max']:
        copia[c] = copia[c].mask(copia[c] < 0)

    copia['n_items'] = copia['n_items'].mask(copia['n_items'] < 1)
    
    for c in ['estado_cliente', 'estado_vendedor']:
        copia[c] = copia[c].where(copia[c].isin(uf_validas))

    return copia

def crear_variables(X):
    """
    Crea variables derivadas para el modelo de mala_resena.
    Recibe el DataFrame completo y lo devuelve con las columnas nuevas.
    Asume fechas ya en datetime64[ns] y tipos ya convertidos.
    """
    d = X.copy()

    # 1. Mismo estado cliente / vendedor (0/1)
    sin_estado = d['estado_cliente'].isna() | d['estado_vendedor'].isna()
    d['mismo_estado'] = ((d['estado_cliente'] == d['estado_vendedor'])
                         .astype('Int64').mask(sin_estado))

    # 2. Cumplimiento de la entrega (el día como referencia) 
    # dif > 0: llegó antes de lo prometido | dif = 0: el mismo día | dif < 0: llegó tarde
    dif = (d['order_estimated_delivery_date'].dt.normalize()
           - d['order_delivered_customer_date'].dt.normalize()).dt.days

    d['dias_dif_entrega'] = dif.astype('Int64')            # estimada - real
    d['dias_retraso']     = (-dif).clip(lower=0).astype('Int64')
    d['dias_adelanto']    = dif.clip(lower=0).astype('Int64')
    d['estado_entrega'] = (
        pd.Series(np.select([dif == 0, dif < 0, dif > 0],
                            ['a_tiempo', 'tardia', 'adelantada'], default=''),
                  index=d.index)
        .replace('', pd.NA).astype('string')
    )
    d['entrega_tardia'] = (dif < 0).astype('Int64').mask(dif.isna())

    # 3. Flete vs. precio 
    precio = d['precio_total']
    flete = d['flete_total']
    sin_monto = precio.isna() | flete.isna()
    d['flete_mayor_precio'] = (flete > precio).astype('Int64').mask(sin_monto)
    d['pct_flete'] = (100 * flete / precio.where(precio > 0)).round(2)

    #  4. Horas entre compra y aprobación del pago 
    d['horas_aprobacion'] = (
        (d['order_approved_at'] - d['order_purchase_timestamp'])
        .dt.total_seconds() / 3600
    )

    # 5. Variables de Tiempo 
    # Tiempos del proceso
    d['dias_entrega_total'] = (
        (d['order_delivered_customer_date'] - d['order_purchase_timestamp'])
        .dt.total_seconds() / 86400
    )
    d['dias_prometidos'] = (
        (d['order_estimated_delivery_date'] - d['order_purchase_timestamp'])
        .dt.total_seconds() / 86400
    )
    d['ratio_entrega'] = d['dias_entrega_total'] / d['dias_prometidos'].where(d['dias_prometidos'] > 0)
    d['dias_despacho'] = (
        (d['order_delivered_carrier_date'] - d['order_approved_at'])
        .dt.total_seconds() / 86400
    )
    d['dias_transito'] = (
        (d['order_delivered_customer_date'] - d['order_delivered_carrier_date'])
        .dt.total_seconds() / 86400
    )

    # Valor por ítem
    items = d['n_items'].astype('float64').where(d['n_items'] > 0)
    d['precio_por_item'] = precio / items
    d['flete_por_item']  = flete / items

    # Complejidad del pedido y del pago
    d['multi_vendedor'] = (d['n_vendedores'] > 1).astype('Int64').mask(d['n_vendedores'].isna())
    d['pago_en_cuotas'] = (d['cuotas_max'] > 1).astype('Int64').mask(d['cuotas_max'].isna())

    # Calendario de la compra
    d['hora_compra'] = d['order_purchase_timestamp'].dt.hour.astype('Int64')
    d['dia_semana_compra'] = d['order_purchase_timestamp'].dt.dayofweek.astype('Int64')   # 0 = lunes
    d['compra_fin_semana'] = (d['dia_semana_compra'] >= 5).astype('Int64')

    return d


NUMERICAS = ['n_items', 'n_vendedores', 'precio_total', 'flete_total', 'peso_total_g',
             'fotos_promedio', 'cuotas_max', 'pct_flete', 'horas_aprobacion',
             'dias_entrega_total', 'dias_prometidos', 'ratio_entrega', 'dias_despacho',
             'dias_transito', 'dias_dif_entrega', 'precio_por_item', 'flete_por_item',
             'hora_compra', 'dia_semana_compra']
BINARIAS  = ['mismo_estado', 'flete_mayor_precio', 'multi_vendedor',
             'pago_en_cuotas', 'compra_fin_semana']
NOMINALES = ['estado_cliente', 'estado_vendedor', 'categoria', 'tipo_pago']


def seleccionar_variables(X):
    """Deja solo las variables del modelo, con tipos que scikit-learn lee bien (NaN, no pd.NA)."""
    d = X[NUMERICAS + BINARIAS + NOMINALES].copy()
    for c in NUMERICAS + BINARIAS:
        d[c] = pd.to_numeric(d[c], errors='coerce').astype('float64')
    for c in NOMINALES:
        s = d[c].astype('object')
        d[c] = s.where(s.notna(), np.nan)
    return d


def crear_pipeline(medianas=None):
    return Pipeline([
        ("fechas",    FunctionTransformer(validar_fechas_pedido, kw_args={"medianas": medianas})),
        ("tipos",     FunctionTransformer(convertir_tipos)),
        ("rangos",    FunctionTransformer(validar_rangos)),
        ("variables", FunctionTransformer(crear_variables)),
    ])
