from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

import preparacion      # el modelo guardado busca aquí sus funciones; debe importarse antes de cargarlo

modelo = joblib.load(Path(__file__).parent / "modelo_riesgo_xgb.joblib")

# Corte del 10% de mayor riesgo, tomado de la validación (0.373).
# Es una aproximación: se revisa si el modelo se vuelve a entrenar.
UMBRAL_RIESGO_ALTO = 0.373

app = FastAPI(title="API de riesgo de mala reseña")


class Pedido(BaseModel):
    order_id: str
    order_purchase_timestamp: str
    order_approved_at: Optional[str] = None
    order_delivered_carrier_date: Optional[str] = None
    order_delivered_customer_date: str            # obligatoria: el modelo predice al entregar
    order_estimated_delivery_date: str
    estado_cliente: Optional[str] = None
    n_items: int
    n_vendedores: int
    precio_total: float
    flete_total: float
    peso_total_g: Optional[float] = None
    fotos_promedio: Optional[float] = None
    categoria: Optional[str] = None
    estado_vendedor: Optional[str] = None
    tipo_pago: Optional[str] = None
    cuotas_max: Optional[int] = None


@app.get("/")
def inicio():
    return {"estado": "ok", "modelo": "HistGradientBoosting"}


@app.post("/predecir")
def predecir(pedidos: list[Pedido]):
    df = pd.DataFrame([p.model_dump() for p in pedidos])
    proba = modelo.predict_proba(df)[:, 1]
    return [
        {"order_id": oid,
         "probabilidad": round(float(p), 4),
         "riesgo_alto": bool(p >= UMBRAL_RIESGO_ALTO)}
        for oid, p in zip(df["order_id"], proba)
    ]