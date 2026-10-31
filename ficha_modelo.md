
### Qué predice
La probabilidad de que un pedido reciba una reseña de 1 o 2 estrellas, calculada **al momento de la
entrega** (no se usa nada de la reseña). Este modelo no es un pronóstico y no puede utilizarse antes
de la entrega.

### Modelo
XGBoost dentro de un pipeline de scikit-learn, con las transformaciones de `preparacion.py`. El modelo
final se entrenó con todos los pedidos disponibles.

### Validación por fecha
Se entrenó con los pedidos anteriores al 28 de febrero de 2018 (56,579) y se validó con los siguientes,
hasta mayo de 2018 (20,724 pedidos; 14.8% de malas reseñas). AUC en validación: 0.794.

### Métrica
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
###  Variables que más pesan
Según la pérdida de AUC al desordenar cada variable, las que más pesan son:
-  **Diferencia entre la fecha estimada y la real de entrega** (los días de retraso): pérdida
  promedio de 0.016 de AUC en los meses con menores tasas de mala reseña.
-  **Cantidad de artículos del pedido:** pérdida de 0.043.
-  **Tiempos de entrega:** pérdida de 0.017.

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

### Límites
-  La fecha de entrega al cliente es obligatoria: el modelo predice al momento de la entrega.
-  Los datos para la predicción por lotes deben subirse en formato JSON (valida el formato de las
  fechas antes de subirlos).
-  El corte de «riesgo alto» (36.4%) viene de la validación y es aproximado.
- Por sí solas, las estrellas no explican la inconformidad del cliente, por lo que no deberían
  usarse como único factor para evaluar el desempeño de un producto o de un vendedor.
- La predicción puede fallar si no se completan los campos obligatorios.

###  ¿A quién puede afectar un error?
-  **Vendedores:** pueden perder comisiones o incluso asumir el costo de una mala predicción.
-  **Desempeño del producto:** la predicción no considera nada de las reseñas, por lo que se
  desconoce la razón por la que un pedido se identifica como de riesgo. La causa queda a interpretación
  de quien reciba la predicción.
-  **Calidad de la predicción:** el modelo desconoce los días feriados, que alargan los días de
  entrega total. También pueden fallar las fechas estimadas cuando las genera un sistema automático
  con sus propios errores.
-  **Consumidores:** una mala predicción puede etiquetarlos como clientes de alto riesgo o
  problemáticos.
-  **Empresas:** quienes confían en el modelo y concentran recursos en retener clientes pueden
  gastarlos en quienes no los necesitaban y perder a quienes sí requieren algún tratamiento.
""")