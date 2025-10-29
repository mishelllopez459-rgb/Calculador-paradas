# =========================
# Resolver ruta
# =========================
def nombre_a_id(nodos_df, nombre):
    m = nodos_df.loc[nodos_df["nombre"] == nombre]
    if m.empty: return None
    return str(m.iloc[0]["id"])

o_id = nombre_a_id(nodos, origen_nombre)
d_id = nombre_a_id(nodos, destino_nombre)

# grafo con pesos (dirigido o no, según el toggle)
graph, weights = build_weighted_graph(aristas_raw, nodos, dirigido=st.session_state.dirigido)

# Dijkstra según criterio
use = "time" if criterio == "tiempo_min" else "dist"
path_ids, costo = dijkstra(graph, o_id, d_id, use=use)

# ---------------- Fallback si NO hay camino ----------------
no_path = (not path_ids or costo == float("inf"))
if no_path and o_id and d_id:
    # inventamos un "camino" de 1 salto para poder calcular métricas
    path_ids = [o_id, d_id]

# Totales base usando pesos del grafo (si hay camino)
total_time = 0.0
total_dist = 0.0
if not no_path and len(path_ids) > 1:
    for i in range(len(path_ids) - 1):
        u, v = path_ids[i], path_ids[i + 1]
        w = weights.get((u, v), {"time": 3.0, "dist": 0.6})
        # evitar ceros patológicos
        total_time += w["time"] if w["time"] and w["time"] > 0 else 3.0
        total_dist += w["dist"] if w["dist"] and w["dist"] > 0 else 0.6

# ---------------- Rutas geométricas (OSRM / grafo / recta) ----------------
fila_o = nodos.loc[nodos["id"] == o_id].iloc[0] if o_id in set(nodos["id"]) else pd.Series({"lat":None,"lon":None})
fila_d = nodos.loc[nodos["id"] == d_id].iloc[0] if d_id in set(nodos["id"]) else pd.Series({"lat":None,"lon":None})

ruta_osrm = None; dist_osrm = None; dur_osrm = None
if usar_osrm and tiene_coords(fila_o) and tiene_coords(fila_d):
    ruta_osrm, dist_osrm, dur_osrm = osrm_route(float(fila_o["lat"]), float(fila_o["lon"]),
                                                float(fila_d["lat"]), float(fila_d["lon"]))

ruta_grafo = ids_a_polyline_lonlat(nodos, path_ids) if len(path_ids) >= 2 else []
ruta_recta = []
if tiene_coords(fila_o) and tiene_coords(fila_d):
    ruta_recta = [[float(fila_o["lon"]), float(fila_o["lat"])],
                  [float(fila_d["lon"]), float(fila_d["lat"])]]

# Ruta final (siempre alguna si hay coords O/D)
if ruta_osrm:
    ruta_final = ruta_osrm
elif ruta_grafo and len(ruta_grafo) >= 2:
    ruta_final = ruta_grafo
else:
    ruta_final = ruta_recta  # puede quedar [] si no hay coords en O y D

# ---------------- Totales definitivos (que no queden en 0.00) ----------------
# Si NO hay camino en el grafo, calculamos los totales con OSRM / Haversine / supuestos
if no_path or len(path_ids) <= 1:
    if ruta_osrm and dist_osrm is not None and dur_osrm is not None:
        total_dist = dist_osrm
        total_time = dur_osrm
    elif ruta_recta:
        # distancia a partir de recta
        (lon1, lat1), (lon2, lat2) = ruta_recta
        hv = haversine_km(lat1, lon1, lat2, lon2)
        total_dist = hv
        total_time = (hv / VEL_KMH) * 60.0
    else:
        # sin coords: asume 1 salto
        total_dist = 0.6
        total_time = 3.0

# =========================
# Capas y mapa
# =========================
RGB_NODES = hex_to_rgb(col_nodes)
RGB_EDGES = hex_to_rgb(col_edges)
RGB_PATH  = hex_to_rgb(col_path)

layers = []; all_coords = []

nodes_layer, nodos_plot = capa_nodos(nodos, RGB_NODES)
if nodes_layer is not None:
    layers.append(nodes_layer)
    all_coords += nodos_plot[["lng","lat"]].values.tolist()

edges_layer, edges_paths = capa_aristas(aristas_raw, nodos, RGB_EDGES, width_px=3)
if edges_layer is not None:
    layers.append(edges_layer)
    for seg in edges_paths: all_coords += seg["path"]

route_layer = capa_ruta(ruta_final, RGB_PATH, width_px=8)
if route_layer is not None:
    layers.append(route_layer)
    all_coords += ruta_final

view_state = fit_view_from_lonlat(all_coords, 0.4) if all_coords else pdk.ViewState(latitude=14.965, longitude=-91.79, zoom=13)

# =========================
# Resumen (estilo 2ª imagen)
# =========================
col1, col2 = st.columns([1,2])
with col1:
    st.subheader("Resumen")
    st.markdown(f"**Origen:** {origen_nombre}")
    st.markdown(f"**Destino:** {destino_nombre}")
    st.markdown(f"**Criterio:** `{'tiempo_min' if use=='time' else 'distancia_km'}`")
    st.markdown(f"**Grafo:** {'Dirigido' if st.session_state.dirigido else 'No dirigido'}")

    paradas_tot = len(path_ids) if len(path_ids) >= 2 else 2
    paradas_int = max(0, paradas_tot - 2)
    st.markdown(f"**Paradas (incluye origen y destino):** {paradas_tot}")
    st.markdown(f"**Paradas intermedias:** {paradas_int}")

    if use == "time":
        st.markdown(f"**Costo total (tiempo_min):** {total_time:.2f}")
        st.markdown(f"**Distancia aprox.:** {total_dist:.2f} km")
    else:
        st.markdown(f"**Costo total (distancia_km):** {total_dist:.2f}")
        st.markdown(f"**Tiempo aprox.:** {total_time:.2f} min")

    if ruta_final:
        export_df = pd.DataFrame(ruta_final, columns=["lon","lat"])
        st.download_button(
            "📥 Descargar ruta (CSV)",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name="ruta.csv",
            mime="text/csv",
        )

    st.markdown("---")
    st.dataframe(nodos.dropna(subset=["lat","lon"])[["id","nombre","lat","lon"]],
                 use_container_width=True)

with col2:
    st.pydeck_chart(
        pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            tooltip={"html": "<b>{origen}</b> → <b>{destino}</b>", "style": {"color": "white"}},
        ),
        use_container_width=True,
    )


