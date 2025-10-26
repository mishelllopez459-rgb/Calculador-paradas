import streamlit as st
import pandas as pd
import pydeck as pdk
import networkx as nx

# ---------------- CONFIG / UI ----------------
st.set_page_config(page_title="Rutas San Marcos", layout="wide")
st.title("🚌 Calculador de paradas y ruta óptima — San Marcos")

# ---------------- CARGA DE DATOS ----------------
# nodos.csv => columnas: id, nombre, lat, lon
# aristas.csv => columnas: origen, destino, tiempo_min, distancia_km, capacidad (opcional)
nodos = pd.read_csv("nodos.csv")
aristas = pd.read_csv("aristas.csv")

# Limpieza mínima
nodos["id"] = nodos["id"].astype(str).str.strip()
nodos["nombre"] = nodos["nombre"].astype(str).str.strip()
aristas["origen"] = aristas["origen"].astype(str).str.strip()
aristas["destino"] = aristas["destino"].astype(str).str.strip()

# Helpers de mapeo
id_por_nombre = {r["nombre"]: r["id"] for _, r in nodos.iterrows()}
nombre_por_id = {r["id"]: r["nombre"] for _, r in nodos.iterrows()}

# ---------------- CONSTANTES DE TIEMPO ----------------
# velocidades estimadas
VEL_BUS_KMH = 30.0     # km/h bus urbano aprox
VEL_BICI_KMH = 15.0    # km/h bicicleta urbana aprox

def hex_to_rgb(h: str):
    """'#RRGGBB' -> [R,G,B]"""
    h = h.lstrip("#")
    return [int(h[i:i+2], 16) for i in (0, 2, 4)]

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("Parámetros")

    dirigido = st.checkbox(
        "Tramos unidireccionales (grafo dirigido)",
        value=False,
        help="Activa para considerar el sentido de las aristas (A→B distinto de B→A)",
    )

    origen_nombre = st.selectbox("Origen", sorted(nodos["nombre"]))
    destino_nombre = st.selectbox("Destino", sorted(nodos["nombre"]), index=1)

    # criterio de optimización
    criterio = st.radio(
        "Optimizar por",
        ["tiempo_min", "distancia_km"],
        index=0,
        help="Elige si quieres la ruta más rápida (tiempo_min) o la más corta (distancia_km).",
    )

    st.markdown("### Colores")
    col_nodes = st.color_picker("Nodos", "#FF007F")
    col_edges = st.color_picker("Aristas", "#F2F2F2")
    col_path  = st.color_picker("Ruta seleccionada", "#007AFF")

    calcular = st.button("Calcular ruta")

# Colores a RGB para pydeck
RGB_NODES = hex_to_rgb(col_nodes)
RGB_EDGES = hex_to_rgb(col_edges)
RGB_PATH  = hex_to_rgb(col_path)

# ---------------- CREAR GRAFO ----------------
G = nx.DiGraph() if dirigido else nx.Graph()

# Agregar nodos con atributos
for _, r in nodos.iterrows():
    G.add_node(
        r["id"],
        nombre=r["nombre"],
        lat=float(r["lat"]),
        lon=float(r["lon"]),
    )

# Agregar aristas con atributos
# Además de lo que viene en CSV, calculamos tiempo_bus_min y tiempo_bici_min
# a partir de la distancia_km
def tiempo_por_dist(dist_km: float, vel_kmh: float) -> float:
    # minutos = (km / kmh) * 60
    if vel_kmh <= 0:
        return 0.0
    return (dist_km / vel_kmh) * 60.0

aristas_enriquecidas = []
for _, r in aristas.iterrows():
    dist = float(r["distancia_km"])
    t_real_min = float(r["tiempo_min"])

    # calculamos tiempos teóricos
    t_bus_min  = tiempo_por_dist(dist, VEL_BUS_KMH)
    t_bici_min = tiempo_por_dist(dist, VEL_BICI_KMH)

    data_edge = {
        "origen": r["origen"],
        "destino": r["destino"],
        "tiempo_min": t_real_min,
        "distancia_km": dist,
        "capacidad": float(r.get("capacidad", 0)),
        "tiempo_bus_min": t_bus_min,
        "tiempo_bici_min": t_bici_min,
    }
    aristas_enriquecidas.append(data_edge)

    # guardamos en el grafo
    G.add_edge(
        r["origen"],
        r["destino"],
        tiempo_min=t_real_min,
        distancia_km=dist,
        capacidad=float(r.get("capacidad", 0)),
        tiempo_bus_min=t_bus_min,
        tiempo_bici_min=t_bici_min,
    )

# DataFrame de aristas enriquecidas
aristas_full_df = pd.DataFrame(aristas_enriquecidas)

# ---------------- FUNCIÓN RUTA ÓPTIMA ----------------
def ruta_optima(o_id: str, d_id: str, peso: str):
    """
    Devuelve:
      path_ids: lista de ids de nodos en orden
      total_peso: suma del peso elegido (tiempo_min o distancia_km) en la ruta
      info_edges: detalle por tramo con distancia y tiempos bus/bici
    """
    # path = [nodo1, nodo2, ..., nodoN]
    path_ids = nx.shortest_path(G, source=o_id, target=d_id, weight=peso)

    # construir detalle por tramo
    detalle_tramos = []
    total_peso = 0.0
    total_dist = 0.0
    total_bus_min = 0.0
    total_bici_min = 0.0

    for u, v in zip(path_ids[:-1], path_ids[1:]):
        edge_data = G[u][v]

        # acumular métricas
        total_peso += edge_data[peso]
        total_dist += edge_data["distancia_km"]
        total_bus_min  += edge_data["tiempo_bus_min"]
        total_bici_min += edge_data["tiempo_bici_min"]

        detalle_tramos.append({
            "origen_id": u,
            "origen_nombre": nombre_por_id[u],
            "destino_id": v,
            "destino_nombre": nombre_por_id[v],
            "distancia_km": edge_data["distancia_km"],
            "tiempo_min_real": edge_data["tiempo_min"],
            "tiempo_bus_min": edge_data["tiempo_bus_min"],
            "tiempo_bici_min": edge_data["tiempo_bici_min"],
        })

    resumen_totales = {
        "total_peso": total_peso,
        "total_distancia_km": total_dist,
        "total_bus_min": total_bus_min,
        "total_bici_min": total_bici_min,
    }

    return path_ids, resumen_totales, pd.DataFrame(detalle_tramos)

# ---------------- PREPARAR CAPAS PARA MAPA ----------------
# Construimos dataframe de segmentos del grafo completo (todas las aristas)
edges_df = (
    aristas_full_df
    .merge(
        nodos[["id", "lat", "lon"]],
        left_on="origen", right_on="id",
        how="left"
    )
    .rename(columns={"lat": "lat_o", "lon": "lon_o"})
    .drop(columns=["id"])
    .merge(
        nodos[["id", "lat", "lon"]],
        left_on="destino", right_on="id",
        how="left"
    )
    .rename(columns={"lat": "lat_d", "lon": "lon_d"})
    .drop(columns=["id"])
)

edges_layer = pdk.Layer(
    "LineLayer",
    data=edges_df,
    get_source_position="[lon_o, lat_o]",
    get_target_position="[lon_d, lat_d]",
    get_width=2,
    width_min_pixels=2,
    get_color=RGB_EDGES,
    pickable=True,
)

nodes_layer = pdk.Layer(
    "ScatterplotLayer",
    data=nodos.rename(columns={"lon": "lng"}),
    get_position="[lng, lat]",
    get_radius=65,
    radius_min_pixels=3,
    get_fill_color=RGB_NODES,
    get_line_color=[30, 30, 30],
    line_width_min_pixels=1,
    pickable=True,
)

center_lat, center_lon = nodos["lat"].mean(), nodos["lon"].mean()
view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=13)

# ---------------- UI PRINCIPAL ----------------
col1, col2 = st.columns([1, 2])

if calcular:
    origen_id = id_por_nombre[origen_nombre]
    destino_id = id_por_nombre[destino_nombre]

    try:
        path_ids, totals, tramos_df = ruta_optima(origen_id, destino_id, criterio)

        # DataFrame de nodos en la ruta (para mostrar y para PathLayer)
        ruta_nodos_df = pd.DataFrame(
            [{
                "id": n,
                "nombre": nombre_por_id[n],
                "lat": G.nodes[n]["lat"],
                "lon": G.nodes[n]["lon"],
            } for n in path_ids]
        )

        # PathLayer para resaltar la ruta óptima
        path_layer = pdk.Layer(
            "PathLayer",
            data=[{"path": ruta_nodos_df[["lon", "lat"]].values.tolist()}],
            get_path="path",
            get_width=6,
            width_scale=8,
            get_color=RGB_PATH,
            pickable=False,
        )

        # ----- PANEL IZQUIERDO -----
        with col1:
            st.subheader("Resumen de la ruta")

            st.markdown(f"**Origen:** {origen_nombre}")
            st.markdown(f"**Destino:** {destino_nombre}")
            st.markdown(f"**Optimizado por:** `{criterio}`")
            st.markdown(f"**Grafo:** {'Dirigido' if dirigido else 'No dirigido'}")

            st.markdown(f"**Paradas totales (incluye origen y destino):** {len(path_ids)}")
            st.markdown(f"**Paradas intermedias:** {max(0, len(path_ids) - 2)}")

            # Distancia total y tiempos totales
            st.markdown(f"**Distancia total:** {totals['total_distancia_km']:.2f} km")
            st.markdown(f"**Tiempo total estimado en bus (@{VEL_BUS_KMH:.0f} km/h):** {totals['total_bus_min']:.1f} min")
            st.markdown(f"**Tiempo total estimado en bici (@{VEL_BICI_KMH:.0f} km/h):** {totals['total_bici_min']:.1f} min")
            st.markdown(f"**Tiempo total real registrado (suma `{criterio}`):** {totals['total_peso']:.2f}")

            st.markdown("#### Nodos en orden (ruta):")
            st.dataframe(ruta_nodos_df, use_container_width=True)

            st.download_button(
                "📥 Descargar nodos de la ruta (CSV)",
                data=ruta_nodos_df.to_csv(index=False).encode("utf-8"),
                file_name=f"ruta_nodos_{origen_id}_{destino_id}_{criterio}.csv",
                mime="text/csv"
            )

            st.markdown("#### Tramos (aristas) en la ruta:")
            # mostramos tabla con distancia y tiempos bus/bici tramo por tramo
            st.dataframe(tramos_df, use_container_width=True)

            st.download_button(
                "📥 Descargar tramos de la ruta (CSV)",
                data=tramos_df.to_csv(index=False).encode("utf-8"),
                file_name=f"ruta_tramos_{origen_id}_{destino_id}_{criterio}.csv",
                mime="text/csv"
            )

        # ----- PANEL DERECHO (MAPA) -----
        with col2:
            st.pydeck_chart(
                pdk.Deck(
                    layers=[edges_layer, nodes_layer, path_layer],
                    initial_view_state=view_state,
                    tooltip={
                        "text": (
                            "Origen: {origen}\n"
                            "Destino: {destino}\n"
                            "Dist: {distancia_km} km\n"
                            "Bus: {tiempo_bus_min} min\n"
                            "Bici: {tiempo_bici_min} min"
                        )
                    },
                ),
                use_container_width=True
            )

    except nx.NetworkXNoPath:
        with col1:
            st.error("No hay camino entre esos nodos con el grafo actual.")
        with col2:
            st.pydeck_chart(
                pdk.Deck(
                    layers=[edges_layer, nodes_layer],
                    initial_view_state=view_state,
                    tooltip={
                        "text": (
                            "Origen: {origen}\n"
                            "Destino: {destino}\n"
                            "Dist: {distancia_km} km\n"
                            "Bus: {tiempo_bus_min} min\n"
                            "Bici: {tiempo_bici_min} min"
                        )
                    },
                ),
                use_container_width=True
            )

else:
    # pantalla inicial sin ruta calculada
    st.info("Elige Origen, Destino y presiona **Calcular ruta**.")
    st.pydeck_chart(
        pdk.Deck(
            layers=[edges_layer, nodes_layer],
            initial_view_state=view_state,
            tooltip={
                "text": (
                    "Origen: {origen}\n"
                    "Destino: {destino}\n"
                    "Dist: {distancia_km} km\n"
                    "Bus: {tiempo_bus_min} min\n"
                    "Bici: {tiempo_bici_min} min"
                )
            },
        ),
        use_container_width=True
    )
