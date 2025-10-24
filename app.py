import streamlit as st
import pandas as pd
import pydeck as pdk
import networkx as nx

# ---------------- UI / CONFIG ----------------
st.set_page_config(page_title="Rutas San Marcos", layout="wide")
st.title("🚌 Calculador de paradas y ruta óptima — San Marcos")

# Carga de datos
nodos = pd.read_csv("nodos.csv")       # id, nombre, lat, lon
aristas = pd.read_csv("aristas.csv")   # origen, destino, tiempo_min, distancia_km, capacidad

# Limpieza mínima
nodos["id"] = nodos["id"].astype(str).str.strip()
nodos["nombre"] = nodos["nombre"].astype(str).str.strip()
aristas["origen"] = aristas["origen"].astype(str).str.strip()
aristas["destino"] = aristas["destino"].astype(str).str.strip()

# Helpers
id_por_nombre = {r["nombre"]: r["id"] for _, r in nodos.iterrows()}
nombre_por_id = {r["id"]: r["nombre"] for _, r in nodos.iterrows()}

def hex_to_rgb(h: str):
    """'#RRGGBB' -> [R,G,B]"""
    h = h.lstrip("#")
    return [int(h[i:i+2], 16) for i in (0, 2, 4)]

def rgba(rgb, alpha=220):
    """[R,G,B] -> [R,G,B,A] con alfa fijo para colores estables."""
    return [int(rgb[0]), int(rgb[1]), int(rgb[2]), int(alpha)]

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("Parámetros")

    # 1) Direccionalidad
    dirigido = st.checkbox(
        "Tramos unidireccionales (grafo dirigido)",
        value=False,
        help="Activa para considerar el sentido de las aristas (A→B distinto de B→A)"
    )

    # 2) Origen/Destino y criterio
    origen_nombre = st.selectbox("Origen", sorted(nodos["nombre"]))
    destino_nombre = st.selectbox("Destino", sorted(nodos["nombre"]), index=1)
    criterio = st.radio("Optimizar por", ["tiempo_min", "distancia_km"], index=0)

    # 3) Colores (por defecto: nodos rosa fuerte, aristas blancas, ruta azul)
    st.markdown("### Colores")
    col_nodes = st.color_picker("Nodos", "#FF007F")
    col_edges = st.color_picker("Aristas", "#F2F2F2")
    col_path  = st.color_picker("Ruta seleccionada", "#007AFF")

    # 4) Mostrar u ocultar la red base
    mostrar_red = st.toggle("Mostrar red base (nodos/aristas)", value=False)

    calcular = st.button("Calcular ruta")

# Convertir colores a RGB
RGB_NODES = hex_to_rgb(col_nodes)
RGB_EDGES = hex_to_rgb(col_edges)
RGB_PATH  = hex_to_rgb(col_path)

# ---------------- GRAFO ----------------
G = nx.DiGraph() if dirigido else nx.Graph()

for _, r in nodos.iterrows():
    G.add_node(r["id"], nombre=r["nombre"], lat=r["lat"], lon=r["lon"])

for _, r in aristas.iterrows():
    G.add_edge(
        r["origen"], r["destino"],
        tiempo_min=float(r["tiempo_min"]),
        distancia_km=float(r["distancia_km"]),
        capacidad=float(r.get("capacidad", 0))
    )

# Centro del mapa
center_lat, center_lon = nodos["lat"].mean(), nodos["lon"].mean()
view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=13)

def ruta_optima(o_id: str, d_id: str, peso: str):
    path = nx.shortest_path(G, source=o_id, target=d_id, weight=peso)
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        total += G[u][v][peso]
    return path, total

# ---------------- CAPAS DE MAPA ----------------
# Edges como segmentos con coords
edges_df = aristas.merge(nodos[["id","lat","lon"]], left_on="origen", right_on="id") \
                  .rename(columns={"lat":"lat_o","lon":"lon_o"}).drop(columns=["id"])
edges_df = edges_df.merge(nodos[["id","lat","lon"]], left_on="destino", right_on="id") \
                   .rename(columns={"lat":"lat_d","lon":"lon_d"}).drop(columns=["id"])

edges_layer = pdk.Layer(
    "LineLayer",
    data=edges_df,
    get_source_position="[lon_o, lat_o]",
    get_target_position="[lon_d, lat_d]",
    get_width=2,
    width_min_pixels=2,
    get_color=rgba(RGB_EDGES, 220),      # color aristas con alfa fijo
    pickable=True,
)

nodes_layer = pdk.Layer(
    "ScatterplotLayer",
    data=nodos.rename(columns={"lon": "lng"}),
    get_position="[lng, lat]",
    get_radius=65,
    radius_min_pixels=3,
    get_fill_color=rgba(RGB_NODES, 230),  # color nodos con alfa fijo
    get_line_color=[30,30,30,255],        # borde estable
    line_width_min_pixels=1,
    pickable=True,
)

# ---------------- UI / RESULTADO ----------------
col1, col2 = st.columns([1, 2])

layers = []  # se llenará según la acción del usuario

if calcular:
    o = id_por_nombre[origen_nombre]
    d = id_por_nombre[destino_nombre]

    try:
        path, total = ruta_optima(o, d, criterio)

        tramo_df = pd.DataFrame([{
            "id": n, "nombre": nombre_por_id[n],
            "lat": G.nodes[n]["lat"], "lon": G.nodes[n]["lon"]
        } for n in path])

        path_layer = pdk.Layer(
            "PathLayer",
            data=[{"path": tramo_df[["lon", "lat"]].values.tolist()}],
            get_path="path",
            get_width=6,
            width_scale=8,
            get_color=rgba(RGB_PATH, 255),   # color ruta con alfa fijo
            pickable=False,
        )

        with col1:
            st.subheader("Resumen")
            st.markdown(f"**Origen:** {origen_nombre}")
            st.markdown(f"**Destino:** {destino_nombre}")
            st.markdown(f"**Criterio:** `{criterio}`")
            st.markdown(f"**Grafo:** {'Dirigido' if dirigido else 'No dirigido'}")
            st.markdown(f"**Paradas (incluye origen y destino):** {len(path)}")
            st.markdown(f"**Paradas intermedias:** {max(0, len(path) - 2)}")
            st.markdown(f"**Costo total ({criterio}):** {total:.2f}")

            st.download_button(
                "📥 Descargar ruta (CSV)",
                data=tramo_df.to_csv(index=False).encode("utf-8"),
                file_name=f"ruta_{o}_{d}_{criterio}.csv",
                mime="text/csv"
            )
            st.dataframe(tramo_df, use_container_width=True)

        # Capas: siempre la ruta; la red base solo si la piden
        layers = [path_layer]
        if mostrar_red:
            layers.extend([edges_layer, nodes_layer])

        with col2:
            st.pydeck_chart(pdk.Deck(
                layers=layers,
                initial_view_state=view_state
            ), use_container_width=True)

    except nx.NetworkXNoPath:
        with col1:
            st.error("No hay camino entre esos nodos con el grafo actual.")

        if mostrar_red:
            layers = [edges_layer, nodes_layer]

        with col2:
            st.pydeck_chart(pdk.Deck(
                layers=layers,
                initial_view_state=view_state
            ), use_container_width=True)

else:
    st.info("Elige origen, destino y presiona **Calcular ruta**.")
    # Al inicio: mapa limpio; si quieren, pueden ver la red base
    if mostrar_red:
        layers = [edges_layer, nodes_layer]
    st.pydeck_chart(pdk.Deck(
        layers=layers,
        initial_view_state=view_state
    ), use_container_width=True)

