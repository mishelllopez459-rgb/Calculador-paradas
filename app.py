import streamlit as st
import pandas as pd
import pydeck as pdk
import networkx as nx

st.set_page_config(page_title="Rutas San Marcos", layout="wide")
st.title("🚌 Calculador de paradas y ruta óptima — San Marcos")

# --- Carga de datos ---
nodos = pd.read_csv("nodos.csv")       # id, nombre, lat, lon
aristas = pd.read_csv("aristas.csv")   # origen, destino, tiempo_min, distancia_km, capacidad

# Limpieza mínima
nodos["id"] = nodos["id"].astype(str).str.strip()
nodos["nombre"] = nodos["nombre"].astype(str).str.strip()
aristas["origen"] = aristas["origen"].astype(str).str.strip()
aristas["destino"] = aristas["destino"].astype(str).str.strip()

# --- Grafo (no dirigido) ---
G = nx.Graph()
for _, r in nodos.iterrows():
    G.add_node(r["id"], nombre=r["nombre"], lat=r["lat"], lon=r["lon"])
for _, r in aristas.iterrows():
    G.add_edge(
        r["origen"], r["destino"],
        tiempo_min=float(r["tiempo_min"]),
        distancia_km=float(r["distancia_km"]),
        capacidad=float(r.get("capacidad", 0))
    )

id_por_nombre = {r["nombre"]: r["id"] for _, r in nodos.iterrows()}
nombre_por_id = {r["id"]: r["nombre"] for _, r in nodos.iterrows()}

# --- UI lateral ---
with st.sidebar:
    st.header("Parámetros")
    origen_nombre = st.selectbox("Origen", sorted(nodos["nombre"]))
    destino_nombre = st.selectbox("Destino", sorted(nodos["nombre"]), index=1)
    criterio = st.radio("Optimizar por", ["tiempo_min", "distancia_km"], index=0)
    calcular = st.button("Calcular ruta")

center_lat, center_lon = nodos["lat"].mean(), nodos["lon"].mean()

def ruta_optima(o_id: str, d_id: str, peso: str):
    path = nx.shortest_path(G, source=o_id, target=d_id, weight=peso)
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        total += G[u][v][peso]
    return path, total

# --- Capas base (todas las aristas/nodos) ---
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
)

nodes_layer = pdk.Layer(
    "ScatterplotLayer",
    data=nodos.rename(columns={"lon":"lng"}),
    get_position="[lng, lat]",
    get_radius=60,
    radius_min_pixels=3,
)

view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=13)

col1, col2 = st.columns([1,2])

if calcular:
    o = id_por_nombre[origen_nombre]
    d = id_por_nombre[destino_nombre]
    path, total = ruta_optima(o, d, criterio)

    tramo_df = pd.DataFrame([{
        "id": n, "nombre": nombre_por_id[n],
        "lat": G.nodes[n]["lat"], "lon": G.nodes[n]["lon"]
    } for n in path])

    path_layer = pdk.Layer(
        "PathLayer",
        data=[{"path": tramo_df[["lon","lat"]].values.tolist()}],
        get_path="path",
        get_width=5,
        width_scale=8,
    )

    with col1:
        st.subheader("Resumen")
        st.markdown(f"**Origen:** {origen_nombre}")
        st.markdown(f"**Destino:** {destino_nombre}")
        st.markdown(f"**Criterio:** `{criterio}`")
        st.markdown(f"**Paradas (incluye origen y destino):** {len(path)}")
        st.markdown(f"**Paradas intermedias:** {max(0, len(path)-2)}")
        st.markdown(f"**Costo total ({criterio}):** {total:.2f}")
        st.dataframe(tramo_df, use_container_width=True)

    with col2:
        st.pydeck_chart(pdk.Deck(
            layers=[edges_layer, nodes_layer, path_layer],
            initial_view_state=view_state
        ), use_container_width=True)
else:
    st.info("Elige origen, destino y presiona **Calcular ruta**.")
    st.pydeck_chart(pdk.Deck(
        layers=[edges_layer, nodes_layer],
        initial_view_state=view_state
    ), use_container_width=True)
