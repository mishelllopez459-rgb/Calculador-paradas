import streamlit as st
import pandas as pd
import pydeck as pdk
import networkx as nx

# ---------------- UI / CONFIG ----------------
st.set_page_config(page_title="Rutas San Marcos", layout="wide")
st.title("🚌 Calculador de paradas y ruta óptima — San Marcos")

# ---------------- CARGA CSV ----------------
# nodos.csv  -> columnas: id,nombre,lat,lon
# aristas.csv-> columnas: origen,destino,tiempo_min,distancia_km,capacidad
nodos = pd.read_csv("nodos.csv")
aristas = pd.read_csv("aristas.csv")

# Limpieza mínima
nodos["id"] = nodos["id"].astype(str).str.strip()
nodos["nombre"] = nodos["nombre"].astype(str).str.strip()
aristas["origen"] = aristas["origen"].astype(str).str.strip()
aristas["destino"] = aristas["destino"].astype(str).str.strip()

# ---------------- EXTRA: LUGARES NUEVOS (edita lat/lon) ----------------
# ⚠️ Rellena lat/lon reales. Si quedan en None, ese lugar no se agrega todavía.
extra_nodos = pd.DataFrame([
    {"id":"F","nombre":"Parque Central","lat":None,"lon":None},
    {"id":"G","nombre":"Catedral","lat":None,"lon":None},
    {"id":"H","nombre":"Terminal de Buses","lat":None,"lon":None},
    {"id":"I","nombre":"Hospital Regional","lat":None,"lon":None},
    {"id":"J","nombre":"Cancha Los Angeles","lat":None,"lon":None},
    {"id":"K","nombre":"Cancha Sintética Golazo","lat":None,"lon":None},
    {"id":"L","nombre":"Aeropuerto Nacional","lat":None,"lon":None},
    {"id":"M","nombre":"Iglesia Candelero de Oro","lat":None,"lon":None},
    {"id":"N","nombre":"Centro de Salud","lat":None,"lon":None},
    {"id":"O","nombre":"Megapaca","lat":None,"lon":None},
    {"id":"P","nombre":"CANICA La Casa de los Niños","lat":None,"lon":None},
    {"id":"Q","nombre":"Aldea San Rafael Soche","lat":None,"lon":None},
    {"id":"R","nombre":"Pollo Campero","lat":None,"lon":None},
    {"id":"S","nombre":"INTECAP San Marcos","lat":None,"lon":None},
    {"id":"T","nombre":"Salón Quetzal","lat":None,"lon":None},
    {"id":"U","nombre":"SAT San Marcos","lat":None,"lon":None},
    {"id":"V","nombre":"Bazar Chino","lat":None,"lon":None},
])

# Agregar solo los que ya tengan coordenadas
extra_nodos = extra_nodos.dropna(subset=["lat", "lon"])
if not extra_nodos.empty:
    nodos = pd.concat([nodos, extra_nodos], ignore_index=True)
    nodos = nodos.drop_duplicates(subset="id", keep="first")

    # Conexiones ejemplo (ajusta tiempos/distancias cuando puedas)
    extra_aristas = pd.DataFrame([
        {"origen":"F","destino":"G","tiempo_min":2,"distancia_km":0.5,"capacidad":30},
        {"origen":"F","destino":"H","tiempo_min":4,"distancia_km":1.2,"capacidad":40},
        {"origen":"G","destino":"H","tiempo_min":3,"distancia_km":0.9,"capacidad":30},
        {"origen":"H","destino":"A","tiempo_min":5,"distancia_km":1.6,"capacidad":40},
        {"origen":"I","destino":"D","tiempo_min":4,"distancia_km":1.1,"capacidad":30},
        {"origen":"J","destino":"K","tiempo_min":3,"distancia_km":0.8,"capacidad":30},
        {"origen":"K","destino":"F","tiempo_min":4,"distancia_km":1.2,"capacidad":30},
        {"origen":"L","destino":"H","tiempo_min":8,"distancia_km":2.6,"capacidad":35},
        {"origen":"M","destino":"F","tiempo_min":4,"distancia_km":1.2,"capacidad":30},
        {"origen":"N","destino":"F","tiempo_min":4,"distancia_km":1.4,"capacidad":30},
        {"origen":"O","destino":"F","tiempo_min":5,"distancia_km":1.6,"capacidad":30},
        {"origen":"P","destino":"N","tiempo_min":6,"distancia_km":2.0,"capacidad":30},
        {"origen":"Q","destino":"F","tiempo_min":12,"distancia_km":4.0,"capacidad":30},
        {"origen":"R","destino":"F","tiempo_min":5,"distancia_km":1.7,"capacidad":30},
        {"origen":"S","destino":"U","tiempo_min":4,"distancia_km":1.2,"capacidad":30},
        {"origen":"T","destino":"F","tiempo_min":6,"distancia_km":2.0,"capacidad":30},
        {"origen":"U","destino":"F","tiempo_min":5,"distancia_km":1.6,"capacidad":30},
        {"origen":"V","destino":"F","tiempo_min":5,"distancia_km":1.6,"capacidad":30},
    ])

    ids_ok = set(nodos["id"])
    extra_aristas = extra_aristas[
        extra_aristas["origen"].isin(ids_ok) & extra_aristas["destino"].isin(ids_ok)
    ]
    if not extra_aristas.empty:
        aristas = pd.concat([aristas, extra_aristas], ignore_index=True)

# ---------------- HELPERS ----------------
id_por_nombre = {r["nombre"]: r["id"] for _, r in nodos.iterrows()}
nombre_por_id = {r["id"]: r["nombre"] for _, r in nodos.iterrows()}

def hex_to_rgb(h: str):
    """'#RRGGBB' -> [R,G,B]"""
    h = h.lstrip("#")
    return [int(h[i:i+2], 16) for i in (0, 2, 4)]

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
    destino_nombre = st.selectbox("Destino", sorted(nodos["nombre"]), index=min(1, len(nodos)-1))
    criterio = st.radio("Optimizar por", ["tiempo_min", "distancia_km"], index=0)

    # 3) Colores
    st.markdown("### Colores")
    col_nodes = st.color_picker("Nodos", "#FF007F")         # rosado fuerte
    col_edges = st.color_picker("Aristas", "#F2F2F2")       # blanco
    col_path  = st.color_picker("Ruta seleccionada", "#007AFF")  # azul

    calcular = st.button("Calcular ruta")

# Convertir colores a RGB
RGB_NODES = hex_to_rgb(col_nodes)
RGB_EDGES = hex_to_rgb(col_edges)
RGB_PATH  = hex_to_rgb(col_path)

# ---------------- GRAFO ----------------
G = nx.DiGraph() if dirigido else nx.Graph()

for _, r in nodos.iterrows():
    G.add_node(r["id"], nombre=r["nombre"], lat=float(r["lat"]), lon=float(r["lon"]))

for _, r in aristas.iterrows():
    G.add_edge(
        r["origen"], r["destino"],
        tiempo_min=float(r["tiempo_min"]),
        distancia_km=float(r["distancia_km"]),
        capacidad=float(r.get("capacidad", 0))
    )

center_lat, center_lon = nodos["lat"].mean(), nodos["lon"].mean()

def ruta_optima(o_id: str, d_id: str, peso: str):
    path = nx.shortest_path(G, source=o_id, target=d_id, weight=peso)
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        total += G[u][v][peso]
    return path, total

# ---------------- CAPAS DE MAPA ----------------
# Aristas como segmentos
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
    get_color=RGB_EDGES,          # color aristas
    pickable=True,
)

nodes_layer = pdk.Layer(
    "ScatterplotLayer",
    data=nodos.rename(columns={"lon": "lng"}),
    get_position="[lng, lat]",
    get_radius=65,
    radius_min_pixels=3,
    get_fill_color=RGB_NODES,     # color nodos (relleno)
    get_line_color=[30,30,30],
    line_width_min_pixels=1,
    pickable=True,
)

view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=13)

# ---------------- UI / RESULTADO ----------------
col1, col2 = st.columns([1, 2])

if calcular:
    o = id_por_nombre[origen_nombre]
    d = id_por_nombre[destino_nombre]

    try:
        path, total = ruta_optima(o, d, criterio)

        tramo_df = pd.DataFrame([{
            "id": n,
            "nombre": nombre_por_id[n],
            "lat": G.nodes[n]["lat"],
            "lon": G.nodes[n]["lon"]
        } for n in path])

        path_layer = pdk.Layer(
            "PathLayer",
            data=[{"path": tramo_df[["lon", "lat"]].values.tolist()}],
            get_path="path",
            get_width=6,
            width_scale=8,
            get_color=RGB_PATH,      # color ruta seleccionada
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

            # Descargar CSV con la ruta
            st.download_button(
                "📥 Descargar ruta (CSV)",
                data=tramo_df.to_csv(index=False).encode("utf-8"),
                file_name=f"ruta_{o}_{d}_{criterio}.csv",
                mime="text/csv"
            )

            st.dataframe(tramo_df, use_container_width=True)

        with col2:
            st.pydeck_chart(pdk.Deck(
                layers=[edges_layer, nodes_layer, path_layer],
                initial_view_state=view_state
            ), use_container_width=True)

    except nx.NetworkXNoPath:
        with col1:
            st.error("No hay camino entre esos nodos con el grafo actual.")
        with col2:
            st.pydeck_chart(pdk.Deck(
                layers=[edges_layer, nodes_layer],
                initial_view_state=view_state
            ), use_container_width=True)

else:
    st.info("Elige origen, destino y presiona **Calcular ruta**.")
    st.pydeck_chart(pdk.Deck(
        layers=[edges_layer, nodes_layer],
        initial_view_state=view_state
    ), use_container_width=True)
