import streamlit as st
import pandas as pd
import pydeck as pdk
import math

# ---------------- UI / CONFIG ----------------
st.set_page_config(page_title="Rutas San Marcos", layout="wide")
st.title("🚌 Calculador de paradas — San Marcos (modo sin grafo)")

# 1) Carga de nodos existentes (si no hay, creamos DataFrame vacío)
try:
    nodos = pd.read_csv("nodos.csv")  # columnas: id, nombre, lat, lon
except Exception:
    nodos = pd.DataFrame(columns=["id", "nombre", "lat", "lon"])

# 2) Lista de lugares (sin coordenadas de momento)
LUGARES_NUEVOS = [
    "Parque Central", "Catedral", "Terminal de Buses", "Hospital Regional",
    "Cancha Los Angeles", "Cancha Sintetica Golazo", "Aeropuerto Nacional",
    "Iglesia Candelero de Oro", "Centro de Salud", "Megapaca",
    "CANICA (Casa de los Niños)", "Aldea San Rafael Soche", "Pollo Campero",
    "INTECAP San Marcos", "Salón Quetzal", "SAT San Marcos", "Bazar Chino"
]

# 3) Normalización mínima
for col in ["id", "nombre"]:
    if col in nodos.columns:
        nodos[col] = nodos[col].astype(str).str.strip()
else:
    nodos = nodos.reindex(columns=["id", "nombre", "lat", "lon"])

# 4) Asegurar que todos los lugares existan en 'nodos' (aunque sea sin lat/lon)
def asegurar_lugares(df: pd.DataFrame, nombres: list) -> pd.DataFrame:
    existentes = set(df["nombre"].str.lower()) if "nombre" in df else set()
    rows = []
    next_id_num = 1
    usados = set(df["id"].astype(str)) if "id" in df else set()
    # generar ids L1, L2, ... que no choquen
    def nuevo_id():
        nonlocal next_id_num
        while f"L{next_id_num}" in usados:
            next_id_num += 1
        nid = f"L{next_id_num}"
        usados.add(nid)
        next_id_num += 1
        return nid

    for nm in nombres:
        if nm.lower() not in existentes:
            rows.append({"id": nuevo_id(), "nombre": nm, "lat": None, "lon": None})
    if rows:
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    return df

nodos = asegurar_lugares(nodos, LUGARES_NUEVOS)

# Guardar en session_state para poder actualizar sin tocar el CSV todavía
if "nodos_mem" not in st.session_state:
    st.session_state.nodos_mem = nodos.copy()

nodos = st.session_state.nodos_mem

# --- GRAFO EN MEMORIA (persistente durante la sesión del usuario) ---
# graph_points = todos los puntos que YA se usaron en rutas calculadas
# columnas: nombre, lat, lon
if "graph_points" not in st.session_state:
    st.session_state.graph_points = pd.DataFrame(columns=["nombre", "lat", "lon"])

# graph_edges = todas las aristas (cada una es {"path": [[lon_o, lat_o],[lon_d, lat_d]]})
if "graph_edges" not in st.session_state:
    st.session_state.graph_edges = []

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("Parámetros")

    origen_nombre = st.selectbox("Origen", sorted(nodos["nombre"]))
    destino_nombre = st.selectbox("Destino", sorted(nodos["nombre"]), index=1)

    st.markdown("### Colores")
    col_nodes = st.color_picker("Nodos", "#FF007F")
    col_path  = st.color_picker("Ruta seleccionada", "#007AFF")

    # >>> IMPORTANTE: ya NO hay editor_de_coords aquí. Lo quité. <<<

    calcular = st.button("Calcular ruta")

# Colores (RGB)
def hex_to_rgb(h: str):
    h = h.lstrip("#")
    return [int(h[i:i+2], 16) for i in (0, 2, 4)]

RGB_NODES = hex_to_rgb(col_nodes)
RGB_PATH  = hex_to_rgb(col_path)

def haversine_km(a_lat, a_lon, b_lat, b_lon):
    R = 6371.0
    lat1, lon1 = math.radians(a_lat), math.radians(a_lon)
    lat2, lon2 = math.radians(b_lat), math.radians(b_lon)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(h))

# ---------------- LÓGICA AL CALCULAR RUTA ----------------
# Vamos a guardar info de la ÚLTIMA ruta calculada para mostrar resumen.
last_tramo_df = None
last_dist_km = None
last_t_min = None
last_origen = None
last_destino = None
update_ok = False

if calcular:
    fila_o = nodos.loc[nodos["nombre"] == origen_nombre].iloc[0]
    fila_d = nodos.loc[nodos["nombre"] == destino_nombre].iloc[0]

    # Verificamos que ambos tengan coordenadas en nodos.csv
    if (
        pd.isna(fila_o["lat"]) or pd.isna(fila_o["lon"]) or
        pd.isna(fila_d["lat"]) or pd.isna(fila_d["lon"])
    ):
        st.error("Ese origen o destino no tiene coordenadas en nodos.csv. Agregalas al CSV y vuelve a intentar.")
    else:
        # Distancia aprox + tiempo aprox
        last_dist_km = haversine_km(fila_o["lat"], fila_o["lon"], fila_d["lat"], fila_d["lon"])
        vel_kmh = 30.0  # supuesta
        last_t_min = (last_dist_km / vel_kmh) * 60.0

        last_origen = origen_nombre
        last_destino = destino_nombre

        # DataFrame de la ruta actual solo (para descargar/tabla)
        last_tramo_df = pd.DataFrame([
            {"nombre": origen_nombre,  "lat": fila_o["lat"], "lon": fila_o["lon"]},
            {"nombre": destino_nombre, "lat": fila_d["lat"], "lon": fila_d["lon"]},
        ])

        # -------- ACTUALIZAR EL GRAFO GLOBAL EN SESSION_STATE --------
        # Agregar origen y destino a graph_points (sin duplicar por nombre)
        nuevos_puntos = pd.DataFrame([
            {"nombre": origen_nombre,  "lat": fila_o["lat"], "lon": fila_o["lon"]},
            {"nombre": destino_nombre, "lat": fila_d["lat"], "lon": fila_d["lon"]},
        ])

        st.session_state.graph_points = (
            pd.concat([st.session_state.graph_points, nuevos_puntos], ignore_index=True)
            .drop_duplicates(subset=["nombre"], keep="last")
        )

        # Agregar la arista origen->destino a graph_edges
        st.session_state.graph_edges.append({
            "path": [
                [fila_o["lon"], fila_o["lat"]],
                [fila_d["lon"], fila_d["lat"]],
            ]
        })

        update_ok = True  # sí se pudo calcular una ruta válida

# ---------------- MAPA (USANDO TODO LO QUE YA SE HA IDO GUARDANDO) ----------------
col1, col2 = st.columns([1, 2])

# Vamos a dibujar:
# - TODOS los nodos que ya se usaron en rutas anteriores (graph_points)
# - TODAS las aristas acumuladas (graph_edges)

if len(st.session_state.graph_points) > 0:
    puntos_plot = st.session_state.graph_points.rename(columns={"lon": "lng"}).copy()

    # Capa de nodos acumulados
    nodes_layer = pdk.Layer(
        "ScatterplotLayer",
        data=puntos_plot,
        get_position="[lng, lat]",
        get_radius=90,
        radius_min_pixels=4,
        get_fill_color=RGB_NODES,
        get_line_color=[0, 0, 0],
        line_width_min_pixels=1,
        pickable=True,
    )

    # Capa de aristas acumuladas
    edges_layer = pdk.Layer(
        "PathLayer",
        data=st.session_state.graph_edges,
        get_path="path",
        get_width=6,
        width_scale=8,
        get_color=RGB_PATH,
        pickable=False,
    )

    # centramos vista en el promedio de todos los puntos visitados hasta ahora
    center_lat = float(puntos_plot["lat"].mean())
    center_lon = float(puntos_plot["lng"].mean())
    view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=13)

    with col2:
        st.pydeck_chart(
            pdk.Deck(
                layers=[edges_layer, nodes_layer],
                initial_view_state=view_state,
                tooltip={"text": "{nombre}\nLat: {lat}\nLon: {lng}"}
            ),
            use_container_width=True
        )

else:
    # Si todavía no se calculó ninguna ruta válida, mostramos mapa vacío
    DEFAULT_LAT = 14.965
    DEFAULT_LON = -91.79
    view_state = pdk.ViewState(latitude=DEFAULT_LAT, longitude=DEFAULT_LON, zoom=13)

    with col2:
        st.pydeck_chart(
            pdk.Deck(
                layers=[],  # sin nodos, sin aristas todavía
                initial_view_state=view_state
            ),
            use_container_width=True
        )

# ---------------- PANEL IZQUIERDO (RESUMEN ÚLTIMA RUTA) ----------------
with col1:
    if update_ok and last_tramo_df is not None:
        st.subheader("Resumen")
        st.markdown(f"**Origen:** {last_origen}")
        st.markdown(f"**Destino:** {last_destino}")
        st.markdown(f"**Distancia directa aprox.:** {last_dist_km:.2f} km")
        st.markdown(f"**Tiempo aprox. (30 km/h):** {last_t_min:.1f} min")

        st.download_button(
            "📥 Descargar puntos (CSV)",
            data=last_tramo_df.to_csv(index=False).encode("utf-8"),
            file_name="puntos_directo.csv",
            mime="text/csv"
        )
        st.dataframe(last_tramo_df, use_container_width=True)
    else:
        st.info(
            "1. Elegí Origen y Destino en la barra lateral.\n"
            "2. Presioná 'Calcular ruta'.\n\n"
            "Cada vez que calcules una ruta:\n"
            "- Se agrega automáticamente el nodo de cada lugar al mapa.\n"
            "- Se dibuja la arista (línea) entre ellos.\n"
            "El mapa va guardando todo lo que ya hiciste en esta sesión."
        )

