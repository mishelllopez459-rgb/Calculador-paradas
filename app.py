import streamlit as st
import pandas as pd
import pydeck as pdk
import math

# ---------------- CONFIG INICIAL ----------------
st.set_page_config(page_title="Rutas San Marcos", layout="wide")
st.title("🚌 Calculador de paradas — San Marcos (grafo automático)")

# Centro base para generar coordenadas cuando falten
BASE_LAT = 14.965
BASE_LON = -91.790

# ---------------- CARGA DE NODOS ----------------
try:
    nodos = pd.read_csv("nodos.csv")  # columnas: id, nombre, lat, lon
except Exception:
    nodos = pd.DataFrame(columns=["id", "nombre", "lat", "lon"])

LUGARES_NUEVOS = [
    "Parque Central", "Catedral", "Terminal de Buses", "Hospital Regional",
    "Cancha Los Angeles", "Cancha Sintetica Golazo", "Aeropuerto Nacional",
    "Iglesia Candelero de Oro", "Centro de Salud", "Megapaca",
    "CANICA (Casa de los Niños)", "Aldea San Rafael Soche", "Pollo Campero",
    "INTECAP San Marcos", "Salón Quetzal", "SAT San Marcos", "Bazar Chino"
]

# Normalizar columnas
for col in ["id", "nombre"]:
    if col in nodos.columns:
        nodos[col] = nodos[col].astype(str).str.strip()
else:
    nodos = nodos.reindex(columns=["id", "nombre", "lat", "lon"])

for col in ["id", "nombre", "lat", "lon"]:
    if col not in nodos.columns:
        nodos[col] = None

# Asegurar que todos los lugares existan en el df
def asegurar_lugares(df: pd.DataFrame, nombres: list) -> pd.DataFrame:
    existentes = set(df["nombre"].astype(str).str.lower()) if "nombre" in df else set()
    rows = []
    next_id_num = 1
    usados = set(df["id"].astype(str)) if "id" in df else set()

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

# Guardar en memoria de sesión (nodos base editables en runtime)
if "nodos_mem" not in st.session_state:
    st.session_state.nodos_mem = nodos.copy()
nodos = st.session_state.nodos_mem

# ---------------- ESTADO DEL GRAFO EN SESIÓN ----------------
# nodos acumulados (toda la red fija que ya se construyó)
if "graph_points" not in st.session_state:
    st.session_state.graph_points = pd.DataFrame(columns=["nombre", "lat", "lon"])

# aristas acumuladas fijas (todas las rutas históricas, azul)
if "graph_edges" not in st.session_state:
    st.session_state.graph_edges = []

# arista resaltada (última ruta calculada, rojo)
if "highlight_edge" not in st.session_state:
    st.session_state.highlight_edge = []

# Métricas mostradas (modo, ruta, distancia, tiempo, notas)
if "metrics" not in st.session_state:
    st.session_state.metrics = {
        "modo": "Bicicleta (15 km/h)",
        "ruta_txt": "—",
        "dist_km": None,
        "t_min": None,
        "nota": "La red es un aproximado visual. Las posiciones son generadas si faltan coordenadas."
    }

# ---------------- FUNCIONES AUX ----------------
def hex_to_rgb(h: str):
    h = h.lstrip("#")
    return [int(h[i:i+2], 16) for i in (0, 2, 4)]

def haversine_km(a_lat, a_lon, b_lat, b_lon):
    R = 6371.0
    lat1, lon1 = math.radians(a_lat), math.radians(a_lon)
    lat2, lon2 = math.radians(b_lat), math.radians(b_lon)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(h))

def pseudo_offset(nombre: str):
    # coords determinísticas cerca del centro BASE_LAT/LON
    s_val = 0
    for i, ch in enumerate(nombre):
        s_val += (i + 1) * ord(ch)
    lat_off = ((s_val % 17) - 8) * 0.0005
    lon_off = (((s_val // 17) % 17) - 8) * 0.0005
    return BASE_LAT + lat_off, BASE_LON + lon_off

def asegurar_coords_en_mem(nombre_lugar: str):
    """
    Devuelve (lat, lon) para el lugar.
    Si no existen, las genera, las guarda en nodos_mem y luego las usa.
    """
    df = st.session_state.nodos_mem
    idx = df.index[df["nombre"] == nombre_lugar]

    if len(idx) == 0:
        lat_new, lon_new = pseudo_offset(nombre_lugar)
        nuevo = {
            "id": f"AUTO_{nombre_lugar}",
            "nombre": nombre_lugar,
            "lat": lat_new,
            "lon": lon_new
        }
        st.session_state.nodos_mem = pd.concat(
            [st.session_state.nodos_mem, pd.DataFrame([nuevo])],
            ignore_index=True
        )
        return lat_new, lon_new
    else:
        i = idx[0]
        lat_val = st.session_state.nodos_mem.at[i, "lat"]
        lon_val = st.session_state.nodos_mem.at[i, "lon"]

        if pd.isna(lat_val) or pd.isna(lon_val):
            lat_new, lon_new = pseudo_offset(nombre_lugar)
            st.session_state.nodos_mem.at[i, "lat"] = lat_new
            st.session_state.nodos_mem.at[i, "lon"] = lon_new
            return lat_new, lon_new
        else:
            return float(lat_val), float(lon_val)

# ---------------- SIDEBAR (CONTROLES) ----------------
with st.sidebar:
    st.header("Parámetros")

    origen_nombre  = st.selectbox(
        "Origen",
        sorted(nodos["nombre"].astype(str)),
        key="sel_origen"
    )
    destino_nombre = st.selectbox(
        "Destino",
        sorted(nodos["nombre"].astype(str)),
        index=1,
        key="sel_destino"
    )

    st.markdown("### Colores de visualización")
    # default como en tu screenshot:
    # nodos = amarillo, red fija = azul, actual = rojo
    col_nodes = st.color_picker("Nodos fijos", "#FFD400")           # amarillo
    col_path_all = st.color_picker("Rutas fijas", "#007BFF")        # azul
    col_path_highlight = st.color_picker("Ruta actual (resaltada)", "#FF0000")  # rojo

    calcular = st.button("Calcular ruta")

RGB_NODES        = hex_to_rgb(col_nodes)
RGB_PATH_ALL     = hex_to_rgb(col_path_all)
RGB_PATH_CURRENT = hex_to_rgb(col_path_highlight)

# ---------------- LÓGICA CUANDO HACES CLICK EN "Calcular ruta" ----------------
if calcular:
    # 1. coords del origen/destino (se generan si faltan)
    o_lat, o_lon = asegurar_coords_en_mem(origen_nombre)
    d_lat, d_lon = asegurar_coords_en_mem(destino_nombre)

    # 2. distancia recta + tiempo
    dist_km_val = haversine_km(o_lat, o_lon, d_lat, d_lon)
    vel_kmh = 15.0  # bicicleta en tu screenshot dice ~15 km/h
    t_min_val = (dist_km_val / vel_kmh) * 60.0

    # 3. actualizar el texto de la métrica tipo "Ruta: C → A → D"
    # en este caso tenemos solo 2 puntos, "Origen → Destino"
    st.session_state.metrics = {
        "modo": "Bicicleta (15 km/h)",
        "ruta_txt": f"{origen_nombre} → {destino_nombre}",
        "dist_km": f"{dist_km_val:.3f} km",
        "t_min": f"{t_min_val:.1f} min",
        "nota": "La red es un aproximado; algunas posiciones fueron generadas."
    }

    # 4. dataframe de este tramo (para tabla/descarga)
    last_tramo_df = pd.DataFrame([
        {"nombre": origen_nombre,  "lat": o_lat, "lon": o_lon},
        {"nombre": destino_nombre, "lat": d_lat, "lon": d_lon},
    ])

    # 5. meter nodos al grafo permanente
    nuevos_puntos = pd.DataFrame([
        {"nombre": origen_nombre,  "lat": o_lat, "lon": o_lon},
        {"nombre": destino_nombre, "lat": d_lat, "lon": d_lon},
    ])
    st.session_state.graph_points = (
        pd.concat([st.session_state.graph_points, nuevos_puntos], ignore_index=True)
        .drop_duplicates(subset=["nombre"], keep="last")
    )

    # 6. agregar arista a la red fija (azul)
    st.session_state.graph_edges.append({
        "path": [[o_lon, o_lat], [d_lon, d_lat]],
        "origen": origen_nombre,
        "destino": destino_nombre,
        "dist_km": f"{dist_km_val:.2f}",
        "t_min": f"{t_min_val:.1f}"
    })

    # 7. guardar la arista actual como highlight (rojo grueso)
    st.session_state.highlight_edge = [{
        "path": [[o_lon, o_lat], [d_lon, d_lat]],
        "origen": origen_nombre,
        "destino": destino_nombre,
        "dist_km": f"{dist_km_val:.2f}",
        "t_min": f"{t_min_val:.1f}"
    }]

    # guardo también el tramo para mostrar tabla/CSV después
    st.session_state.metrics["last_tramo_df_csv"] = last_tramo_df.to_csv(index=False).encode("utf-8")
    st.session_state.metrics["last_tramo_df_table"] = last_tramo_df

# ---------------- DIBUJO DEL MAPA ----------------
col_info, col_map = st.columns([1, 2])

if len(st.session_state.graph_points) > 0:
    puntos_plot = st.session_state.graph_points.rename(columns={"lon": "lng"}).copy()

    # capa de NODOS (amarillos con aro negro, como tus circulitos)
    nodes_layer = pdk.Layer(
        "ScatterplotLayer",
        data=puntos_plot,
        get_position="[lng, lat]",
        get_radius=90,
        radius_min_pixels=5,
        get_fill_color=RGB_NODES,
        get_line_color=[0, 0, 0],   # borde negro
        line_width_min_pixels=2,
        pickable=True,
    )

    # capa de TODAS las aristas históricas (azules finas)
    edges_layer_all = pdk.Layer(
        "PathLayer",
        data=st.session_state.graph_edges,
        get_path="path",
        get_width=4,
        width_scale=6,
        get_color=RGB_PATH_ALL,
        pickable=False,
    )

    # capa de la RUTA ACTUAL (roja gruesa encima)
    layers_to_draw = [edges_layer_all]
    if len(st.session_state.highlight_edge) > 0:
        edges_layer_highlight = pdk.Layer(
            "PathLayer",
            data=st.session_state.highlight_edge,
            get_path="path",
            get_width=8,          # más gruesa
            width_scale=10,
            get_color=RGB_PATH_CURRENT,
            pickable=False,
        )
        layers_to_draw.append(edges_layer_highlight)

    layers_to_draw.append(nodes_layer)

    # centro del mapa = promedio de todos los puntos
    center_lat = float(puntos_plot["lat"].mean())
    center_lon = float(puntos_plot["lng"].mean())

    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=14,
        pitch=0
    )

    with col_map:
        st.pydeck_chart(
            pdk.Deck(
                map_style="mapbox://styles/mapbox/light-v10",  # mapa claro tipo calle
                layers=layers_to_draw,
                initial_view_state=view_state,
                tooltip={
                    "text": "{nombre}\nLat: {lat}\nLon: {lng}"
                }
            ),
            use_container_width=True
        )

else:
    # mapa vacío inicial
    view_state = pdk.ViewState(
        latitude=BASE_LAT,
        longitude=BASE_LON,
        zoom=14,
        pitch=0
    )

    with col_map:
        st.pydeck_chart(
            pdk.Deck(
                map_style="mapbox://styles/mapbox/light-v10",
                layers=[],
                initial_view_state=view_state
            ),
            use_container_width=True
        )

# ---------------- PANEL IZQUIERDO (INFO ESTILO PANELOTE) ----------------
with col_info:
    st.subheader("Resumen de la última ruta")

    st.markdown(f"**Modo:** {st.session_state.metrics['modo']}")
    st.markdown(f"**Ruta:** {st.session_state.metrics['ruta_txt']}")
    if st.session_state.metrics["dist_km"] is not None:
        st.markdown(f"**Distancia:** {st.session_state.metrics['dist_km']}")
    if st.session_state.metrics["t_min"] is not None:
        st.markdown(f"**Tiempo estimado:** {st.session_state.metrics['t_min']}")
    st.markdown(f"**Notas:** {st.session_state.metrics['nota']}")

    # si ya hay tabla/CSV guardada, la mostramos
    if "last_tramo_df_table" in st.session_state.metrics:
        st.download_button(
            "📥 Descargar puntos (CSV)",
            data=st.session_state.metrics["last_tramo_df_csv"],
            file_name="puntos_directo.csv",
            mime="text/csv"
        )

        st.dataframe(
            st.session_state.metrics["last_tramo_df_table"],
            use_container_width=True
        )
    else:
        st.info(
            "El mapa muestra:\n"
            "• Azul = toda la red que ya marcaste\n"
            "• Rojo = la última ruta que calculaste\n"
            "• Amarillo = paradas/nodos"
        )



