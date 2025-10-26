import streamlit as st
import pandas as pd
import pydeck as pdk
import math

# ---------------- CONFIG INICIAL ----------------
st.set_page_config(page_title="Rutas San Marcos", layout="wide")

# Estilos básicos (panel más bonito + tarjetita de leyenda)
st.markdown("""
<style>
.summary-header {
    font-size: 1.4rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
    color: white;
}
.summary-label {
    font-weight: 600;
    color: white;
}
.summary-text {
    color: white;
    margin-bottom: 0.5rem;
    line-height: 1.4;
}
.legend-card {
    background-color: rgba(100,150,200,0.15);
    border: 1px solid rgba(100,150,200,0.4);
    color: white;
    padding: 0.75rem 1rem;
    border-radius: 0.5rem;
    font-size: 0.9rem;
    line-height: 1.4;
    margin-top: 0.75rem;
}
.small-label {
    font-size: 0.9rem;
    line-height: 1.4;
    color: white;
}
.bold-line {
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

st.title("🚌 Calculador de paradas — San Marcos")

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

# aristas acumuladas fijas (todas las rutas históricas)
if "graph_edges" not in st.session_state:
    st.session_state.graph_edges = []

# arista resaltada (última ruta calculada)
if "highlight_edge" not in st.session_state:
    st.session_state.highlight_edge = []

# Métricas mostradas en el panel izquierdo
if "metrics" not in st.session_state:
    st.session_state.metrics = {
        "modo": "Bicicleta (15 km/h)",
        "ruta_txt": "—",
        "dist_km": None,
        "t_min": None,
        "nota": "Red aproximada. Coordenadas generadas si faltan.",
        "last_tramo_df_csv": None,
        "last_tramo_df_table": None,
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

    # Colores fijos tipo screenshot:
    # nodos: amarillo, grafo base: celeste, highlight: rojo
    col_nodes = "#FFD400"
    col_path_all = "#00CFFF"
    col_path_highlight = "#FF0033"

    calcular = st.button("Calcular ruta")

RGB_NODES        = hex_to_rgb(col_nodes)          # amarillo nodos
RGB_PATH_ALL     = hex_to_rgb(col_path_all)       # azul/celeste conexiones base
RGB_PATH_CURRENT = hex_to_rgb(col_path_highlight) # rojo ruta actual

# ---------------- LÓGICA CUANDO HACES CLICK EN "Calcular ruta" ----------------
if calcular:
    # 1. coords del origen/destino (se generan si faltan)
    o_lat, o_lon = asegurar_coords_en_mem(origen_nombre)
    d_lat, d_lon = asegurar_coords_en_mem(destino_nombre)

    # 2. distancia recta + tiempo
    dist_km_val = haversine_km(o_lat, o_lon, d_lat, d_lon)
    vel_kmh = 15.0  # igual que "Bicicleta (15 km/h)" en tu panel
    t_min_val = (dist_km_val / vel_kmh) * 60.0

    # 3. dataframe de este tramo (para tabla y CSV)
    tramo_df = pd.DataFrame([
        {"nombre": origen_nombre,  "lat": o_lat, "lon": o_lon},
        {"nombre": destino_nombre, "lat": d_lat, "lon": d_lon},
    ])

    # 4. actualizar métricas del panel
    st.session_state.metrics["modo"] = "Bicicleta (15 km/h)"
    st.session_state.metrics["ruta_txt"] = f"{origen_nombre} → {destino_nombre}"
    st.session_state.metrics["dist_km"] = f"{dist_km_val:.3f} km"
    st.session_state.metrics["t_min"]   = f"{t_min_val:.1f} min"
    st.session_state.metrics["nota"]    = "Red aproximada. Coordenadas generadas si faltan."
    st.session_state.metrics["last_tramo_df_csv"]   = tramo_df.to_csv(index=False).encode("utf-8")
    st.session_state.metrics["last_tramo_df_table"] = tramo_df

    # 5. meter nodos de esta ruta a la red fija
    nuevos_puntos = pd.DataFrame([
        {"nombre": origen_nombre,  "lat": o_lat, "lon": o_lon},
        {"nombre": destino_nombre, "lat": d_lat, "lon": d_lon},
    ])
    st.session_state.graph_points = (
        pd.concat([st.session_state.graph_points, nuevos_puntos], ignore_index=True)
        .drop_duplicates(subset=["nombre"], keep="last")
    )

    # 6. agregar esta arista a la red fija (todas las conexiones, celeste)
    st.session_state.graph_edges.append({
        "path": [[o_lon, o_lat], [d_lon, d_lat]],
        "origen": origen_nombre,
        "destino": destino_nombre,
        "dist_km": f"{dist_km_val:.2f}",
        "t_min": f"{t_min_val:.1f}"
    })

    # 7. guardar también como arista resaltada actual (rojo)
    st.session_state.highlight_edge = [{
        "path": [[o_lon, o_lat], [d_lon, d_lat]],
        "origen": origen_nombre,
        "destino": destino_nombre,
        "dist_km": f"{dist_km_val:.2f}",
        "t_min": f"{t_min_val:.1f}"
    }]

# ---------------- DIBUJO DEL MAPA ----------------
col_info, col_map = st.columns([1, 2])

if len(st.session_state.graph_points) > 0:
    puntos_plot = st.session_state.graph_points.rename(columns={"lon": "lng"}).copy()

    # Capa de NODOS (amarillos con borde negro, círculos grandes)
    nodes_layer = pdk.Layer(
        "ScatterplotLayer",
        data=puntos_plot,
        get_position="[lng, lat]",
        get_radius=120,
        radius_min_pixels=6,
        get_fill_color=RGB_NODES,
        get_line_color=[0, 0, 0],
        line_width_min_pixels=2,
        pickable=True,
    )

    # Capa de TODAS las aristas históricas (celeste, más delgada)
    edges_layer_all = pdk.Layer(
        "PathLayer",
        data=st.session_state.graph_edges,
        get_path="path",
        get_width=4,
        width_scale=6,
        get_color=RGB_PATH_ALL,
        pickable=False,
    )

    # Capa de la RUTA ACTUAL (roja, más gruesa, por encima)
    layers_to_draw = [edges_layer_all]
    if len(st.session_state.highlight_edge) > 0:
        edges_layer_highlight = pdk.Layer(
            "PathLayer",
            data=st.session_state.highlight_edge,
            get_path="path",
            get_width=8,
            width_scale=10,
            get_color=RGB_PATH_CURRENT,
            pickable=False,
        )
        layers_to_draw.append(edges_layer_highlight)

    # Nodos al final para que queden encima de las líneas
    layers_to_draw.append(nodes_layer)

    # Centro del mapa = promedio de todos los puntos
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
                # mapa oscuro como en tu captura
                map_style="mapbox://styles/mapbox/dark-v10",
                layers=layers_to_draw,
                initial_view_state=view_state,
                tooltip={"text": "{nombre}\nLat: {lat}\nLon: {lng}"}
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
                map_style="mapbox://styles/mapbox/dark-v10",
                layers=[],
                initial_view_state=view_state
            ),
            use_container_width=True
        )

# ---------------- PANEL IZQUIERDO (ORDENADO + LEYENDA) ----------------
m = st.session_state.metrics

with col_info:
    st.markdown('<div class="summary-header">Resumen de la última selección</div>', unsafe_allow_html=True)

    st.markdown(f'<div class="summary-text"><span class="summary-label">Modo:</span> {m["modo"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="summary-text"><span class="summary-label">Ruta:</span> {m["ruta_txt"]}</div>', unsafe_allow_html=True)

    # Sólo mostramos distancia / tiempo si ya hay ruta calculada
    if m["dist_km"] and m["t_min"]:
        st.markdown(f'<div class="summary-text"><span class="summary-label">Distancia:</span> {m["dist_km"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="summary-text"><span class="summary-label">Tiempo estimado:</span> {m["t_min"]}</div>', unsafe_allow_html=True)

    st.markdown(f'<div class="summary-text"><span class="summary-label">Notas:</span> {m["nota"]}</div>', unsafe_allow_html=True)

    # Leyenda bonita tipo tarjetita
    st.markdown(
        f"""
        <div class="legend-card">
        <div class="small-label">
        <span class="bold-line" style="color:{col_path_all};">Azul</span> = grafo base (todas las conexiones)<br/>
        <span class="bold-line" style="color:{col_path_highlight};">Rojo</span> = arista seleccionada ahora<br/>
        <span class="bold-line" style="color:{col_nodes};">Amarillo</span> = nodos / lugares
        </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Si ya hay una ruta calculada, mostramos tabla de puntos y botón CSV
    if m["last_tramo_df_table"] is not None:
        st.download_button(
            "📥 Descargar puntos (CSV)",
            data=m["last_tramo_df_csv"],
            file_name="puntos_directo.csv",
            mime="text/csv"
        )

        st.dataframe(m["last_tramo_df_table"], use_container_width=True)
