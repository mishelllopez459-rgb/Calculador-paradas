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

# Guardar en memoria de sesión
if "nodos_mem" not in st.session_state:
    st.session_state.nodos_mem = nodos.copy()

nodos = st.session_state.nodos_mem

# ---------------- ESTADO DEL GRAFO EN SESIÓN ----------------
# puntos ya visitados (para pintarlos siempre)
if "graph_points" not in st.session_state:
    st.session_state.graph_points = pd.DataFrame(columns=["nombre", "lat", "lon"])

# aristas ya creadas (guardamos también nombres para grafo lógico)
if "graph_edges" not in st.session_state:
    # cada item: { "path":[[lon1,lat1],[lon2,lat2]], "a":nombre_origen, "b":nombre_destino }
    st.session_state.graph_edges = []

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
    """
    Genera un offset pequeño y estable en base al texto del nombre.
    Así podemos inventar coords si no existen en el CSV.
    """
    s_val = 0
    for i, ch in enumerate(nombre):
        s_val += (i + 1) * ord(ch)

    # off pequeño (±0.004 aprox) para que no se encimen todos
    lat_off = ((s_val % 17) - 8) * 0.0005
    lon_off = (((s_val // 17) % 17) - 8) * 0.0005

    return BASE_LAT + lat_off, BASE_LON + lon_off

def asegurar_coords_en_mem(nombre_lugar: str):
    """
    Devuelve (lat, lon) para ese lugar.
    - Si ya tiene lat/lon en st.session_state.nodos_mem => usamos eso.
    - Si NO tiene, le generamos coords pseudo y ACTUALIZAMOS st.session_state.nodos_mem.
    """
    df = st.session_state.nodos_mem
    idx = df.index[df["nombre"] == nombre_lugar]
    if len(idx) == 0:
        # no existe? lo creamos con coords pseudo
        lat_new, lon_new = pseudo_offset(nombre_lugar)
        nuevo = {
            "id": f"AUTO_{nombre_lugar}",
            "nombre": nombre_lugar,
            "lat": lat_new,
            "lon": lon_new,
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

        # si faltan coords, las genero y guardo
        if pd.isna(lat_val) or pd.isna(lon_val):
            lat_new, lon_new = pseudo_offset(nombre_lugar)
            st.session_state.nodos_mem.at[i, "lat"] = lat_new
            st.session_state.nodos_mem.at[i, "lon"] = lon_new
            return lat_new, lon_new
        else:
            return float(lat_val), float(lon_val)

def build_graph_graphviz(rgb_nodes, rgb_edges):
    """
    Construye el grafo lógico usando Graphviz (que Streamlit puede renderizar sin librerías externas).
    Cada nodo = lugar. Cada arista = conexión calculada.
    """

    # nodos únicos
    pts = st.session_state.graph_points[["nombre"]].drop_duplicates()

    if pts.empty:
        return None

    # Armamos el código DOT de Graphviz
    # Usamos 'graph' (no 'digraph') porque las conexiones son no dirigidas.
    # También metemos estilos básicos con los colores elegidos.

    # color de nodos y aristas en formato #RRGGBB
    nodes_hex = "#{:02X}{:02X}{:02X}".format(*rgb_nodes)
    edges_hex = "#{:02X}{:02X}{:02X}".format(*rgb_edges)

    dot_lines = []
    dot_lines.append('graph G {')
    dot_lines.append('  layout=dot;')         # que intente ordenarlos
    dot_lines.append('  rankdir=LR;')         # izquierda->derecha (más tipo ruta)
    dot_lines.append('  splines=true;')
    dot_lines.append('  node [shape=circle, style=filled, color="#000000", fontname="Helvetica", fontsize=10, fillcolor="' + nodes_hex + '"];')
    dot_lines.append('  edge [color="' + edges_hex + '", penwidth=2];')

    # declarar nodos
    for _, row in pts.iterrows():
        nombre = row["nombre"].replace('"', '\\"')
        dot_lines.append(f'  "{nombre}";')

    # declarar aristas
    # usamos pares (a -- b)
    # evitamos duplicados con un set ordenado
    edges_set = set()
    for e in st.session_state.graph_edges:
        a = e.get("a")
        b = e.get("b")
        if not a or not b:
            continue
        key = tuple(sorted([a, b]))
        if key not in edges_set:
            edges_set.add(key)

    for (a, b) in edges_set:
        a_esc = a.replace('"', '\\"')
        b_esc = b.replace('"', '\\"')
        dot_lines.append(f'  "{a_esc}" -- "{b_esc}";')

    dot_lines.append('}')

    dot_src = "\n".join(dot_lines)
    return dot_src

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("Parámetros")

    origen_nombre = st.selectbox("Origen", sorted(nodos["nombre"].astype(str)))
    destino_nombre = st.selectbox("Destino", sorted(nodos["nombre"].astype(str)), index=1)

    st.markdown("### Colores")
    col_nodes = st.color_picker("Nodos", "#FF007F")
    col_path  = st.color_picker("Ruta seleccionada", "#007AFF")

    calcular = st.button("Calcular ruta")

RGB_NODES = hex_to_rgb(col_nodes)
RGB_PATH  = hex_to_rgb(col_path)

# ---------------- LÓGICA DE RUTA ----------------
last_tramo_df = None
last_dist_km = None
last_t_min = None
last_origen = None
last_destino = None
update_ok = False

if calcular:
    # agarrar / generar coords del origen
    o_lat, o_lon = asegurar_coords_en_mem(origen_nombre)
    # agarrar / generar coords del destino
    d_lat, d_lon = asegurar_coords_en_mem(destino_nombre)

    # distancia y tiempo aprox
    last_dist_km = haversine_km(o_lat, o_lon, d_lat, d_lon)
    vel_kmh = 30.0  # velocidad supuesta
    last_t_min = (last_dist_km / vel_kmh) * 60.0

    last_origen = origen_nombre
    last_destino = destino_nombre

    # tramo actual (tabla resumen)
    last_tramo_df = pd.DataFrame([
        {"nombre": origen_nombre,  "lat": o_lat, "lon": o_lon},
        {"nombre": destino_nombre, "lat": d_lat, "lon": d_lon},
    ])

    # agregar nodos al grafo global (sin duplicar)
    nuevos_puntos = pd.DataFrame([
        {"nombre": origen_nombre,  "lat": o_lat, "lon": o_lon},
        {"nombre": destino_nombre, "lat": d_lat, "lon": d_lon},
    ])
    st.session_state.graph_points = (
        pd.concat([st.session_state.graph_points, nuevos_puntos], ignore_index=True)
        .drop_duplicates(subset=["nombre"], keep="last")
    )

    # agregar arista al grafo global
    st.session_state.graph_edges.append({
        "path": [
            [o_lon, o_lat],
            [d_lon, d_lat],
        ],
        "a": origen_nombre,
        "b": destino_nombre,
    })

    update_ok = True

# ---------------- UI PRINCIPAL ----------------
# Tabs: Mapa geográfico vs Grafo abstracto
tab_mapa, tab_grafo = st.tabs(["🗺️ Mapa geográfico", "🔗 Grafo de conexiones"])

# ---- TAB MAPA ----
with tab_mapa:
    col1, col2 = st.columns([1, 2])

    if len(st.session_state.graph_points) > 0:
        puntos_plot = st.session_state.graph_points.rename(columns={"lon": "lng"}).copy()

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

        edges_layer = pdk.Layer(
            "PathLayer",
            data=st.session_state.graph_edges,
            get_path="path",
            get_width=6,
            width_scale=8,
            get_color=RGB_PATH,
            pickable=False,
        )

        center_lat = float(puntos_plot["lat"].mean())
        center_lon = float(puntos_plot["lng"].mean())

        view_state = pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=14,
            pitch=0
        )

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
        # mapa vacío antes de cualquier ruta
        view_state = pdk.ViewState(
            latitude=BASE_LAT,
            longitude=BASE_LON,
            zoom=14,
            pitch=0
        )

        with col2:
            st.pydeck_chart(
                pdk.Deck(
                    layers=[],
                    initial_view_state=view_state,
                ),
                use_container_width=True
            )

    # ---------------- PANEL IZQUIERDO (INFO DE TRAMO) ----------------
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
                "1. Elegí Origen y Destino en la izquierda.\n"
                "2. Dale 'Calcular ruta'.\n\n"
                "Si un lugar no tiene coordenadas, se le generan solas.\n"
                "Cada vez que calculás una ruta:\n"
                "   • Se agrega el nodo al mapa.\n"
                "   • Se dibuja la arista.\n"
                "El mapa va guardando todo lo que hiciste en esta sesión."
            )

# ---- TAB GRAFO ----
with tab_grafo:
    st.markdown("### 🔎 Vista lógica de las conexiones")
    st.caption(
        "Esto NO es el mapa real: es una vista limpia tipo red. "
        "Cada punto es una parada y cada línea es una conexión que ya calculaste."
    )

    dot_src = build_graph_graphviz(RGB_NODES, RGB_PATH)

    if dot_src is None:
        st.warning("Todavía no hay suficientes datos para dibujar el grafo. Calculá al menos una ruta 👇")
    else:
        st.graphviz_chart(dot_src, use_container_width=True)
