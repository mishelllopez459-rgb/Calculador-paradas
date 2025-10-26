import streamlit as st
import pandas as pd
import pydeck as pdk
import math
from itertools import combinations

# ---------------- CONFIG INICIAL ----------------
st.set_page_config(page_title="Rutas San Marcos", layout="wide")
st.title("🚌 Calculador de paradas — San Marcos (grafo + highlight)")

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

# Normalizar columnas básicas
for col in ["id", "nombre"]:
    if col in nodos.columns:
        nodos[col] = nodos[col].astype(str).str.strip()
else:
    # si viene sin columnas bien, las forzamos
    nodos = nodos.reindex(columns=["id", "nombre", "lat", "lon"])

for col in ["id", "nombre", "lat", "lon"]:
    if col not in nodos.columns:
        nodos[col] = None

# Asegurar que todos los lugares estén listados aunque no tengan coordenadas
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

# Guardar nodos en session_state para poder modificarlos (asignar coords faltantes)
if "nodos_mem" not in st.session_state:
    st.session_state.nodos_mem = nodos.copy()

# --- estado persistente para el highlight rojo (última arista seleccionada)
if "highlight_edge" not in st.session_state:
    st.session_state.highlight_edge = []

# --- estado persistente para métricas mostradas
if "metrics" not in st.session_state:
    st.session_state.metrics = {
        "modo": "Bicicleta (15 km/h)",
        "ruta_txt": "—",
        "dist_km": None,
        "t_min": None,
        "nota": "Red aproximada. Coordenadas generadas si faltan."
    }

# refrescamos referencia local
nodos_mem = st.session_state.nodos_mem

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
    Crea coordenadas falsas pero estables cerca de BASE_LAT/LON
    usando el nombre como semilla.
    Así cada lugar siempre cae más o menos en el mismo sitio.
    """
    s_val = 0
    for i, ch in enumerate(nombre):
        s_val += (i + 1) * ord(ch)
    lat_off = ((s_val % 17) - 8) * 0.0005  # ~ +/-0.004
    lon_off = (((s_val // 17) % 17) - 8) * 0.0005
    return BASE_LAT + lat_off, BASE_LON + lon_off

def coord_de_lugar(nombre_lugar: str):
    """
    Asegura que un lugar tenga lat/lon en st.session_state.nodos_mem.
    Devuelve lat, lon.
    """
    df = st.session_state.nodos_mem
    idx_list = df.index[df["nombre"] == nombre_lugar].tolist()

    if not idx_list:
        # Si ni siquiera existe, lo creamos desde cero
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

    i = idx_list[0]
    lat_val = st.session_state.nodos_mem.at[i, "lat"]
    lon_val = st.session_state.nodos_mem.at[i, "lon"]

    if pd.isna(lat_val) or pd.isna(lon_val):
        lat_new, lon_new = pseudo_offset(nombre_lugar)
        st.session_state.nodos_mem.at[i, "lat"] = lat_new
        st.session_state.nodos_mem.at[i, "lon"] = lon_new
        return lat_new, lon_new

    return float(lat_val), float(lon_val)

def construir_grafo_base():
    """
    Construye:
    - puntos_base: todos los nodos con coords (nombre, lat, lon)
    - aristas_base: TODAS las conexiones entre pares de nodos (para que se vea grafo)
      Cada arista es {"path": [[lonA, latA],[lonB, latB]], ... }
    Esto es lo azul del mapa.
    """
    # asegurar coords para todos los lugares listados
    for nm in st.session_state.nodos_mem["nombre"]:
        coord_de_lugar(str(nm))

    df_ok = st.session_state.nodos_mem.copy()
    df_ok = df_ok.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    # puntos_base
    puntos_base = df_ok[["nombre", "lat", "lon"]].copy()

    # aristas_base: conectamos todos contra todos (combinations)
    aristas = []
    for i, j in combinations(df_ok.index.tolist(), 2):
        a_lat = df_ok.at[i, "lat"]; a_lon = df_ok.at[i, "lon"]; a_nom = df_ok.at[i, "nombre"]
        b_lat = df_ok.at[j, "lat"]; b_lon = df_ok.at[j, "lon"]; b_nom = df_ok.at[j, "nombre"]

        aristas.append({
            "path": [[a_lon, a_lat], [b_lon, b_lat]],
            "origen": a_nom,
            "destino": b_nom,
            "dist_km": f"{haversine_km(a_lat, a_lon, b_lat, b_lon):.2f}",
            "t_min": ""
        })

    return puntos_base, aristas

# ---------------- SIDEBAR (CONTROLES) ----------------
with st.sidebar:
    st.header("Parámetros")

    origen_nombre = st.selectbox(
        "Origen",
        sorted(st.session_state.nodos_mem["nombre"].astype(str)),
        key="sel_origen"
    )
    destino_nombre = st.selectbox(
        "Destino",
        sorted(st.session_state.nodos_mem["nombre"].astype(str)),
        index=1,
        key="sel_destino"
    )

    st.markdown("### Colores del grafo")
    # Amarillo nodos, Azul grafo base, Rojo highlight actual (igual que screenshot vibe)
    col_nodes = st.color_picker("Nodos (grafo base)", "#FFD400")
    col_path_all = st.color_picker("Aristas del grafo base", "#007BFF")
    col_path_highlight = st.color_picker("Arista seleccionada", "#FF0000")

    calcular = st.button("Calcular / Resaltar esta arista")

RGB_NODES        = hex_to_rgb(col_nodes)
RGB_PATH_ALL     = hex_to_rgb(col_path_all)
RGB_PATH_CURRENT = hex_to_rgb(col_path_highlight)

# ---------------- SI EL USUARIO PIDE UNA RUTA ----------------
if calcular:
    # coords reales (o generadas) para los 2 nodos elegidos
    o_lat, o_lon = coord_de_lugar(origen_nombre)
    d_lat, d_lon = coord_de_lugar(destino_nombre)

    dist_km_val = haversine_km(o_lat, o_lon, d_lat, d_lon)
    vel_kmh = 15.0  # bici en tu captura
    t_min_val = (dist_km_val / vel_kmh) * 60.0

    # guardamos highlight_edge (ROJO)
    st.session_state.highlight_edge = [{
        "path": [[o_lon, o_lat], [d_lon, d_lat]],
        "origen": origen_nombre,
        "destino": destino_nombre,
        "dist_km": f"{dist_km_val:.2f}",
        "t_min": f"{t_min_val:.1f}"
    }]

    # guardamos métricas que se ven a la izquierda
    st.session_state.metrics = {
        "modo": "Bicicleta (15 km/h)",
        "ruta_txt": f"{origen_nombre} → {destino_nombre}",
        "dist_km": f"{dist_km_val:.3f} km",
        "t_min": f"{t_min_val:.1f} min",
        "nota": "Red aproximada. Coordenadas generadas si faltan."
    }

    # para mostrar en la tabla y CSV
    last_tramo_df = pd.DataFrame([
        {"nombre": origen_nombre,  "lat": o_lat, "lon": o_lon},
        {"nombre": destino_nombre, "lat": d_lat, "lon": d_lon},
    ])
    st.session_state.metrics["last_tramo_df_csv"] = last_tramo_df.to_csv(index=False).encode("utf-8")
    st.session_state.metrics["last_tramo_df_table"] = last_tramo_df

# ---------------- CONSTRUIR EL GRAFO BASE (AZUL + NODOS AMARILLOS) ----------------
puntos_base, aristas_base = construir_grafo_base()

# ---------------- DIBUJAR ----------------
col_info, col_map = st.columns([1, 2])

if len(puntos_base) > 0:
    puntos_plot = puntos_base.rename(columns={"lon": "lng"}).copy()

    # Capa de NODOS AMARILLOS con borde negro
    nodes_layer = pdk.Layer(
        "ScatterplotLayer",
        data=puntos_plot,
        get_position="[lng, lat]",
        get_radius=90,
        radius_min_pixels=5,
        get_fill_color=RGB_NODES,
        get_line_color=[0, 0, 0],
        line_width_min_pixels=2,
        pickable=True,
    )

    # Capa de TODAS LAS ARISTAS DEL GRAFO (AZUL)
    edges_layer_all = pdk.Layer(
        "PathLayer",
        data=aristas_base,
        get_path="path",
        get_width=4,
        width_scale=6,
        get_color=RGB_PATH_ALL,
        pickable=False,
    )

    # Capa de LA ARISTA ACTUAL (ROJO, MÁS GRUESA) si existe highlight
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

    # nodos encima de las líneas
    layers_to_draw.append(nodes_layer)

    # centramos el mapa en el promedio de TODOS los nodos del grafo
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
                # SIN map_style forzado -> usa default de pydeck que sí te estaba funcionando
                layers=layers_to_draw,
                initial_view_state=view_state,
                tooltip={
                    "text": "{nombre}\nLat: {lat}\nLon: {lng}"
                }
            ),
            use_container_width=True
        )
else:
    # Si por alguna razón no hay puntos (no debería pasar), centramos en base
    view_state = pdk.ViewState(
        latitude=BASE_LAT,
        longitude=BASE_LON,
        zoom=14,
        pitch=0
    )
    with col_map:
        st.pydeck_chart(
            pdk.Deck(
                layers=[],
                initial_view_state=view_state
            ),
            use_container_width=True
        )

# ---------------- PANEL IZQUIERDO (INFO COMO EN LA FOTO) ----------------
with col_info:
    st.subheader("Resumen de la última selección")

    st.markdown(f"**Modo:** {st.session_state.metrics['modo']}")
    st.markdown(f"**Ruta:** {st.session_state.metrics['ruta_txt']}")
    if st.session_state.metrics["dist_km"] is not None:
        st.markdown(f"**Distancia:** {st.session_state.metrics['dist_km']}")
    if st.session_state.metrics["t_min"] is not None:
        st.markdown(f"**Tiempo estimado:** {st.session_state.metrics['t_min']}")
    st.markdown(f"**Notas:** {st.session_state.metrics['nota']}")

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
            "Azul = grafo base (todas las conexiones)\n"
            "Rojo = arista seleccionada ahora\n"
            "Amarillo = nodos/lugares"
        )
