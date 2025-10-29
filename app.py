# app.py
import math
import requests
import pandas as pd
import pydeck as pdk
import streamlit as st

# ---------------- UI / CONFIG ----------------
st.set_page_config(page_title="Rutas San Marcos", layout="wide")
st.title("🚌 Calculador de paradas — San Marcos (modo grafo + OSRM)")

# ---------------- CARGA DE DATOS ----------------
# nodos.csv -> columnas esperadas: id, nombre, lat, lon
try:
    nodos_raw = pd.read_csv("nodos.csv")
except Exception:
    nodos_raw = pd.DataFrame(columns=["id", "nombre", "lat", "lon"])

# aristas.csv -> columnas esperadas: origen, destino, (peso opcional)
try:
    aristas_raw = pd.read_csv("aristas.csv")
except Exception:
    aristas_raw = pd.DataFrame(columns=["origen", "destino", "peso"])

# ---------------- LISTA BASE DE LUGARES ----------------
LUGARES_NUEVOS = [
    "Parque Central", "Catedral", "Terminal de Buses", "Hospital Regional",
    "Cancha Los Angeles", "Cancha Sintetica Golazo", "Aeropuerto Nacional",
    "Iglesia Candelero de Oro", "Centro de Salud", "Megapaca",
    "CANICA (Casa de los Niños)", "Aldea San Rafael Soche", "Pollo Campero",
    "INTECAP San Marcos", "Salón Quetzal", "SAT San Marcos", "Bazar Chino"
]

# ---------------- NORMALIZACIÓN DE NODOS ----------------
nodos = nodos_raw.copy()

# limpiar strings en columnas clave
for col in ["id", "nombre"]:
    if col in nodos.columns:
        nodos[col] = nodos[col].astype(str).str.strip()
    else:
        nodos[col] = None

# asegurar columnas mínimas
for c in ["id", "nombre", "lat", "lon"]:
    if c not in nodos.columns:
        nodos[c] = None
nodos = nodos[["id", "nombre", "lat", "lon"]]

def asegurar_lugares(df: pd.DataFrame, nombres: list) -> pd.DataFrame:
    """
    Si en LUGARES_NUEVOS hay nombres que no están en el CSV,
    se agregan filas nuevas con ids tipo L1, L2, ...
    """
    existentes = set(df["nombre"].astype(str).str.lower()) if "nombre" in df else set()
    usados = set(df["id"].astype(str)) if "id" in df else set()

    def nuevo_id(start=1):
        i = start
        while True:
            candidato = f"L{i}"
            if candidato not in usados:
                usados.add(candidato)
                return candidato
            i += 1

    filas_nuevas = []
    for nm in nombres:
        if nm.lower() not in existentes:
            filas_nuevas.append(
                {"id": nuevo_id(), "nombre": nm, "lat": None, "lon": None}
            )

    if filas_nuevas:
        df = pd.concat([df, pd.DataFrame(filas_nuevas)], ignore_index=True)

    return df

nodos = asegurar_lugares(nodos, LUGARES_NUEVOS)

# Guardar nodos "vivos" en sesión para poder editar coords sin pisar el CSV original
if "nodos_mem" not in st.session_state:
    st.session_state.nodos_mem = nodos.copy()

# siempre trabajamos sobre la versión en memoria
nodos = st.session_state.nodos_mem

# ---------------- HELPERS ----------------
def hex_to_rgb(h: str):
    h = h.lstrip("#")
    return [int(h[i:i+2], 16) for i in (0, 2, 4)]

def haversine_km(a_lat, a_lon, b_lat, b_lon):
    R = 6371.0
    lat1, lon1 = math.radians(a_lat), math.radians(a_lon)
    lat2, lon2 = math.radians(b_lat), math.radians(b_lon)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(h))

def tiene_coords(row) -> bool:
    return pd.notna(row["lat"]) and pd.notna(row["lon"])

def osrm_route(o_lat, o_lon, d_lat, d_lon):
    """
    Devuelve: (path_lonlat, dist_km, dur_min) o (None, None, None) si falla.
    path_lonlat: lista [[lon, lat], ...] siguiendo la calle.
    """
    url = (
        f"https://router.project-osrm.org/route/v1/driving/"
        f"{o_lon},{o_lat};{d_lon},{d_lat}?overview=full&geometries=geojson"
    )
    try:
        r = requests.get(url, timeout=12)
        r.raise_for_status()
        data = r.json()
        coords = data["routes"][0]["geometry"]["coordinates"]  # [lon, lat]
        dist_km = data["routes"][0]["distance"] / 1000.0
        dur_min = data["routes"][0]["duration"] / 60.0
        return coords, dist_km, dur_min
    except Exception:
        return None, None, None

def fit_view_from_lonlat(coords_lonlat: list, extra_zoom_out: float = 0.35):
    """
    Calcula un ViewState que encuadra todas las coordenadas (lon,lat) usando
    pdk.data_utils.compute_view y aplica un pequeño padding con 'extra_zoom_out'.
    """
    if not coords_lonlat:
        # fallback: centro aproximado San Marcos
        return pdk.ViewState(latitude=14.965, longitude=-91.79, zoom=13)

    df_bounds = pd.DataFrame(coords_lonlat, columns=["lon", "lat"])
    view = pdk.data_utils.compute_view(df_bounds[["lon", "lat"]])
    # alejamos un poquito para no cortar bordes
    view["zoom"] = max(1, view["zoom"] - extra_zoom_out)
    return pdk.ViewState(**view, pitch=0, bearing=0)

def construir_capa_nodos(df_nodos: pd.DataFrame, rgb_color):
    """ScatterplotLayer con todos los nodos que ya tienen lat/lon."""
    nodos_plot = df_nodos.dropna(subset=["lat", "lon"]).copy()
    if nodos_plot.empty:
        return None, pd.DataFrame()
    nodos_plot.rename(columns={"lon": "lng"}, inplace=True)

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=nodos_plot,
        get_position="[lng, lat]",
        get_radius=65,
        radius_min_pixels=3,
        get_fill_color=rgb_color,
        get_line_color=[30, 30, 30],
        line_width_min_pixels=1,
        pickable=True,
        tooltip=True,
    )
    return layer, nodos_plot

def construir_capa_aristas(df_aristas: pd.DataFrame, df_nodos: pd.DataFrame, rgb_color):
    """
    Genera una PathLayer con TODAS las aristas del grafo.
    Cada arista se dibuja como un segmento recto: [ [lon_o, lat_o], [lon_d, lat_d] ].
    Ignora aristas cuyos nodos no tengan coordenadas.
    """
    if df_aristas.empty:
        return None, []

    # prepararnos para join rápido por id
    nod_idx = df_nodos.set_index("id")[["lat", "lon"]]

    paths = []
    for _, row in df_aristas.iterrows():
        o_id = str(row.get("origen", "")).strip()
        d_id = str(row.get("destino", "")).strip()
        if o_id in nod_idx.index and d_id in nod_idx.index:
            o_lat, o_lon = nod_idx.loc[o_id, ["lat", "lon"]]
            d_lat, d_lon = nod_idx.loc[d_id, ["lat", "lon"]]
            if pd.notna(o_lat) and pd.notna(o_lon) and pd.notna(d_lat) and pd.notna(d_lon):
                paths.append({
                    "path": [[o_lon, o_lat], [d_lon, d_lat]],
                    "origen": o_id,
                    "destino": d_id,
                })

    if not paths:
        return None, []

    layer = pdk.Layer(
        "PathLayer",
        data=paths,
        get_path="path",
        get_width=4,
        width_scale=8,
        get_color=rgb_color,
        pickable=True,
        tooltip=True,
    )
    return layer, paths

def construir_capa_ruta(path_lonlat, rgb_color):
    """
    PathLayer para la ruta calculada (origen->destino).
    path_lonlat: lista [[lon,lat], ...]
    """
    if not path_lonlat:
        return None
    layer = pdk.Layer(
        "PathLayer",
        data=[{"path": path_lonlat}],
        get_path="path",
        get_width=6,
        width_scale=8,
        get_color=rgb_color,
        pickable=False,
    )
    return layer

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("Parámetros")

    origen_nombre = st.selectbox("Origen", sorted(nodos["nombre"]))
    destino_nombre = st.selectbox(
        "Destino",
        sorted(nodos["nombre"]),
        index=1 if len(nodos["nombre"]) > 1 else 0
    )

    st.markdown("### Visualización")

    show_nodes = st.toggle("Mostrar nodos del grafo", value=True)
    show_edges = st.toggle("Mostrar aristas del grafo", value=True)

    col_nodes = st.color_picker("Color nodos", "#FF007F")
    col_edges = st.color_picker("Color aristas", "#FFA500")
    col_path  = st.color_picker("Color ruta origen→destino", "#007AFF")

    usar_osrm = st.toggle("Ruta real por calle (OSRM)", value=True)

    st.markdown("---")
    st.markdown("### Agregar / editar coordenadas")

    def editor_de_coords(etiqueta, nombre_sel):
        fila_sel = nodos.loc[nodos["nombre"] == nombre_sel].iloc[0]
        col1, col2 = st.columns(2)
        with col1:
            lat_txt = st.text_input(
                f"Lat ({etiqueta})",
                value="" if pd.isna(fila_sel["lat"]) else str(fila_sel["lat"]),
                key=f"lat_{etiqueta}",
            )
        with col2:
            lon_txt = st.text_input(
                f"Lon ({etiqueta})",
                value="" if pd.isna(fila_sel["lon"]) else str(fila_sel["lon"]),
                key=f"lon_{etiqueta}",
            )
        if st.button(f"Guardar coords de {etiqueta}"):
            try:
                lat = float(str(lat_txt).replace(",", "."))
                lon = float(str(lon_txt).replace(",", "."))
                st.session_state.nodos_mem.loc[
                    nodos["nombre"] == nombre_sel, ["lat", "lon"]
                ] = [lat, lon]
                st.success(
                    f"Coordenadas guardadas para {nombre_sel}: ({lat}, {lon})"
                )
            except ValueError:
                st.error("Lat/Lon inválidos. Usa números (ej. 14.9712 y -91.7815)")

    editor_de_coords("Origen", origen_nombre)
    editor_de_coords("Destino", destino_nombre)

# ---------------- PREPARAR CAPAS DEL MAPA ----------------
RGB_NODES = hex_to_rgb(col_nodes)
RGB_EDGES = hex_to_rgb(col_edges)
RGB_PATH  = hex_to_rgb(col_path)

layers = []
all_coords_for_view = []  # para calcular el zoom automático

# 1. Capa de NODOS
nodes_layer, nodos_plot = construir_capa_nodos(nodos, RGB_NODES)
if show_nodes and nodes_layer is not None:
    layers.append(nodes_layer)
    # coords de nodos para el encuadre
    all_coords_for_view.extend(
        nodos_plot[["lng", "lat"]].values.tolist()
    )

# 2. Capa de ARISTAS DEL GRAFO
edges_layer, paths_edges = construir_capa_aristas(aristas_raw, nodos, RGB_EDGES)
if show_edges and edges_layer is not None:
    layers.append(edges_layer)
    # coords de aristas para encuadre
    for seg in paths_edges:
        all_coords_for_view.extend(seg["path"])

# 3. Calcular ruta entre origen y destino (si ambos tienen coords)
fila_o = nodos.loc[nodos["nombre"] == origen_nombre].iloc[0]
fila_d = nodos.loc[nodos["nombre"] == destino_nombre].iloc[0]

dist_km = None
dur_min = None
ruta_layer = None
ruta_path_lonlat = []

if tiene_coords(fila_o) and tiene_coords(fila_d):
    o_lat, o_lon = float(fila_o["lat"]), float(fila_o["lon"])
    d_lat, d_lon = float(fila_d["lat"]), float(fila_d["lon"])

    # Intentar OSRM (ruta real calle)
    ruta_path_lonlat, dist_km, dur_min = (None, None, None)
    if usar_osrm:
        ruta_path_lonlat, dist_km, dur_min = osrm_route(o_lat, o_lon, d_lat, d_lon)

    # Fallback: línea recta + tiempo estimado por velocidad 30 km/h
    if ruta_path_lonlat is None:
        dist_km = haversine_km(o_lat, o_lon, d_lat, d_lon)
        vel_kmh = 30.0
        dur_min = (dist_km / vel_kmh) * 60.0
        ruta_path_lonlat = [[o_lon, o_lat], [d_lon, d_lat]]

    ruta_layer = construir_capa_ruta(ruta_path_lonlat, RGB_PATH)
    if ruta_layer is not None:
        layers.append(ruta_layer)
        all_coords_for_view.extend(ruta_path_lonlat)

# ---------------- VIEWSTATE (ZOOOOM) ----------------
if all_coords_for_view:
    view_state = fit_view_from_lonlat(all_coords_for_view, extra_zoom_out=0.4)
else:
    # fallback si no hay nada con coords
    view_state = pdk.ViewState(latitude=14.965, longitude=-91.79, zoom=13)

# ---------------- LAYOUT PRINCIPAL ----------------
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Resumen")

    st.markdown(f"**Origen:** {origen_nombre}")
    st.markdown(f"**Destino:** {destino_nombre}")

    if dist_km is not None and dur_min is not None:
        st.markdown(f"**Distancia aprox.:** {dist_km:.2f} km")
        st.markdown(f"**Tiempo aprox.:** {dur_min:.1f} min")
    else:
        st.markdown("**Distancia aprox.:** —")
        st.markdown("**Tiempo aprox.:** —")

    if ruta_path_lonlat:
        export_df = pd.DataFrame(ruta_path_lonlat, columns=["lon", "lat"])
        st.download_button(
            "📥 Descargar ruta (CSV)",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name="ruta.csv",
            mime="text/csv",
        )

    st.markdown("---")
    st.markdown("**Nodos cargados (con coordenadas):**")
    nodos_show = nodos.dropna(subset=["lat", "lon"])[["id", "nombre", "lat", "lon"]]
    st.dataframe(nodos_show, use_container_width=True)

    st.markdown("**Aristas cargadas:**")
    st.dataframe(aristas_raw, use_container_width=True)

with col2:
    st.pydeck_chart(
        pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            tooltip={
                "html": "<b>{origen}</b> → <b>{destino}</b>",
                "style": {"color": "white"}
            },
        ),
        use_container_width=True,
    )

# ---------------- MENSAJE SI FALTAN COORDS ----------------
if not (tiene_coords(fila_o) and tiene_coords(fila_d)):
    st.info(
        "Asigna lat/lon al Origen y al Destino en la barra lateral para ver la ruta.\n"
        "Los nodos y aristas del grafo se muestran igual si ya tienen coordenadas."
    )

