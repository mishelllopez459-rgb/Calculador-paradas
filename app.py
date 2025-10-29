import math
import requests
import pandas as pd
import pydeck as pdk
import streamlit as st
from collections import deque

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Rutas San Marcos", layout="wide")
st.title("🚌 Calculador de paradas y ruta óptima — San Marcos")

# ---------------- COORDENADAS PREDEFINIDAS ----------------
# Lugares que ya tenemos ubicados físicamente (lo que mandó Andrea)
PRESET_COORDS = {
    "Megapaca": (14.963451, -91.791009),
    "Pollo Campero": (14.965879, -91.811366),
    "SAT San Marcos": (14.965911, -91.793319),
    "Salón Quetzal": (14.964171, -91.795312),
    "Canchas Santo Domingo": (14.960118, -91.798793),
    "Parque Benito Juarez": (14.958947, -91.802114),
    "Iglesia Cristiana Buena Tierra": (14.949966, -91.809321),

    "Salón Terracota": (14.966691, -91.797232),
    "Contraloría General de Cuentas": (14.966359, -91.797010),
    "Centro Medico de Especialidades": (14.964610, -91.797402),
    "Cementerio San Marcos": (14.964338, -91.800597),

    "Hotel y Restaurante Santa Barbara": (14.963654, -91.796771),
    "Domino's Pizza SM": (14.963560, -91.7914700),
    "Ministerio de Ambiente": (14.963989, -91.793348),
    "Ministerio Público de la Mujer": (14.968695, -91.798307),
    "Juzgado de Primera Instancia": (14.969025, -91.797968),
    "Edificio Tribunales San Marcos": (14.964409, -91.794494),

    "Gobernación San Marcos": (14.966084, -91.794452),
    "Downtown Cafe y Discoteca San Marcos": (14.964449, -91.794986),
    "Municipalidad de San Marcos": (14.964601, -91.793954),
    "Centro Universitario CUSAM San Marcos": (14.965126, -91.799426),
    "CLICOLOR": (14.966102, -91.799505),
    "Fundap Microcrédito San Marcos": (14.962564, -91.799215),
    "ACREDICOM R. L San Marcos": (14.966154, -91.793269),
    "Banrural San Marcos": (14.965649, -91.795858),
}

# ---------------- CARGA DE ARCHIVOS ----------------
try:
    nodos_raw = pd.read_csv("nodos.csv")  # columnas esperadas: id, nombre, lat, lon
except Exception:
    nodos_raw = pd.DataFrame(columns=["id", "nombre", "lat", "lon"])

try:
    aristas_raw = pd.read_csv("aristas.csv")  # columnas: origen, destino, peso(opc)
except Exception:
    aristas_raw = pd.DataFrame(columns=["origen", "destino", "peso"])

# ---------------- LUGARES BASE (también queremos que aparezcan en la lista) ----------------
LUGARES_NUEVOS = [
    "Parque Central", "Catedral", "Terminal de Buses", "Hospital Regional",
    "Cancha Los Angeles", "Cancha Sintetica Golazo", "Aeropuerto Nacional",
    "Iglesia Candelero de Oro", "Centro de Salud", "Megapaca",
    "CANICA (Casa de los Niños)", "Aldea San Rafael Soche", "Pollo Campero",
    "INTECAP San Marcos", "Salón Quetzal", "SAT San Marcos",
    "Bazar Chino",

    # nuevos aportes de Andrea 👇
    "Canchas Santo Domingo",
    "Parque Benito Juarez",
    "Iglesia Cristiana Buena Tierra",
    "Salón Terracota",
    "Contraloría General de Cuentas",
    "Centro Medico de Especialidades",
    "Cementerio San Marcos",
    "Hotel y Restaurante Santa Barbara",
    "Domino's Pizza SM",
    "Ministerio de Ambiente",
    "Ministerio Público de la Mujer",
    "Juzgado de Primera Instancia",
    "Edificio Tribunales San Marcos",
    "Gobernación San Marcos",
    "Downtown Cafe y Discoteca San Marcos",
    "Municipalidad de San Marcos",
    "Centro Universitario CUSAM San Marcos",
    "CLICOLOR",
    "Fundap Microcrédito San Marcos",
    "ACREDICOM R. L San Marcos",
    "Banrural San Marcos",
]

# ---------------- NORMALIZACIÓN INICIAL DE NODOS ----------------
nodos = nodos_raw.copy()

# limpiar strings
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
    - Agrega filas para lugares que faltan.
    - Si el lugar está en PRESET_COORDS, le ponemos esas coords de una vez.
    """
    existentes_lower = set(df["nombre"].astype(str).str.lower()) if "nombre" in df else set()
    usados_ids = set(df["id"].astype(str)) if "id" in df else set()

    def nuevo_id(start=1):
        i = start
        while True:
            cand = f"L{i}"
            if cand not in usados_ids:
                usados_ids.add(cand)
                return cand
            i += 1

    nuevas_filas = []
    for nm in nombres:
        if nm.lower() not in existentes_lower:
            lat, lon = (None, None)
            if nm in PRESET_COORDS:
                lat, lon = PRESET_COORDS[nm]
            nuevas_filas.append({
                "id": nuevo_id(),
                "nombre": nm,
                "lat": lat,
                "lon": lon,
            })

    if nuevas_filas:
        df = pd.concat([df, pd.DataFrame(nuevas_filas)], ignore_index=True)

    # además: si ya existía pero no tenía coords y PRESET_COORDS sí tiene, completamos
    for nm, (plat, plon) in PRESET_COORDS.items():
        mask = df["nombre"].astype(str).str.strip().str.lower() == nm.lower()
        if mask.any():
            # sólo poner coords si faltan
            if df.loc[mask, "lat"].isna().any() or df.loc[mask, "lon"].isna().any():
                df.loc[mask, ["lat", "lon"]] = [plat, plon]

    return df

nodos = asegurar_lugares(nodos, LUGARES_NUEVOS)

# memoria editable viva
if "nodos_mem" not in st.session_state:
    st.session_state.nodos_mem = nodos.copy()
nodos = st.session_state.nodos_mem  # usar en adelante

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

def tiene_coords_row(row) -> bool:
    return pd.notna(row["lat"]) and pd.notna(row["lon"])

def osrm_route(o_lat, o_lon, d_lat, d_lon):
    """
    Devuelve (camino_lonlat, dist_km, dur_min) o (None, None, None)
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
    Ajusta el mapa para ver todo. Compatible con pydeck dict / ViewState.
    """
    if not coords_lonlat:
        return pdk.ViewState(latitude=14.965, longitude=-91.79, zoom=13, pitch=0, bearing=0)

    df_bounds = pd.DataFrame(coords_lonlat, columns=["lon", "lat"])
    raw_view = pdk.data_utils.compute_view(df_bounds[["lon", "lat"]])

    if isinstance(raw_view, dict):
        lat_center = raw_view.get("latitude", 14.965)
        lon_center = raw_view.get("longitude", -91.79)
        zoom_val = raw_view.get("zoom", 13)
    else:
        lat_center = getattr(raw_view, "latitude", 14.965)
        lon_center = getattr(raw_view, "longitude", -91.79)
        zoom_val   = getattr(raw_view, "zoom", 13)

    try:
        zoom_val = max(1, float(zoom_val) - float(extra_zoom_out))
    except Exception:
        zoom_val = 13

    return pdk.ViewState(
        latitude=lat_center,
        longitude=lon_center,
        zoom=zoom_val,
        pitch=0,
        bearing=0,
    )

# --------- Grafo básico (para paradas / hops / falbacks) ---------
def build_graph_edges(df_aristas: pd.DataFrame):
    """
    Grafo no dirigido {nodo_id: set(vecinos)} usando columnas origen,destino.
    """
    graph = {}
    for _, row in df_aristas.iterrows():
        a = str(row.get("origen", "")).strip()
        b = str(row.get("destino", "")).strip()
        if not a or not b:
            continue
        graph.setdefault(a, set()).add(b)
        graph.setdefault(b, set()).add(a)
    return graph

def bfs_shortest_path(graph: dict, start: str, goal: str):
    """
    Devuelve lista de IDs [start,...,goal] por BFS mínimo saltos.
    Si no hay camino -> [].
    """
    if not start or not goal:
        return []
    if start not in graph or goal not in graph:
        return []
    if start == goal:
        return [start]

    q = deque([[start]])
    seen = {start}

    while q:
        path = q.popleft()
        u = path[-1]
        for v in graph.get(u, []):
            if v in seen:
                continue
            new_path = path + [v]
            if v == goal:
                return new_path
            seen.add(v)
            q.append(new_path)

    return []

def nombre_a_id(df_nodos: pd.DataFrame, nombre_busca: str):
    fila = df_nodos.loc[df_nodos["nombre"] == nombre_busca]
    if fila.empty:
        return None
    return str(fila.iloc[0]["id"])

def ids_a_polyline_lonlat(df_nodos: pd.DataFrame, path_ids: list):
    """
    Convierte una lista de IDs de nodos en [[lon,lat], ...] solo
    para los que tienen coords. Si menos de 2 puntos útiles -> [].
    """
    out = []
    idx = df_nodos.set_index("id")
    for nid in path_ids:
        if nid in idx.index:
            lat = idx.loc[nid, "lat"]
            lon = idx.loc[nid, "lon"]
            if pd.notna(lat) and pd.notna(lon):
                out.append([float(lon), float(lat)])
    return out if len(out) >= 2 else []

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("Parámetros")

    origen_nombre = st.selectbox("Origen", sorted(nodos["nombre"]))
    destino_nombre = st.selectbox(
        "Destino",
        sorted(nodos["nombre"]),
        index=1 if len(nodos["nombre"]) > 1 else 0,
    )

    st.markdown("### Visualización")
    show_nodes = st.toggle("Mostrar nodos del grafo", value=True)
    show_edges = st.toggle("Mostrar aristas del grafo", value=True)

    col_nodes = st.color_picker("Color nodos", "#FF007F")
    col_edges = st.color_picker("Color aristas", "#FFA500")
    col_path  = st.color_picker("Color ruta origen→destino", "#007AFF")

    usar_osrm = st.toggle("Ruta real por calle (OSRM)", value=True)

    st.markdown("---")
    st.markdown("### Coordenadas actuales")

    # Mostrar coords ya guardadas (vienen de PRESET_COORDS si existían)
    fila_o = nodos.loc[nodos["nombre"] == origen_nombre].iloc[0]
    fila_d = nodos.loc[nodos["nombre"] == destino_nombre].iloc[0]

    col1, col2 = st.columns(2)
    with col1:
        st.text_input(
            "Lat (Origen)",
            value="" if pd.isna(fila_o["lat"]) else str(fila_o["lat"]),
            key="lat_origen_box",
        )
    with col2:
        st.text_input(
            "Lon (Origen)",
            value="" if pd.isna(fila_o["lon"]) else str(fila_o["lon"]),
            key="lon_origen_box",
        )

    col3, col4 = st.columns(2)
    with col3:
        st.text_input(
            "Lat (Destino)",
            value="" if pd.isna(fila_d["lat"]) else str(fila_d["lat"]),
            key="lat_destino_box",
        )
    with col4:
        st.text_input(
            "Lon (Destino)",
            value="" if pd.isna(fila_d["lon"]) else str(fila_d["lon"]),
            key="lon_destino_box",
        )

# ---------------- CÁLCULO DE RUTA ----------------
# 1. ids de origen/destino
origen_id = nombre_a_id(nodos, origen_nombre)
destino_id = nombre_a_id(nodos, destino_nombre)

# 2. grafo (para hops y paradas)
graph = build_graph_edges(aristas_raw)
path_ids = bfs_shortest_path(graph, origen_id, destino_id) if (origen_id and destino_id) else []

# 3. métrica de paradas
if path_ids:
    paradas_totales = len(path_ids)
    paradas_intermedias = max(0, paradas_totales - 2)
    saltos = len(path_ids) - 1
else:
    paradas_totales = 2
    paradas_intermedias = 0
    saltos = 1

# 4. tiempo estimado SI NO tenemos OSRM
#    suponemos 3 min por salto de una parada a otra
estimado_min_por_grafo = saltos * 3.0 if saltos >= 1 else None

# 5. ruta real si hay coords
dist_km = None
dur_min = None
ruta_path_lonlat_osrm = None

if tiene_coords_row(fila_o) and tiene_coords_row(fila_d):
    o_lat, o_lon = float(fila_o["lat"]), float(fila_o["lon"])
    d_lat, d_lon = float(fila_d["lat"]), float(fila_d["lon"])

    # intentar OSRM
    if usar_osrm:
        ruta_path_lonlat_osrm, dist_km, dur_min = osrm_route(o_lat, o_lon, d_lat, d_lon)

    # fallback: línea recta si OSRM falló
    if ruta_path_lonlat_osrm is None:
        dist_km = haversine_km(o_lat, o_lon, d_lat, d_lon)
        VEL_KMH = 30.0
        dur_min = (dist_km / VEL_KMH) * 60.0
        ruta_path_lonlat_osrm = [[o_lon, o_lat], [d_lon, d_lat]]

# 6. si no logramos ruta_path_lonlat_osrm, intentamos dibujar la ruta por el grafo (si hay coords en cada parada)
ruta_path_lonlat_grafo = []
if not ruta_path_lonlat_osrm:
    ruta_path_lonlat_grafo = ids_a_polyline_lonlat(nodos, path_ids)

# ---------------- CAPAS MAPA ----------------
RGB_NODES = hex_to_rgb(col_nodes)
RGB_EDGES = hex_to_rgb(col_edges)
RGB_PATH  = hex_to_rgb(col_path)

layers = []
all_coords_for_view = []

# nodos (scatter)
nodos_plot = nodos.dropna(subset=["lat", "lon"]).copy()
if not nodos_plot.empty:
    nodos_plot.rename(columns={"lon": "lng"}, inplace=True)
    if show_nodes:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=nodos_plot,
                get_position="[lng, lat]",
                get_radius=65,
                radius_min_pixels=3,
                get_fill_color=RGB_NODES,
                get_line_color=[30, 30, 30],
                line_width_min_pixels=1,
                pickable=True,
            )
        )
    all_coords_for_view.extend(nodos_plot[["lng", "lat"]].values.tolist())

# aristas (segmentos rectos entre nodos conectados)
if show_edges and not aristas_raw.empty:
    idx = nodos.set_index("id")[["lat", "lon"]]
    aristas_paths = []
    for _, row in aristas_raw.iterrows():
        a = str(row.get("origen", "")).strip()
        b = str(row.get("destino", "")).strip()
        if a in idx.index and b in idx.index:
            a_lat, a_lon = idx.loc[a, ["lat", "lon"]]
            b_lat, b_lon = idx.loc[b, ["lat", "lon"]]
            if pd.notna(a_lat) and pd.notna(a_lon) and pd.notna(b_lat) and pd.notna(b_lon):
                seg = [[a_lon, a_lat], [b_lon, b_lat]]
                aristas_paths.append({"path": seg, "origen": a, "destino": b})
                all_coords_for_view.extend(seg)

    if aristas_paths:
        layers.append(
            pdk.Layer(
                "PathLayer",
                data=aristas_paths,
                get_path="path",
                get_width=4,
                width_scale=8,
                get_color=RGB_EDGES,
                pickable=True,
            )
        )

# ruta final a dibujar (preferimos OSRM/haversine; si no, grafo)
ruta_final_lonlat = ruta_path_lonlat_osrm if ruta_path_lonlat_osrm else ruta_path_lonlat_grafo
if ruta_final_lonlat:
    layers.append(
        pdk.Layer(
            "PathLayer",
            data=[{"path": ruta_final_lonlat}],
            get_path="path",
            get_width=6,
            width_scale=8,
            get_color=RGB_PATH,
            pickable=False,
        )
    )
    all_coords_for_view.extend(ruta_final_lonlat)

# viewstate para el mapa
if all_coords_for_view:
    view_state = fit_view_from_lonlat(all_coords_for_view, extra_zoom_out=0.4)
else:
    view_state = pdk.ViewState(latitude=14.965, longitude=-91.79, zoom=13, pitch=0, bearing=0)

# ---------------- MÉTRICAS PARA RESUMEN ----------------
criterio_texto = "⏱ tiempo mín"
grafo_texto = "No dirigido"

# texto distancia
distancia_txt = f"{dist_km:.2f} km" if dist_km is not None else "—"

# texto tiempo / costo
if dur_min is not None:
    tiempo_txt = f"{dur_min:.1f} min"
    costo_total_txt = f"{dur_min:.1f} min"
elif estimado_min_por_grafo is not None:
    tiempo_txt = f"~{estimado_min_por_grafo:.1f} min"
    costo_total_txt = f"~{estimado_min_por_grafo:.1f} min"
else:
    tiempo_txt = "—"
    costo_total_txt = "—"

# ---------------- LAYOUT: IZQ info / DER mapa ----------------
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Resumen")

    st.markdown(f"**Origen:** {origen_nombre}")
    st.markdown(f"**Destino:** {destino_nombre}")
    st.markdown(f"**Criterio:** {criterio_texto}")
    st.markdown(f"**Grafo:** {grafo_texto}")

    st.markdown(f"**Paradas (incluye origen y destino):** {paradas_totales}")
    st.markdown(f"**Paradas intermedias:** {paradas_intermedias}")

    st.markdown(f"**Distancia aprox.:** {distancia_txt}")
    st.markdown(f"**Tiempo aprox.:** {tiempo_txt}")
    st.markdown(f"**Costo total (tiempo mín):** {costo_total_txt}")

    if ruta_final_lonlat:
        export_df = pd.DataFrame(ruta_final_lonlat, columns=["lon", "lat"])
        st.download_button(
            "📥 Descargar ruta (CSV)",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name="ruta.csv",
            mime="text/csv",
        )

    st.markdown("---")
    st.markdown("**Nodos cargados (con coordenadas):**")
    st.dataframe(
        nodos.dropna(subset=["lat", "lon"])[["id", "nombre", "lat", "lon"]],
        use_container_width=True,
    )

    st.markdown("**Aristas cargadas:**")
    st.dataframe(aristas_raw, use_container_width=True)

with col2:
    st.pydeck_chart(
        pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            tooltip={
                "html": "<b>{origen}</b> → <b>{destino}</b>",
                "style": {"color": "white"},
            },
        ),
        use_container_width=True,
    )

if not ruta_final_lonlat:
    st.info(
        "No se pudo dibujar la ruta en el mapa (falta coordenada en alguna parada del camino). "
        "Pero el cálculo de tiempo y paradas ya está arriba."
    )
