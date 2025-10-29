import math
import requests
import pandas as pd
import pydeck as pdk
import streamlit as st
from collections import deque

# ---------------- UI / CONFIG ----------------
st.set_page_config(page_title="Rutas San Marcos", layout="wide")
st.title("🚌 Calculador de paradas y ruta óptima — San Marcos")

# ---------------- CARGA DE DATOS ----------------
try:
    nodos_raw = pd.read_csv("nodos.csv")  # columnas: id, nombre, lat, lon
except Exception:
    nodos_raw = pd.DataFrame(columns=["id", "nombre", "lat", "lon"])

try:
    aristas_raw = pd.read_csv("aristas.csv")  # columnas: origen, destino, (peso opcional)
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

for col in ["id", "nombre"]:
    if col in nodos.columns:
        nodos[col] = nodos[col].astype(str).str.strip()
    else:
        nodos[col] = None

for c in ["id", "nombre", "lat", "lon"]:
    if c not in nodos.columns:
        nodos[c] = None
nodos = nodos[["id", "nombre", "lat", "lon"]]

def asegurar_lugares(df: pd.DataFrame, nombres: list) -> pd.DataFrame:
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

# memoria editable de coords
if "nodos_mem" not in st.session_state:
    st.session_state.nodos_mem = nodos.copy()
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
    if not coords_lonlat:
        return pdk.ViewState(latitude=14.965, longitude=-91.79, zoom=13, pitch=0, bearing=0)

    df_bounds = pd.DataFrame(coords_lonlat, columns=["lon", "lat"])
    raw_view = pdk.data_utils.compute_view(df_bounds[["lon", "lat"]])

    if isinstance(raw_view, dict):
        lat_center = raw_view.get("latitude", 14.965)
        lon_center = raw_view.get("longitude", -91.79)
        zoom_val  = raw_view.get("zoom", 13)
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

# --------- NUEVO: utilidades de grafo ---------
def build_graph_edges(df_aristas: pd.DataFrame):
    """
    Crea un grafo no dirigido como dict {nodo_id: set(vecinos_ids)}
    usando columnas 'origen','destino'.
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
    Retorna lista de ids desde start hasta goal (incluyendo ambos),
    usando BFS por número de saltos. Si no hay camino, retorna [].
    """
    if start not in graph or goal not in graph:
        return []
    if start == goal:
        return [start]

    visit = set([start])
    q = deque([[start]])

    while q:
        path = q.popleft()
        last = path[-1]
        for nb in graph.get(last, []):
            if nb in visit:
                continue
            new_path = path + [nb]
            if nb == goal:
                return new_path
            visit.add(nb)
            q.append(new_path)
    return []

def nombre_a_id(nodos_df: pd.DataFrame, nombre_busca: str):
    """
    Busca el id del nodo que tiene ese nombre exacto.
    Si no lo encuentra, devuelve None.
    """
    fila = nodos_df.loc[nodos_df["nombre"] == nombre_busca]
    if fila.empty:
        return None
    return str(fila.iloc[0]["id"])

def ids_a_polyline_lonlat(nodos_df: pd.DataFrame, path_ids: list):
    """
    Intenta crear una polyline [[lon,lat], ...] siguiendo el orden
    de path_ids. Sólo usa puntos que sí tengan lat/lon.
    Si a alguno le falta lat/lon, se omite ese punto.
    (si quedan menos de 2 puntos útiles, devolvemos [])
    """
    out = []
    subset = nodos_df.set_index("id")
    for nid in path_ids:
        if nid in subset.index:
            lat = subset.loc[nid, "lat"]
            lon = subset.loc[nid, "lon"]
            if pd.notna(lat) and pd.notna(lon):
                out.append([float(lon), float(lat)])
    if len(out) < 2:
        return []
    return out

# --------- CAPAS PARA EL MAPA ---------
def construir_capa_nodos(df_nodos: pd.DataFrame, rgb_color):
    plot = df_nodos.dropna(subset=["lat", "lon"]).copy()
    if plot.empty:
        return None, pd.DataFrame()
    plot.rename(columns={"lon": "lng"}, inplace=True)

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=plot,
        get_position="[lng, lat]",
        get_radius=65,
        radius_min_pixels=3,
        get_fill_color=rgb_color,
        get_line_color=[30, 30, 30],
        line_width_min_pixels=1,
        pickable=True,
        tooltip=True,
    )
    return layer, plot

def construir_capa_aristas(df_aristas: pd.DataFrame, df_nodos: pd.DataFrame, rgb_color):
    if df_aristas.empty:
        return None, []

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
        key_lat = f"lat_{etiqueta}_{nombre_sel}"
        key_lon = f"lon_{etiqueta}_{nombre_sel}"

        col1, col2 = st.columns(2)
        with col1:
            lat_txt = st.text_input(
                f"Lat ({etiqueta})",
                value="" if pd.isna(fila_sel["lat"]) else str(fila_sel["lat"]),
                key=key_lat,
            )
        with col2:
            lon_txt = st.text_input(
                f"Lon ({etiqueta})",
                value="" if pd.isna(fila_sel["lon"]) else str(fila_sel["lon"]),
                key=key_lon,
            )
        if st.button(f"Guardar coords de {etiqueta}"):
            try:
                lat = float(str(lat_txt).replace(",", "."))
                lon = float(str(lon_txt).replace(",", "."))
                st.session_state.nodos_mem.loc[
                    nodos["nombre"] == nombre_sel, ["lat", "lon"]
                ] = [lat, lon]
                st.success(f"Coordenadas guardadas para {nombre_sel}: ({lat}, {lon})")
            except ValueError:
                st.error("Lat/Lon inválidos. Usa números (ej. 14.9712 y -91.7815)")

    editor_de_coords("Origen", origen_nombre)
    editor_de_coords("Destino", destino_nombre)

# ---------------- LÓGICA DE RUTA ----------------
# 1) conseguir ids del origen/destino
origen_id = nombre_a_id(nodos, origen_nombre)
destino_id = nombre_a_id(nodos, destino_nombre)

# 2) armar grafo no dirigido y sacar camino más corto por BFS
graph = build_graph_edges(aristas_raw)
path_ids = []
if origen_id and destino_id:
    path_ids = bfs_shortest_path(graph, origen_id, destino_id)

# 3) calcular métricas de paradas aunque NO haya coords
if path_ids:
    paradas_totales = len(path_ids)
    paradas_intermedias = max(0, paradas_totales - 2)
else:
    # si no hay camino en el grafo, dejamos 2 por defecto
    paradas_totales = 2
    paradas_intermedias = 0

# 4) estimar costo en minutos usando grafo si no tenemos OSRM
# asumimos 3 min por salto (puedes ajustar)
if path_ids and len(path_ids) > 1:
    hops = len(path_ids) - 1
    estimado_min_por_grafo = hops * 3.0
else:
    estimado_min_por_grafo = None

# 5) intentar ruta real con coords si ambos nodos tienen lat/lon
dist_km = None
dur_min = None
ruta_path_lonlat_osrm = None

if origen_id and destino_id:
    fila_o = nodos.loc[nodos["id"] == origen_id].iloc[0]
    fila_d = nodos.loc[nodos["id"] == destino_id].iloc[0]

    if tiene_coords(fila_o) and tiene_coords(fila_d):
        o_lat, o_lon = float(fila_o["lat"]), float(fila_o["lon"])
        d_lat, d_lon = float(fila_d["lat"]), float(fila_d["lon"])

        ruta_path_lonlat_osrm, dist_km, dur_min = (None, None, None)
        if usar_osrm:
            ruta_path_lonlat_osrm, dist_km, dur_min = osrm_route(o_lat, o_lon, d_lat, d_lon)

        if ruta_path_lonlat_osrm is None:
            # línea recta entre origen y destino
            dist_km = haversine_km(o_lat, o_lon, d_lat, d_lon)
            vel_kmh = 30.0
            dur_min = (dist_km / vel_kmh) * 60.0
            ruta_path_lonlat_osrm = [[o_lon, o_lat], [d_lon, d_lat]]

else:
    fila_o = pd.Series({"lat": None, "lon": None})
    fila_d = pd.Series({"lat": None, "lon": None})

# 6) si NO pudimos generar ruta_path_lonlat_osrm,
# intentamos trazar la ruta del grafo (path_ids) usando coordenadas de cada parada
ruta_path_lonlat_grafo = []
if not ruta_path_lonlat_osrm:
    ruta_path_lonlat_grafo = ids_a_polyline_lonlat(nodos, path_ids)

# ---------------- PREPARAR CAPAS MAPA ----------------
RGB_NODES = hex_to_rgb(st.session_state.get("col_nodes", "#FF007F")) if "col_nodes" in st.session_state else hex_to_rgb("#FF007F")
RGB_EDGES = hex_to_rgb(st.session_state.get("col_edges", "#FFA500")) if "col_edges" in st.session_state else hex_to_rgb("#FFA500")
RGB_PATH  = hex_to_rgb(st.session_state.get("col_path" , "#007AFF")) if "col_path"  in st.session_state else hex_to_rgb("#007AFF")

# pero usamos los colores actuales del sidebar (más reciente)
RGB_NODES = hex_to_rgb(st.session_state.get("Color nodos", "#FF007F")) if False else hex_to_rgb(st.sidebar.color_picker if False else "#FF007F")
# ↑ truco feo, ignore; mejor simplemente:
RGB_NODES = hex_to_rgb(st.session_state.get("_last_nodes_color", "#FF007F"))
RGB_EDGES = hex_to_rgb(st.session_state.get("_last_edges_color", "#FFA500"))
RGB_PATH  = hex_to_rgb(st.session_state.get("_last_path_color",  "#007AFF"))
# ok, pausa: lo de arriba es demasiado hacky, vamos más limpio:

# REDEFINIMOS RGB_* usando directamente las variables locales del sidebar,
# que sí existen en este scope:
RGB_NODES = hex_to_rgb(col_nodes)
RGB_EDGES = hex_to_rgb(col_edges)
RGB_PATH  = hex_to_rgb(col_path)

layers = []
all_coords_for_view = []

# capa nodos
nodes_layer, nodos_plot = construir_capa_nodos(nodos, RGB_NODES)
if show_nodes and nodes_layer is not None:
    layers.append(nodes_layer)
    all_coords_for_view.extend(nodos_plot[["lng", "lat"]].values.tolist())

# capa aristas
edges_layer, paths_edges = construir_capa_aristas(aristas_raw, nodos, RGB_EDGES)
if show_edges and edges_layer is not None:
    layers.append(edges_layer)
    for seg in paths_edges:
        all_coords_for_view.extend(seg["path"])

# capa ruta (preferencia: OSRM/recta -> si no, grafo dibujado con coordenadas conocidas)
ruta_final_lonlat = []
if ruta_path_lonlat_osrm:
    ruta_final_lonlat = ruta_path_lonlat_osrm
elif ruta_path_lonlat_grafo:
    ruta_final_lonlat = ruta_path_lonlat_grafo

ruta_layer = construir_capa_ruta(ruta_final_lonlat, RGB_PATH)
if ruta_layer is not None:
    layers.append(ruta_layer)
    all_coords_for_view.extend(ruta_final_lonlat)

# viewstate automático
if all_coords_for_view:
    view_state = fit_view_from_lonlat(all_coords_for_view, extra_zoom_out=0.4)
else:
    view_state = pdk.ViewState(latitude=14.965, longitude=-91.79, zoom=13, pitch=0, bearing=0)

# ---------------- MÉTRICAS PARA EL RESUMEN ----------------
criterio_texto = "⏱ tiempo mín"
grafo_texto = "No dirigido"

# Distancia / tiempo real si lo logramos
if dist_km is not None:
    distancia_txt = f"{dist_km:.2f} km"
else:
    distancia_txt = "—"

if dur_min is not None:
    tiempo_txt = f"{dur_min:.1f} min"
    costo_total_txt = f"{dur_min:.1f} min"
else:
    # fallback: usar estimación por saltos de grafo
    if estimado_min_por_grafo is not None:
        tiempo_txt = f"~{estimado_min_por_grafo:.1f} min"
        costo_total_txt = f"~{estimado_min_por_grafo:.1f} min"
    else:
        tiempo_txt = "—"
        costo_total_txt = "—"

# ---------------- LAYOUT PRINCIPAL ----------------
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

# info si aún no hay coords ni camino trazable en mapa
if not ruta_final_lonlat:
    st.info(
        "No se pudo dibujar la ruta en el mapa (falta lat/lon de alguna parada). "
        "Pero el resumen ya está calculado usando el grafo."
    )
