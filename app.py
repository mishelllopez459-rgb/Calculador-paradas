import math
import requests
import pandas as pd
import pydeck as pdk
import streamlit as st
from collections import defaultdict
import heapq

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Rutas San Marcos", layout="wide")

# =========================
# CARGA CSV
# =========================
def cargar_csv(path, cols_min):
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.DataFrame(columns=cols_min)
    for c in cols_min:
        if c not in df.columns:
            df[c] = None
    return df

nodos_raw   = cargar_csv("nodos.csv",   ["id", "nombre", "lat", "lon"])
aristas_raw = cargar_csv("aristas.csv", ["origen", "destino", "peso"])

# normalizar texto en nodos
for col in ["id", "nombre"]:
    nodos_raw[col] = nodos_raw[col].astype(str).str.strip()

nodos = nodos_raw[["id", "nombre", "lat", "lon"]].copy()

# si faltan lugares conocidos, agrégalos
LUGARES_NUEVOS = [
    "Parque Central","Catedral","Terminal de Buses","Hospital Regional",
    "Cancha Los Angeles","Cancha Sintetica Golazo","Aeropuerto Nacional",
    "Iglesia Candelero de Oro","Centro de Salud","Megapaca",
    "CANICA (Casa de los Niños)","Aldea San Rafael Soche","Pollo Campero",
    "INTECAP San Marcos","Salón Quetzal","SAT San Marcos","Bazar Chino"
]
def asegurar_lugares(df, nombres):
    existentes = set(df["nombre"].astype(str).str.lower())
    usados     = set(df["id"].astype(str))
    def nuevo_id():
        i = 1
        while True:
            cand = f"L{i}"
            if cand not in usados:
                usados.add(cand)
                return cand
            i += 1
    filas = []
    for nm in nombres:
        if nm.lower() not in existentes:
            filas.append({"id": nuevo_id(), "nombre": nm, "lat": None, "lon": None})
    if filas:
        df = pd.concat([df, pd.DataFrame(filas)], ignore_index=True)
    df["id"] = df["id"].astype(str).str.strip()
    df["nombre"] = df["nombre"].astype(str).str.strip()
    return df

nodos = asegurar_lugares(nodos, LUGARES_NUEVOS)

# guardamos en sesión por si luego editas coords en otra versión
if "nodos_mem" not in st.session_state:
    st.session_state.nodos_mem = nodos.copy()
nodos = st.session_state.nodos_mem

# =========================
# HELPERS GEO
# =========================
VEL_KMH = 30.0  # velocidad promedio asumida

def hex_to_rgb(h: str):
    h = h.lstrip("#")
    return [int(h[i:i+2], 16) for i in (0,2,4)]

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
    Ruta real por calle entre origen y destino (OSRM público).
    Devuelve (lista [[lon,lat]...], dist_km, dur_min)
    """
    url = (
        "https://router.project-osrm.org/route/v1/driving/"
        f"{o_lon},{o_lat};{d_lon},{d_lat}?overview=full&geometries=geojson"
    )
    try:
        r = requests.get(url, timeout=12)
        r.raise_for_status()
        data = r.json()
        coords  = data["routes"][0]["geometry"]["coordinates"]
        dist_km = data["routes"][0]["distance"] / 1000.0
        dur_min = data["routes"][0]["duration"] / 60.0
        return coords, dist_km, dur_min
    except Exception:
        return None, None, None

def fit_view_from_lonlat(coords_lonlat: list, extra_zoom_out: float = 0.4):
    """
    Centra y hace zoom para que se vea toda la red/ruta.
    """
    if not coords_lonlat:
        return pdk.ViewState(
            latitude=14.965,
            longitude=-91.79,
            zoom=13,
            pitch=0,
            bearing=0,
        )
    df_bounds = pd.DataFrame(coords_lonlat, columns=["lon","lat"])
    raw_view = pdk.data_utils.compute_view(df_bounds[["lon","lat"]])
    if isinstance(raw_view, dict):
        lat_c = raw_view.get("latitude", 14.965)
        lon_c = raw_view.get("longitude", -91.79)
        zoom_v = raw_view.get("zoom", 13)
    else:
        lat_c = getattr(raw_view, "latitude", 14.965)
        lon_c = getattr(raw_view, "longitude", -91.79)
        zoom_v = getattr(raw_view, "zoom", 13)
    try:
        zoom_v = max(1, float(zoom_v) - float(extra_zoom_out))
    except Exception:
        zoom_v = 13
    return pdk.ViewState(
        latitude=lat_c,
        longitude=lon_c,
        zoom=zoom_v,
        pitch=0,
        bearing=0,
    )

# =========================
# GRAFO CON PESOS
# =========================
def build_weighted_graph(aristas_df: pd.DataFrame, nodos_df: pd.DataFrame, dirigido: bool):
    """
    graph[u] = [(v, t_min, d_km)]
    weights[(u,v)] = {"time": t_min, "dist": d_km}

    Reglas:
    - Si aristas.csv trae 'peso', lo tomamos como tiempo en minutos.
    - Si no trae peso, calculamos distancia entre nodos y estimamos tiempo con VEL_KMH.
    - Si un nodo no tiene coords => asumimos ~0.6 km / ~3 min.
    """
    from collections import defaultdict
    graph = defaultdict(list)
    weights = {}

    if "origen" not in aristas_df.columns:
        aristas_df["origen"] = ""
    if "destino" not in aristas_df.columns:
        aristas_df["destino"] = ""

    idx = nodos_df.set_index("id")[["lat","lon"]]

    def add_edge(u, v, peso):
        # distancia
        d_km = None
        if u in idx.index and v in idx.index:
            la, lo  = idx.loc[u, ["lat","lon"]]
            lb, lo2 = idx.loc[v, ["lat","lon"]]
            if pd.notna(la) and pd.notna(lo) and pd.notna(lb) and pd.notna(lo2):
                d_km = haversine_km(float(la), float(lo), float(lb), float(lo2))
        if d_km is None:
            d_km = 0.6  # fallback si no hay coords

        # tiempo
        if pd.notna(peso):
            t_min = float(peso)
        else:
            t_min = (d_km / VEL_KMH) * 60.0
            if t_min <= 0:
                t_min = 3.0  # fallback

        graph[u].append((v, t_min, d_km))
        weights[(u, v)] = {"time": t_min, "dist": d_km}

    for _, r in aristas_df.iterrows():
        u = str(r.get("origen","")).strip()
        v = str(r.get("destino","")).strip()
        if not u or not v:
            continue
        peso = r.get("peso", None)
        add_edge(u, v, peso)
        if not dirigido:
            add_edge(v, u, peso)

    return graph, weights

# =========================
# DIJKSTRA
# =========================
def dijkstra(graph, start, goal, use="time"):
    """
    use = "time"  -> minimizar minutos
    use = "dist"  -> minimizar km
    graph[u] = [(v, t_min, d_km)]
    """
    if start is None or goal is None:
        return [], float("inf")
    if start == goal:
        return [start], 0.0

    dist_cost = {start: 0.0}
    prev = {}
    pq = [(0.0, start)]
    visited = set()

    while pq:
        cost_u, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)

        if u == goal:
            break

        for v, t_min, d_km in graph.get(u, []):
            w = t_min if use == "time" else d_km
            if w is None:
                w = float("inf")
            new_cost = cost_u + w
            if v not in dist_cost or new_cost < dist_cost[v]:
                dist_cost[v] = new_cost
                prev[v] = u
                heapq.heappush(pq, (new_cost, v))

    if goal not in dist_cost or dist_cost[goal] == float("inf"):
        return [], float("inf")

    # reconstruir
    path = [goal]
    while path[-1] != start:
        path.append(prev[path[-1]])
    path.reverse()
    return path, dist_cost[goal]

# =========================
# UTILIDADES RUTA
# =========================
def ids_a_polyline_lonlat(nodos_df, ids):
    """
    Devuelve [[lon,lat], ...] con las coords de cada parada del camino (las que tengan coords).
    Aunque falte alguna, usamos las que sí tienen.
    """
    pts = []
    idx = nodos_df.set_index("id")
    for nid in ids:
        if nid in idx.index:
            lat = idx.loc[nid, "lat"]
            lon = idx.loc[nid, "lon"]
            if pd.notna(lat) and pd.notna(lon):
                pts.append([float(lon), float(lat)])
    return pts

def distancia_km_sobre_polyline(poly_lonlat):
    """
    Suma haversine entre cada tramo consecutivo de la polilínea.
    """
    if not poly_lonlat or len(poly_lonlat) < 2:
        return 0.0
    total = 0.0
    for i in range(len(poly_lonlat)-1):
        lon1, lat1 = poly_lonlat[i]
        lon2, lat2 = poly_lonlat[i+1]
        total += haversine_km(lat1, lon1, lat2, lon2)
    return total

# =========================
# CAPAS MAPA
# =========================
def capa_nodos(df_nodos, rgb):
    pts = df_nodos.dropna(subset=["lat","lon"]).copy()
    if pts.empty:
        return None, pd.DataFrame()
    pts.rename(columns={"lon":"lng"}, inplace=True)
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=pts,
        get_position="[lng, lat]",
        get_radius=65,
        radius_min_pixels=3,
        get_fill_color=rgb,
        get_line_color=[30,30,30],
        line_width_min_pixels=1,
        pickable=True,
    )
    return layer, pts

def capa_aristas(aristas_df, nodos_df, rgb, width_px=3):
    idx = nodos_df.set_index("id")[["lat","lon"]]
    segs = []
    for _, r in aristas_df.iterrows():
        u = str(r.get("origen","")).strip()
        v = str(r.get("destino","")).strip()
        if u in idx.index and v in idx.index:
            la, lo  = idx.loc[u, ["lat","lon"]]
            lb, lo2 = idx.loc[v, ["lat","lon"]]
            if pd.notna(la) and pd.notna(lo) and pd.notna(lb) and pd.notna(lo2):
                segs.append({
                    "path": [[lo,la],[lo2,lb]],
                    "origen": u,
                    "destino": v
                })
    if not segs:
        return None, []
    layer = pdk.Layer(
        "PathLayer",
        data=segs,
        get_path="path",
        get_width=width_px,
        width_scale=8,
        get_color=rgb,
        pickable=True,
    )
    return layer, segs

def capa_ruta(poly_lonlat, rgb, width_px=6):
    if not poly_lonlat or len(poly_lonlat) < 2:
        return None
    layer = pdk.Layer(
        "PathLayer",
        data=[{"path": poly_lonlat}],
        get_path="path",
        get_width=width_px,
        width_scale=8,
        get_color=rgb,
        pickable=False,
    )
    return layer

# =========================
# SIDEBAR (tu estilo)
# =========================
with st.sidebar:
    st.markdown("### Parámetros")

    dirigido_flag = st.checkbox(
        "Tramos unidireccionales (grafo dirigido)",
        value=False,
        help="Si está activo, las aristas solo valen en la dirección origen→destino."
    )

    origen_nombre = st.selectbox("Origen", sorted(nodos["nombre"]))
    destino_nombre = st.selectbox(
        "Destino",
        sorted(nodos["nombre"]),
        index=1 if len(nodos) > 1 else 0,
    )

    criterio_radio = st.radio(
        "Optimizar por",
        ["tiempo_min", "distancia_km"],
        index=0,
        help="Qué quieres minimizar: tiempo (min) o distancia (km)."
    )

    st.markdown("### Colores")
    color_nodes = st.color_picker("Nodos (puntos)", "#FF008C")     # rosa fuerte
    color_edges = st.color_picker("Red general (aristas)", "#FFFFFF")  # blanco
    color_path  = st.color_picker("Ruta seleccionada", "#00B2FF")  # celeste
    usar_osrm   = st.toggle("Ruta real por calle (OSRM)", value=True)

    st.button("Calcular ruta")  # decorativo, recálculo es automático

# =========================
# CALCULAR RUTA
# =========================
def nombre_a_id(nodos_df, nombre):
    m = nodos_df.loc[nodos_df["nombre"] == nombre]
    if m.empty:
        return None
    return str(m.iloc[0]["id"])

origen_id  = nombre_a_id(nodos, origen_nombre)
destino_id = nombre_a_id(nodos, destino_nombre)

graph, weights = build_weighted_graph(aristas_raw, nodos, dirigido_flag)

use_metric = "time" if criterio_radio == "tiempo_min" else "dist"
path_ids, _cost_used = dijkstra(graph, origen_id, destino_id, use=use_metric)

# sumar costo real total_time / total_dist desde weights
total_time = 0.0
total_dist = 0.0
if len(path_ids) >= 2:
    for i in range(len(path_ids)-1):
        u = path_ids[i]
        v = path_ids[i+1]
        w = weights.get((u, v), {"time": 3.0, "dist": 0.6})
        total_time += float(w["time"])
        total_dist += float(w["dist"])
else:
    # sin ruta válida, dejamos 0 para que abajo podamos mostrar "—"
    total_time = None
    total_dist = None

# filas origen/destino para OSRM / fallback recta
if origen_id in set(nodos["id"]):
    fila_o = nodos.loc[nodos["id"] == origen_id].iloc[0]
else:
    fila_o = pd.Series({"lat": None, "lon": None})

if destino_id in set(nodos["id"]):
    fila_d = nodos.loc[nodos["id"] == destino_id].iloc[0]
else:
    fila_d = pd.Series({"lat": None, "lon": None})

ruta_por_paradas = ids_a_polyline_lonlat(nodos, path_ids)

ruta_recta = []
if tiene_coords(fila_o) and tiene_coords(fila_d):
    ruta_recta = [
        [float(fila_o["lon"]), float(fila_o["lat"])],
        [float(fila_d["lon"]), float(fila_d["lat"])],
    ]
    # si no había pesos (total_time None), calculamos aprox con recta
    if total_time is None or total_dist is None:
        line_km = haversine_km(
            float(fila_o["lat"]), float(fila_o["lon"]),
            float(fila_d["lat"]), float(fila_d["lon"])
        )
        total_dist = line_km
        total_time = (line_km / VEL_KMH) * 60.0 if line_km is not None else None

ruta_osrm, dist_osrm, dur_osrm = (None, None, None)
if usar_osrm and tiene_coords(fila_o) and tiene_coords(fila_d):
    ruta_osrm, dist_osrm, dur_osrm = osrm_route(
        float(fila_o["lat"]), float
