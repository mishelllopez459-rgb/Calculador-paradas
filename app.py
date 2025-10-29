import math
import heapq
import requests
import pandas as pd
import pydeck as pdk
import streamlit as st
from collections import defaultdict

# =========================
# CONFIG STREAMLIT
# =========================
st.set_page_config(page_title="Rutas San Marcos", layout="wide")

# =========================
# CARGA DE CSV
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

# normalizar strings
for col in ["id", "nombre"]:
    nodos_raw[col] = nodos_raw[col].astype(str).str.strip()

nodos = nodos_raw[["id", "nombre", "lat", "lon"]].copy()

# asegurar lugares básicos aunque no tengan coords (para que aparezcan en los selects)
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

    nuevos = []
    for nm in nombres:
        if nm.lower() not in existentes:
            nuevos.append({
                "id": nuevo_id(),
                "nombre": nm,
                "lat": None,
                "lon": None,
            })
    if nuevos:
        df = pd.concat([df, pd.DataFrame(nuevos)], ignore_index=True)

    df["id"] = df["id"].astype(str).str.strip()
    df["nombre"] = df["nombre"].astype(str).str.strip()
    return df

nodos = asegurar_lugares(nodos, LUGARES_NUEVOS)

# mantenemos nodos en la sesión por si luego quieres editor de coords
if "nodos_mem" not in st.session_state:
    st.session_state.nodos_mem = nodos.copy()
nodos = st.session_state.nodos_mem

# =========================
# HELPERS GEO
# =========================
VEL_KMH = 30.0  # velocidad estimada bus urbano

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
    Pide ruta real por calle (OSRM).
    Devuelve: (coords_lonlat, dist_km, dur_min)
    coords_lonlat = [[lon,lat], ...]
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
    Encuadra todos los puntos dados.
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

    # pydeck puede dar dict o ViewState
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
    graph[u] = [(v, t_min, d_km), ...]
    weights[(u,v)] = {"time": t_min, "dist": d_km}

    Reglas:
    - Si 'peso' existe en la arista => lo usamos como tiempo_min (en minutos).
    - Si no hay 'peso', calculamos distancia haversine y de ahí tiempo usando VEL_KMH.
    - Si faltan coords => fallback distancia 0.6 km / tiempo 3.0 min.
    """
    graph = defaultdict(list)
    weights = {}

    if "origen" not in aristas_df.columns:
        aristas_df["origen"] = ""
    if "destino" not in aristas_df.columns:
        aristas_df["destino"] = ""

    idx = nodos_df.set_index("id")[["lat","lon"]]

    def add_edge(u, v, peso):
        # distancia entre nodos
        d_km = None
        if u in idx.index and v in idx.index:
            la, lo  = idx.loc[u, ["lat","lon"]]
            lb, lo2 = idx.loc[v, ["lat","lon"]]
            if pd.notna(la) and pd.notna(lo) and pd.notna(lb) and pd.notna(lo2):
                d_km = haversine_km(float(la), float(lo), float(lb), float(lo2))

        if d_km is None:
            d_km = 0.6  # fallback en km

        # tiempo estimado
        if pd.notna(peso):
            t_min = float(peso)
        else:
            t_min = (d_km / VEL_KMH) * 60.0
            if t_min <= 0:
                t_min = 3.0  # fallback en minutos

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
    Calcula camino mínimo entre start y goal.
    use: "time" (minutos) o "dist" (km)
    graph[u] = [(v, t_min, d_km), ...]
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

    # reconstruir el camino
    path = [goal]
    while path[-1] != start:
        path.append(prev[path[-1]])
    path.reverse()
    return path, dist_cost[goal]

# =========================
# POLYLINES
# =========================
def ids_a_polyline_lonlat(nodos_df, ids):
    """
    Devuelve [[lon,lat], ...] de los nodos del camino que sí tienen coords.
    (si falta alguno intermedio igual seguimos)
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
    Distancia total sobre una polilínea, sumando haversine tramo a tramo.
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
# CAPAS PARA EL MAPA
# =========================
def capa_nodos(df_nodos, rgb):
    plot = df_nodos.dropna(subset=["lat","lon"]).copy()
    if plot.empty:
        return None, pd.DataFrame()
    plot.rename(columns={"lon":"lng"}, inplace=True)
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=plot,
        get_position="[lng, lat]",
        get_radius=65,
        radius_min_pixels=3,
        get_fill_color=rgb,
        get_line_color=[30,30,30],
        line_width_min_pixels=1,
        pickable=True,
    )
    return layer, plot

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
                    "destino": v,
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
# SIDEBAR (UI)
# =========================
with st.sidebar:
    st.markdown("### Parámetros")

    dirigido_flag = st.checkbox(
        "Tramos unidireccionales (grafo dirigido)",
        value=False,
        help="Si está activo, las aristas solo cuentan en la dirección origen→destino."
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
    color_nodes = st.color_picker("Nodos (puntos)", "#FF008C")          # rosa fuerte
    color_edges = st.color_picker("Red general (aristas)", "#FFFFFF")   # blanco
    color_path  = st.color_picker("Ruta seleccionada", "#00B2FF")       # celeste
    usar_osrm   = st.toggle("Ruta real por calle (OSRM)", value=True)

    st.button("Calcular ruta")  # decorativo, el cálculo corre igual

# =========================
# CALCULAR LA RUTA
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

# sumar métricas reales de la ruta elegida (tiempo total y distancia total)
if len(path_ids) >= 2:
    total_time = 0.0
    total_dist = 0.0
    for i in range(len(path_ids) - 1):
        u = path_ids[i]
        v = path_ids[i + 1]
        w = weights.get((u, v), {"time": 3.0, "dist": 0.6})
        total_time += float(w["time"])
        total_dist += float(w["dist"])
else:
    total_time = None
    total_dist = None

# filas origen/destino para OSRM y fallback recta
if origen_id in set(nodos["id"]):
    fila_o = nodos.loc[nodos["id"] == origen_id].iloc[0]
else:
    fila_o = pd.Series({"lat": None, "lon": None})
if destino_id in set(nodos["id"]):
    fila_d = nodos.loc[nodos["id"] == destino_id].iloc[0]
else:
    fila_d = pd.Series({"lat": None, "lon": None})

# polyline por paradas del grafo
ruta_por_paradas = ids_a_polyline_lonlat(nodos, path_ids)

# polyline recta O->D por si no hay coords intermedias
ruta_recta = []
if tiene_coords(fila_o) and tiene_coords(fila_d):
    ruta_recta = [
        [float(fila_o["lon"]), float(fila_o["lat"])],
        [float(fila_d["lon"]), float(fila_d["lat"])],
    ]
    # si no había totales, calcúlalos con la recta
    if total_time is None or total_dist is None:
        line_km = haversine_km(
            float(fila_o["lat"]), float(fila_o["lon"]),
            float(fila_d["lat"]), float(fila_d["lon"]),
        )
        total_dist = line_km
        total_time = (line_km / VEL_KMH) * 60.0 if line_km is not None else None

# intento OSRM (ruta real por calle)
ruta_osrm, dist_osrm, dur_osrm = (None, None, None)
if usar_osrm and tiene_coords(fila_o) and tiene_coords(fila_d):
    ruta_osrm, dist_osrm, dur_osrm = osrm_route(
        float(fila_o["lat"]), float(fila_o["lon"]),
        float(fila_d["lat"]), float(fila_d["lon"]),
    )
    # si OSRM devolvió buenos valores, usamos esos como costo/distancia mostrada
    if ruta_osrm and dist_osrm is not None and dur_osrm is not None:
        total_dist = dist_osrm
        total_time = dur_osrm

# polyline final a dibujar
if ruta_osrm:
    ruta_final = ruta_osrm
elif ruta_por_paradas and len(ruta_por_paradas) >= 2:
    ruta_final = ruta_por_paradas
else:
    ruta_final = ruta_recta

# =========================
# CAPAS MAPA
# =========================
RGB_NODES = hex_to_rgb(color_nodes)
RGB_EDGES = hex_to_rgb(color_edges)
RGB_PATH  = hex_to_rgb(color_path)

layers = []
all_coords = []

# nodos (puntos)
layer_nodes, nodos_plot = capa_nodos(nodos, RGB_NODES)
if layer_nodes is not None:
    layers.append(layer_nodes)
    if not nodos_plot.empty:
        all_coords.extend(nodos_plot[["lng","lat"]].values.tolist())

# red general (todas las aristas)
layer_edges, edges_paths = capa_aristas(aristas_raw, nodos, RGB_EDGES, width_px=3)
if layer_edges is not None:
    layers.append(layer_edges)
    for seg in edges_paths:
        all_coords.extend(seg["path"])

# ruta seleccionada (arriba, más gruesa)
layer_route = capa_ruta(ruta_final, RGB_PATH, width_px=6)
if layer_route is not None:
    layers.append(layer_route)
    all_coords.extend(ruta_final)

# vista inicial del mapa
view_state = fit_view_from_lonlat(all_coords, extra_zoom_out=0.4)

# =========================
# UI FINAL
# =========================
st.title("🚌 Calculador de paradas y ruta óptima — San Marcos")

st.subheader("Resumen")
st.markdown(f"**Origen:** {origen_nombre}")
st.markdown(f"**Destino:** {destino_nombre}")
st.markdown(f"**Criterio:** `{criterio_radio}`")
st.markdown(f"**Grafo:** {'Dirigido' if dirigido_flag else 'No dirigido'}")

paradas_tot = len(path_ids) if path_ids else 2
paradas_int = max(0, paradas_tot - 2)

st.markdown(f"**Paradas (incluye origen y destino):** {paradas_tot}")
st.markdown(f"**Paradas intermedias:** {paradas_int}")

if criterio_radio == "tiempo_min":
    # costo principal = tiempo
    if total_time is not None:
        st.markdown(f"**Costo total (tiempo_min):** {total_time:.2f} min")
    else:
        st.markdown(f"**Costo total (tiempo_min):** —")

    if total_dist is not None:
        st.markdown(f"**Distancia aprox.:** {total_dist:.2f} km")
    else:
        st.markdown(f"**Distancia aprox.:** —")
else:
    # costo principal = distancia
    if total_dist is not None:
        st.markdown(f"**Costo total (distancia_km):** {total_dist:.2f} km")
    else:
        st.markdown(f"**Costo total (distancia_km):** —")

    if total_time is not None:
        st.markdown(f"**Tiempo aprox.:** {total_time:.2f} min")
    else:
        st.markdown(f"**Tiempo aprox.:** —")

# botón para bajar CSV
if ruta_final and len(ruta_final) >= 2:
    export_df = pd.DataFrame(ruta_final, columns=["lon","lat"])
    st.download_button(
        "📥 Descargar ruta (CSV)",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name="ruta.csv",
        mime="text/csv",
    )

# mapa grandote
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
