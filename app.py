import math
import requests
import pandas as pd
import pydeck as pdk
import streamlit as st
from collections import defaultdict
import heapq

# =========================
# CONFIG / TÍTULO
# =========================
st.set_page_config(page_title="Rutas San Marcos", layout="wide")
st.title("🚌 Calculador de paradas y ruta óptima — San Marcos")

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

# =========================
# ASEGURAR LUGARES (por si faltan algunos nombres fijos)
# =========================
LUGARES_NUEVOS = [
    "Parque Central","Catedral","Terminal de Buses","Hospital Regional",
    "Cancha Los Angeles","Cancha Sintetica Golazo","Aeropuerto Nacional",
    "Iglesia Candelero de Oro","Centro de Salud","Megapaca",
    "CANICA (Casa de los Niños)","Aldea San Rafael Soche","Pollo Campero",
    "INTECAP San Marcos","Salón Quetzal","SAT San Marcos","Bazar Chino"
]

def asegurar_lugares(df, nombres):
    existentes = set(df["nombre"].astype(str).str.lower())
    usados = set(df["id"].astype(str))

    def nuevo_id():
        i = 1
        while True:
            cand = f"L{i}"
            if cand not in usados:
                usados.add(cand)
                return cand
            i += 1

    nuevas = []
    for nm in nombres:
        if nm.lower() not in existentes:
            nuevas.append({
                "id": nuevo_id(),
                "nombre": nm,
                "lat": None,
                "lon": None
            })

    if nuevas:
        df = pd.concat([df, pd.DataFrame(nuevas)], ignore_index=True)

    df["id"] = df["id"].astype(str).str.strip()
    df["nombre"] = df["nombre"].astype(str).str.strip()
    return df

nodos = asegurar_lugares(nodos, LUGARES_NUEVOS)

# guardamos nodos en sesión (por si luego quieres editar coords en otra versión)
if "nodos_mem" not in st.session_state:
    st.session_state.nodos_mem = nodos.copy()
nodos = st.session_state.nodos_mem

# =========================
# HELPERS GEO
# =========================
VEL_KMH = 30.0  # velocidad promedio para estimar tiempo

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
    Devuelve (coords_lonlat, dist_km, dur_min) o (None,None,None) si falla.
    coords_lonlat = [[lon,lat], ...] siguiendo calles reales.
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

def fit_view_from_lonlat(coords_lonlat: list, extra_zoom_out: float = 0.35):
    """
    Calcula una vista que encuadre todas las coords y le baja un poquito el zoom
    para que no corte los bordes.
    """
    if not coords_lonlat:
        return pdk.ViewState(latitude=14.965, longitude=-91.79, zoom=13, pitch=0, bearing=0)

    df_bounds = pd.DataFrame(coords_lonlat, columns=["lon","lat"])
    raw_view = pdk.data_utils.compute_view(df_bounds[["lon","lat"]])

    if isinstance(raw_view, dict):
        lat_center = raw_view.get("latitude", 14.965)
        lon_center = raw_view.get("longitude", -91.79)
        zoom_val   = raw_view.get("zoom", 13)
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

# =========================
# GRAFO CON PESOS
# =========================
def build_weighted_graph(aristas_df: pd.DataFrame, nodos_df: pd.DataFrame, dirigido: bool):
    """
    graph[u] = [(v, tiempo_min, distancia_km), ...]
    weights[(u,v)] = {"time": t_min, "dist": d_km}

    Reglas:
    - si hay 'peso' en aristas.csv => usamos eso como tiempo_min (costo en min)
    - si no hay peso pero hay coords en ambos nodos => distancia Haversine
      y tiempo = distancia / vel_promedio
    - si no hay coords => distancia ~0.6 km, tiempo ~3.0 min (fallback)
    """
    graph = defaultdict(list)
    weights = {}

    idx = nodos_df.set_index("id")[["lat","lon"]]

    def add_edge(u, v, peso_original):
        # distancia
        d_km = None
        if u in idx.index and v in idx.index:
            la, lo  = idx.loc[u, ["lat","lon"]]
            lb, lo2 = idx.loc[v, ["lat","lon"]]
            if pd.notna(la) and pd.notna(lo) and pd.notna(lb) and pd.notna(lo2):
                d_km = haversine_km(float(la), float(lo), float(lb), float(lo2))
        if d_km is None:
            d_km = 0.6  # fallback

        # tiempo
        if pd.notna(peso_original):
            try:
                t_min = float(peso_original)
            except:
                t_min = (d_km / VEL_KMH) * 60.0
        else:
            t_min = (d_km / VEL_KMH) * 60.0 if d_km else 3.0
            if not t_min or t_min <= 0:
                t_min = 3.0

        graph[u].append((v, t_min, d_km))
        weights[(u, v)] = {"time": t_min, "dist": d_km}

    # construir
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
    start -> goal minimizando:
      - "time": minutos totales
      - "dist": km totales
    graph es {u: [(v, tiempo_min, distancia_km), ...]}
    """
    if start is None or goal is None:
        return [], float("inf")
    if start == goal:
        return [start], 0.0

    # costo acumulado
    dist_cost = defaultdict(lambda: float("inf"))
    prev = {}
    dist_cost[start] = 0.0

    pq = [(0.0, start)]
    visit = set()

    while pq:
        cost, u = heapq.heappop(pq)
        if u in visit:
            continue
        visit.add(u)

        if u == goal:
            break

        for v, t_min, d_km in graph.get(u, []):
            w = t_min if use == "time" else d_km
            if w is None:
                continue
            new_cost = cost + w
            if new_cost < dist_cost[v]:
                dist_cost[v] = new_cost
                prev[v] = u
                heapq.heappush(pq, (new_cost, v))

    if goal not in dist_cost or dist_cost[goal] == float("inf"):
        # no hay camino
        return [], float("inf")

    # reconstruir ruta
    path = [goal]
    while path[-1] != start:
        path.append(prev[path[-1]])
    path.reverse()
    return path, dist_cost[goal]

# =========================
# UTILIDADES PARA RUTA
# =========================
def nombre_a_id(nodos_df, nombre):
    m = nodos_df.loc[nodos_df["nombre"] == nombre]
    if m.empty:
        return None
    return str(m.iloc[0]["id"])

def ids_a_polyline_lonlat(nodos_df, ids):
    """
    Devuelve [[lon,lat], ...] SOLO de los nodos que tienen coords.
    Puede quedar vacía si ningún nodo tiene coords.
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
    Distancia total sumando segmento a segmento sobre la polyline.
    """
    if not poly_lonlat or len(poly_lonlat) < 2:
        return None
    tot = 0.0
    for i in range(len(poly_lonlat)-1):
        lon1, lat1 = poly_lonlat[i]
        lon2, lat2 = poly_lonlat[i+1]
        tot += haversine_km(lat1, lon1, lat2, lon2)
    return tot

# =========================
# CAPAS PYDECK
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
    if aristas_df.empty:
        return None, []
    idx = nodos_df.set_index("id")[["lat","lon"]]
    paths = []
    for _, r in aristas_df.iterrows():
        u = str(r.get("origen","")).strip()
        v = str(r.get("destino","")).strip()
        if u in idx.index and v in idx.index:
            la, lo  = idx.loc[u, ["lat","lon"]]
            lb, lo2 = idx.loc[v, ["lat","lon"]]
            if pd.notna(la) and pd.notna(lo) and pd.notna(lb) and pd.notna(lo2):
                paths.append({
                    "path": [[lo,la],[lo2,lb]],
                    "origen": u,
                    "destino": v,
                })
    if not paths:
        return None, []
    layer = pdk.Layer(
        "PathLayer",
        data=paths,
        get_path="path",
        get_width=width_px,
        width_scale=8,
        get_color=rgb,
        pickable=True,
    )
    return layer, paths

def capa_ruta(poly_lonlat, rgb, width_px=8):
    if not poly_lonlat:
        return None
    return pdk.Layer(
        "PathLayer",
        data=[{"path": poly_lonlat}],
        get_path="path",
        get_width=width_px,
        width_scale=8,
        get_color=rgb,
        pickable=False,
    )

# =========================
# SIDEBAR (UI estilo tus capturas)
# =========================
with st.sidebar:
    dirigido_flag = st.checkbox("Tramos unidireccionales (grafo dirigido)", value=False)
    origen_nombre = st.selectbox("Origen", sorted(nodos["nombre"]))
    destino_nombre = st.selectbox(
        "Destino",
        sorted(nodos["nombre"]),
        index=1 if len(nodos) > 1 else 0,
    )

    criterio = st.radio(
        "Optimizar por",
        ["tiempo_min", "distancia_km"],
        index=0
    )

    st.markdown("### Colores")
    col_nodes = st.color_picker("Nodos", "#FF5CA8")         # rosa
    col_edges = st.color_picker("Aristas", "#FFFFFF")       # blanco
    col_path  = st.color_picker("Ruta seleccionada", "#2F80ED")  # azul
    usar_osrm = st.toggle("Ruta real por calle (OSRM)", value=True)

    # botón "Calcular ruta" tipo decorativo (trigger visual)
    st.button("Calcular ruta")

# =========================
# LÓGICA DE RUTA
# =========================
# 1. sacar ids
o_id = nombre_a_id(nodos, origen_nombre)
d_id = nombre_a_id(nodos, destino_nombre)

# 2. grafo con pesos (dirigido o no)
graph, weights = build_weighted_graph(aristas_raw, nodos, dirigido=dirigido_flag)

# 3. correr Dijkstra según criterio elegido
use_metric = "time" if criterio == "tiempo_min" else "dist"
path_ids, costo = dijkstra(graph, o_id, d_id, use=use_metric)

# si no hay camino, fingimos que el camino es [origen,destino] para que igual se dibuje algo / se calcule algo
no_path = (not path_ids or costo == float("inf"))
if no_path and o_id and d_id:
    path_ids = [o_id, d_id]

# 4. sumar pesos reales a lo largo del path_ids
total_time = 0.0
total_dist = 0.0
if len(path_ids) >= 2:
    for i in range(len(path_ids) - 1):
        u, v = path_ids[i], path_ids[i + 1]
        w = weights.get((u, v), {"time": 3.0, "dist": 0.6})
        seg_t = w["time"] if w["time"] and w["time"] > 0 else 3.0
        seg_d = w["dist"] if w["dist"] and w["dist"] > 0 else 0.6
        total_time += seg_t
        total_dist += seg_d

# 5. ruta geométrica a dibujar
#    prioridad: OSRM > polyline por stops > recta O→D

# filas de origen/destino
fila_o = nodos.loc[nodos["id"] == o_id].iloc[0] if o_id in set(nodos["id"]) else pd.Series({"lat":None,"lon":None})
fila_d = nodos.loc[nodos["id"] == d_id].iloc[0] if d_id in set(nodos["id"]) else pd.Series({"lat":None,"lon":None})

ruta_osrm = None
dist_osrm = None
dur_osrm = None
if usar_osrm and tiene_coords(fila_o) and tiene_coords(fila_d):
    ruta_osrm, dist_osrm, dur_osrm = osrm_route(
        float(fila_o["lat"]), float(fila_o["lon"]),
        float(fila_d["lat"]), float(fila_d["lon"])
    )

ruta_grafo = ids_a_polyline_lonlat(nodos, path_ids) if len(path_ids) >= 2 else []

ruta_recta = []
if tiene_coords(fila_o) and tiene_coords(fila_d):
    ruta_recta = [
        [float(fila_o["lon"]), float(fila_o["lat"])],
        [float(fila_d["lon"]), float(fila_d["lat"])],
    ]

# elegimos ruta_final
if ruta_osrm:
    ruta_final = ruta_osrm
elif ruta_grafo and len(ruta_grafo) >= 2:
    ruta_final = ruta_grafo
else:
    ruta_final = ruta_recta

# 6. si NO había camino en el grafo,
#    usamos OSRM / recta / fallback 1 salto para ajustar totales
if no_path or len(path_ids) < 2:
    if ruta_osrm and dist_osrm is not None and dur_osrm is not None:
        total_dist = dist_osrm
        total_time = dur_osrm
    elif ruta_recta:
        (lon1, lat1), (lon2, lat2) = ruta_recta
        hv = haversine_km(lat1, lon1, lat2, lon2)
        total_dist = hv
        total_time = (hv / VEL_KMH) * 60.0
    else:
        # sin coords en absoluto, inventamos 1 salto
        total_dist = 0.6
        total_time = 3.0

# =========================
# CAPAS PARA EL MAPA
# =========================
RGB_NODES = hex_to_rgb(col_nodes)
RGB_EDGES = hex_to_rgb(col_edges)
RGB_PATH  = hex_to_rgb(col_path)

layers = []
all_coords = []

nodes_layer, nodos_plot = capa_nodos(nodos, RGB_NODES)
if nodes_layer is not None:
    layers.append(nodes_layer)
    all_coords += nodos_plot[["lng","lat"]].values.tolist()

edges_layer, edges_paths = capa_aristas(aristas_raw, nodos, RGB_EDGES, width_px=3)
if edges_layer is not None:
    layers.append(edges_layer)
    for seg in edges_paths:
        all_coords += seg["path"]

route_layer = capa_ruta(ruta_final, RGB_PATH, width_px=8)
if route_layer is not None:
    layers.append(route_layer)
    all_coords += ruta_final

view_state = (
    fit_view_from_lonlat(all_coords, extra_zoom_out=0.4)
    if all_coords else
    pdk.ViewState(latitude=14.965, longitude=-91.79, zoom=13, pitch=0, bearing=0)
)

# =========================
# RESUMEN (IGUAL A TU SCREENSHOT 2)
# =========================
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Resumen")

    st.markdown(f"**Origen:** {origen_nombre}")
    st.markdown(f"**Destino:** {destino_nombre}")
    st.markdown(f"**Criterio:** `{criterio}`")
    st.markdown(f"**Grafo:** {'Dirigido' if dirigido_flag else 'No dirigido'}")

    paradas_tot = len(path_ids) if len(path_ids) >= 2 else 2
    paradas_int = max(0, paradas_tot - 2)
    st.markdown(f"**Paradas (incluye origen y destino):** {paradas_tot}")
    st.markdown(f"**Paradas intermedias:** {paradas_int}")

    # mostramos costo según criterio, y la otra métrica abajo
    if criterio == "tiempo_min":
        st.markdown(f"**Costo total (tiempo_min):** {total_time:.2f}")
        st.markdown(f"**Distancia aprox.:** {total_dist:.2f} km")
    else:
        st.markdown(f"**Costo total (distancia_km):** {total_dist:.2f}")
        st.markdown(f"**Tiempo aprox.:** {total_time:.2f} min")

    if ruta_final:
        export_df = pd.DataFrame(ruta_final, columns=["lon","lat"])
        st.download_button(
            "📥 Descargar ruta (CSV)",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name="ruta.csv",
            mime="text/csv",
        )

    st.markdown("---")
    st.dataframe(
        nodos.dropna(subset=["lat","lon"])[["id","nombre","lat","lon"]],
        use_container_width=True,
    )

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


