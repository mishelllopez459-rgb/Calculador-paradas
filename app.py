import math
import heapq
import requests
import pandas as pd
import pydeck as pdk
import streamlit as st
from collections import defaultdict

# =========================================
# CONFIG STREAMLIT
# =========================================
st.set_page_config(page_title="Rutas San Marcos", layout="wide")

# =========================================
# CONSTANTES
# =========================================
VEL_KMH = 30.0  # velocidad promedio estimada para calcular tiempo si no hay OSRM
CENTER_LAT = 14.965
CENTER_LON = -91.79

# =========================================
# HELPERS BÁSICOS
# =========================================
def hex_to_rgb(h: str):
    h = h.lstrip("#")
    return [int(h[i:i+2], 16) for i in (0,2,4)]

def haversine_km(a_lat, a_lon, b_lat, b_lon):
    R = 6371.0
    lat1, lon1 = math.radians(a_lat), math.radians(a_lon)
    lat2, lon2 = math.radians(b_lat), math.radians(b_lon)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = (
        math.sin(dlat/2)**2
        + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    )
    return 2 * R * math.asin(math.sqrt(h))

def tiene_coords(row) -> bool:
    return (
        row is not None
        and "lat" in row and "lon" in row
        and pd.notna(row["lat"])
        and pd.notna(row["lon"])
    )

def osrm_route(o_lat, o_lon, d_lat, d_lon):
    """
    Ruta real por calle usando el OSRM público.
    Devuelve:
      coords_lonlat -> [[lon,lat], ...]
      dist_km       -> float
      dur_min       -> float
    o falla y devuelve (None, None, None)
    """
    url = (
        "https://router.project-osrm.org/route/v1/driving/"
        f"{o_lon},{o_lat};{d_lon},{d_lat}?overview=full&geometries=geojson"
    )
    try:
        r = requests.get(url, timeout=12)
        r.raise_for_status()
        data = r.json()
        coords  = data["routes"][0]["geometry"]["coordinates"]  # [[lon,lat], ...]
        dist_km = data["routes"][0]["distance"] / 1000.0
        dur_min = data["routes"][0]["duration"] / 60.0
        return coords, dist_km, dur_min
    except Exception:
        return None, None, None

def fit_view_from_lonlat(coords_lonlat, extra_zoom_out=0.4):
    """
    Saca la vista inicial del mapa para que encuadre todo.
    """
    if not coords_lonlat:
        return pdk.ViewState(
            latitude=CENTER_LAT,
            longitude=CENTER_LON,
            zoom=13,
            pitch=0,
            bearing=0,
        )
    df_bounds = pd.DataFrame(coords_lonlat, columns=["lon","lat"])
    raw_view = pdk.data_utils.compute_view(df_bounds[["lon","lat"]])

    if isinstance(raw_view, dict):
        lat_c = raw_view.get("latitude", CENTER_LAT)
        lon_c = raw_view.get("longitude", CENTER_LON)
        zoom_v = raw_view.get("zoom", 13)
    else:
        lat_c = getattr(raw_view, "latitude", CENTER_LAT)
        lon_c = getattr(raw_view, "longitude", CENTER_LON)
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

# =========================================
# CARGA CSVs
# nodos.csv : id,nombre,lat,lon
# aristas.csv : origen,destino,peso  (peso = minutos aprox)
# =========================================
def cargar_csv_nodos():
    try:
        df = pd.read_csv("nodos.csv")
    except Exception:
        df = pd.DataFrame(columns=["id","nombre","lat","lon"])

    for col in ["id","nombre"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
        else:
            df[col] = ""

    for c in ["id","nombre","lat","lon"]:
        if c not in df.columns:
            df[c] = None

    df = df[["id","nombre","lat","lon"]]

    # asegurar que lugares comunes estén aunque no tengan coords aún
    LUGARES_NUEVOS = [
        "Parque Central","Catedral","Terminal de Buses","Hospital Regional",
        "Cancha Los Angeles","Cancha Sintetica Golazo","Aeropuerto Nacional",
        "Iglesia Candelero de Oro","Centro de Salud","Megapaca",
        "CANICA (Casa de los Niños)","Aldea San Rafael Soche","Pollo Campero",
        "INTECAP San Marcos","Salón Quetzal","SAT San Marcos","Bazar Chino"
    ]

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

    nuevos_rows = []
    for nm in LUGARES_NUEVOS:
        if nm.lower() not in existentes:
            nuevos_rows.append(
                {"id": nuevo_id(), "nombre": nm, "lat": None, "lon": None}
            )
    if nuevos_rows:
        df = pd.concat([df, pd.DataFrame(nuevos_rows)], ignore_index=True)

    return df

def cargar_csv_aristas():
    try:
        df = pd.read_csv("aristas.csv")
    except Exception:
        df = pd.DataFrame(columns=["origen","destino","peso"])
    for c in ["origen","destino"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
        else:
            df[c] = ""
    if "peso" not in df.columns:
        df["peso"] = None
    return df[["origen","destino","peso"]]

nodos = cargar_csv_nodos()
aristas = cargar_csv_aristas()

# mantener nodos en sesión para consistencia entre reruns
if "nodos_mem" not in st.session_state:
    st.session_state.nodos_mem = nodos.copy()
nodos = st.session_state.nodos_mem

# =========================================
# GRAFO + DIJKSTRA
# =========================================
def build_weighted_graph(aristas_df, nodos_df, dirigido: bool):
    """
    graph[u] = [(v, tiempo_min, dist_km), ...]
    weights[(u,v)] = {"time": tiempo_min, "dist": dist_km}

    peso (si viene en CSV) = tiempo_min directo.
    si no hay peso: calculamos distancia entre coords y estimamos tiempo.
    fallback si no hay coords: 0.6 km y 3.0 min.
    """
    graph = defaultdict(list)
    weights = {}

    idx = nodos_df.set_index("id")[["lat","lon"]]

    def add_edge(u, v, peso):
        # distancia
        d_km = None
        if u in idx.index and v in idx.index:
            la, lo = idx.loc[u, ["lat","lon"]]
            lb, lo2 = idx.loc[v, ["lat","lon"]]
            if pd.notna(la) and pd.notna(lo) and pd.notna(lb) and pd.notna(lo2):
                d_km = haversine_km(float(la), float(lo), float(lb), float(lo2))
        if d_km is None:
            d_km = 0.6  # fallback distancia si no hay coords

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
        u = str(r["origen"]).strip()
        v = str(r["destino"]).strip()
        if not u or not v:
            continue
        peso = r.get("peso", None)
        add_edge(u, v, peso)
        if not dirigido:
            add_edge(v, u, peso)

    return graph, weights

def dijkstra(graph, start, goal, use="time"):
    """
    Camino mínimo entre start y goal,
    use="time"  => minimizar minutos
    use="dist"  => minimizar km
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

# =========================================
# UTILIDADES RUTA
# =========================================
def nombre_a_id(nodos_df, nombre):
    m = nodos_df.loc[nodos_df["nombre"] == nombre]
    if m.empty:
        return None
    return str(m.iloc[0]["id"])

def ids_a_polyline_lonlat(nodos_df, ids):
    """
    Devuelve lista [[lon,lat], ...] usando coords de cada nodo del camino.
    Solo añade los que tienen coords válidas.
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

# =========================================
# SIDEBAR CONTROLES
# =========================================
with st.sidebar:
    st.markdown("### Parámetros")

    dirigido_flag = st.checkbox(
        "Tramos unidireccionales (grafo dirigido)",
        value=False,
        help="Si está activo, cada arista cuenta solo en la dirección origen→destino."
    )

    origen_nombre = st.selectbox("Origen", sorted(nodos["nombre"].astype(str)))
    destino_nombre = st.selectbox(
        "Destino",
        sorted(nodos["nombre"].astype(str)),
        index=1 if len(nodos) > 1 else 0,
    )

    criterio_radio = st.radio(
        "Optimizar por",
        ["tiempo_min", "distancia_km"],
        index=0,
        help="Qué quieres minimizar en el grafo."
    )

    st.markdown("### Colores")
    color_nodes = st.color_picker("Nodos (puntos)", "#FF008C")        # rosa
    color_edges = st.color_picker("Red general (aristas)", "#FFFFFF") # blanco
    color_path  = st.color_picker("Ruta seleccionada", "#00B2FF")     # celeste
    usar_osrm   = st.toggle("Ruta real por calle (OSRM)", value=True)

    st.button("Calcular ruta")  # decorativo, Streamlit recalcula igual solo

# =========================================
# CALCULAR RUTA ÓPTIMA
# =========================================
origen_id  = nombre_a_id(nodos, origen_nombre)
destino_id = nombre_a_id(nodos, destino_nombre)

# obtener filas completas para coords
fila_o = nodos.loc[nodos["id"] == origen_id].iloc[0] if origen_id in set(nodos["id"]) else None
fila_d = nodos.loc[nodos["id"] == destino_id].iloc[0] if destino_id in set(nodos["id"]) else None

# construir grafo
graph, weights = build_weighted_graph(aristas, nodos, dirigido_flag)

# seleccionar métrica
use_metric = "time" if criterio_radio == "tiempo_min" else "dist"

# buscar camino más corto en el grafo
path_ids, _ = dijkstra(graph, origen_id, destino_id, use=use_metric)

# fallback: si el grafo NO tiene ruta, igual usamos directo origen->destino
if len(path_ids) < 2:
    path_ids = []
    if origen_id:  path_ids.append(origen_id)
    if destino_id and destino_id != origen_id:
        path_ids.append(destino_id)

# ahora calculamos métricas totales para ese camino
total_time = 0.0
total_dist = 0.0
if len(path_ids) >= 2:
    for i in range(len(path_ids)-1):
        u = path_ids[i]
        v = path_ids[i+1]
        if (u, v) in weights:
            total_time += float(weights[(u,v)]["time"])
            total_dist += float(weights[(u,v)]["dist"])
        else:
            # si no hay peso en el grafo entre esos dos (ej: fallback directo),
            # calculamos por coordenadas:
            if u in set(nodos["id"]) and v in set(nodos["id"]):
                ro = nodos.loc[nodos["id"] == u].iloc[0]
                rd = nodos.loc[nodos["id"] == v].iloc[0]
                if tiene_coords(ro) and tiene_coords(rd):
                    d_km = haversine_km(float(ro["lat"]), float(ro["lon"]),
                                        float(rd["lat"]), float(rd["lon"]))
                    t_min = (d_km / VEL_KMH) * 60.0
                    total_time += t_min
                    total_dist += d_km
                else:
                    # si ni coords, no podemos sumar, dejamos 0 para ese tramo
                    pass

# polyline desde los nodos del camino
poly_paradas = ids_a_polyline_lonlat(nodos, path_ids)

# recta directa O->D (por si el camino tiene coords en O y D)
poly_recta = []
if tiene_coords(fila_o) and tiene_coords(fila_d):
    poly_recta = [
        [float(fila_o["lon"]), float(fila_o["lat"])],
        [float(fila_d["lon"]), float(fila_d["lat"])],
    ]

# probar OSRM entre origen y destino (solo si tiene coords y toggle activo)
poly_osrm, dist_osrm, dur_osrm = (None, None, None)
if usar_osrm and tiene_coords(fila_o) and tiene_coords(fila_d):
    poly_osrm, dist_osrm, dur_osrm = osrm_route(
        float(fila_o["lat"]), float(fila_o["lon"]),
        float(fila_d["lat"]), float(fila_d["lon"]),
    )
    # si OSRM respondió bien, usamos esos valores reales
    if poly_osrm and dist_osrm is not None and dur_osrm is not None:
        ruta_final = poly_osrm
        total_dist = dist_osrm
        total_time = dur_osrm
    else:
        # si OSRM falló, usamos poly_paradas > recta
        if len(poly_paradas) >= 2:
            ruta_final = poly_paradas
        elif len(poly_recta) >= 2:
            ruta_final = poly_recta
        else:
            ruta_final = []
else:
    # no usar OSRM -> grafo -> recta
    if len(poly_paradas) >= 2:
        ruta_final = poly_paradas
    elif len(poly_recta) >= 2:
        ruta_final = poly_recta
    else:
        ruta_final = []

# =========================================
# CAPAS PARA EL MAPA
# =========================================

# capa de nodos (puntos)
nodos_plot = nodos.dropna(subset=["lat","lon"]).copy()
nodos_plot.rename(columns={"lon": "lng"}, inplace=True)

layer_nodes = None
if not nodos_plot.empty:
    layer_nodes = pdk.Layer(
        "ScatterplotLayer",
        data=nodos_plot,
        get_position="[lng, lat]",
        get_radius=65,
        radius_min_pixels=3,
        get_fill_color=hex_to_rgb(color_nodes),
        get_line_color=[30,30,30],
        line_width_min_pixels=1,
        pickable=True,
    )

# capa de TODA la red (todas las aristas en blanco)
idx_coords = nodos.set_index("id")[["lat","lon"]]
edge_segments = []
for _, r in aristas.iterrows():
    u = str(r["origen"]).strip()
    v = str(r["destino"]).strip()
    if u in idx_coords.index and v in idx_coords.index:
        la, lo = idx_coords.loc[u, ["lat","lon"]]
        lb, lb_lo = idx_coords.loc[v, ["lat","lon"]]
        # cuidado con nombres: lb_lo es lon de v
        if pd.notna(la) and pd.notna(lo) and pd.notna(lb) and pd.notna(lb_lo):
            edge_segments.append({
                "path": [[lo, la],[lb_lo, lb]],
                "origen": u,
                "destino": v,
            })

layer_edges = None
if edge_segments:
    layer_edges = pdk.Layer(
        "PathLayer",
        data=edge_segments,
        get_path="path",
        get_width=3,
        width_scale=8,
        get_color=hex_to_rgb(color_edges),
        pickable=False,
    )

# capa de la ruta seleccionada (gordita celeste)
layer_route = None
if len(ruta_final) >= 2:
    layer_route = pdk.Layer(
        "PathLayer",
        data=[{"path": ruta_final}],
        get_path="path",
        get_width=6,
        width_scale=8,
        get_color=hex_to_rgb(color_path),
        pickable=False,
    )

# juntar capas
layers = []
all_coords_for_view = []

if layer_nodes is not None:
    layers.append(layer_nodes)
    if not nodos_plot.empty:
        all_coords_for_view.extend(nodos_plot[["lng","lat"]].values.tolist())

if layer_edges is not None:
    layers.append(layer_edges)
    for seg in edge_segments:
        all_coords_for_view.extend(seg["path"])

if layer_route is not None:
    layers.append(layer_route)
    all_coords_for_view.extend(ruta_final)

view_state = fit_view_from_lonlat(all_coords_for_view, extra_zoom_out=0.4)

# =========================================
# UI PRINCIPAL (RESUMEN + MAPA)
# =========================================
st.title("🚌 Calculador de paradas y ruta óptima — San Marcos")

st.subheader("Resumen")
st.markdown(f"**Origen:** {origen_nombre}")
st.markdown(f"**Destino:** {destino_nombre}")
st.markdown(f"**Criterio:** `{criterio_radio}`")
st.markdown(f"**Grafo:** {'Dirigido' if dirigido_flag else 'No dirigido'}")

paradas_tot = len(path_ids) if len(path_ids) >= 1 else 0
paradas_int = max(0, paradas_tot - 2)

st.markdown(f"**Paradas (incluye origen y destino):** {paradas_tot}")
st.markdown(f"**Paradas intermedias:** {paradas_int}")

# mostramos métricas siempre
if criterio_radio == "tiempo_min":
    st.markdown(f"**Costo total (tiempo_min):** {total_time:.2f} min")
    st.markdown(f"**Distancia aprox.:** {total_dist:.2f} km")
else:
    st.markdown(f"**Costo total (distancia_km):** {total_dist:.2f} km")
    st.markdown(f"**Tiempo aprox.:** {total_time:.2f} min")

# permitir bajar la línea de la ruta
if len(ruta_final) >= 2:
    export_df = pd.DataFrame(ruta_final, columns=["lon","lat"])
    st.download_button(
        "📥 Descargar ruta (CSV)",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name="ruta.csv",
        mime="text/csv",
    )

# mapa grande
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
