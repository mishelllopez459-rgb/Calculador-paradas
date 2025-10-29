import math
import requests
import pandas as pd
import pydeck as pdk
import streamlit as st
from collections import deque

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Rutas San Marcos", layout="wide")
st.title("🚌 Calculador de paradas y ruta óptima — San Marcos")

# ---------------- CARGA CSV ----------------
try:
    nodos_raw = pd.read_csv("nodos.csv")   # columnas: id,nombre,lat,lon
except Exception:
    nodos_raw = pd.DataFrame(columns=["id", "nombre", "lat", "lon"])

try:
    aristas_raw = pd.read_csv("aristas.csv")  # columnas: origen,destino,(peso)
except Exception:
    aristas_raw = pd.DataFrame(columns=["origen", "destino", "peso"])

# ---------------- LUGARES DEFINIDOS ----------------
LUGARES_NUEVOS = [
    "Parque Central","Catedral","Terminal de Buses","Hospital Regional",
    "Cancha Los Angeles","Cancha Sintetica Golazo","Aeropuerto Nacional",
    "Iglesia Candelero de Oro","Centro de Salud","Megapaca",
    "CANICA (Casa de los Niños)","Aldea San Rafael Soche","Pollo Campero",
    "INTECAP San Marcos","Salón Quetzal","SAT San Marcos","Bazar Chino"
]

# ---------------- NORMALIZACIÓN DE NODOS ----------------
nodos = nodos_raw.copy()

# asegurar columnas base existen antes de tocarlas
for base_col in ["id", "nombre", "lat", "lon"]:
    if base_col not in nodos.columns:
        nodos[base_col] = None

# limpiar strings en columnas clave id/nombre
for col in ["id", "nombre"]:
    nodos[col] = nodos[col].astype(str).str.strip()

# dejar solo columnas que usamos
nodos = nodos[["id", "nombre", "lat", "lon"]]

def asegurar_lugares(df, nombres):
    """
    Garantiza que todos los nombres de LUGARES_NUEVOS existan en df.
    Si falta uno, lo crea con id L# y lat/lon = None.
    """
    # asegurar columnas por si acaso
    for c in ["id", "nombre", "lat", "lon"]:
        if c not in df.columns:
            df[c] = None

    existentes = set(df["nombre"].astype(str).str.lower())
    usados     = set(df["id"].astype(str))

    def nuevo_id(start=1):
        i = start
        while True:
            cand = f"L{i}"
            if cand not in usados:
                usados.add(cand)
                return cand
            i += 1

    nuevas = []
    for nm in nombres:
        if nm.lower() not in existentes:
            nuevas.append(
                {"id": nuevo_id(), "nombre": nm, "lat": None, "lon": None}
            )

    if nuevas:
        df = pd.concat([df, pd.DataFrame(nuevas)], ignore_index=True)

    # normalizar otra vez (strip)
    df["id"] = df["id"].astype(str).str.strip()
    df["nombre"] = df["nombre"].astype(str).str.strip()
    return df

nodos = asegurar_lugares(nodos, LUGARES_NUEVOS)

# mantener nodos editables en sesión
if "nodos_mem" not in st.session_state:
    st.session_state.nodos_mem = nodos.copy()
nodos = st.session_state.nodos_mem

# ---------------- HELPERS GEOGRAFÍA ----------------
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
    url = (
        "https://router.project-osrm.org/route/v1/driving/"
        f"{o_lon},{o_lat};{d_lon},{d_lat}?overview=full&geometries=geojson"
    )
    try:
        r = requests.get(url, timeout=12)
        r.raise_for_status()
        data = r.json()
        coords   = data["routes"][0]["geometry"]["coordinates"]  # [lon, lat]
        dist_km  = data["routes"][0]["distance"] / 1000.0
        dur_min  = data["routes"][0]["duration"] / 60.0
        return coords, dist_km, dur_min
    except Exception:
        return None, None, None

def fit_view_from_lonlat(coords_lonlat: list, extra_zoom_out: float = 0.35):
    if not coords_lonlat:
        return pdk.ViewState(latitude=14.965, longitude=-91.79, zoom=13, pitch=0, bearing=0)

    df_bounds = pd.DataFrame(coords_lonlat, columns=["lon","lat"])
    raw_view  = pdk.data_utils.compute_view(df_bounds[["lon","lat"]])

    # pydeck puede devolver dict ó ViewState
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

# ---------------- GRAFO / BFS ----------------
def build_graph_edges(df_aristas: pd.DataFrame):
    """
    Grafo no dirigido: { nodo_id: set(vecinos) }
    """
    g = {}
    # aseguramos columnas mínimo
    if "origen" not in df_aristas.columns:
        df_aristas["origen"] = ""
    if "destino" not in df_aristas.columns:
        df_aristas["destino"] = ""

    for _, r in df_aristas.iterrows():
        a = str(r.get("origen","")).strip()
        b = str(r.get("destino","")).strip()
        if not a or not b:
            continue
        g.setdefault(a,set()).add(b)
        g.setdefault(b,set()).add(a)
    return g

def bfs_shortest_path(graph: dict, start: str, goal: str):
    """
    Camino más corto en saltos (lista de ids). [] si no se puede.
    """
    if not start or not goal or start not in graph or goal not in graph:
        return []
    if start == goal:
        return [start]

    q = deque([[start]])
    visit = {start}
    while q:
        path = q.popleft()
        u = path[-1]
        for v in graph.get(u, []):
            if v in visit:
                continue
            new_path = path + [v]
            if v == goal:
                return new_path
            visit.add(v)
            q.append(new_path)
    return []

def nombre_a_id(nodos_df, nombre):
    match = nodos_df.loc[nodos_df["nombre"] == nombre]
    if match.empty:
        return None
    return str(match.iloc[0]["id"])

def ids_a_polyline_lonlat(nodos_df, ids):
    """
    Devuelve [[lon,lat], ...] siguiendo ids,
    ignorando nodos sin coords,
    Si quedan <2 puntos válidos, devuelve [].
    """
    pts = []
    if "id" not in nodos_df.columns:
        return []
    idx = nodos_df.set_index("id")
    for nid in ids:
        if nid in idx.index:
            lat = idx.loc[nid, "lat"]
            lon = idx.loc[nid, "lon"]
            if pd.notna(lat) and pd.notna(lon):
                pts.append([float(lon), float(lat)])
    return pts if len(pts) >= 2 else []

def distancia_km_sobre_polyline(poly_lonlat):
    """
    Suma haversine sobre todos los segmentos consecutivos.
    """
    if not poly_lonlat or len(poly_lonlat) < 2:
        return None
    total = 0.0
    for i in range(len(poly_lonlat)-1):
        lon1, lat1 = poly_lonlat[i]
        lon2, lat2 = poly_lonlat[i+1]
        total += haversine_km(lat1, lon1, lat2, lon2)
    return total

# ---------------- CAPAS DE MAPA ----------------
def capa_nodos(df_nodos, rgb):
    df_plot = df_nodos.dropna(subset=["lat","lon"]).copy()
    if df_plot.empty:
        return None, pd.DataFrame()
    df_plot.rename(columns={"lon":"lng"}, inplace=True)
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_plot,
        get_position="[lng, lat]",
        get_radius=65,
        radius_min_pixels=3,
        get_fill_color=rgb,
        get_line_color=[30,30,30],
        line_width_min_pixels=1,
        pickable=True,
    )
    return layer, df_plot

def capa_aristas(df_aristas, df_nodos, rgb, width_px=3):
    if df_aristas.empty:
        return None, []
    # proteger contra nodos sin columna id/lat/lon
    for need in ["id","lat","lon"]:
        if need not in df_nodos.columns:
            df_nodos[need] = None

    idx = df_nodos.set_index("id")[["lat","lon"]]
    segs = []
    for _, r in df_aristas.iterrows():
        a = str(r.get("origen","")).strip()
        b = str(r.get("destino","")).strip()
        if a in idx.index and b in idx.index:
            la, lo = idx.loc[a, ["lat","lon"]]
            lb, lo2 = idx.loc[b, ["lat","lon"]]
            if pd.notna(la) and pd.notna(lo) and pd.notna(lb) and pd.notna(lo2):
                segs.append({"path": [[lo,la],[lo2,lb]],
                             "origen": a,
                             "destino": b})
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

def capa_pin(lat, lon, rgb, radius=150):
    return pdk.Layer(
        "ScatterplotLayer",
        data=[{"lat":lat,"lng":lon}],
        get_position="[lng, lat]",
        get_radius=radius,
        radius_min_pixels=5,
        get_fill_color=rgb,
        get_line_color=[20,20,20],
        line_width_min_pixels=1,
        pickable=False,
    )

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
    show_nodes = st.toggle("Mostrar nodos del grafo", True)
    show_edges = st.toggle("Mostrar aristas del grafo", True)

    col_nodes = st.color_picker("Color nodos", "#FF5CA8")
    col_edges = st.color_picker("Color aristas", "#FFC400")
    col_path  = st.color_picker("Color ruta origen→destino", "#FF5733")
    usar_osrm = st.toggle("Ruta real por calle (OSRM)", value=True)

    # ---------- COORDENADAS: sincronizar estado con selección ----------
    st.markdown("---")
    st.markdown("### Coordenadas y edición")

    def sync_inputs(tipo, nombre_sel):
        row = nodos.loc[nodos["nombre"] == nombre_sel].iloc[0]
        lat_val = "" if pd.isna(row["lat"]) else str(row["lat"])
        lon_val = "" if pd.isna(row["lon"]) else str(row["lon"])

        lat_key  = f"{tipo}_lat_input"
        lon_key  = f"{tipo}_lon_input"
        last_key = f"last_{tipo}_nombre"

        # si cambió el nodo seleccionado, refrescamos los inputs visibles
        if (last_key not in st.session_state) or (st.session_state[last_key] != nombre_sel):
            st.session_state[lat_key] = lat_val
            st.session_state[lon_key] = lon_val
            st.session_state[last_key] = nombre_sel

    sync_inputs("origen", origen_nombre)
    sync_inputs("destino", destino_nombre)

    st.caption("Origen")
    c1, c2 = st.columns(2)
    with c1:
        st.session_state["origen_lat_input"] = st.text_input(
            "Lat (Origen)",
            value=st.session_state["origen_lat_input"],
            key="origen_lat_input_key",
        )
    with c2:
        st.session_state["origen_lon_input"] = st.text_input(
            "Lon (Origen)",
            value=st.session_state["origen_lon_input"],
            key="origen_lon_input_key",
        )

    if st.button("💾 Guardar coords Origen"):
        try:
            lat = float(str(st.session_state["origen_lat_input"]).replace(",", "."))
            lon = float(str(st.session_state["origen_lon_input"]).replace(",", "."))
            st.session_state.nodos_mem.loc[
                nodos["nombre"] == origen_nombre, ["lat","lon"]
            ] = [lat, lon]
            st.success(f"Origen actualizado: ({lat}, {lon})")
        except ValueError:
            st.error("Lat/Lon inválidos. Ej: 14.9712  y  -91.7815")

    st.caption("Destino")
    d1, d2 = st.columns(2)
    with d1:
        st.session_state["destino_lat_input"] = st.text_input(
            "Lat (Destino)",
            value=st.session_state["destino_lat_input"],
            key="destino_lat_input_key",
        )
    with d2:
        st.session_state["destino_lon_input"] = st.text_input(
            "Lon (Destino)",
            value=st.session_state["destino_lon_input"],
            key="destino_lon_input_key",
        )

    if st.button("💾 Guardar coords Destino"):
        try:
            lat = float(str(st.session_state["destino_lat_input"]).replace(",", "."))
            lon = float(str(st.session_state["destino_lon_input"]).replace(",", "."))
            st.session_state.nodos_mem.loc[
                nodos["nombre"] == destino_nombre, ["lat","lon"]
            ] = [lat, lon]
            st.success(f"Destino actualizado: ({lat}, {lon})")
        except ValueError:
            st.error("Lat/Lon inválidos. Ej: 14.9712  y  -91.7815")

# ---------------- RUTA AUTOMÁTICA ----------------
# IDs origen/destino
origen_id  = nombre_a_id(nodos, origen_nombre)
destino_id = nombre_a_id(nodos, destino_nombre)

graph = build_graph_edges(aristas_raw)
path_ids = bfs_shortest_path(graph, origen_id, destino_id) if (origen_id and destino_id) else []

# Paradas
if path_ids:
    paradas_tot = len(path_ids)
    paradas_int = max(0, paradas_tot - 2)
else:
    paradas_tot = 2
    paradas_int = 0

# Coordenadas filas origen/destino (para pins y OSRM)
fila_o = nodos.loc[nodos["id"] == origen_id].iloc[0] if origen_id in set(nodos["id"]) else pd.Series({"lat":None,"lon":None})
fila_d = nodos.loc[nodos["id"] == destino_id].iloc[0] if destino_id in set(nodos["id"]) else pd.Series({"lat":None,"lon":None})

# Ruta OSRM entre origen y destino si hay coords en ambos
ruta_osrm = None
dist_km_osrm = None
dur_min_osrm = None
if (
    origen_id and destino_id and
    tiene_coords(fila_o) and tiene_coords(fila_d) and usar_osrm
):
    ruta_osrm, dist_km_osrm, dur_min_osrm = osrm_route(
        float(fila_o["lat"]), float(fila_o["lon"]),
        float(fila_d["lat"]), float(fila_d["lon"])
    )

# Polyline del grafo siguiendo path_ids con coords disponibles
ruta_grafo = ids_a_polyline_lonlat(nodos, path_ids) if path_ids else []

# Distancia / tiempo final
VEL_KMH = 30.0
dist_km_final = None
dur_min_final = None
estimado = False  # para poner "~" si es aproximación

if ruta_osrm and dist_km_osrm is not None:
    # 1. mejor caso: OSRM
    dist_km_final = dist_km_osrm
    dur_min_final = dur_min_osrm
    estimado = False
elif ruta_grafo:
    # 2. sumar tramo a tramo del grafo
    dist_lineal = distancia_km_sobre_polyline(ruta_grafo)
    if dist_lineal is not None:
        dist_km_final = dist_lineal
        dur_min_final = (dist_lineal / VEL_KMH) * 60.0
        estimado = True
elif tiene_coords(fila_o) and tiene_coords(fila_d):
    # 3. distancia recta O->D
    dist_lineal = haversine_km(
        float(fila_o["lat"]), float(fila_o["lon"]),
        float(fila_d["lat"]), float(fila_d["lon"]),
    )
    dist_km_final = dist_lineal
    dur_min_final = (dist_lineal / VEL_KMH) * 60.0
    estimado = True
else:
    # 4. sin coords suficientes, fallback por saltos
    if path_ids and len(path_ids) > 1:
        hops = len(path_ids) - 1
        dist_km_final = hops * 0.6    # ~0.6 km por salto asumida
        dur_min_final = hops * 3.0    # ~3 min por salto asumida
        estimado = True

# ---------------- CAPAS DE MAPA ----------------
RGB_NODES = hex_to_rgb(col_nodes)
RGB_EDGES = hex_to_rgb(col_edges)
RGB_PATH  = hex_to_rgb(col_path)

layers = []
all_coords = []

# nodos (rosados)
layer_nodes, nodos_plot = capa_nodos(nodos, RGB_NODES)
if show_nodes and layer_nodes is not None:
    layers.append(layer_nodes)
    all_coords.extend(nodos_plot[["lng","lat"]].values.tolist())

# aristas del grafo (amarillas finas)
layer_edges, edges_paths = capa_aristas(aristas_raw, nodos, RGB_EDGES, width_px=3)
if show_edges and layer_edges is not None:
    layers.append(layer_edges)
    for seg in edges_paths:
        all_coords.extend(seg["path"])

# ruta elegida (naranja gordo)
ruta_final_poly = ruta_osrm if ruta_osrm else ruta_grafo
layer_route = capa_ruta(ruta_final_poly, RGB_PATH, width_px=8)
if layer_route:
    layers.append(layer_route)
    all_coords.extend(ruta_final_poly)

# pines origen/destino (verde/rojo)
if tiene_coords(fila_o):
    layers.append(
        capa_pin(float(fila_o["lat"]), float(fila_o["lon"]), [0,255,0], radius=180)
    )
    all_coords.append([float(fila_o["lon"]), float(fila_o["lat"])])
if tiene_coords(fila_d):
    layers.append(
        capa_pin(float(fila_d["lat"]), float(fila_d["lon"]), [255,0,0], radius=180)
    )
    all_coords.append([float(fila_d["lon"]), float(fila_d["lat"])])

# vista inicial mapa
if all_coords:
    view_state = fit_view_from_lonlat(all_coords, extra_zoom_out=0.4)
else:
    view_state = pdk.ViewState(latitude=14.965, longitude=-91.79, zoom=13, pitch=0, bearing=0)

# ---------------- TEXTO RESUMEN ----------------
criterio_texto = "⏱ tiempo mín"
grafo_texto    = "No dirigido"

if dist_km_final is not None:
    dist_txt = f"{dist_km_final:.2f} km"
else:
    dist_txt = "—"

if dur_min_final is not None:
    if estimado:
        tiempo_txt = f"~{dur_min_final:.1f} min"
        costo_txt  = f"~{dur_min_final:.1f} min"
    else:
        tiempo_txt = f"{dur_min_final:.1f} min"
        costo_txt  = f"{dur_min_final:.1f} min"
else:
    tiempo_txt = "—"
    costo_txt  = "—"

# ---------------- LAYOUT ----------------
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Resumen")

    st.markdown(f"**Origen:** {origen_nombre}")
    st.markdown(f"**Destino:** {destino_nombre}")
    st.markdown(f"**Criterio:** {criterio_texto}")
    st.markdown(f"**Grafo:** {grafo_texto}")

    st.markdown(f"**Paradas (incluye origen y destino):** {paradas_tot}")
    st.markdown(f"**Paradas intermedias:** {paradas_int}")

    st.markdown(f"**Distancia aprox.:** {dist_txt}")
    st.markdown(f"**Tiempo aprox.:** {tiempo_txt}")
    st.markdown(f"**Costo total (tiempo mín):** {costo_txt}")

    if ruta_final_poly:
        export_df = pd.DataFrame(ruta_final_poly, columns=["lon","lat"])
        st.download_button(
            "📥 Descargar ruta (CSV)",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name="ruta.csv",
            mime="text/csv",
        )

    st.markdown("---")
    st.markdown("**Nodos cargados (con coordenadas):**")
    st.dataframe(
        nodos.dropna(subset=["lat","lon"])[["id","nombre","lat","lon"]],
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

if not ruta_final_poly:
    st.info(
        "No se pudo dibujar la ruta en el mapa (faltan coordenadas en alguna parada), "
        "pero igual se calculó la ruta mínima con el grafo."
    )
