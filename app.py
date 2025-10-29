import math
import requests
import pandas as pd
import pydeck as pdk
import streamlit as st
from collections import deque

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Rutas San Marcos", layout="wide")
st.title("🚌 Calculador de paradas y ruta óptima — San Marcos")

# ---------------- CARGA DE DATOS ----------------
try:
    nodos_raw = pd.read_csv("nodos.csv")  # columnas esperadas: id, nombre, lat, lon
except Exception:
    nodos_raw = pd.DataFrame(columns=["id", "nombre", "lat", "lon"])

try:
    aristas_raw = pd.read_csv("aristas.csv")  # columnas esperadas: origen, destino, (peso opcional)
except Exception:
    aristas_raw = pd.DataFrame(columns=["origen", "destino", "peso"])

# ---------------- LUGARES BASE ----------------
LUGARES_NUEVOS = [
    "Parque Central", "Catedral", "Terminal de Buses", "Hospital Regional",
    "Cancha Los Angeles", "Cancha Sintetica Golazo", "Aeropuerto Nacional",
    "Iglesia Candelero de Oro", "Centro de Salud", "Megapaca",
    "CANICA (Casa de los Niños)", "Aldea San Rafael Soche", "Pollo Campero",
    "INTECAP San Marcos", "Salón Quetzal", "SAT San Marcos", "Bazar Chino"
]

# ---------------- NORMALIZACIÓN NODOS ----------------
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

    nuevas = []
    for nm in nombres:
        if nm.lower() not in existentes:
            nuevas.append({"id": nuevo_id(), "nombre": nm, "lat": None, "lon": None})

    if nuevas:
        df = pd.concat([df, pd.DataFrame(nuevas)], ignore_index=True)

    return df

nodos = asegurar_lugares(nodos, LUGARES_NUEVOS)

# sesión editable
if "nodos_mem" not in st.session_state:
    st.session_state.nodos_mem = nodos.copy()

# trabajamos SIEMPRE sobre memoria
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

def tiene_coords_row(row) -> bool:
    return pd.notna(row["lat"]) and pd.notna(row["lon"])

def osrm_route(o_lat, o_lon, d_lat, d_lon):
    """
    Devuelve: (path_lonlat, dist_km, dur_min) o (None, None, None)
    path_lonlat = [[lon,lat], ...] siguiendo calle
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
    Hace zoom al conjunto de puntos (lon,lat) que le pasemos.
    Funciona en pydeck viejo y nuevo.
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

# --------- Geocodificador automático ---------
def geocode_nominatim(nombre: str, base_hint: str = "San Marcos, Guatemala"):
    """
    Intenta conseguir lat/lon desde OpenStreetMap.
    Devuelve (lat, lon) o None.
    """
    try:
        q = f"{nombre}, {base_hint}".strip().replace("  ", " ")
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": q, "format": "json", "limit": 1}
        headers = {"User-Agent": "RutasSanMarcos/1.0 (educational)"}
        r = requests.get(url, params=params, headers=headers, timeout=12)
        r.raise_for_status()
        js = r.json()
        if not js:
            return None
        lat = float(js[0]["lat"])
        lon = float(js[0]["lon"])
        return (lat, lon)
    except Exception:
        return None

def asegurar_coords_para(nombre_lugar: str, base_hint: str):
    """
    Retorna (lat, lon) ya listos para ese lugar.
    Si el nodo ya tiene coords en memoria -> las usa.
    Si NO tiene, intenta geocodificar -> guarda en memoria -> devuelve.
    """
    fila_idx = nodos["nombre"] == nombre_lugar
    if not fila_idx.any():
        return (None, None)

    lat_actual = nodos.loc[fila_idx, "lat"].iloc[0]
    lon_actual = nodos.loc[fila_idx, "lon"].iloc[0]

    if pd.notna(lat_actual) and pd.notna(lon_actual):
        return (float(lat_actual), float(lon_actual))

    # no hay coords => intentar geocodificar
    geo = geocode_nominatim(nombre_lugar, base_hint)
    if geo:
        lat_new, lon_new = geo
        st.session_state.nodos_mem.loc[fila_idx, ["lat", "lon"]] = [lat_new, lon_new]
        return (lat_new, lon_new)

    # si ni así, devolvemos None
    return (None, None)

# --------- Grafo ---------
def build_graph_edges(df_aristas: pd.DataFrame):
    """
    Grafo NO dirigido {nodo_id: set(vecinos)}.
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
    Devuelve lista de ids [start,...,goal] con BFS.
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
    Convierte una lista de IDs de nodos en una polilínea [[lon,lat],...]
    sólo con nodos que tengan coords. Si queda <2 puntos útiles -> [].
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

    # Selección origen/destino
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
    st.markdown("### Coordenadas automáticas")
    base_hint = st.text_input("Ámbito geográfico", "San Marcos, Guatemala")

    # >>> AQUÍ ES DONDE LLENAMOS AUTOMÁTICAMENTE <<<
    o_lat, o_lon = asegurar_coords_para(origen_nombre, base_hint)
    d_lat, d_lon = asegurar_coords_para(destino_nombre, base_hint)

    # Mostramos SIEMPRE las coords actuales (guardadas o geocodificadas)
    st.markdown("#### Origen")
    col1, col2 = st.columns(2)
    with col1:
        new_o_lat = st.text_input("Lat (Origen)", value="" if o_lat is None else str(o_lat), key="lat_origen_box")
    with col2:
        new_o_lon = st.text_input("Lon (Origen)", value="" if o_lon is None else str(o_lon), key="lon_origen_box")

    if st.button("💾 Guardar Origen"):
        try:
            lat_f = float(str(new_o_lat).replace(",", "."))
            lon_f = float(str(new_o_lon).replace(",", "."))
            st.session_state.nodos_mem.loc[nodos["nombre"] == origen_nombre, ["lat", "lon"]] = [lat_f, lon_f]
            st.success(f"Guardado {origen_nombre}: ({lat_f}, {lon_f})")
        except ValueError:
            st.error("Lat/Lon inválidos en Origen")

    st.markdown("#### Destino")
    col3, col4 = st.columns(2)
    with col3:
        new_d_lat = st.text_input("Lat (Destino)", value="" if d_lat is None else str(d_lat), key="lat_destino_box")
    with col4:
        new_d_lon = st.text_input("Lon (Destino)", value="" if d_lon is None else str(d_lon), key="lon_destino_box")

    if st.button("💾 Guardar Destino"):
        try:
            lat_f = float(str(new_d_lat).replace(",", "."))
            lon_f = float(str(new_d_lon).replace(",", "."))
            st.session_state.nodos_mem.loc[nodos["nombre"] == destino_nombre, ["lat", "lon"]] = [lat_f, lon_f]
            st.success(f"Guardado {destino_nombre}: ({lat_f}, {lon_f})")
        except ValueError:
            st.error("Lat/Lon inválidos en Destino")

# ---------------- LÓGICA DE RUTA ----------------
# 1. obtener IDs de esos nombres
origen_id = nombre_a_id(nodos, origen_nombre)
destino_id = nombre_a_id(nodos, destino_nombre)

# 2. grafo no dirigido
graph = build_graph_edges(aristas_raw)

# 3. ruta por grafo (BFS)
path_ids = bfs_shortest_path(graph, origen_id, destino_id) if (origen_id and destino_id) else []

# 4. métricas del grafo
if path_ids:
    paradas_totales = len(path_ids)
    paradas_intermedias = max(0, paradas_totales - 2)
    saltos = len(path_ids) - 1
else:
    paradas_totales = 2
    paradas_intermedias = 0
    saltos = 1

# tiempo estimado por grafo (asumimos 3 min por salto)
estimado_min_por_grafo = saltos * 3.0 if saltos >= 1 else None

# 5. ruta real si tenemos coords de origen y destino
dist_km = None
dur_min = None
ruta_path_lonlat_osrm = None

if origen_id and destino_id:
    fila_o = nodos.loc[nodos["id"] == origen_id].iloc[0]
    fila_d = nodos.loc[nodos["id"] == destino_id].iloc[0]

    if tiene_coords_row(fila_o) and tiene_coords_row(fila_d):
        o_lat2, o_lon2 = float(fila_o["lat"]), float(fila_o["lon"])
        d_lat2, d_lon2 = float(fila_d["lat"]), float(fila_d["lon"])

        ruta_path_lonlat_osrm, dist_km, dur_min = (None, None, None)
        if usar_osrm:
            ruta_path_lonlat_osrm, dist_km, dur_min = osrm_route(o_lat2, o_lon2, d_lat2, d_lon2)

        if ruta_path_lonlat_osrm is None:
            # fallback línea recta
            dist_km = haversine_km(o_lat2, o_lon2, d_lat2, d_lon2)
            vel_kmh = 30.0
            dur_min = (dist_km / vel_kmh) * 60.0
            ruta_path_lonlat_osrm = [[o_lon2, o_lat2], [d_lon2, d_lat2]]
else:
    fila_o = pd.Series({"lat": None, "lon": None})
    fila_d = pd.Series({"lat": None, "lon": None})

# 6. si no hay ruta OSRM, intentamos dibujar la ruta del grafo con coords de cada parada
ruta_path_lonlat_grafo = []
if not ruta_path_lonlat_osrm:
    ruta_path_lonlat_grafo = ids_a_polyline_lonlat(nodos, path_ids)

# ---------------- CAPAS PARA MAPA ----------------
RGB_NODES = hex_to_rgb(col_nodes)
RGB_EDGES = hex_to_rgb(col_edges)
RGB_PATH  = hex_to_rgb(col_path)

layers = []
all_coords_for_view = []

# capa de nodos
nodos_plot = nodos.dropna(subset=["lat", "lon"]).copy()
if not nodos_plot.empty:
    nodos_plot = nodos_plot.rename(columns={"lon": "lng"})
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

# capa de aristas
if show_edges and not aristas_raw.empty:
    idx = nodos.set_index("id")[["lat", "lon"]]
    aristas_paths = []
    for _, row in aristas_raw.iterrows():
        o_id = str(row.get("origen", "")).strip()
        d_id = str(row.get("destino", "")).strip()
        if o_id in idx.index and d_id in idx.index:
            o_lat3, o_lon3 = idx.loc[o_id, ["lat", "lon"]]
            d_lat3, d_lon3 = idx.loc[d_id, ["lat", "lon"]]
            if pd.notna(o_lat3) and pd.notna(o_lon3) and pd.notna(d_lat3) and pd.notna(d_lon3):
                seg = [[o_lon3, o_lat3], [d_lon3, d_lat3]]
                aristas_paths.append({"path": seg, "origen": o_id, "destino": d_id})
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

# capa ruta final
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

# viewstate
if all_coords_for_view:
    view_state = fit_view_from_lonlat(all_coords_for_view, extra_zoom_out=0.4)
else:
    view_state = pdk.ViewState(latitude=14.965, longitude=-91.79, zoom=13, pitch=0, bearing=0)

# ---------------- MÉTRICAS PARA EL RESUMEN ----------------
criterio_texto = "⏱ tiempo mín"
grafo_texto = "No dirigido"

if dist_km is not None:
    distancia_txt = f"{dist_km:.2f} km"
else:
    distancia_txt = "—"

if dur_min is not None:
    tiempo_txt = f"{dur_min:.1f} min"
    costo_total_txt = f"{dur_min:.1f} min"
elif estimado_min_por_grafo is not None:
    tiempo_txt = f"~{estimado_min_por_grafo:.1f} min"
    costo_total_txt = f"~{estimado_min_por_grafo:.1f} min"
else:
    tiempo_txt = "—"
    costo_total_txt = "—"

paradas_intermedias_txt = paradas_intermedias
paradas_totales_txt = paradas_totales

# ---------------- LAYOUT MAIN ----------------
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Resumen")

    st.markdown(f"**Origen:** {origen_nombre}")
    st.markdown(f"**Destino:** {destino_nombre}")

    st.markdown(f"**Criterio:** {criterio_texto}")
    st.markdown(f"**Grafo:** {grafo_texto}")

    st.markdown(f"**Paradas (incluye origen y destino):** {paradas_totales_txt}")
    st.markdown(f"**Paradas intermedias:** {paradas_intermedias_txt}")

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
                "style": {"color": "white"},
            },
        ),
        use_container_width=True,
    )

if not ruta_final_lonlat:
    st.info(
        "No se pudo dibujar la ruta en el mapa (falta lat/lon de alguna parada). "
        "El resumen se calculó con el grafo. Las coordenadas ya se intentaron autocompletar arriba."
    )

