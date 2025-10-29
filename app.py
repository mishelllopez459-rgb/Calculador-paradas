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
    nodos_raw = pd.read_csv("nodos.csv")   # id,nombre,lat,lon
except Exception:
    nodos_raw = pd.DataFrame(columns=["id", "nombre", "lat", "lon"])

try:
    aristas_raw = pd.read_csv("aristas.csv")  # origen,destino,(peso)
except Exception:
    aristas_raw = pd.DataFrame(columns=["origen", "destino", "peso"])

# ---------------- LISTA BASE ----------------
LUGARES_NUEVOS = [
    "Parque Central","Catedral","Terminal de Buses","Hospital Regional",
    "Cancha Los Angeles","Cancha Sintetica Golazo","Aeropuerto Nacional",
    "Iglesia Candelero de Oro","Centro de Salud","Megapaca",
    "CANICA (Casa de los Niños)","Aldea San Rafael Soche","Pollo Campero",
    "INTECAP San Marcos","Salón Quetzal","SAT San Marcos","Bazar Chino"
]

# ---------------- NORMALIZACIÓN ----------------
nodos = nodos_raw.copy()
for col in ["id","nombre"]:
    if col in nodos.columns:
        nodos[col] = nodos[col].astype(str).str.strip()
    else:
        nodos[col] = None
for c in ["id","nombre","lat","lon"]:
    if c not in nodos.columns:
        nodos[c] = None
nodos = nodos[["id","nombre","lat","lon"]]

def asegurar_lugares(df, nombres):
    existentes = set(df["nombre"].astype(str).str.lower()) if "nombre" in df else set()
    usados = set(df["id"].astype(str)) if "id" in df else set()
    def nuevo_id(start=1):
        i = start
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
    return df

nodos = asegurar_lugares(nodos, LUGARES_NUEVOS)

# Memoria editable
if "nodos_mem" not in st.session_state:
    st.session_state.nodos_mem = nodos.copy()
nodos = st.session_state.nodos_mem

# ---------------- HELPERS ----------------
def hex_to_rgb(h: str):
    h = h.lstrip("#")
    return [int(h[i:i+2], 16) for i in (0,2,4)]

def haversine_km(a_lat, a_lon, b_lat, b_lon):
    R = 6371.0
    lat1, lon1 = math.radians(a_lat), math.radians(a_lon)
    lat2, lon2 = math.radians(b_lat), math.radians(b_lon)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(h))

def tiene_coords(row) -> bool:
    return pd.notna(row["lat"]) and pd.notna(row["lon"])

def osrm_route(o_lat, o_lon, d_lat, d_lon):
    url = (f"https://router.project-osrm.org/route/v1/driving/"
           f"{o_lon},{o_lat};{d_lon},{d_lat}?overview=full&geometries=geojson")
    try:
        r = requests.get(url, timeout=12)
        r.raise_for_status()
        data = r.json()
        coords = data["routes"][0]["geometry"]["coordinates"]  # [lon,lat]
        dist_km = data["routes"][0]["distance"]/1000.0
        dur_min = data["routes"][0]["duration"]/60.0
        return coords, dist_km, dur_min
    except Exception:
        return None, None, None

def fit_view_from_lonlat(coords_lonlat: list, extra_zoom_out: float = 0.35):
    if not coords_lonlat:
        return pdk.ViewState(latitude=14.965, longitude=-91.79, zoom=13, pitch=0, bearing=0)
    df_bounds = pd.DataFrame(coords_lonlat, columns=["lon","lat"])
    raw_view = pdk.data_utils.compute_view(df_bounds[["lon","lat"]])
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
    return pdk.ViewState(latitude=lat_center, longitude=lon_center, zoom=zoom_val, pitch=0, bearing=0)

# --------- GRAFO / BFS ---------
def build_graph_edges(df_aristas: pd.DataFrame):
    g = {}
    for _, r in df_aristas.iterrows():
        a = str(r.get("origen","")).strip()
        b = str(r.get("destino","")).strip()
        if not a or not b: 
            continue
        g.setdefault(a,set()).add(b)
        g.setdefault(b,set()).add(a)   # no dirigido
    return g

def bfs_shortest_path(graph: dict, start: str, goal: str):
    if not start or not goal or start not in graph or goal not in graph:
        return []
    if start == goal:
        return [start]
    q = deque([[start]])
    vis = {start}
    while q:
        path = q.popleft()
        u = path[-1]
        for v in graph.get(u, []):
            if v in vis: 
                continue
            npth = path + [v]
            if v == goal:
                return npth
            vis.add(v); q.append(npth)
    return []

def nombre_a_id(nodos_df, nombre):
    m = nodos_df.loc[nodos_df["nombre"] == nombre]
    if m.empty: return None
    return str(m.iloc[0]["id"])

def ids_a_polyline_lonlat(nodos_df, ids):
    out = []
    idx = nodos_df.set_index("id")
    for nid in ids:
        if nid in idx.index:
            lat, lon = idx.loc[nid, ["lat","lon"]]
            if pd.notna(lat) and pd.notna(lon):
                out.append([float(lon), float(lat)])
    return out if len(out) >= 2 else []

def distancia_km_sobre_polyline(poly_lonlat):
    """Suma Haversine a lo largo de la polilínea [ [lon,lat], ... ]."""
    if not poly_lonlat or len(poly_lonlat) < 2:
        return None
    tot = 0.0
    for i in range(len(poly_lonlat)-1):
        lon1, lat1 = poly_lonlat[i]
        lon2, lat2 = poly_lonlat[i+1]
        tot += haversine_km(lat1, lon1, lat2, lon2)
    return tot

# --------- CAPAS MAPA ---------
def capa_nodos(df_nodos, rgb):
    plot = df_nodos.dropna(subset=["lat","lon"]).copy()
    if plot.empty: return None, pd.DataFrame()
    plot.rename(columns={"lon":"lng"}, inplace=True)
    layer = pdk.Layer("ScatterplotLayer", data=plot,
                      get_position="[lng, lat]", get_radius=65,
                      radius_min_pixels=3, get_fill_color=rgb,
                      get_line_color=[30,30,30], line_width_min_pixels=1,
                      pickable=True, tooltip=True)
    return layer, plot

def capa_aristas(df_aristas, df_nodos, rgb, width_px=3):
    if df_aristas.empty: return None, []
    idx = df_nodos.set_index("id")[["lat","lon"]]
    paths = []
    for _, r in df_aristas.iterrows():
        a = str(r.get("origen","")).strip()
        b = str(r.get("destino","")).strip()
        if a in idx.index and b in idx.index:
            latA, lonA = idx.loc[a, ["lat","lon"]]
            latB, lonB = idx.loc[b, ["lat","lon"]]
            if pd.notna(latA) and pd.notna(lonA) and pd.notna(latB) and pd.notna(lonB):
                paths.append({"path": [[lonA,latA],[lonB,latB]], "origen": a, "destino": b})
    if not paths: return None, []
    layer = pdk.Layer("PathLayer", data=paths, get_path="path",
                      get_width=width_px, width_scale=8,
                      get_color=rgb, pickable=True, tooltip=True)
    return layer, paths

def capa_ruta(poly_lonlat, rgb, width_px=8):
    if not poly_lonlat: return None
    layer = pdk.Layer("PathLayer", data=[{"path": poly_lonlat}],
                      get_path="path", get_width=width_px,
                      width_scale=8, get_color=rgb,
                      pickable=False)
    return layer

def capa_pin(lat, lon, rgb, radius=120):
    return pdk.Layer("ScatterplotLayer",
                     data=[{"lat":lat,"lng":lon}],
                     get_position="[lng, lat]",
                     get_radius=radius, radius_min_pixels=5,
                     get_fill_color=rgb, get_line_color=[20,20,20],
                     line_width_min_pixels=1, pickable=False)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("Parámetros")
    origen_nombre  = st.selectbox("Origen",  sorted(nodos["nombre"]))
    destino_nombre = st.selectbox("Destino", sorted(nodos["nombre"]),
                                  index=1 if len(nodos["nombre"])>1 else 0)

    st.markdown("### Visualización")
    show_nodes = st.toggle("Mostrar nodos del grafo", True)
    show_edges = st.toggle("Mostrar aristas del grafo", True)

    col_nodes = st.color_picker("Color nodos", "#FF5CA8")
    col_edges = st.color_picker("Color aristas", "#FFB000")
    col_path  = st.color_picker("Color ruta origen→destino", "#2F80ED")
    usar_osrm = st.toggle("Ruta real por calle (OSRM)", value=True)

    st.markdown("---")
    st.markdown("### Coordenadas actuales")
    # MOSTRAR (solo lectura) las coords actuales de O/D
    fila_o_view = nodos.loc[nodos["nombre"]==origen_nombre].iloc[0]
    fila_d_view = nodos.loc[nodos["nombre"]==destino_nombre].iloc[0]
    col1, col2 = st.columns(2)
    with col1:
        st.text_input("Lat (Origen)", value="" if pd.isna(fila_o_view["lat"]) else str(fila_o_view["lat"]), disabled=True)
    with col2:
        st.text_input("Lon (Origen)", value="" if pd.isna(fila_o_view["lon"]) else str(fila_o_view["lon"]), disabled=True)
    col1, col2 = st.columns(2)
    with col1:
        st.text_input("Lat (Destino)", value="" if pd.isna(fila_d_view["lat"]) else str(fila_d_view["lat"]), disabled=True)
    with col2:
        st.text_input("Lon (Destino)", value="" if pd.isna(fila_d_view["lon"]) else str(fila_d_view["lon"]), disabled=True)

    with st.expander("Editar coordenadas"):
        def editor(etq, nombre_sel):
            fila = nodos.loc[nodos["nombre"]==nombre_sel].iloc[0]
            c1,c2 = st.columns(2)
            with c1:
                lat_txt = st.text_input(f"Lat ({etq})", value="" if pd.isna(fila["lat"]) else str(fila["lat"]), key=f"lat_{etq}")
            with c2:
                lon_txt = st.text_input(f"Lon ({etq})", value="" if pd.isna(fila["lon"]) else str(fila["lon"]), key=f"lon_{etq}")
            if st.button(f"Guardar coords de {etq}"):
                try:
                    lat = float(str(lat_txt).replace(",","."))
                    lon = float(str(lon_txt).replace(",","."))
                    st.session_state.nodos_mem.loc[nodos["nombre"]==nombre_sel, ["lat","lon"]] = [lat,lon]
                    st.success(f"Coordenadas guardadas para {nombre_sel}: ({lat}, {lon})")
                except ValueError:
                    st.error("Lat/Lon inválidos. Ej: 14.9712  y  -91.7815")
        editor("Origen", origen_nombre)
        editor("Destino", destino_nombre)

# ---------------- RUTA / MÉTRICAS ----------------
origen_id  = nombre_a_id(nodos, origen_nombre)
destino_id = nombre_a_id(nodos, destino_nombre)
graph = build_graph_edges(aristas_raw)
path_ids = bfs_shortest_path(graph, origen_id, destino_id) if (origen_id and destino_id) else []

# paradas (si no hay camino, contamos 2 por defecto)
paradas_tot = len(path_ids) if path_ids else 2
paradas_int = max(0, paradas_tot-2)

# intentamos OSRM si hay coords en O/D
dist_km_osrm = dur_min_osrm = None
ruta_osrm = None
if origen_id and destino_id:
    fila_o = nodos.loc[nodos["id"]==origen_id].iloc[0]
    fila_d = nodos.loc[nodos["id"]==destino_id].iloc[0]
    if tiene_coords(fila_o) and tiene_coords(fila_d) and usar_osrm:
        ruta_osrm, dist_km_osrm, dur_min_osrm = osrm_route(float(fila_o["lat"]), float(fila_o["lon"]),
                                                            float(fila_d["lat"]), float(fila_d["lon"]))

# polilínea del grafo (si tiene coords)
ruta_grafo = ids_a_polyline_lonlat(nodos, path_ids) if path_ids else []

# Distancia y tiempo “finales”
VEL_KMH = 30.0  # velocidad promedio urbana
dist_km_final = None
dur_min_final = None

if ruta_osrm and dist_km_osrm is not None:
    dist_km_final = dist_km_osrm
    dur_min_final = dur_min_osrm
elif ruta_grafo:
    # sumar por tramos del grafo
    dist_grafo = distancia_km_sobre_polyline(ruta_grafo)
    if dist_grafo is not None:
        dist_km_final = dist_grafo
        dur_min_final = (dist_grafo / VEL_KMH) * 60.0
elif origen_id and destino_id and tiene_coords(fila_o) and tiene_coords(fila_d):
    # al menos Haversine directo
    dist_km_final = haversine_km(float(fila_o["lat"]), float(fila_o["lon"]),
                                 float(fila_d["lat"]), float(fila_d["lon"]))
    dur_min_final = (dist_km_final / VEL_KMH) * 60.0
else:
    # último fallback: usar saltos del grafo si los hay
    if path_ids and len(path_ids) > 1:
        hops = len(path_ids) - 1
        dur_min_final = hops * 3.0  # 3 min por salto

# ---------------- CAPAS Y MAPA ----------------
RGB_NODES = hex_to_rgb(col_nodes)
RGB_EDGES = hex_to_rgb(col_edges)
RGB_PATH  = hex_to_rgb(col_path)

layers = []
all_coords = []

# nodos
layer_nodes, nodos_plot = capa_nodos(nodos, RGB_NODES)
if show_nodes and layer_nodes:
    layers.append(layer_nodes)
    all_coords.extend(nodos_plot[["lng","lat"]].values.tolist())

# aristas (delgadas)
layer_edges, edges_paths = capa_aristas(aristas_raw, nodos, RGB_EDGES, width_px=3)
if show_edges and layer_edges:
    layers.append(layer_edges)
    for seg in edges_paths: all_coords.extend(seg["path"])

# ruta (GRUESA, por encima de todo)
ruta_final = ruta_osrm if ruta_osrm else ruta_grafo
layer_route = capa_ruta(ruta_final, RGB_PATH, width_px=8)
if layer_route:
    layers.append(layer_route)
    all_coords.extend(ruta_final)

# pines O/D si hay coords
if origen_id and destino_id:
    if tiene_coords(fila_o):
        layers.append(capa_pin(float(fila_o["lat"]), float(fila_o["lon"]), [0,200,0], radius=150))
    if tiene_coords(fila_d):
        layers.append(capa_pin(float(fila_d["lat"]), float(fila_d["lon"]), [200,0,0], radius=150))

# vista
view_state = fit_view_from_lonlat(all_coords, 0.4) if all_coords else pdk.ViewState(latitude=14.965, longitude=-91.79, zoom=13)

# ---------------- RESUMEN ----------------
criterio = "⏱ tiempo mín"
grafoTxt = "No dirigido"

dist_txt  = f"{dist_km_final:.2f} km" if dist_km_final is not None else "—"
tiempo_txt = f"{dur_min_final:.1f} min" if dur_min_final is not None else "—"
costo_txt  = tiempo_txt

col1, col2 = st.columns([1,2])
with col1:
    st.subheader("Resumen")
    st.markdown(f"**Origen:** {origen_nombre}")
    st.markdown(f"**Destino:** {destino_nombre}")
    st.markdown(f"**Criterio:** {criterio}")
    st.markdown(f"**Grafo:** {grafoTxt}")
    st.markdown(f"**Paradas (incluye origen y destino):** {paradas_tot}")
    st.markdown(f"**Paradas intermedias:** {paradas_int}")
    st.markdown(f"**Distancia aprox.:** {dist_txt}")
    st.markdown(f"**Tiempo aprox.:** {tiempo_txt}")
    st.markdown(f"**Costo total (tiempo mín):** {costo_txt}")

    if ruta_final:
        export_df = pd.DataFrame(ruta_final, columns=["lon","lat"])
        st.download_button("📥 Descargar ruta (CSV)",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name="ruta.csv", mime="text/csv")

    st.markdown("---")
    st.markdown("**Nodos cargados (con coordenadas):**")
    st.dataframe(nodos.dropna(subset=["lat","lon"])[["id","nombre","lat","lon"]], use_container_width=True)
    st.markdown("**Aristas cargadas:**")
    st.dataframe(aristas_raw, use_container_width=True)

with col2:
    st.pydeck_chart(
        pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            tooltip={"html": "<b>{origen}</b> → <b>{destino}</b>", "style": {"color": "white"}},
        ),
        use_container_width=True,
    )

# aviso si no se pudo dibujar
if not ruta_final:
    st.info("No se pudo dibujar la ruta en el mapa (faltan coordenadas en alguna parada). El resumen se calculó con el grafo.")

