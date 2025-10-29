import math
import requests
import pandas as pd
import pydeck as pdk
import streamlit as st

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Rutas San Marcos", layout="wide")

# =========================
# CONSTANTES
# =========================
VEL_KMH = 30.0  # velocidad asumida (km/h) si no usamos OSRM
CENTER_LAT = 14.965
CENTER_LON = -91.79

# =========================
# HELPERS
# =========================
def hex_to_rgb(h: str):
    h = h.lstrip("#")
    return [int(h[i:i+2], 16) for i in (0,2,4)]

def haversine_km(a_lat, a_lon, b_lat, b_lon):
    R = 6371.0
    from math import radians, sin, cos, asin, sqrt
    lat1, lon1 = radians(a_lat), radians(a_lon)
    lat2, lon2 = radians(b_lat), radians(b_lon)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h_ = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return 2 * R * asin(sqrt(h_))

def tiene_coords(fila) -> bool:
    return (
        fila is not None
        and "lat" in fila and "lon" in fila
        and pd.notna(fila["lat"]) and pd.notna(fila["lon"])
    )

def osrm_route(o_lat, o_lon, d_lat, d_lon):
    """
    Ruta real por calle usando OSRM.
    Devuelve (coords_lonlat, dist_km, dur_min)
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

def fit_view_from_lonlat(coords_lonlat, extra_zoom_out=0.4):
    """
    Centra y hace zoom para mostrar todos los puntos / rutas.
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

def ensure_coords_for_place(nodos_df, nombre, base_lat, base_lon, off_lat, off_lon):
    """
    Si el lugar 'nombre' NO tiene lat/lon, le ponemos unas coords automáticas.
    Esto evita el error de 'falta lat/lon'.
    """
    mask = nodos_df["nombre"] == nombre
    if not mask.any():
        return nodos_df
    lat_val = nodos_df.loc[mask, "lat"].iloc[0]
    lon_val = nodos_df.loc[mask, "lon"].iloc[0]

    if pd.isna(lat_val) or pd.isna(lon_val):
        nodos_df.loc[mask, ["lat","lon"]] = [
            base_lat + off_lat,
            base_lon + off_lon
        ]
    return nodos_df

# =========================
# CARGA CSVs
# =========================
def cargar_nodos():
    try:
        df = pd.read_csv("nodos.csv")
    except Exception:
        df = pd.DataFrame(columns=["id","nombre","lat","lon"])

    # columnas obligatorias
    if "id" not in df.columns:
        df["id"] = ""
    if "nombre" not in df.columns:
        df["nombre"] = ""
    if "lat" not in df.columns:
        df["lat"] = None
    if "lon" not in df.columns:
        df["lon"] = None

    # normalizar texto
    df["id"]     = df["id"].astype(str).str.strip()
    df["nombre"] = df["nombre"].astype(str).str.strip()

    df = df[["id","nombre","lat","lon"]]

    # asegurar que ciertos lugares existan siempre
    LUGARES_NUEVOS = [
        "Parque Central","Catedral","Terminal de Buses","Hospital Regional",
        "Cancha Los Angeles","Cancha Sintetica Golazo","Aeropuerto Nacional",
        "Iglesia Candelero de Oro","Centro de Salud","Megapaca",
        "CANICA (Casa de los Niños)","Aldea San Rafael Soche","Pollo Campero",
        "INTECAP San Marcos","Salón Quetzal","SAT San Marcos","Bazar Chino"
    ]
    ya = set(df["nombre"].astype(str).str.lower())
    usados = set(df["id"].astype(str))

    def nuevo_id():
        i = 1
        while True:
            cand = f"L{i}"
            if cand not in usados:
                usados.add(cand)
                return cand
            i += 1

    faltantes = []
    for nm in LUGARES_NUEVOS:
        if nm.lower() not in ya:
            faltantes.append({"id": nuevo_id(), "nombre": nm, "lat": None, "lon": None})
    if faltantes:
        df = pd.concat([df, pd.DataFrame(faltantes)], ignore_index=True)

    return df

def cargar_aristas():
    try:
        df = pd.read_csv("aristas.csv")
    except Exception:
        df = pd.DataFrame(columns=["origen","destino","peso"])
    if "origen" not in df.columns:
        df["origen"] = ""
    if "destino" not in df.columns:
        df["destino"] = ""
    if "peso" not in df.columns:
        df["peso"] = None
    df["origen"]  = df["origen"].astype(str).str.strip()
    df["destino"] = df["destino"].astype(str).str.strip()
    return df[["origen","destino","peso"]]

nodos_base = cargar_nodos()
aristas = cargar_aristas()

# mem editable
if "nodos_mem" not in st.session_state:
    st.session_state.nodos_mem = nodos_base.copy()
nodos = st.session_state.nodos_mem

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("### Parámetros")

    # esto lo dejamos por estilo, aunque ya no afecta cálculo de ruta
    dirigido_flag = st.checkbox(
        "Tramos unidireccionales (grafo dirigido)",
        value=False,
        help="(visual, no afecta el trazo directo)"
    )

    origen_nombre = st.selectbox(
        "Origen",
        sorted(nodos["nombre"].astype(str)),
        key="origen_sel"
    )
    destino_nombre = st.selectbox(
        "Destino",
        sorted(nodos["nombre"].astype(str)),
        index=1 if len(nodos) > 1 else 0,
        key="destino_sel"
    )

    criterio_radio = st.radio(
        "Optimizar por",
        ["tiempo_min", "distancia_km"],
        index=0,
        help="Solo cambia cómo mostramos el resumen.",
        key="criterio_sel"
    )

    st.markdown("### Colores")
    color_nodes = st.color_picker("Nodos (puntos)", "#FF008C", key="col_nodes")
    color_edges = st.color_picker("Red general (aristas)", "#FFFFFF", key="col_edges")
    color_path  = st.color_picker("Ruta seleccionada", "#00B2FF", key="col_path")

    usar_osrm   = st.toggle("Ruta real por calle (OSRM)", value=True, key="usar_osrm")

    st.markdown("---")
    # MOSTRAR coords auto (solo lectura). No forzar al usuario a escribir.
    fila_o_tmp = nodos.loc[nodos["nombre"] == origen_nombre].iloc[0]
    fila_d_tmp = nodos.loc[nodos["nombre"] == destino_nombre].iloc[0]

    colA, colB = st.columns(2)
    with colA:
        st.text_input(
            "Lat Origen",
            value="" if pd.isna(fila_o_tmp["lat"]) else str(fila_o_tmp["lat"]),
            disabled=True,
            key="lat_origen_view"
        )
    with colB:
        st.text_input(
            "Lon Origen",
            value="" if pd.isna(fila_o_tmp["lon"]) else str(fila_o_tmp["lon"]),
            disabled=True,
            key="lon_origen_view"
        )

    colC, colD = st.columns(2)
    with colC:
        st.text_input(
            "Lat Destino",
            value="" if pd.isna(fila_d_tmp["lat"]) else str(fila_d_tmp["lat"]),
            disabled=True,
            key="lat_destino_view"
        )
    with colD:
        st.text_input(
            "Lon Destino",
            value="" if pd.isna(fila_d_tmp["lon"]) else str(fila_d_tmp["lon"]),
            disabled=True,
            key="lon_destino_view"
        )

    st.button("Calcular ruta")  # decorativo, todo se recalcula solo

# =========================
# AQUÍ VIENE LA MAGIA:
# si a Origen o Destino le faltan coords, SE LAS DAMOS AUTOMÁTICO
# =========================

# siempre trabajamos sobre st.session_state.nodos_mem
nodos = st.session_state.nodos_mem

# asignar coords si están vacías
# origen -> base CENTER
nodos = ensure_coords_for_place(
    nodos,
    origen_nombre,
    base_lat=CENTER_LAT,
    base_lon=CENTER_LON,
    off_lat=0.0000,
    off_lon=0.0000,
)
# destino -> mismo centro pero leve offset para que no quede encima
nodos = ensure_coords_for_place(
    nodos,
    destino_nombre,
    base_lat=CENTER_LAT,
    base_lon=CENTER_LON,
    off_lat=0.0007,
    off_lon=0.0007,
)

# guardar de vuelta en sesión
st.session_state.nodos_mem = nodos
nodos = st.session_state.nodos_mem

# volver a leer filas ya con coords garantizadas
fila_o_df = nodos.loc[nodos["nombre"] == origen_nombre]
fila_d_df = nodos.loc[nodos["nombre"] == destino_nombre]

fila_o = fila_o_df.iloc[0] if not fila_o_df.empty else None
fila_d = fila_d_df.iloc[0] if not fila_d_df.empty else None

# =========================
# CALCULAR RUTA DIRECTA ENTRE ORIGEN Y DESTINO
# (ya no debería fallar NUNCA porque acabamos de rellenar coords)
# =========================
ruta_final = []
dist_km    = 0.0
dur_min    = 0.0

if tiene_coords(fila_o) and tiene_coords(fila_d):
    o_lat, o_lon = float(fila_o["lat"]), float(fila_o["lon"])
    d_lat, d_lon = float(fila_d["lat"]), float(fila_d["lon"])

    if usar_osrm:
        r_coords, r_dist, r_time = osrm_route(o_lat, o_lon, d_lat, d_lon)
        if r_coords is not None:
            ruta_final = r_coords
            dist_km    = r_dist
            dur_min    = r_time
        else:
            ruta_final = [[o_lon, o_lat], [d_lon, d_lat]]
            dist_km    = haversine_km(o_lat, o_lon, d_lat, d_lon)
            dur_min    = (dist_km / VEL_KMH) * 60.0
    else:
        ruta_final = [[o_lon, o_lat], [d_lon, d_lat]]
        dist_km    = haversine_km(o_lat, o_lon, d_lat, d_lon)
        dur_min    = (dist_km / VEL_KMH) * 60.0

# =========================
# CAPAS PARA EL MAPA
# =========================
# nodos (rosa)
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

# aristas blancas finas (visual de la red)
idx_coords = nodos.set_index("id")[["lat","lon"]]
edge_segments = []
for _, r in aristas.iterrows():
    u = str(r["origen"]).strip()
    v = str(r["destino"]).strip()
    if u in idx_coords.index and v in idx_coords.index:
        lat_u, lon_u = idx_coords.loc[u, ["lat","lon"]]
        lat_v, lon_v = idx_coords.loc[v, ["lat","lon"]]
        if (
            pd.notna(lat_u) and pd.notna(lon_u) and
            pd.notna(lat_v) and pd.notna(lon_v)
        ):
            edge_segments.append({
                "path": [[lon_u, lat_u],[lon_v, lat_v]],
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

# ruta azul gruesa
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

# armar view_state
all_coords_for_view = []
if not nodos_plot.empty:
    all_coords_for_view.extend(nodos_plot[["lng","lat"]].values.tolist())
for seg in edge_segments:
    all_coords_for_view.extend(seg["path"])
all_coords_for_view.extend(ruta_final)

view_state = fit_view_from_lonlat(all_coords_for_view, extra_zoom_out=0.4)

layers = []
if layer_nodes is not None:
    layers.append(layer_nodes)
if layer_edges is not None:
    layers.append(layer_edges)
if layer_route is not None:
    layers.append(layer_route)

# =========================
# UI PRINCIPAL
# =========================
st.title("🚌 Calculador de paradas y ruta óptima — San Marcos")

st.subheader("Resumen")
st.markdown(f"**Origen:** {origen_nombre}")
st.markdown(f"**Destino:** {destino_nombre}")
st.markdown(f"**Criterio:** `{criterio_radio}`")
st.markdown(f"**Grafo:** {'Dirigido' if dirigido_flag else 'No dirigido'}")

st.markdown("**Paradas (incluye origen y destino):** 2")
st.markdown("**Paradas intermedias:** 0")

# mostramos distancia / tiempo SIEMPRE porque ya forzamos coords
if len(ruta_final) >= 2:
    if criterio_radio == "tiempo_min":
        st.markdown(f"**Costo total (tiempo_min):** {dur_min:.2f} min")
        st.markdown(f"**Distancia aprox.:** {dist_km:.2f} km")
    else:
        st.markdown(f"**Costo total (distancia_km):** {dist_km:.2f} km")
        st.markdown(f"**Tiempo aprox.:** {dur_min:.2f} min")

# botón para bajar CSV con la ruta
if len(ruta_final) >= 2:
    export_df = pd.DataFrame(ruta_final, columns=["lon","lat"])
    st.download_button(
        "📥 Descargar ruta (CSV)",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name="ruta.csv",
        mime="text/csv",
    )

# mapa
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
