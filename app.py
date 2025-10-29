import math
import pandas as pd
import pydeck as pdk
import streamlit as st

# =========================
# CONFIG BÁSICA
# =========================
st.set_page_config(page_title="Rutas San Marcos", layout="wide")

# =========================
# CONSTANTES
# =========================
VEL_KMH = 30.0  # velocidad estimada en km/h para tiempo aprox
CENTER_LAT = 14.965
CENTER_LON = -91.79

# =========================
# HELPERS
# =========================
def hex_to_rgb(h: str):
    h = h.lstrip("#")
    return [int(h[i:i+2], 16) for i in (0, 2, 4)]

def haversine_km(a_lat, a_lon, b_lat, b_lon):
    R = 6371.0
    from math import radians, sin, cos, asin, sqrt
    lat1, lon1 = radians(a_lat), radians(a_lon)
    lat2, lon2 = radians(b_lat), radians(b_lon)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h_ = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return 2 * R * asin(sqrt(h_))

def fit_view_from_lonlat(coords_lonlat, extra_zoom_out=0.4):
    # arma viewstate para centrar el mapa
    if not coords_lonlat:
        return pdk.ViewState(
            latitude=CENTER_LAT,
            longitude=CENTER_LON,
            zoom=13,
            pitch=0,
            bearing=0,
        )
    df_bounds = pd.DataFrame(coords_lonlat, columns=["lon", "lat"])
    raw_view = pdk.data_utils.compute_view(df_bounds[["lon", "lat"]])

    # pydeck a veces da dict, a veces objeto
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

def jitter_from_name(name: str):
    """
    Generamos un pequeño 'desplazamiento' único por nombre
    para que cada parada sin coords tenga una ubicación distinta cerca del centro.
    """
    if name is None:
        name = "X"

    h = abs(hash(name))
    off_lat_raw = (h % 200) - 100         # -100 .. 99
    off_lon_raw = ((h // 200) % 200) - 100

    off_lat = off_lat_raw * 0.00005       # ±0.005 grados aprox
    off_lon = off_lon_raw * 0.00005

    return off_lat, off_lon

def fill_missing_coords_for_all_nodes(nodos_df: pd.DataFrame) -> pd.DataFrame:
    """
    Recorre TODOS los nodos.
    Si algún nodo no tiene lat/lon, le damos coords = centro + jitter único.
    Devuelve SIEMPRE nodos con lat/lon válidos.
    """
    nodos_df = nodos_df.copy()
    for idx, row in nodos_df.iterrows():
        lat_val = row.get("lat", None)
        lon_val = row.get("lon", None)

        if pd.isna(lat_val) or pd.isna(lon_val):
            off_lat, off_lon = jitter_from_name(str(row.get("nombre", row.get("id", ""))))
            nodos_df.at[idx, "lat"] = CENTER_LAT + off_lat
            nodos_df.at[idx, "lon"] = CENTER_LON + off_lon

    return nodos_df

def get_row_by_name(df: pd.DataFrame, nombre: str):
    sel = df.loc[df["nombre"] == nombre]
    if sel.empty:
        return None
    return sel.iloc[0]

# =========================
# CARGA CSVs
# =========================
def cargar_nodos():
    try:
        df = pd.read_csv("nodos.csv")
    except Exception:
        df = pd.DataFrame(columns=["id", "nombre", "lat", "lon"])

    # columnas mínimas
    if "id" not in df.columns:
        df["id"] = ""
    if "nombre" not in df.columns:
        df["nombre"] = ""
    if "lat" not in df.columns:
        df["lat"] = None
    if "lon" not in df.columns:
        df["lon"] = None

    df["id"] = df["id"].astype(str).str.strip()
    df["nombre"] = df["nombre"].astype(str).str.strip()

    df = df[["id", "nombre", "lat", "lon"]]

    # asegurar que estos lugares existan (aunque sin coords en el CSV original)
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

    nuevos_rows = []
    for nm in LUGARES_NUEVOS:
        if nm.lower() not in ya:
            nuevos_rows.append({
                "id": nuevo_id(),
                "nombre": nm,
                "lat": None,
                "lon": None,
            })

    if nuevos_rows:
        df = pd.concat([df, pd.DataFrame(nuevos_rows)], ignore_index=True)

    return df

def cargar_aristas():
    try:
        df = pd.read_csv("aristas.csv")
    except Exception:
        df = pd.DataFrame(columns=["origen", "destino", "peso"])

    if "origen" not in df.columns:
        df["origen"] = ""
    if "destino" not in df.columns:
        df["destino"] = ""
    if "peso" not in df.columns:
        df["peso"] = None

    df["origen"] = df["origen"].astype(str).str.strip()
    df["destino"] = df["destino"].astype(str).str.strip()

    return df[["origen", "destino", "peso"]]

# =========================
# CARGAMOS DATOS BASE
# =========================
nodos_raw = cargar_nodos()
aristas_raw = cargar_aristas()

if "nodos_full" not in st.session_state:
    st.session_state.nodos_full = nodos_raw.copy()

nodos_session = st.session_state.nodos_full.copy()

# =========================
# SIDEBAR (UI)
# =========================
with st.sidebar:
    st.markdown("### Parámetros")

    dirigido_flag = st.checkbox(
        "Tramos unidireccionales (grafo dirigido)",
        value=False,
        help="(solo visual / informativo, ya no afecta el cálculo de la línea azul)"
    )

    origen_nombre = st.selectbox(
        "Origen",
        sorted(nodos_session["nombre"].astype(str)),
        key="origen_sel"
    )
    destino_nombre = st.selectbox(
        "Destino",
        sorted(nodos_session["nombre"].astype(str)),
        index=1 if len(nodos_session) > 1 else 0,
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

    st.button("Calcular ruta")  # decorativo, el cálculo corre solo

# =========================
# COMPLETAMOS COORDENADAS PARA *TODOS* LOS NODOS
# =========================
nodos_full = fill_missing_coords_for_all_nodes(nodos_session)
st.session_state.nodos_full = nodos_full.copy()

fila_o = get_row_by_name(nodos_full, origen_nombre)
fila_d = get_row_by_name(nodos_full, destino_nombre)

o_lat, o_lon = float(fila_o["lat"]), float(fila_o["lon"])
d_lat, d_lon = float(fila_d["lat"]), float(fila_d["lon"])

# Línea azul simple origen -> destino
ruta_final = [
    [o_lon, o_lat],
    [d_lon, d_lat],
]

# distancia y tiempo aprox
dist_km = haversine_km(o_lat, o_lon, d_lat, d_lon)
dur_min = (dist_km / VEL_KMH) * 60.0 if dist_km > 0 else 0.0

# =========================
# ARMAR CAPAS DEL MAPA
# =========================

# ----------- Capa nodos (rosa) -----------
nodos_plot = nodos_full.copy()
nodos_plot.rename(columns={"lon": "lng"}, inplace=True)

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

# ----------- Capa aristas reales (aristas.csv) -> líneas blancas existentes -----------
idx_coords = nodos_full.set_index("id")[["lat","lon"]]
edge_segments = []
for _, r in aristas_raw.iterrows():
    u = str(r["origen"]).strip()
    v = str(r["destino"]).strip()
    if u in idx_coords.index and v in idx_coords.index:
        lat_u, lon_u = idx_coords.loc[u, ["lat","lon"]]
        lat_v, lon_v = idx_coords.loc[v, ["lat","lon"]]
        edge_segments.append({
            "path": [[lon_u, lat_u], [lon_v, lat_v]],
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

# ----------- NUEVO: Cadena que conecta TODOS los nodos en orden -----------
# Idea: ordenar por lon, luego lat. Eso nos da una "ruta" estable
# que pasa por todos los puntos rosa para que no queden sueltos.
sorted_nodes = nodos_full.sort_values(by=["lon", "lat"], ascending=[True, True]).reset_index(drop=True)

all_nodes_path = [
    [float(row["lon"]), float(row["lat"])]
    for _, row in sorted_nodes.iterrows()
]

layer_allnodes_chain = None
if len(all_nodes_path) > 1:
    layer_allnodes_chain = pdk.Layer(
        "PathLayer",
        data=[{"path": all_nodes_path}],
        get_path="path",
        get_width=2,          # más delgado que la ruta azul
        width_scale=8,
        get_color=hex_to_rgb(color_edges),  # mismo color que las aristas blancas
        pickable=False,
    )

# ----------- Capa ruta azul gruesa (origen -> destino) -----------
layer_route = pdk.Layer(
    "PathLayer",
    data=[{"path": ruta_final}],
    get_path="path",
    get_width=6,
    width_scale=8,
    get_color=hex_to_rgb(color_path),
    pickable=False,
)

# =========================
# VIEWSTATE (ZOOM AUTO)
# =========================
all_coords_for_view = []
all_coords_for_view.extend([[float(x["lng"]), float(x["lat"])] for _, x in nodos_plot.iterrows()])
for seg in edge_segments:
    all_coords_for_view.extend(seg["path"])
all_coords_for_view.extend(ruta_final)
all_coords_for_view.extend(all_nodes_path)  # NUEVO: usar también la cadena completa

view_state = fit_view_from_lonlat(all_coords_for_view, extra_zoom_out=0.4)

# =========================
# JUNTAR CAPAS (el orden importa visualmente)
# nodes abajo de todo no importa, lo que importa es dejar la azul encima
# =========================
layers = []
layers.append(layer_nodes)

if layer_edges is not None:
    layers.append(layer_edges)

if layer_allnodes_chain is not None:
    layers.append(layer_allnodes_chain)

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

# en este modo siempre son Origen y Destino directos
st.markdown("**Paradas (incluye origen y destino):** 2")
st.markdown("**Paradas intermedias:** 0")

if criterio_radio == "tiempo_min":
    st.markdown(f"**Costo total (tiempo_min):** {dur_min:.2f} min")
    st.markdown(f"**Distancia aprox.:** {dist_km:.2f} km")
else:
    st.markdown(f"**Costo total (distancia_km):** {dist_km:.2f} km")
    st.markdown(f"**Tiempo aprox.:** {dur_min:.2f} min")

# botón CSV con la línea azul actual
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
