import math
import requests
import pandas as pd
import pydeck as pdk
import streamlit as st

# ---------------------------------
# CONFIG STREAMLIT
# ---------------------------------
st.set_page_config(page_title="Rutas San Marcos", layout="wide")

# ---------------------------------
# HELPERS BÁSICOS
# ---------------------------------

VEL_KMH = 30.0  # velocidad asumida para estimar tiempo cuando no hay OSRM
CENTER_LAT = 14.965
CENTER_LON = -91.79

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
    h = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return 2 * R * asin(sqrt(h))

def osrm_route(o_lat, o_lon, d_lat, d_lon):
    """
    Llama OSRM público para sacar ruta calle a calle.
    Devuelve (coords_lonlat, dist_km, dur_min)
    coords_lonlat = [[lon,lat], ...]
    Si falla: (None, None, None)
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
    Calcula el encuadre del mapa para que se vea toda la ruta.
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

    # pydeck puede devolver dict o un objeto con attrs
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

def tiene_coords(row):
    return (row is not None
            and "lat" in row and "lon" in row
            and pd.notna(row["lat"]) and pd.notna(row["lon"]))

# ---------------------------------
# CARGA DE NODOS
# Debe existir nodos.csv con columnas: id,nombre,lat,lon
# ---------------------------------

def cargar_nodos():
    try:
        df = pd.read_csv("nodos.csv")
    except Exception:
        df = pd.DataFrame(columns=["id","nombre","lat","lon"])

    # normalizar texto
    for col in ["id", "nombre"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
        else:
            df[col] = ""

    # asegurar columnas
    for c in ["id","nombre","lat","lon"]:
        if c not in df.columns:
            df[c] = None

    df = df[["id","nombre","lat","lon"]]

    # si quieres que ciertos lugares siempre aparezcan aunque estén sin coords:
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
    rows_to_add = []
    for nm in LUGARES_NUEVOS:
        if nm.lower() not in existentes:
            rows_to_add.append(
                {"id": nuevo_id(), "nombre": nm, "lat": None, "lon": None}
            )
    if rows_to_add:
        df = pd.concat([df, pd.DataFrame(rows_to_add)], ignore_index=True)

    return df

nodos = cargar_nodos()

# guardar en sesión para estabilidad (en caso de ediciones futuras)
if "nodos_mem" not in st.session_state:
    st.session_state.nodos_mem = nodos.copy()
nodos = st.session_state.nodos_mem

# ---------------------------------
# SIDEBAR (lo que el usuario controla)
# ---------------------------------

with st.sidebar:
    st.markdown("### Parámetros")

    origen_nombre = st.selectbox(
        "Origen",
        sorted(nodos["nombre"].astype(str))
    )

    destino_nombre = st.selectbox(
        "Destino",
        sorted(nodos["nombre"].astype(str)),
        index=1 if len(nodos) > 1 else 0
    )

    color_nodes = st.color_picker("Color nodos (puntos)", "#FF008C")   # rosa
    color_path  = st.color_picker("Color ruta origen→destino", "#00B2FF")  # celeste
    usar_osrm   = st.toggle("Ruta real por calle (OSRM)", value=True)

    # Mostrar coordenadas actuales de los seleccionados (solo lectura)
    st.markdown("---")
    st.markdown("#### Coordenadas actuales")

    fila_o_tmp = nodos.loc[nodos["nombre"] == origen_nombre]
    fila_d_tmp = nodos.loc[nodos["nombre"] == destino_nombre]

    lat_o = fila_o_tmp.iloc[0]["lat"] if not fila_o_tmp.empty else ""
    lon_o = fila_o_tmp.iloc[0]["lon"] if not fila_o_tmp.empty else ""
    lat_d = fila_d_tmp.iloc[0]["lat"] if not fila_d_tmp.empty else ""
    lon_d = fila_d_tmp.iloc[0]["lon"] if not fila_d_tmp.empty else ""

    colA, colB = st.columns(2)
    with colA:
        st.text_input("Lat (Origen)", value="" if pd.isna(lat_o) else str(lat_o), disabled=True)
    with colB:
        st.text_input("Lon (Origen)", value="" if pd.isna(lon_o) else str(lon_o), disabled=True)

    colC, colD = st.columns(2)
    with colC:
        st.text_input("Lat (Destino)", value="" if pd.isna(lat_d) else str(lat_d), disabled=True)
    with colD:
        st.text_input("Lon (Destino)", value="" if pd.isna(lon_d) else str(lon_d), disabled=True)

    # Botón "Calcular ruta" solo para que se mire bonito, pero realmente
    # Streamlit recalcula automático
    st.button("Calcular ruta")

# ---------------------------------
# SACAR FILAS ORIGEN / DESTINO
# ---------------------------------

fila_o = nodos.loc[nodos["nombre"] == origen_nombre]
fila_d = nodos.loc[nodos["nombre"] == destino_nombre]

fila_o = fila_o.iloc[0] if not fila_o.empty else None
fila_d = fila_d.iloc[0] if not fila_d.empty else None

# ---------------------------------
# CALCULAR RUTA ENTRE ESOS 2 PUNTOS
# ---------------------------------

ruta_lonlat = []   # [[lon,lat], ...] que vamos a dibujar
dist_km     = None
dur_min     = None
mensaje     = None

if tiene_coords(fila_o) and tiene_coords(fila_d):
    o_lat, o_lon = float(fila_o["lat"]), float(fila_o["lon"])
    d_lat, d_lon = float(fila_d["lat"]), float(fila_d["lon"])

    if usar_osrm:
        # intentar ruta real por calle
        rcoords, rdist, rtime = osrm_route(o_lat, o_lon, d_lat, d_lon)
        if rcoords is not None:
            ruta_lonlat = rcoords
            dist_km = rdist
            dur_min = rtime
        else:
            # fallback recta
            ruta_lonlat = [[o_lon, o_lat], [d_lon, d_lat]]
            dist_km = haversine_km(o_lat, o_lon, d_lat, d_lon)
            dur_min = (dist_km / VEL_KMH) * 60.0
    else:
        # recta directa (sin OSRM)
        ruta_lonlat = [[o_lon, o_lat], [d_lon, d_lat]]
        dist_km = haversine_km(o_lat, o_lon, d_lat, d_lon)
        dur_min = (dist_km / VEL_KMH) * 60.0
else:
    # no se puede porque falta coord en alguno
    mensaje = "No se pudo calcular la ruta (a uno de los dos puntos le faltan coordenadas lat/lon)."

# ---------------------------------
# ARMAR CAPAS DEL MAPA
# ---------------------------------

# capa nodos (scatter)
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
        get_line_color=[30, 30, 30],
        line_width_min_pixels=1,
        pickable=True,
    )

# capa ruta (PathLayer)
layer_path = None
if len(ruta_lonlat) >= 2:
    layer_path = pdk.Layer(
        "PathLayer",
        data=[{"path": ruta_lonlat}],
        get_path="path",
        get_width=6,
        width_scale=8,
        get_color=hex_to_rgb(color_path),
        pickable=False,
    )

# vista mapa
coords_all = []
if not nodos_plot.empty:
    coords_all.extend(nodos_plot[["lng","lat"]].values.tolist())
coords_all.extend(ruta_lonlat)

view_state = fit_view_from_lonlat(coords_all, extra_zoom_out=0.4)

layers = []
if layer_nodes is not None:
    layers.append(layer_nodes)
if layer_path is not None:
    layers.append(layer_path)

# ---------------------------------
# UI PRINCIPAL
# ---------------------------------

st.title("🚌 Calculador de ruta — San Marcos")

st.subheader("Resumen")

st.markdown(f"**Origen:** {origen_nombre}")
st.markdown(f"**Destino:** {destino_nombre}")

if dist_km is not None and dur_min is not None:
    st.markdown(f"**Distancia aprox.:** {dist_km:.2f} km")
    st.markdown(f"**Tiempo aprox.:** {dur_min:.1f} min")
else:
    st.markdown("**Distancia aprox.:** —")
    st.markdown("**Tiempo aprox.:** —")

if mensaje:
    st.warning(mensaje)

# Botón descarga CSV de la ruta dibujada
if len(ruta_lonlat) >= 2:
    export_df = pd.DataFrame(ruta_lonlat, columns=["lon","lat"])
    st.download_button(
        "📥 Descargar ruta (CSV)",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name="ruta.csv",
        mime="text/csv",
    )

# mapa abajo
st.pydeck_chart(
    pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        tooltip={
            "html": "<b>Parada</b>",
            "style": {"color": "white"},
        },
    ),
    use_container_width=True,
)
