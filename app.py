# app.py
import math
import requests
import pandas as pd
import pydeck as pdk
import streamlit as st

# ---------------- UI / CONFIG ----------------
st.set_page_config(page_title="Rutas San Marcos", layout="wide")
st.title("🚌 Calculador de paradas — San Marcos (modo sin grafo)")

# ---------------- Datos base ----------------
try:
    nodos = pd.read_csv("nodos.csv")  # columnas: id, nombre, lat, lon
except Exception:
    nodos = pd.DataFrame(columns=["id", "nombre", "lat", "lon"])

LUGARES_NUEVOS = [
    "Parque Central", "Catedral", "Terminal de Buses", "Hospital Regional",
    "Cancha Los Angeles", "Cancha Sintetica Golazo", "Aeropuerto Nacional",
    "Iglesia Candelero de Oro", "Centro de Salud", "Megapaca",
    "CANICA (Casa de los Niños)", "Aldea San Rafael Soche", "Pollo Campero",
    "INTECAP San Marcos", "Salón Quetzal", "SAT San Marcos", "Bazar Chino"
]

# Normalización y columnas obligatorias
for col in ["id", "nombre"]:
    if col in nodos.columns:
        nodos[col] = nodos[col].astype(str).str.strip()

for c in ["id", "nombre", "lat", "lon"]:
    if c not in nodos.columns:
        nodos[c] = None
nodos = nodos[["id", "nombre", "lat", "lon"]]

# Asegurar que todos los lugares existan en 'nodos'
def asegurar_lugares(df: pd.DataFrame, nombres: list) -> pd.DataFrame:
    existentes = set(df["nombre"].astype(str).str.lower()) if "nombre" in df else set()
    usados = set(df["id"].astype(str)) if "id" in df else set()

    def nuevo_id(start=1):
        i = start
        while True:
            candidate = f"L{i}"
            if candidate not in usados:
                usados.add(candidate)
                return candidate
            i += 1

    filas = []
    for nm in nombres:
        if nm.lower() not in existentes:
            filas.append({"id": nuevo_id(), "nombre": nm, "lat": None, "lon": None})
    if filas:
        df = pd.concat([df, pd.DataFrame(filas)], ignore_index=True)
    return df

nodos = asegurar_lugares(nodos, LUGARES_NUEVOS)

# Memoria (para editar coordenadas sin tocar CSV)
if "nodos_mem" not in st.session_state:
    st.session_state.nodos_mem = nodos.copy()
nodos = st.session_state.nodos_mem

# ---------------- Helpers ----------------
def hex_to_rgb(h: str):
    h = h.lstrip("#")
    return [int(h[i:i+2], 16) for i in (0, 2, 4)]

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
    """
    Devuelve: (path_lonlat, dist_km, dur_min) o (None, None, None) si falla.
    path_lonlat: lista [[lon, lat], ...] siguiendo la calle.
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
    Calcula un ViewState que encuadra todas las coordenadas (lon,lat) usando
    pdk.data_utils.compute_view y aplica un pequeño padding con 'extra_zoom_out'.
    """
    if not coords_lonlat:
        # fallback a San Marcos aprox.
        return pdk.ViewState(latitude=14.965, longitude=-91.79, zoom=13)
    df_bounds = pd.DataFrame(coords_lonlat, columns=["lon", "lat"])
    view = pdk.data_utils.compute_view(df_bounds[["lon", "lat"]])
    # padding pequeño para que no corte los extremos
    view["zoom"] = max(1, view["zoom"] - extra_zoom_out)
    return pdk.ViewState(**view, pitch=0, bearing=0)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("Parámetros")

    origen_nombre = st.selectbox("Origen", sorted(nodos["nombre"]))
    destino_nombre = st.selectbox("Destino", sorted(nodos["nombre"]), index=1)

    st.markdown("### Visualización")
    show_nodes = st.toggle("Mostrar nodos", value=False)  # por defecto ocultos
    col_nodes = st.color_picker("Color de nodos", "#FF007F")
    col_path  = st.color_picker("Color de ruta", "#007AFF")
    usar_osrm = st.toggle("Ruta real por calle (OSRM)", value=True)

    st.markdown("---")
    st.markdown("### Agregar/editar coordenadas del lugar seleccionado")

    def editor_de_coords(etiqueta, nombre_sel):
        fila = nodos.loc[nodos["nombre"] == nombre_sel].iloc[0]
        col1, col2 = st.columns(2)
        with col1:
            lat_txt = st.text_input(
                f"Lat ({etiqueta})",
                value="" if pd.isna(fila["lat"]) else str(fila["lat"]),
                key=f"lat_{etiqueta}",
            )
        with col2:
            lon_txt = st.text_input(
                f"Lon ({etiqueta})",
                value="" if pd.isna(fila["lon"]) else str(fila["lon"]),
                key=f"lon_{etiqueta}",
            )
        if st.button(f"Guardar coords de {etiqueta}"):
            try:
                lat = float(str(lat_txt).replace(",", "."))
                lon = float(str(lon_txt).replace(",", "."))
                st.session_state.nodos_mem.loc[nodos["nombre"] == nombre_sel, ["lat", "lon"]] = [lat, lon]
                st.success(f"Coordenadas guardadas para {nombre_sel}: ({lat}, {lon})")
            except ValueError:
                st.error("Lat/Lon inválidos. Usa números (ej. 14.9712 y -91.7815)")

    editor_de_coords("Origen", origen_nombre)
    editor_de_coords("Destino", destino_nombre)

# ---------------- MAPA ----------------
RGB_NODES = hex_to_rgb(col_nodes)
RGB_PATH  = hex_to_rgb(col_path)

# Capa de nodos (solo los que ya tengan coordenadas)
nodos_plot = nodos.dropna(subset=["lat", "lon"]).copy()
nodos_plot.rename(columns={"lon": "lng"}, inplace=True)

layers = []
if show_nodes and not nodos_plot.empty:
    nodes_layer = pdk.Layer(
        "ScatterplotLayer",
        data=nodos_plot,
        get_position="[lng, lat]",
        get_radius=65,
        radius_min_pixels=3,
        get_fill_color=RGB_NODES,
        get_line_color=[30, 30, 30],
        line_width_min_pixels=1,
        pickable=False,
    )
    layers.append(nodes_layer)

# Obtener filas origen/destino
fila_o = nodos.loc[nodos["nombre"] == origen_nombre].iloc[0]
fila_d = nodos.loc[nodos["nombre"] == destino_nombre].iloc[0]

# Si hay coords en ambos, dibujar ruta automáticamente
if tiene_coords(fila_o) and tiene_coords(fila_d):
    o_lat, o_lon = float(fila_o["lat"]), float(fila_o["lon"])
    d_lat, d_lon = float(fila_d["lat"]), float(fila_d["lon"])

    path_lonlat, dist_km, dur_min = (None, None, None)
    if usar_osrm:
        path_lonlat, dist_km, dur_min = osrm_route(o_lat, o_lon, d_lat, d_lon)

    # Fallback: línea recta
    if path_lonlat is None:
        dist_km = haversine_km(o_lat, o_lon, d_lat, d_lon)
        vel_kmh = 30.0
        dur_min = (dist_km / vel_kmh) * 60.0
        path_lonlat = [[o_lon, o_lat], [d_lon, d_lat]]

    # Capa de camino
    path_layer = pdk.Layer(
        "PathLayer",
        data=[{"path": path_lonlat}],
        get_path="path",
        get_width=6,
        width_scale=8,
        get_color=RGB_PATH,
        pickable=False,
    )
    layers.append(path_layer)

    # Zoom/encuadre automático a la ruta
    view_state = fit_view_from_lonlat(path_lonlat, extra_zoom_out=0.45)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Resumen")
        st.markdown(f"**Origen:** {origen_nombre}")
        st.markdown(f"**Destino:** {destino_nombre}")
        st.markdown(f"**Distancia aprox.:** {dist_km:.2f} km")
        st.markdown(f"**Tiempo aprox.:** {dur_min:.1f} min")
        export_df = pd.DataFrame(path_lonlat, columns=["lon", "lat"])
        st.download_button(
            "📥 Descargar ruta (CSV)",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name="ruta.csv",
            mime="text/csv",
        )
    with col2:
        st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view_state), use_container_width=True)

else:
    # Sin ruta: si hay nodos visibles, encuadra esos; si no, fallback a centro
    if show_nodes and not nodos_plot.empty:
        coords = nodos_plot[["lng", "lat"]].values.tolist()
        view_state = fit_view_from_lonlat(coords, extra_zoom_out=0.25)
    else:
        view_state = pdk.ViewState(latitude=14.965, longitude=-91.79, zoom=13)

    st.info("Selecciona origen y destino, y asigna lat/lon a ambos en la barra lateral. La ruta se dibuja automáticamente.")
    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view_state), use_container_width=True)
