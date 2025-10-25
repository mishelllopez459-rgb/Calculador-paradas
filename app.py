import streamlit as st
import pandas as pd
import pydeck as pdk
import math

# ---------------- UI / CONFIG ----------------
st.set_page_config(page_title="Rutas San Marcos", layout="wide")

st.markdown("## 🚌 Calculador de paradas — San Marcos")

# ---------------- CARGA / PREPARACIÓN DE DATOS ----------------
# 1) Carga nodos existentes
try:
    nodos = pd.read_csv("nodos.csv")  # columnas: id, nombre, lat, lon
except Exception:
    nodos = pd.DataFrame(columns=["id", "nombre", "lat", "lon"])

# 2) Lista de lugares base
LUGARES_NUEVOS = [
    "Parque Central", "Catedral", "Terminal de Buses", "Hospital Regional",
    "Cancha Los Angeles", "Cancha Sintetica Golazo", "Aeropuerto Nacional",
    "Iglesia Candelero de Oro", "Centro de Salud", "Megapaca",
    "CANICA (Casa de los Niños)", "Aldea San Rafael Soche", "Pollo Campero",
    "INTECAP San Marcos", "Salón Quetzal", "SAT San Marcos", "Bazar Chino"
]

# 3) Normalización mínima
if "id" in nodos.columns:
    nodos["id"] = nodos["id"].astype(str).str.strip()
if "nombre" in nodos.columns:
    nodos["nombre"] = nodos["nombre"].astype(str).str.strip()
# Asegurar columnas
for col in ["id", "nombre", "lat", "lon"]:
    if col not in nodos.columns:
        nodos[col] = None

# 4) Asegurar que todos los lugares existan aunque no tengan coord
def asegurar_lugares(df: pd.DataFrame, nombres: list) -> pd.DataFrame:
    existentes = set(df["nombre"].str.lower()) if "nombre" in df else set()
    rows = []
    usados = set(df["id"].astype(str)) if "id" in df else set()

    def nuevo_id_local():
        # genera L1, L2, ... que no choquen
        i = 1
        while f"L{i}" in usados:
            i += 1
        nid = f"L{i}"
        usados.add(nid)
        return nid

    for nm in nombres:
        if nm.lower() not in existentes:
            rows.append({"id": nuevo_id_local(), "nombre": nm, "lat": None, "lon": None})

    if rows:
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)

    return df

nodos = asegurar_lugares(nodos, LUGARES_NUEVOS)

# Guardar copia viva en session_state
if "nodos_mem" not in st.session_state:
    st.session_state.nodos_mem = nodos.copy()

# trabajamos SIEMPRE sobre st.session_state.nodos_mem
nodos = st.session_state.nodos_mem

# helper para generar id nuevo (para el formulario "Nuevo punto")
def generar_nuevo_id(df: pd.DataFrame) -> str:
    usados = set(df["id"].astype(str))
    i = 1
    while f"L{i}" in usados:
        i += 1
    return f"L{i}"

# ---------------- PRIMERA FILA: FORMULARIOS ----------------
col_nuevo, col_origen, col_destino = st.columns([1.1, 1, 1])

# ------ Columna: crear punto nuevo manual ------
with col_nuevo:
    st.subheader("📍 Nuevo punto manual")

    nombre_nuevo = st.text_input("Nombre del punto", key="nombre_nuevo")
    lat_nuevo_txt = st.text_input("Lat (nuevo)", key="lat_nuevo_txt")
    lon_nuevo_txt = st.text_input("Lon (nuevo)", key="lon_nuevo_txt")

    if st.button("➕ Agregar punto", key="btn_add_punto"):
        if nombre_nuevo.strip() == "":
            st.error("Ponle nombre al punto.")
        else:
            try:
                lat_val = float(lat_nuevo_txt.replace(",", "."))
                lon_val = float(lon_nuevo_txt.replace(",", "."))
                nuevo_row = {
                    "id": generar_nuevo_id(nodos),
                    "nombre": nombre_nuevo.strip(),
                    "lat": lat_val,
                    "lon": lon_val,
                }
                st.session_state.nodos_mem = pd.concat(
                    [st.session_state.nodos_mem, pd.DataFrame([nuevo_row])],
                    ignore_index=True
                )
                st.success(f"Punto '{nombre_nuevo}' agregado.")
            except ValueError:
                st.error("Lat/Lon inválidos. Usa números (ej. 14.9712 y -91.7815)")

# Necesitamos refrescar `nodos` después de posible alta
nodos = st.session_state.nodos_mem

# ------ Columna: ORIGEN ------
with col_origen:
    st.subheader("🟢 Origen")

    origen_nombre = st.selectbox(
        "Lugar origen",
        sorted(nodos["nombre"]),
        key="origen_nombre"
    )

    fila_o = nodos.loc[nodos["nombre"] == origen_nombre].iloc[0]

    lat_origen_txt = st.text_input(
        "Lat (Origen)",
        value="" if pd.isna(fila_o["lat"]) else str(fila_o["lat"]),
        key="lat_origen_txt"
    )
    lon_origen_txt = st.text_input(
        "Lon (Origen)",
        value="" if pd.isna(fila_o["lon"]) else str(fila_o["lon"]),
        key="lon_origen_txt"
    )

    if st.button("Guardar coords de Origen", key="btn_save_origen"):
        try:
            lat_val = float(lat_origen_txt.replace(",", "."))
            lon_val = float(lon_origen_txt.replace(",", "."))
            st.session_state.nodos_mem.loc[
                st.session_state.nodos_mem["nombre"] == origen_nombre, ["lat", "lon"]
            ] = [lat_val, lon_val]
            st.success(f"Coordenadas guardadas para {origen_nombre}.")
        except ValueError:
            st.error("Lat/Lon inválidos en Origen.")

# ------ Columna: DESTINO ------
with col_destino:
    st.subheader("🔴 Destino")

    destino_nombre = st.selectbox(
        "Lugar destino",
        sorted(nodos["nombre"]),
        key="destino_nombre"
    )

    fila_d = nodos.loc[nodos["nombre"] == destino_nombre].iloc[0]

    lat_destino_txt = st.text_input(
        "Lat (Destino)",
        value="" if pd.isna(fila_d["lat"]) else str(fila_d["lat"]),
        key="lat_destino_txt"
    )
    lon_destino_txt = st.text_input(
        "Lon (Destino)",
        value="" if pd.isna(fila_d["lon"]) else str(fila_d["lon"]),
        key="lon_destino_txt"
    )

    if st.button("Guardar coords de Destino", key="btn_save_destino"):
        try:
            lat_val = float(lat_destino_txt.replace(",", "."))
            lon_val = float(lon_destino_txt.replace(",", "."))
            st.session_state.nodos_mem.loc[
                st.session_state.nodos_mem["nombre"] == destino_nombre, ["lat", "lon"]
            ] = [lat_val, lon_val]
            st.success(f"Coordenadas guardadas para {destino_nombre}.")
        except ValueError:
            st.error("Lat/Lon inválidos en Destino.")

# refrescar de nuevo por si modificaron coords
nodos = st.session_state.nodos_mem

st.divider()

# ---------------- SEGUNDA FILA: ACCIONES GENERALES ----------------
col_accion, col_zoom, col_color = st.columns([1, 1, 1])

with col_accion:
    st.subheader("Ruta")
    calcular = st.button("🚍 Calcular ruta", key="btn_calcular")

with col_zoom:
    st.subheader("Zoom mapa")
    zoom_value = st.slider(
        "Nivel de zoom",
        min_value=10.0,
        max_value=20.0,
        value=14.0,
        step=0.1,
        key="zoom_slider"
    )

with col_color:
    st.subheader("Colores")
    col_nodes = st.color_picker("Color nodos", "#FF007F", key="col_nodes")
    col_path  = st.color_picker("Color ruta", "#007AFF", key="col_path")

st.divider()

# ---------------- FUNCIONES AUX ----------------
def hex_to_rgb(h: str):
    h = h.lstrip("#")
    return [int(h[i:i+2], 16) for i in (0, 2, 4)]

RGB_NODES = hex_to_rgb(col_nodes)
RGB_PATH  = hex_to_rgb(col_path)

def haversine_km(a_lat, a_lon, b_lat, b_lon):
    R = 6371.0
    lat1, lon1 = math.radians(a_lat), math.radians(a_lon)
    lat2, lon2 = math.radians(b_lat), math.radians(b_lon)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(h))

# ---------------- MAPA ----------------
st.subheader("🗺️ Mapa")

# Solo plotear puntos que sí tienen coords
nodos_plot = nodos.dropna(subset=["lat", "lon"]).copy()
nodos_plot = nodos_plot.rename(columns={"lon": "lng"})

nodes_layer = pdk.Layer(
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

# Centro del mapa -> promedio o fallback
if len(nodos_plot):
    center_lat = float(nodos_plot["lat"].mean())
    center_lon = float(nodos_plot["lng"].mean())
else:
    center_lat = 14.965
    center_lon = -91.79

view_state = pdk.ViewState(
    latitude=center_lat,
    longitude=center_lon,
    zoom=zoom_value,
    pitch=0
)

col_info, col_map = st.columns([1, 2])

# Lógica al presionar "Calcular ruta"
if calcular:
    fila_o = nodos.loc[nodos["nombre"] == origen_nombre].iloc[0]
    fila_d = nodos.loc[nodos["nombre"] == destino_nombre].iloc[0]

    # Validar coords
    if (
        pd.isna(fila_o["lat"]) or pd.isna(fila_o["lon"]) or
        pd.isna(fila_d["lat"]) or pd.isna(fila_d["lon"])
    ):
        with col_info:
            st.error("Faltan coordenadas en origen o destino. Completa lat/lon arriba y vuelve a calcular.")
        with col_map:
            st.pydeck_chart(
                pdk.Deck(
                    layers=[nodes_layer],
                    initial_view_state=view_state,
                    tooltip={"text": "{nombre}\nLat: {lat}\nLon: {lng}"}
                ),
                use_container_width=True
            )

    else:
        # Distancia y tiempo aprox
        dist_km = haversine_km(fila_o["lat"], fila_o["lon"], fila_d["lat"], fila_d["lon"])
        vel_kmh = 30.0  # suposición bus urbano
        t_min = (dist_km / vel_kmh) * 60.0

        # DataFrame tramo directo
        tramo_df = pd.DataFrame([
            {"nombre": origen_nombre,  "lat": fila_o["lat"], "lon": fila_o["lon"]},
            {"nombre": destino_nombre, "lat": fila_d["lat"], "lon": fila_d["lon"]},
        ])

        path_layer = pdk.Layer(
            "PathLayer",
            data=[{"path": tramo_df[["lon", "lat"]].values.tolist()}],
            get_path="path",
            get_width=6,
            width_scale=8,
            get_color=RGB_PATH,
            pickable=False,
        )

        with col_info:
            st.markdown("### Resumen de la ruta")
            st.markdown(f"**Origen:** {origen_nombre}")
            st.markdown(f"**Destino:** {destino_nombre}")
            st.markdown(f"**Distancia directa aprox.:** {dist_km:.2f} km")
            st.markdown(f"**Tiempo aprox. (30 km/h):** {t_min:.1f} min")

            st.download_button(
                "📥 Descargar puntos (CSV)",
                data=tramo_df.to_csv(index=False).encode("utf-8"),
                file_name="puntos_directo.csv",
                mime="text/csv"
            )

            st.dataframe(tramo_df, use_container_width=True)

        with col_map:
            st.pydeck_chart(
                pdk.Deck(
                    layers=[nodes_layer, path_layer],
                    initial_view_state=view_state,
                    tooltip={"text": "{nombre}\nLat: {lat}\nLon: {lng}"}
                ),
                use_container_width=True
            )

else:
    with col_info:
        st.info("👈 Selecciona Origen y Destino, edita coordenadas si hace falta y luego presiona 'Calcular ruta'.")
    with col_map:
        st.pydeck_chart(
            pdk.Deck(
                layers=[nodes_layer],
                initial_view_state=view_state,
                tooltip={"text": "{nombre}\nLat: {lat}\nLon: {lng}"}
            ),
            use_container_width=True
        )

# (Opcional) Si quieres persistir cambios a disco cada vez que cambie algo:
# st.session_state.nodos_mem.to_csv("nodos.csv", index=False)


