import streamlit as st
import pandas as pd
import pydeck as pdk
import math

# ---------------- UI / CONFIG ----------------
st.set_page_config(page_title="Rutas San Marcos", layout="wide")
st.title("🚌 Calculador de paradas — San Marcos (modo sin grafo)")

# 1) Carga de nodos existentes (si no hay, creamos DataFrame vacío)
try:
    nodos = pd.read_csv("nodos.csv")  # columnas: id, nombre, lat, lon
except Exception:
    nodos = pd.DataFrame(columns=["id", "nombre", "lat", "lon"])

# 2) Lista de lugares (sin coordenadas de momento)
LUGARES_NUEVOS = [
    "Parque Central", "Catedral", "Terminal de Buses", "Hospital Regional",
    "Cancha Los Angeles", "Cancha Sintetica Golazo", "Aeropuerto Nacional",
    "Iglesia Candelero de Oro", "Centro de Salud", "Megapaca",
    "CANICA (Casa de los Niños)", "Aldea San Rafael Soche", "Pollo Campero",
    "INTECAP San Marcos", "Salón Quetzal", "SAT San Marcos", "Bazar Chino"
]

# 3) Normalización mínima
for col in ["id", "nombre"]:
    if col in nodos.columns:
        nodos[col] = nodos[col].astype(str).str.strip()
else:
    nodos = nodos.reindex(columns=["id", "nombre", "lat", "lon"])

# 4) Asegurar que todos los lugares existan en 'nodos' (aunque sea sin lat/lon)
def asegurar_lugares(df: pd.DataFrame, nombres: list) -> pd.DataFrame:
    existentes = set(df["nombre"].str.lower()) if "nombre" in df else set()
    rows = []
    next_id_num = 1
    usados = set(df["id"].astype(str)) if "id" in df else set()
    # generar ids L1, L2, ... que no choquen
    def nuevo_id():
        nonlocal next_id_num
        while f"L{next_id_num}" in usados:
            next_id_num += 1
        nid = f"L{next_id_num}"
        usados.add(nid)
        next_id_num += 1
        return nid

    for nm in nombres:
        if nm.lower() not in existentes:
            rows.append({"id": nuevo_id(), "nombre": nm, "lat": None, "lon": None})
    if rows:
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    return df

nodos = asegurar_lugares(nodos, LUGARES_NUEVOS)

# Guardar en session_state para poder actualizar sin tocar el CSV todavía
if "nodos_mem" not in st.session_state:
    st.session_state.nodos_mem = nodos.copy()

nodos = st.session_state.nodos_mem

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("Parámetros")

    origen_nombre = st.selectbox("Origen", sorted(nodos["nombre"]))
    destino_nombre = st.selectbox("Destino", sorted(nodos["nombre"]), index=1)

    st.markdown("### Colores")
    col_nodes = st.color_picker("Nodos", "#FF007F")
    col_path  = st.color_picker("Ruta seleccionada", "#007AFF")

    st.markdown("---")
    st.markdown("### Agregar/editar coordenadas del lugar seleccionado")

    def editor_de_coords(etiqueta, nombre_sel):
        fila = nodos.loc[nodos["nombre"] == nombre_sel].iloc[0]
        col1, col2 = st.columns(2)
        with col1:
            lat_txt = st.text_input(
                f"Lat ({etiqueta})",
                value="" if pd.isna(fila["lat"]) else str(fila["lat"]),
                key=f"lat_{etiqueta}"
            )
        with col2:
            lon_txt = st.text_input(
                f"Lon ({etiqueta})",
                value="" if pd.isna(fila["lon"]) else str(fila["lon"]),
                key=f"lon_{etiqueta}"
            )
        if st.button(f"Guardar coords de {etiqueta}"):
            try:
                lat = float(str(lat_txt).replace(",", "."))
                lon = float(str(lon_txt).replace(",", "."))
                st.session_state.nodos_mem.loc[
                    nodos["nombre"] == nombre_sel, ["lat", "lon"]
                ] = [lat, lon]
                st.success(
                    f"Coordenadas guardadas para {nombre_sel}: ({lat}, {lon})"
                )
            except ValueError:
                st.error("Lat/Lon inválidos. Usa números (ej. 14.9712 y -91.7815)")

    editor_de_coords("Origen", origen_nombre)
    editor_de_coords("Destino", destino_nombre)

    calcular = st.button("Calcular ruta")

# ---------------- UTILIDADES ----------------
def hex_to_rgb(h: str):
    h = h.lstrip("#")
    return [int(h[i:i+2], 16) for i in (0, 2, 4)]

RGB_NODES = hex_to_rgb(col_nodes)
RGB_PATH  = hex_to_rgb(col_path)

def haversine_km(a_lat, a_lon, b_lat, b_lon):
    R = 6371.0
    lat1, lon1 = math.radians(a_lat), math.radians(a_lon)
    lat2, lon2 = math.radians(b_lat), math.radians(b_lon)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(h))

# ---------------- MAPA / LAYOUT ----------------
col1, col2 = st.columns([1, 2])

# valores por defecto del mapa (sin marcar ruta todavía)
DEFAULT_LAT = 14.965
DEFAULT_LON = -91.79
DEFAULT_ZOOM = 13

# Ahora ya NO creamos la capa de todos los nodos aquí.
# En lugar de eso vamos a construir dinámicamente según si se calculó ruta o no.

if calcular:
    # agarrar filas de origen y destino
    fila_o = nodos.loc[nodos["nombre"] == origen_nombre].iloc[0]
    fila_d = nodos.loc[nodos["nombre"] == destino_nombre].iloc[0]

    # validar coords
    if (
        pd.isna(fila_o["lat"]) or pd.isna(fila_o["lon"]) or
        pd.isna(fila_d["lat"]) or pd.isna(fila_d["lon"])
    ):
        # coords incompletas => no mostramos puntos
        st.error(
            "Faltan coordenadas en uno o ambos lugares. Completa lat/lon en la barra lateral y vuelve a calcular."
        )

        # centro por defecto
        view_state = pdk.ViewState(latitude=DEFAULT_LAT,
                                   longitude=DEFAULT_LON,
                                   zoom=DEFAULT_ZOOM)

        # Mapa SIN nodos SIN ruta
        deck_layers = []

        st.pydeck_chart(
            pdk.Deck(layers=deck_layers, initial_view_state=view_state),
            use_container_width=True
        )

    else:
        # Tenemos coords => construimos ruta directa y mostramos SOLO esos nodos
        dist_km = haversine_km(
            fila_o["lat"], fila_o["lon"],
            fila_d["lat"], fila_d["lon"]
        )
        vel_kmh = 30.0  # supuesta velocidad bus
        t_min = (dist_km / vel_kmh) * 60.0

        tramo_df = pd.DataFrame([
            {"nombre": origen_nombre, "lat": fila_o["lat"], "lon": fila_o["lon"]},
            {"nombre": destino_nombre, "lat": fila_d["lat"], "lon": fila_d["lon"]},
        ])

        # capa nodos SOLO origen y destino
        tramo_plot = tramo_df.rename(columns={"lon": "lng"})
        nodes_layer = pdk.Layer(
            "ScatterplotLayer",
            data=tramo_plot,
            get_position="[lng, lat]",
            get_radius=65,
            radius_min_pixels=3,
            get_fill_color=RGB_NODES,
            get_line_color=[30, 30, 30],
            line_width_min_pixels=1,
            pickable=True,
        )

        # capa ruta (arista entre origen y destino)
        path_layer = pdk.Layer(
            "PathLayer",
            data=[{"path": tramo_df[["lon", "lat"]].values.tolist()}],
            get_path="path",
            get_width=6,
            width_scale=8,
            get_color=RGB_PATH,
            pickable=False,
        )

        # centramos vista en el promedio de origen/destino
        center_lat = tramo_df["lat"].mean()
        center_lon = tramo_df["lon"].mean()
        view_state = pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=DEFAULT_ZOOM
        )

        with col1:
            st.subheader("Resumen")
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

        with col2:
            st.pydeck_chart(
                pdk.Deck(
                    layers=[nodes_layer, path_layer],
                    initial_view_state=view_state
                ),
                use_container_width=True
            )

else:
    # todavía NO se ha calculado ruta:
    #  - NO mostramos nodos
    #  - NO mostramos aristas
    # solo el mapa vacío centrado en default y un mensajito
    st.info(
        "Selecciona origen y destino en la barra lateral. "
        "Guarda coordenadas si hace falta y presiona 'Calcular ruta'."
    )

    view_state = pdk.ViewState(
        latitude=DEFAULT_LAT,
        longitude=DEFAULT_LON,
        zoom=DEFAULT_ZOOM
    )

    # sin capas
    deck_layers = []

    st.pydeck_chart(
        pdk.Deck(layers=deck_layers, initial_view_state=view_state),
        use_container_width=True
    )


