import streamlit as st
import pandas as pd
import pydeck as pdk

st.set_page_config(page_title="🚌 Rutas San Marcos", layout="wide")

# =========================
# DATOS BASE
# =========================
# Carga tus puntos guardados
try:
    nodos = pd.read_csv("nodos.csv")  # id, nombre, lat, lon
except Exception:
    nodos = pd.DataFrame(columns=["id", "nombre", "lat", "lon"])

# Punto inicial del mapa (centro de San Marcos más o menos)
MAP_CENTER_LAT = 14.9635
MAP_CENTER_LON = -91.7960

# =========================
# SIDEBAR / CABECERA
# =========================
st.markdown("### 🗺️ Calculador de paradas en San Marcos")

# Hacemos 3 columnas arriba para que no se vea todo en una pila gigante
col_sel, col_origen, col_dest = st.columns([1.2, 1, 1])

# -------- Columna: punto seleccionado --------
with col_sel:
    st.subheader("📍 Lugar seleccionado / manual")
    lat_sel = st.text_input("Lat", key="lat_sel")
    lon_sel = st.text_input("Lon", key="lon_sel")

    if st.button("Guardar/Actualizar este punto"):
        # TODO: aquí agregas/actualizas en tu DataFrame `nodos`
        # Ejemplo básico:
        # nuevo = {"id": len(nodos)+1, "nombre": f"Punto {len(nodos)+1}",
        #          "lat": float(lat_sel), "lon": float(lon_sel)}
        # nodos = pd.concat([nodos, pd.DataFrame([nuevo])], ignore_index=True)
        # nodos.to_csv("nodos.csv", index=False)
        st.success("Punto guardado (ejemplo).")

# -------- Columna: ORIGEN --------
with col_origen:
    st.subheader("🟢 Origen")
    lat_origen = st.text_input("Lat (Origen)", key="lat_origen")
    lon_origen = st.text_input("Lon (Origen)", key="lon_origen")

    if st.button("Guardar coords de Origen"):
        # TODO: guarda origen en session_state o variable global
        st.session_state["origen"] = (lat_origen, lon_origen)
        st.success("Origen guardado.")

# -------- Columna: DESTINO --------
with col_dest:
    st.subheader("🔴 Destino")
    lat_dest = st.text_input("Lat (Destino)", key="lat_dest")
    lon_dest = st.text_input("Lon (Destino)", key="lon_dest")

    if st.button("Guardar coords de Destino"):
        st.session_state["destino"] = (lat_dest, lon_dest)
        st.success("Destino guardado.")

# =========================
# SEGUNDA FILA: acciones generales
# =========================
col_accion, col_zoom = st.columns([1,1])

with col_accion:
    if st.button("Calcular ruta"):
        # TODO: acá llamas tu lógica de ruta
        # Ejemplo:
        # ruta_df = calcular_ruta(st.session_state["origen"], st.session_state["destino"])
        st.info("(Demo) Aquí se calcularía la ruta con tus aristas.")

with col_zoom:
    zoom_value = st.slider("Zoom del mapa", min_value=10.0, max_value=20.0, value=14.5, step=0.1)

st.divider()

# =========================
# MAPA
# =========================
st.subheader("Mapa")

# Capa de puntos (paradas / nodos)
if not nodos.empty:
    layer_puntos = pdk.Layer(
        "ScatterplotLayer",
        data=nodos,
        get_position='[lon, lat]',
        get_radius=25,
        pickable=True,
        get_fill_color='[255, 100, 100]',  # puedes ajustar color si querés
    )
    capas = [layer_puntos]
else:
    capas = []

# Estado de vista (usa el zoom del slider)
view_state = pdk.ViewState(
    latitude=MAP_CENTER_LAT,
    longitude=MAP_CENTER_LON,
    zoom=zoom_value,
    pitch=0,
)

st.pydeck_chart(
    pdk.Deck(
        map_style="mapbox://styles/mapbox/dark-v11",
        initial_view_state=view_state,
        layers=capas,
        tooltip={"text": "{nombre}\nLat: {lat}\nLon: {lon}"},
    )
)

