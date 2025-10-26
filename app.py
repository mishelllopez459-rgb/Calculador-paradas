import streamlit as st
import pandas as pd
import pydeck as pdk
import networkx as nx

# ---------------- CONFIG / UI ----------------
st.set_page_config(page_title="Rutas San Marcos", layout="wide")
st.title("🚌 Calculador de paradas y ruta óptima — San Marcos")

# ---------------- CARGA DE DATOS ----------------
# nodos.csv => columnas: id, nombre, lat, lon
# aristas.csv => columnas: origen, destino, tiempo_min, distancia_km, capacidad (opcional)
nodos = pd.read_csv("nodos.csv")
aristas = pd.read_csv("aristas.csv")

# Limpieza mínima
nodos["id"] = nodos["id"].astype(str).str.strip()
nodos["nombre"] = nodos["nombre"].astype(str).str.strip()
aristas["origen"] = aristas["origen"].astype(str).str.strip()
aristas["destino"] = aristas["destino"].astype(str).str.strip()

# Helpers de mapeo id <-> nombre
id_por_nombre = {r["nombre"]: r["id"] for _, r in nodos.iterrows()}
nombre_por_id = {r["id"]: r["nombre"] for _, r in nodos.iterrows()}

# ---------------- CONSTANTES DE TIEMPO ----------------
VEL_BUS_KMH = 30.0     # km/h bus
VEL_BICI_KMH = 15.0    # km/h bici

def tiempo_por_dist(dist_km: float, vel_kmh: float) -> float:
    # minutos = (km / kmh) * 60
    if vel_kmh <= 0:
        return 0.0
    return (dist_km / vel_kmh) * 60.0

def hex_to_rgb(h: str):
    """'#RRGGBB' -> [R,G,B]"""
    h = h.lstrip("#")
    return [int(h[i:i+2], 16) for i in (0, 2, 4)]

def rgb_to_hex(rgb):
    """[R,G,B] -> '#RRGGBB'"""
    return "#{:02X}{:02X}{:02X}".format(rgb[0], rgb[1], rgb[2])

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("Parámetros")

    dirigido = st.checkbox(
        "Tramos unidireccionales (grafo dirigido)",
        value=False,
        help="Activa para considerar el sentido de las aristas (A→B distinto de B→A)",
    )

    origen_nombre = st.selectbox("Origen", sorted(nodos["nombre"]))
    destino_nombre = st.selectbox("Destino", sorted(nodos["nombre"]), index=1)

    criterio = st.radio(
        "Optimizar por",
        ["tiempo_min", "distancia_km"],
        index=0,
        help="Ruta más rápida (tiempo_min) o más corta (distancia_km).",
    )

    st.markdown("### Colores")
    col_nodes = st.color_picker("Nodos (puntos)", "#FF007F")
    col_edges = st.color_picker("Red general (aristas)", "#9B9B9B")
    col_path  = st.color_picker("Ruta seleccionada", "#007AFF")

    calcular = st.button("Calcular ruta")

# Convertir colores a RGB (para pydeck) y HEX (para graphviz)
RGB_NODES = hex_to_rgb(col_nodes)
RGB_EDGES = hex_to_rgb(col_edges)
RGB_PATH  = hex_to_rgb(col_path)

HEX_NODES = rgb_to_hex(RGB_NODES)
HEX_EDGES = rgb_to_hex(RGB_EDGES)
HEX_PATH  = rgb_to_hex(RGB_PATH)

# ---------------- CREAR GRAFO NETWORKX ----------------
G = nx.DiGraph() if dirigido else nx.Graph()

# Agregar nodos a G con atributos
for _, r in nodos.iterrows():
    G.add_node(
        r["id"],
        nombre=r["nombre"],
        lat=float(r["lat"]),
        lon=float(r["lon"]),
    )

# Agregar aristas a G con atributos + enriquecer con tiempos bus/bici
aristas_enriquecidas = []
for _, r in aristas.iterrows():
    dist = float(r["distancia_km"])
    t_real_min = float(r["tiempo_min"])

    t_bus_min  = tiempo_por_dist(dist, VEL_BUS_KMH)
    t_bici_min = tiempo_por_dist(dist, VEL_BICI_KMH)

    data_edge = {
        "origen": r["origen"],
        "destino": r["destino"],
        "distancia_km": dist,
        "tiempo_min": t_real_min,
        "tiempo_bus_min": t_bus_min,
        "tiempo_bici_min": t_bici_min,
        "capacidad": float(r.get("capacidad", 0)),
    }
    aristas_enriquecidas.append(data_edge)

    G.add_edge(
        r["origen"],
        r["destino"],
        distancia_km=dist,
        tiempo_min=t_real_min,
        tiempo_bus_min=t_bus_min,
        tiempo_bici_min=t_bici_min,
        capacidad=float(r.get("capacidad", 0)),
    )

# DataFrame final de TODAS las aristas enriquecidas
aristas_full_df = pd.DataFrame(aristas_enriquecidas)

# ---------------- FUNCIÓN RUTA ÓPTIMA ----------------
def ruta_optima(o_id: str, d_id: str, peso: str):
    """
    Devuelve:
      path_ids: lista de ids de nodos en la ruta
      totals: dict con sumas totales
      tramos_df: DataFrame tramo a tramo con distancias y tiempos
    """
    path_ids = nx.shortest_path(G, source=o_id, target=d_id, weight=peso)

    detalle_tramos = []
    total_peso = 0.0
    total_dist = 0.0
    total_bus_min = 0.0
    total_bici_min = 0.0

    for u, v in zip(path_ids[:-1], path_ids[1:]):
        edge_data = G[u][v]

        total_peso     += edge_data[peso]
        total_dist     += edge_data["distancia_km"]
        total_bus_min  += edge_data["tiempo_bus_min"]
        total_bici_min += edge_data["tiempo_bici_min"]

        detalle_tramos.append({
            "origen_id": u,
            "origen_nombre": nombre_por_id[u],
            "destino_id": v,
            "destino_nombre": nombre_por_id[v],
            "distancia_km": edge_data["distancia_km"],
            "tiempo_min_real": edge_data["tiempo_min"],
            "tiempo_bus_min": edge_data["tiempo_bus_min"],
            "tiempo_bici_min": edge_data["tiempo_bici_min"],
        })

    resumen_totales = {
        "total_peso": total_peso,
        "total_distancia_km": total_dist,
        "total_bus_min": total_bus_min,
        "total_bici_min": total_bici_min,
    }

    return path_ids, resumen_totales, pd.DataFrame(detalle_tramos)

# ---------------- DATA PARA EL MAPA ----------------
# edges_df con coords y nombres para dibujar LineLayer
edges_df = (
    aristas_full_df
    .merge(
        nodos[["id", "lat", "lon", "nombre"]],
        left_on="origen", right_on="id", how="left"
    )
    .rename(columns={
        "lat": "lat_o",
        "lon": "lon_o",
        "nombre": "origen_nombre"
    })
    .drop(columns=["id"])
    .merge(
        nodos[["id", "lat", "lon", "nombre"]],
        left_on="destino", right_on="id", how="left"
    )
    .rename(columns={
        "lat": "lat_d",
        "lon": "lon_d",
        "nombre": "destino_nombre"
    })
    .drop(columns=["id"])
)

edges_layer = pdk.Layer(
    "LineLayer",
    data=edges_df,
    get_source_position="[lon_o, lat_o]",
    get_target_position="[lon_d, lat_d]",
    get_width=2,
    width_min_pixels=2,
    get_color=RGB_EDGES,
    pickable=True,
)

nodes_layer = pdk.Layer(
    "ScatterplotLayer",
    data=nodos.rename(columns={"lon": "lng"}),
    get_position="[lng, lat]",
    get_radius=65,
    radius_min_pixels=3,
    get_fill_color=RGB_NODES,
    get_line_color=[30, 30, 30],
    line_width_min_pixels=1,
    pickable=True,
)

center_lat, center_lon = nodos["lat"].mean(), nodos["lon"].mean()
view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=13)

# ---------------- GRAFO LÓGICO (GRAPHVIZ) ----------------
def build_graph_graphviz(nodos_df: pd.DataFrame,
                         aristas_df: pd.DataFrame,
                         hex_nodes: str,
                         hex_edges: str,
                         dirigido_flag: bool):
    """
    Grafo lógico (no mapa) que combina:
    - nodos/aristas reales del CSV
    - MÁS los nodos y conexiones urbanas que querías ver (Parque Central, Megapaca, etc.)
    Cada arista lleva label con:
      distancia_km
      min bus
      min bici
    """

    # 1. Nodos extra (aseguramos que estos lugares aparezcan visualmente)
    nodos_extra = [
        "Parque Central",
        "Catedral",
        "Pollo Campero",
        "Megapaca",
        "Bazar Chino",
        "Terminal de Buses",
        "SAT San Marcos",
        "INTECAP San Marcos",
        "Salón Quetzal",
        "Centro de Salud",
        "Hospital Regional",
        "Cancha Los Angeles",
        "Cancha Sintetica Golazo",
        "Iglesia Candelero de Oro",
        "CANICA (Casa de los Niños)",
        "Aldea San Rafael Soche",
        "Aeropuerto Nacional",
    ]

    # 2. Conexiones extra entre esos nodos (tu red urbana tipo ruta)
    conexiones_extra = [
        ("Parque Central", "Catedral"),
        ("Parque Central", "Pollo Campero"),
        ("Parque Central", "Megapaca"),
        ("Megapaca", "Bazar Chino"),
        ("Bazar Chino", "Terminal de Buses"),
        ("Pollo Campero", "Terminal de Buses"),

        ("Parque Central", "SAT San Marcos"),
        ("SAT San Marcos", "INTECAP San Marcos"),
        ("INTECAP San Marcos", "Salón Quetzal"),
        ("Salón Quetzal", "Centro de Salud"),
        ("Centro de Salud", "Hospital Regional"),

        ("Hospital Regional", "Cancha Los Angeles"),
        ("Cancha Los Angeles", "Cancha Sintetica Golazo"),
        ("Cancha Sintetica Golazo", "Iglesia Candelero de Oro"),
        ("Iglesia Candelero de Oro", "CANICA (Casa de los Niños)"),
        ("CANICA (Casa de los Niños)", "Aldea San Rafael Soche"),
        ("Aldea San Rafael Soche", "Aeropuerto Nacional"),

        ("Terminal de Buses", "Aeropuerto Nacional"),
        ("Hospital Regional", "Terminal de Buses"),
    ]

    # 3. Todos los nombres de nodos del CSV + los extra
    nombres_csv = [str(x) for x in nodos_df["nombre"].tolist()]
    nodos_unificados = sorted(set(nombres_csv + nodos_extra))

    # 4. Construir lista de TODAS las aristas:
    #    a) reales del CSV
    aristas_unificadas = []
    for _, r in aristas_df.iterrows():
        o_id = r["origen"]
        d_id = r["destino"]
        if o_id not in nombre_por_id or d_id not in nombre_por_id:
            continue

        o_name = nombre_por_id[o_id]
        d_name = nombre_por_id[d_id]

        dist_km = float(r["distancia_km"])
        bus_min = tiempo_por_dist(dist_km, VEL_BUS_KMH)
        bici_min = tiempo_por_dist(dist_km, VEL_BICI_KMH)

        aristas_unificadas.append({
            "o": o_name,
            "d": d_name,
            "dist_km": dist_km,
            "bus_min": bus_min,
            "bici_min": bici_min,
            "is_extra": False,
        })

    #    b) conexiones extra (si no existen ya)
    ya = set()
    for e in aristas_unificadas:
        key = tuple(sorted([e["o"], e["d"]]))
        ya.add(key)

    for a, b in conexiones_extra:
        key2 = tuple(sorted([a, b]))
        if key2 in ya and not dirigido_flag:
            continue
        ya.add(key2)

        # Distancia ficticia 1.0 km para que podamos calcular tiempo bus/bici
        dist_km = 1.0
        bus_min = tiempo_por_dist(dist_km, VEL_BUS_KMH)
        bici_min = tiempo_por_dist(dist_km, VEL_BICI_KMH)

        aristas_unificadas.append({
            "o": a,
            "d": b,
            "dist_km": dist_km,
            "bus_min": bus_min,
            "bici_min": bici_min,
            "is_extra": True,
        })

    # 5. Generar DOT
    dot_lines = []
    dot_lines.append("graph G {")
    dot_lines.append('  graph [layout=neato, overlap=false, splines=true];')
    dot_lines.append(
        f'  node [shape=circle, style=filled, fontname="Helvetica", '
        f'fontsize=10, color="#000000", fillcolor="{hex_nodes}"];'
    )
    dot_lines.append(
        f'  edge [color="{hex_edges}", fontname="Helvetica", fontsize=9, penwidth=2];'
    )

    # Nodos
    for name in nodos_unificados:
        safe_name = name.replace('"', '\\"')
        dot_lines.append(f'  "{safe_name}";')

    # Aristas con label "dist / bus / bici"
    usados_dot = set()
    for e in aristas_unificadas:
        o_name = e["o"].replace('"', '\\"')
        d_name = e["d"].replace('"', '\\"')

        if not dirigido_flag:
            key_d = tuple(sorted([o_name, d_name]))
            if key_d in usados_dot:
                continue
            usados_dot.add(key_d)

        edge_label = f'{e["dist_km"]:.2f} km / {e["bus_min"]:.1f}m bus / {e["bici_min"]:.1f}m bici'
        dot_lines.append(f'  "{o_name}" -- "{d_name}" [label="{edge_label}"];')

    dot_lines.append("}")
    return "\n".join(dot_lines)

# construimos el DOT para el grafo lógico
dot_src = build_graph_graphviz(
    nodos_df=nodos,
    aristas_df=aristas_full_df,
    hex_nodes=HEX_NODES,
    hex_edges=HEX_PATH,   # usamos color de la ruta seleccionada para resaltar aristas
    dirigido_flag=dirigido,
)

# ---------------- UI PRINCIPAL CON TABS ----------------
tab_mapa, tab_grafo = st.tabs(["🗺️ Mapa geográfico", "🔗 Grafo de conexiones"])

with tab_mapa:
    col1, col2 = st.columns([1, 2])

    if calcular:
        origen_id = id_por_nombre[origen_nombre]
        destino_id = id_por_nombre[destino_nombre]

        try:
            path_ids, totals, tramos_df = ruta_optima(origen_id, destino_id, criterio)

            ruta_nodos_df = pd.DataFrame(
                [{
                    "id": n,
                    "nombre": nombre_por_id[n],
                    "lat": G.nodes[n]["lat"],
                    "lon": G.nodes[n]["lon"],
                } for n in path_ids]
            )

            # capa resaltada para la mejor ruta
            path_layer = pdk.Layer(
                "PathLayer",
                data=[{"path": ruta_nodos_df[["lon", "lat"]].values.tolist()}],
                get_path="path",
                get_width=6,
                width_scale=8,
                get_color=RGB_PATH,
                pickable=False,
            )

            # ----------- PANEL IZQUIERDO -----------
            with col1:
                st.subheader("Resumen de la ruta")
                st.markdown(f"**Origen:** {origen_nombre}")
                st.markdown(f"**Destino:** {destino_nombre}")
                st.markdown(f"**Optimizado por:** `{criterio}`")
                st.markdown(f"**Grafo:** {'Dirigido' if dirigido else 'No dirigido'}")

                st.markdown(f"**Paradas totales (incluye origen y destino):** {len(path_ids)}")
                st.markdown(f"**Paradas intermedias:** {max(0, len(path_ids) - 2)}")

                st.markdown(f"**Distancia total:** {totals['total_distancia_km']:.2f} km")
                st.markdown(f"**Tiempo total estimado en bus (@{VEL_BUS_KMH:.0f} km/h):** {totals['total_bus_min']:.1f} min")
                st.markdown(f"**Tiempo total estimado en bici (@{VEL_BICI_KMH:.0f} km/h):** {totals['total_bici_min']:.1f} min")
                st.markdown(f"**Suma del criterio `{criterio}`:** {totals['total_peso']:.2f}")

                st.markdown("#### Nodos en orden (ruta):")
                st.dataframe(ruta_nodos_df, use_container_width=True)

                st.download_button(
                    "📥 Descargar nodos de la ruta (CSV)",
                    data=ruta_nodos_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"ruta_nodos_{origen_id}_{destino_id}_{criterio}.csv",
                    mime="text/csv"
                )

                st.markdown("#### Tramos (aristas) en la ruta:")
                st.dataframe(tramos_df, use_container_width=True)

                st.download_button(
                    "📥 Descargar tramos de la ruta (CSV)",
                    data=tramos_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"ruta_tramos_{origen_id}_{destino_id}_{criterio}.csv",
                    mime="text/csv"
                )

            # ----------- PANEL DERECHO (MAPA) -----------
            with col2:
                st.pydeck_chart(
                    pdk.Deck(
                        layers=[edges_layer, nodes_layer, path_layer],
                        initial_view_state=view_state,
                    ),
                    use_container_width=True
                )

        except nx.NetworkXNoPath:
            with col1:
                st.error("No hay camino entre esos nodos con el grafo actual.")
            with col2:
                st.pydeck_chart(
                    pdk.Deck(
                        layers=[edges_layer, nodes_layer],
                        initial_view_state=view_state,
                    ),
                    use_container_width=True
                )

    else:
        # estado inicial sin ruta seleccionada
        st.info("Elegí Origen y Destino y presioná **Calcular ruta** 👇")
        st.pydeck_chart(
            pdk.Deck(
                layers=[edges_layer, nodes_layer],
                initial_view_state=view_state,
            ),
            use_container_width=True
        )

with tab_grafo:
    st.markdown("### 🔗 Grafo de conexiones (vista lógica)")
    st.caption(
        "🔴 Cada círculo es una parada.\n"
        "🔗 Cada línea es una conexión.\n"
        "📝 Cada etiqueta = km / min bus / min bici.\n"
        "Incluye las paradas extra como Parque Central, Megapaca, Terminal de Buses, etc."
    )

    st.graphviz_chart(dot_src, use_container_width=True)



