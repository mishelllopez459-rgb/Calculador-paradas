import streamlit as st
import pandas as pd
import pydeck as pdk
import networkx as nx
import math

# ---------------- CONFIG / UI ----------------
st.set_page_config(page_title="Rutas San Marcos", layout="wide")
st.title("🚌 Calculador de paradas y ruta óptima — San Marcos")

# ---------------- CONSTANTES DE TIEMPO / VELOCIDADES ----------------
VEL_BUS_KMH = 30.0     # km/h bus urbano aprox
VEL_BICI_KMH = 15.0    # km/h bicicleta urbana aprox

def tiempo_por_dist(dist_km: float, vel_kmh: float) -> float:
    # minutos = (km / kmh) * 60
    if vel_kmh <= 0:
        return 0.0
    return (dist_km / vel_kmh) * 60.0

def haversine_km(lat1, lon1, lat2, lon2):
    """distancia en km usando la fórmula de Haversine"""
    R = 6371.0
    la1 = math.radians(lat1)
    lo1 = math.radians(lon1)
    la2 = math.radians(lat2)
    lo2 = math.radians(lon2)
    dlat = la2 - la1
    dlon = lo2 - lo1
    h = math.sin(dlat/2)**2 + math.cos(la1)*math.cos(la2)*math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(h))

def hex_to_rgb(h: str):
    """'#RRGGBB' -> [R,G,B]"""
    h = h.lstrip("#")
    return [int(h[i:i+2], 16) for i in (0, 2, 4)]

def rgb_to_hex(rgb):
    """[R,G,B] -> '#RRGGBB'"""
    return "#{:02X}{:02X}{:02X}".format(rgb[0], rgb[1], rgb[2])

# ---------------- CARGA DE CSV BASE ----------------
# nodos.csv => columnas: id, nombre, lat, lon
# aristas.csv => columnas: origen, destino, tiempo_min, distancia_km, capacidad (opcional)
nodos = pd.read_csv("nodos.csv")
aristas = pd.read_csv("aristas.csv")

# Limpieza mínima
nodos["id"] = nodos["id"].astype(str).str.strip()
nodos["nombre"] = nodos["nombre"].astype(str).str.strip()
aristas["origen"] = aristas["origen"].astype(str).str.strip()
aristas["destino"] = aristas["destino"].astype(str).str.strip()

# ---------------- AGREGAR NODOS EXTRA CON COORDENADAS ----------------
# Lugares nuevos que me diste, con lat/lon reales
EXTRA_NODOS_COORDS = [
    ("Megapaca",                         14.963451, -91.791009),
    ("Pollo Campero",                    14.965879, -91.811366),
    ("SAT San Marcos",                   14.965911, -91.793319),
    ("Salón Quetzal",                    14.964171, -91.795312),
    ("Canchas Santo Domingo",            14.960118, -91.798793),
    ("Parque Benito Juarez",             14.958947, -91.802114),
    ("Iglesia Cristiana Buena Tierra",   14.949966, -91.809321),
]

def ensure_extra_nodes(nodos_df: pd.DataFrame,
                       extra_nodes_with_coords: list[tuple[str, float, float]]) -> pd.DataFrame:
    """
    Asegura que cada lugar extra exista como nodo con id único y coords.
    Si ya existe por nombre, no lo duplica, pero si le faltaban coords, se las pone.
    """
    # set de nombres ya existentes en lower para comparar
    existentes_lower = {nm.lower(): idx for idx, nm in enumerate(nodos_df["nombre"].astype(str))}
    used_ids = set(nodos_df["id"].astype(str))

    def new_auto_id(base_name: str):
        base = "AUTO_" + base_name.replace(" ", "_").upper()
        cand = base
        i = 1
        while cand in used_ids:
            cand = f"{base}_{i}"
            i += 1
        used_ids.add(cand)
        return cand

    # vamos a mutar sobre copia
    nodos_work = nodos_df.copy()

    for nombre, lat, lon in extra_nodes_with_coords:
        key = nombre.strip().lower()
        if key in existentes_lower:
            # ya existe ese nombre -> asegurar coords
            idxs = nodos_work.index[nodos_work["nombre"].str.lower() == key]
            if len(idxs) > 0:
                i = idxs[0]
                # si estaban NaN, las rellenamos
                if pd.isna(nodos_work.at[i, "lat"]):
                    nodos_work.at[i, "lat"] = lat
                if pd.isna(nodos_work.at[i, "lon"]):
                    nodos_work.at[i, "lon"] = lon
        else:
            # no existe -> crearlo
            nuevo_id = new_auto_id(nombre)
            nodos_work = pd.concat([
                nodos_work,
                pd.DataFrame([{
                    "id": nuevo_id,
                    "nombre": nombre,
                    "lat": lat,
                    "lon": lon,
                }])
            ], ignore_index=True)

    return nodos_work

nodos = ensure_extra_nodes(nodos, EXTRA_NODOS_COORDS)

# helpers después de meter nodos extra
id_por_nombre = {r["nombre"]: r["id"] for _, r in nodos.iterrows()}
nombre_por_id = {r["id"]: r["nombre"] for _, r in nodos.iterrows()}

# ---------------- AGREGAR ARISTAS EXTRA BASADAS EN ESOS NODOS ----------------
# Conexiones nuevas que queremos que existan físicamente en el mapa / grafo,
# usando las paradas que ya te importan.
# Nota: algunas conexiones necesitan que ambos nodos tengan coords.
EXTRA_ARISTAS_PARES = [
    # conexiones principales alrededor del centro / gobierno
    ("Parque Central", "SAT San Marcos"),
    ("SAT San Marcos", "Salón Quetzal"),

    # zona Megapaca / Pollo / etc
    ("Parque Central", "Megapaca"),
    ("Parque Central", "Pollo Campero"),

    # nuevas que pediste
    ("Pollo Campero", "Parque Benito Juarez"),
    ("Parque Benito Juarez", "Canchas Santo Domingo"),
    ("Canchas Santo Domingo", "Salón Quetzal"),
    ("Pollo Campero", "Iglesia Cristiana Buena Tierra"),
    ("Iglesia Cristiana Buena Tierra", "Parque Benito Juarez"),
]

def ensure_extra_edges(aristas_df: pd.DataFrame,
                       nodos_df: pd.DataFrame,
                       pares: list[tuple[str, str]]) -> pd.DataFrame:
    """
    Asegura que existan aristas entre los pares dados.
    Calcula distancia haversine usando coords reales si existen.
    Si no hay coords para uno de los nodos, ese par se salta.
    tiempo_min aproximado = tiempo en bus (30 km/h).
    capacidad = 0 por defecto.
    """
    work = aristas_df.copy()
    existing_pairs = set(
        (row["origen"], row["destino"])
        for _, row in work.iterrows()
    )

    # mapa rápido nombre -> (id, lat, lon)
    info_por_nombre = {}
    for _, r in nodos_df.iterrows():
        info_por_nombre[r["nombre"]] = (
            r["id"],
            float(r["lat"]),
            float(r["lon"]),
        )

    new_rows = []
    for a_name, b_name in pares:
        if a_name not in info_por_nombre or b_name not in info_por_nombre:
            continue

        a_id, a_lat, a_lon = info_por_nombre[a_name]
        b_id, b_lat, b_lon = info_por_nombre[b_name]

        # si coords vienen NaN de CSV y no las logramos, salta
        if pd.isna(a_lat) or pd.isna(a_lon) or pd.isna(b_lat) or pd.isna(b_lon):
            continue

        pair_key = (a_id, b_id)
        if pair_key in existing_pairs:
            # ya existe arista exacta
            continue

        dist_km = haversine_km(a_lat, a_lon, b_lat, b_lon)

        # estimaciones de tiempo usando distancia
        t_bus_min  = tiempo_por_dist(dist_km, VEL_BUS_KMH)   # bus
        # nota: vamos a usar bus_min como tiempo_min base
        t_min_real = t_bus_min

        row = {
            "origen": a_id,
            "destino": b_id,
            "distancia_km": dist_km,
            "tiempo_min": t_min_real,
            "capacidad": 0.0,
        }
        new_rows.append(row)

    if new_rows:
        work = pd.concat([work, pd.DataFrame(new_rows)], ignore_index=True)

    return work

aristas = ensure_extra_edges(aristas, nodos, EXTRA_ARISTAS_PARES)

# ---------------- SIDEBAR (después de nodos/edges actualizados) ----------------
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

# Rebuild helpers (después de agregar extras): importante
id_por_nombre = {r["nombre"]: r["id"] for _, r in nodos.iterrows()}
nombre_por_id = {r["id"]: r["nombre"] for _, r in nodos.iterrows()}

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

# Enriquecer aristas con tiempos bus/bici y guardarlas en una lista
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
                         hex_nodes: str,
                         hex_edges: str,
                         dirigido_flag: bool):
    """
    Grafo lógico (no mapa) que combina:
    - nodos/aristas originales
    - TODOS los lugares clave que vos querés ver en el diagrama
    - y conexiones urbanas entre ellos.
    Cada arista lleva label con:
      distancia_km / min bus / min bici
    Si no tenemos coords -> le ponemos distancia 1 km como base.
    """

    # nodos urbanos importantes (incluyendo los nuevos que me diste)
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
        "Canchas Santo Domingo",
        "Parque Benito Juarez",
        "Iglesia Cristiana Buena Tierra",
        "Iglesia Candelero de Oro",
        "CANICA (Casa de los Niños)",
        "Aldea San Rafael Soche",
        "Aeropuerto Nacional",
    ]

    # conexiones lógicas de la red urbana
    conexiones_extra = [
        # centro / comercio
        ("Parque Central", "Catedral"),
        ("Parque Central", "Pollo Campero"),
        ("Parque Central", "Megapaca"),
        ("Megapaca", "Bazar Chino"),
        ("Bazar Chino", "Terminal de Buses"),
        ("Pollo Campero", "Terminal de Buses"),

        # gobierno / servicios
        ("Parque Central", "SAT San Marcos"),
        ("SAT San Marcos", "INTECAP San Marcos"),
        ("INTECAP San Marcos", "Salón Quetzal"),
        ("Salón Quetzal", "Centro de Salud"),
        ("Centro de Salud", "Hospital Regional"),

        # periferia vieja
        ("Hospital Regional", "Cancha Los Angeles"),
        ("Cancha Los Angeles", "Cancha Sintetica Golazo"),
        ("Cancha Sintetica Golazo", "Iglesia Candelero de Oro"),
        ("Iglesia Candelero de Oro", "CANICA (Casa de los Niños)"),
        ("CANICA (Casa de los Niños)", "Aldea San Rafael Soche"),
        ("Aldea San Rafael Soche", "Aeropuerto Nacional"),

        # periferia nueva que me diste
        ("Pollo Campero", "Parque Benito Juarez"),
        ("Parque Benito Juarez", "Canchas Santo Domingo"),
        ("Canchas Santo Domingo", "Salón Quetzal"),
        ("Pollo Campero", "Iglesia Cristiana Buena Tierra"),
        ("Iglesia Cristiana Buena Tierra", "Parque Benito Juarez"),

        # conexiones largas
        ("Terminal de Buses", "Aeropuerto Nacional"),
        ("Hospital Regional", "Terminal de Buses"),
    ]

    # usamos coords reales si las tenemos en nodos_df
    coords_por_nombre = {}
    id_por_nombre_local = {}
    for _, r in nodos_df.iterrows():
        coords_por_nombre[r["nombre"]] = (r["lat"], r["lon"])
        id_por_nombre_local[r["nombre"]] = r["id"]

    # armamos lista de nodos finales (union CSV + extra)
    nombres_csv = [str(x) for x in nodos_df["nombre"].tolist()]
    nodos_unificados = sorted(set(nombres_csv + nodos_extra))

    # preparamos aristas_unificadas con etiqueta
    usados = set()
    aristas_dot = []
    for (a_name, b_name) in conexiones_extra:
        # evitamos duplicar si no es dirigido
        key_pair = tuple(sorted([a_name, b_name]))
        if (not dirigido_flag) and (key_pair in usados):
            continue
        usados.add(key_pair)

        # si tenemos coords de ambos -> distancia real
        if a_name in coords_por_nombre and b_name in coords_por_nombre:
            la, loa = coords_por_nombre[a_name]
            lb, lob = coords_por_nombre[b_name]
            if not (pd.isna(la) or pd.isna(loa) or pd.isna(lb) or pd.isna(lob)):
                dist_km = haversine_km(float(la), float(loa), float(lb), float(lob))
            else:
                dist_km = 1.0
        else:
            dist_km = 1.0  # fallback si no hay coords

        bus_min  = tiempo_por_dist(dist_km, VEL_BUS_KMH)
        bici_min = tiempo_por_dist(dist_km, VEL_BICI_KMH)

        label_txt = f"{dist_km:.2f} km / {bus_min:.1f}m bus / {bici_min:.1f}m bici"
        aristas_dot.append((a_name, b_name, label_txt))

    # ahora generamos el source DOT
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

    # declarar nodos
    for name in nodos_unificados:
        safe = name.replace('"', '\\"')
        dot_lines.append(f'  "{safe}";')

    # declarar aristas con label
    for (a_name, b_name, lbl) in aristas_dot:
        a_safe = a_name.replace('"', '\\"')
        b_safe = b_name.replace('"', '\\"')
        lbl_safe = lbl.replace('"', '\\"')
        dot_lines.append(
            f'  "{a_safe}" -- "{b_safe}" [label="{lbl_safe}"];'
        )

    dot_lines.append("}")
    return "\n".join(dot_lines)


dot_src = build_graph_graphviz(
    nodos_df=nodos,
    hex_nodes=HEX_NODES,
    hex_edges=HEX_PATH,   # usamos color ruta seleccionada porque se ve más vivo
    dirigido_flag=False  # el diagrama lógico se ve mejor no dirigido
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

            path_layer = pdk.Layer(
                "PathLayer",
                data=[{"path": ruta_nodos_df[["lon", "lat"]].values.tolist()}],
                get_path="path",
                get_width=6,
                width_scale=8,
                get_color=RGB_PATH,
                pickable=False,
            )

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
        "Cada nodo es una parada (incluye las nuevas con coordenadas). "
        "Cada arista tiene etiqueta con distancia y minutos en bus/bici."
    )
    st.graphviz_chart(dot_src, use_container_width=True)
