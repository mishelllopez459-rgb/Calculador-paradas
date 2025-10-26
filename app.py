import streamlit as st
import pandas as pd
import pydeck as pdk
import networkx as nx
import math

# ---------------- CONFIG / UI ----------------
st.set_page_config(page_title="Rutas San Marcos", layout="wide")
st.title("🚌 Calculador de paradas y ruta óptima — San Marcos")

# ---------------- CONSTANTES / HELPERS GLOBALES ----------------
VEL_BUS_KMH = 30.0     # km/h bus urbano aprox
VEL_BICI_KMH = 15.0    # km/h bicicleta urbana aprox

def tiempo_por_dist(dist_km: float, vel_kmh: float) -> float:
    """Convierte distancia km -> minutos dada velocidad km/h."""
    if vel_kmh <= 0:
        return 0.0
    return (dist_km / vel_kmh) * 60.0

def haversine_km(lat1, lon1, lat2, lon2):
    """Distancia en km usando Haversine."""
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

def ensure_extra_nodes(nodos_df: pd.DataFrame,
                       extra_nodes_with_coords: list[tuple[str, float, float]]) -> pd.DataFrame:
    """
    Asegura que cada lugar extra exista como nodo con id único y coords.
    Si ya existe por nombre (case-insensitive), no lo duplica. Si le faltaban coords, las rellena.
    """
    nodos_work = nodos_df.copy()

    existentes_lower = {nm.lower(): idx for idx, nm in enumerate(nodos_work["nombre"].astype(str))}
    used_ids = set(nodos_work["id"].astype(str))

    def new_auto_id(base_name: str):
        base = "AUTO_" + base_name.replace(" ", "_").upper()
        cand = base
        i = 1
        while cand in used_ids:
            cand = f"{base}_{i}"
            i += 1
        used_ids.add(cand)
        return cand

    for nombre, lat, lon in extra_nodes_with_coords:
        key = nombre.strip().lower()
        if key in existentes_lower:
            # Ya existe -> asegurar coords.
            idxs = nodos_work.index[nodos_work["nombre"].str.lower() == key]
            if len(idxs) > 0:
                i = idxs[0]
                if pd.isna(nodos_work.at[i, "lat"]):
                    nodos_work.at[i, "lat"] = lat
                if pd.isna(nodos_work.at[i, "lon"]):
                    nodos_work.at[i, "lon"] = lon
        else:
            # No existe -> crearlo
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

def ensure_extra_edges(aristas_df: pd.DataFrame,
                       nodos_df: pd.DataFrame,
                       pares: list[tuple[str, str]]) -> pd.DataFrame:
    """
    Asegura que existan aristas entre cada par dado por nombre.
    Calcula distancia haversine y pone tiempo_min aprox (bus) si no existe.
    Si algún nodo no tiene coords válidas, se salta ese par.
    """
    work = aristas_df.copy()

    existing_pairs = set(
        (row["origen"], row["destino"])
        for _, row in work.iterrows()
    )

    # nombre -> (id, lat, lon)
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

        # coords deben existir
        if pd.isna(a_lat) or pd.isna(a_lon) or pd.isna(b_lat) or pd.isna(b_lon):
            continue

        pair_key = (a_id, b_id)
        if pair_key in existing_pairs:
            # ya hay esa arista exacta
            continue

        dist_km = haversine_km(a_lat, a_lon, b_lat, b_lon)
        t_bus_min  = tiempo_por_dist(dist_km, VEL_BUS_KMH)
        # usamos bus como "tiempo_min" base aproximado si no había uno real
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

def build_graph_graphviz(nodos_df: pd.DataFrame,
                         hex_nodes: str,
                         hex_edges: str,
                         dirigido_flag: bool):
    """
    Grafo lógico (no mapa) que enseña TODAS las paradas importantes
    y las conexiones tipo red urbana.

    Cada arista va etiquetada con:
      distancia_km / min bus / min bici
    usando coordenadas reales si las tenemos. Si no hay coords de ambos nodos,
    se asume 1 km.
    """

    # nodos urbanos clave (incluye TODOS los nuevos)
    nodos_extra = [
        "Parque Central",
        "Catedral",
        "Pollo campero",
        "Megapaca",
        "Bazar Chino",
        "Terminal de Buses",
        "Sat san marcos",
        "INTECAP San Marcos",
        "Salon quetzal",
        "Centro de Salud",
        "Hospital Regional",
        "Cancha Los Angeles",
        "Cancha Sintetica Golazo",
        "Canchas Santo Domingo",
        "Parque benito juarez",
        "Iglesia Cristiana Buena Tierra",
        "Iglesia Candelero de Oro",
        "CANICA (Casa de los Niños)",
        "Aldea San Rafael Soche",
        "Aeropuerto Nacional",

        # nuevos que pediste ahora:
        "Gobernación San Marcos",
        "DOWNTOWN CAFE Y DISCOTECA san marcos",
        "Municipalidad de san marcos",
        "Centro universitario CUSAM san marcos",
        "CLICOLOR",
        "Fundap microcredito san marcos",
        "ACREDICOM R. L san marcos",
        "Banrural san marcos",
    ]

    # conexiones base de la red (anteriores + nuevas conexiones centro)
    conexiones_extra = [
        # centro / comercio
        ("Parque Central", "Catedral"),
        ("Parque Central", "Pollo campero"),
        ("Parque Central", "Megapaca"),
        ("Megapaca", "Bazar Chino"),
        ("Bazar Chino", "Terminal de Buses"),
        ("Pollo campero", "Terminal de Buses"),

        # gobierno / servicios "viejos"
        ("Parque Central", "Sat san marcos"),
        ("Sat san marcos", "INTECAP San Marcos"),
        ("INTECAP San Marcos", "Salon quetzal"),
        ("Salon quetzal", "Centro de Salud"),
        ("Centro de Salud", "Hospital Regional"),

        # periferia vieja
        ("Hospital Regional", "Cancha Los Angeles"),
        ("Cancha Los Angeles", "Cancha Sintetica Golazo"),
        ("Cancha Sintetica Golazo", "Iglesia Candelero de Oro"),
        ("Iglesia Candelero de Oro", "CANICA (Casa de los Niños)"),
        ("CANICA (Casa de los Niños)", "Aldea San Rafael Soche"),
        ("Aldea San Rafael Soche", "Aeropuerto Nacional"),

        # periferia nueva que me diste en el turno anterior
        ("Pollo campero", "Parque benito juarez"),
        ("Parque benito juarez", "Canchas Santo Domingo"),
        ("Canchas Santo Domingo", "Salon quetzal"),
        ("Pollo campero", "Iglesia Cristiana Buena Tierra"),
        ("Iglesia Cristiana Buena Tierra", "Parque benito juarez"),

        # conexiones largas
        ("Terminal de Buses", "Aeropuerto Nacional"),
        ("Hospital Regional", "Terminal de Buses"),

        # 🔥 conexiones nuevas que me acabás de dar del centro cívico / bancos / zona nocturna:
        ("Parque Central", "Municipalidad de san marcos"),
        ("Municipalidad de san marcos", "Gobernación San Marcos"),
        ("Gobernación San Marcos", "Sat san marcos"),
        ("Municipalidad de san marcos", "Banrural san marcos"),
        ("Banrural san marcos", "DOWNTOWN CAFE Y DISCOTECA san marcos"),
        ("DOWNTOWN CAFE Y DISCOTECA san marcos", "Salon quetzal"),
        ("Salon quetzal", "CLICOLOR"),
        ("CLICOLOR", "Centro universitario CUSAM san marcos"),
        ("Centro universitario CUSAM san marcos", "Fundap microcredito san marcos"),
        ("Fundap microcredito san marcos", "CLICOLOR"),
        ("Sat san marcos", "ACREDICOM R. L san marcos"),
        ("ACREDICOM R. L san marcos", "Municipalidad de san marcos"),
    ]

    # coords reales si existen
    coords_por_nombre = {}
    for _, r in nodos_df.iterrows():
        coords_por_nombre[r["nombre"]] = (r["lat"], r["lon"])

    # lista final de nodos = nodos CSV + todos los extra
    nombres_csv = [str(x) for x in nodos_df["nombre"].tolist()]
    nodos_unificados = sorted(set(nombres_csv + nodos_extra))

    # calcular aristas con etiquetas
    usados = set()
    aristas_dot = []

    for (a_name, b_name) in conexiones_extra:
        # evitar duplicado si el grafo lógico es no dirigido
        key_pair = tuple(sorted([a_name, b_name]))
        if (not dirigido_flag) and (key_pair in usados):
            continue
        usados.add(key_pair)

        # distancia real si tenemos coords. Sino 1 km
        if a_name in coords_por_nombre and b_name in coords_por_nombre:
            la, loa = coords_por_nombre[a_name]
            lb, lob = coords_por_nombre[b_name]
            if not (pd.isna(la) or pd.isna(loa) or pd.isna(lb) or pd.isna(lob)):
                dist_km = haversine_km(float(la), float(loa), float(lb), float(lob))
            else:
                dist_km = 1.0
        else:
            dist_km = 1.0

        bus_min  = tiempo_por_dist(dist_km, VEL_BUS_KMH)
        bici_min = tiempo_por_dist(dist_km, VEL_BICI_KMH)

        label_txt = f"{dist_km:.2f} km / {bus_min:.1f}m bus / {bici_min:.1f}m bici"
        aristas_dot.append((a_name, b_name, label_txt))

    # construir DOT para graphviz
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

    # nodos
    for name in nodos_unificados:
        safe = name.replace('"', '\\"')
        dot_lines.append(f'  "{safe}";')

    # aristas con label distancia/bus/bici
    for (a_name, b_name, lbl) in aristas_dot:
        a_safe = a_name.replace('"', '\\"')
        b_safe = b_name.replace('"', '\\"')
        lbl_safe = lbl.replace('"', '\\"')
        dot_lines.append(
            f'  "{a_safe}" -- "{b_safe}" [label="{lbl_safe}"];'
        )

    dot_lines.append("}")
    return "\n".join(dot_lines)

# ---------------- 1. CARGA CSV ORIGINALES ----------------
nodos = pd.read_csv("nodos.csv")
aristas = pd.read_csv("aristas.csv")

# limpieza
nodos["id"] = nodos["id"].astype(str).str.strip()
nodos["nombre"] = nodos["nombre"].astype(str).str.strip()
aristas["origen"] = aristas["origen"].astype(str).str.strip()
aristas["destino"] = aristas["destino"].astype(str).str.strip()

# ---------------- 2. AÑADIR NODOS NUEVOS CON COORDS REALES ----------------
EXTRA_NODOS_COORDS = [
    # nodos que ya habíamos metido
    ("Megapaca",                        14.963451, -91.791009),
    ("Pollo campero",                   14.965879, -91.811366),
    ("Sat san marcos",                  14.965911, -91.793319),
    ("Salon quetzal",                   14.964171, -91.795312),
    ("Canchas Santo Domingo",           14.960118, -91.798793),
    ("Parque benito juarez",            14.958947, -91.802114),
    ("Iglesia Cristiana Buena Tierra",  14.949966, -91.809321),

    # nodos NUEVOS que acabas de dar:
    ("Gobernación San Marcos",                  14.966084, -91.794452),
    ("DOWNTOWN CAFE Y DISCOTECA san marcos",    14.964449, -91.794986),
    ("Municipalidad de san marcos",             14.964601, -91.793954),
    ("Centro universitario CUSAM san marcos",   14.965126, -91.799426),
    ("CLICOLOR",                                14.966102, -91.799505),
    ("Fundap microcredito san marcos",          14.962564, -91.799215),
    ("ACREDICOM R. L san marcos",               14.966154, -91.793269),
    ("Banrural san marcos",                     14.965649, -91.795858),
]
nodos = ensure_extra_nodes(nodos, EXTRA_NODOS_COORDS)

# reconstruimos maps de ayuda (nombre <-> id)
id_por_nombre = {r["nombre"]: r["id"] for _, r in nodos.iterrows()}
nombre_por_id = {r["id"]: r["nombre"] for _, r in nodos.iterrows()}

# ---------------- 3. AÑADIR ARISTAS ENTRE ESOS NODOS NUEVOS ----------------
EXTRA_ARISTAS_PARES = [
    # centro / comercio
    ("Parque Central",        "Megapaca"),
    ("Parque Central",        "Pollo campero"),

    # zona nueva anterior
    ("Pollo campero",         "Parque benito juarez"),
    ("Parque benito juarez",  "Canchas Santo Domingo"),
    ("Canchas Santo Domingo", "Salon quetzal"),
    ("Pollo campero",         "Iglesia Cristiana Buena Tierra"),
    ("Iglesia Cristiana Buena Tierra", "Parque benito juarez"),

    # gobierno / servicios viejo
    ("Sat san marcos",        "Salon quetzal"),

    # 🔥 conexiones centro cívico / comercios / nocturno NUEVAS
    ("Parque Central",                         "Municipalidad de san marcos"),
    ("Municipalidad de san marcos",            "Gobernación San Marcos"),
    ("Gobernación San Marcos",                 "Sat san marcos"),
    ("Municipalidad de san marcos",            "Banrural san marcos"),
    ("Banrural san marcos",                    "DOWNTOWN CAFE Y DISCOTECA san marcos"),
    ("DOWNTOWN CAFE Y DISCOTECA san marcos",   "Salon quetzal"),
    ("Salon quetzal",                          "CLICOLOR"),
    ("CLICOLOR",                               "Centro universitario CUSAM san marcos"),
    ("Centro universitario CUSAM san marcos",  "Fundap microcredito san marcos"),
    ("Fundap microcredito san marcos",         "CLICOLOR"),
    ("Sat san marcos",                         "ACREDICOM R. L san marcos"),
    ("ACREDICOM R. L san marcos",              "Municipalidad de san marcos"),
]
aristas = ensure_extra_edges(aristas, nodos, EXTRA_ARISTAS_PARES)

# ---------------- 4. SIDEBAR ----------------
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

# convertir colores a rgb/hex
RGB_NODES = hex_to_rgb(col_nodes)
RGB_EDGES = hex_to_rgb(col_edges)
RGB_PATH  = hex_to_rgb(col_path)

HEX_NODES = rgb_to_hex(RGB_NODES)
HEX_EDGES = rgb_to_hex(RGB_EDGES)
HEX_PATH  = rgb_to_hex(RGB_PATH)

# ---------------- 5. RECONSTRUIR MAPAS DE AYUDA (por si cambió algo arriba) ----------------
id_por_nombre = {r["nombre"]: r["id"] for _, r in nodos.iterrows()}
nombre_por_id = {r["id"]: r["nombre"] for _, r in nodos.iterrows()}

# ---------------- 6. CONSTRUIR GRAFO NETWORKX (para ruta óptima) ----------------
G = nx.DiGraph() if dirigido else nx.Graph()

# nodos -> grafo
for _, r in nodos.iterrows():
    G.add_node(
        r["id"],
        nombre=r["nombre"],
        lat=float(r["lat"]),
        lon=float(r["lon"]),
    )

# aristas -> grafo (enriquecer con bus/bici)
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

# dataframe de TODAS las aristas finales
aristas_full_df = pd.DataFrame(aristas_enriquecidas)

# ---------------- 7. FUNCIÓN RUTA ÓPTIMA ----------------
def ruta_optima(o_id: str, d_id: str, peso: str):
    """
    Devuelve:
      path_ids: lista de ids de nodos en la ruta
      totals: dict con sumas totales
      tramos_df: detalle tramo a tramo (dist/km, bus/min, bici/min)
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

# ---------------- 8. DATA PARA EL MAPA ----------------
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
    get_color=hex_to_rgb("#9B9B9B"),  # usa color neutro para red base
    pickable=True,
)

nodes_layer = pdk.Layer(
    "ScatterplotLayer",
    data=nodos.rename(columns={"lon": "lng"}),
    get_position="[lng, lat]",
    get_radius=65,
    radius_min_pixels=3,
    get_fill_color=hex_to_rgb("#FF007F"),
    get_line_color=[30, 30, 30],
    line_width_min_pixels=1,
    pickable=True,
)

center_lat, center_lon = nodos["lat"].mean(), nodos["lon"].mean()
view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=13)

# ---------------- 9. GRAFO LÓGICO GRAPHVIZ ----------------
dot_src = build_graph_graphviz(
    nodos_df=nodos,
    hex_nodes=rgb_to_hex(hex_to_rgb("#FF007F")),
    hex_edges=rgb_to_hex(hex_to_rgb("#007AFF")),
    dirigido_flag=False  # para diagrama lógico lo dejamos no dirigido
)

# ---------------- 10. UI PRINCIPAL ----------------
tab_mapa, tab_grafo = st.tabs(["🗺️ Mapa geográfico", "🔗 Grafo de conexiones"])

with tab_mapa:
    col1, col2 = st.columns([1, 2])

    if calcular:
        origen_id = id_por_nombre[origen_nombre]
        destino_id = id_por_nombre[destino_nombre]

        try:
            # Intento ruta óptima usando el grafo conectado
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
                get_color=hex_to_rgb("#007AFF"),
                pickable=False,
            )

            with col1:
                st.subheader("Resumen de la ruta (grafo conectado)")
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
            # fallback cuando no hay camino -> línea recta con dist/bus/bici
            o_row = nodos[nodos["id"] == origen_id].iloc[0]
            d_row = nodos[nodos["id"] == destino_id].iloc[0]

            o_lat, o_lon = float(o_row["lat"]), float(o_row["lon"])
            d_lat, d_lon = float(d_row["lat"]), float(d_row["lon"])

            dist_dir_km = haversine_km(o_lat, o_lon, d_lat, d_lon)
            dir_bus_min  = tiempo_por_dist(dist_dir_km, VEL_BUS_KMH)
            dir_bici_min = tiempo_por_dist(dist_dir_km, VEL_BICI_KMH)

            fallback_df = pd.DataFrame([
                {
                    "id": origen_id,
                    "nombre": origen_nombre,
                    "lat": o_lat,
                    "lon": o_lon,
                },
                {
                    "id": destino_id,
                    "nombre": destino_nombre,
                    "lat": d_lat,
                    "lon": d_lon,
                },
            ])

            fallback_layer = pdk.Layer(
                "PathLayer",
                data=[{"path": fallback_df[["lon", "lat"]].values.tolist()}],
                get_path="path",
                get_width=6,
                width_scale=8,
                get_color=hex_to_rgb("#007AFF"),
                pickable=False,
            )

            with col1:
                st.subheader("Ruta directa aproximada (grafo NO conectado)")
                st.error("No hay camino entre esos nodos con el grafo actual, así que mostramos la línea directa.")

                st.markdown(f"**Origen:** {origen_nombre}")
                st.markdown(f"**Destino:** {destino_nombre}")
                st.markdown(f"**Distancia en línea recta:** {dist_dir_km:.2f} km")
                st.markdown(f"**Tiempo estimado en bus (@{VEL_BUS_KMH:.0f} km/h):** {dir_bus_min:.1f} min")
                st.markdown(f"**Tiempo estimado en bici (@{VEL_BICI_KMH:.0f} km/h):** {dir_bici_min:.1f} min")

                st.markdown("#### Puntos usados:")
                st.dataframe(fallback_df, use_container_width=True)

                st.download_button(
                    "📥 Descargar tramo directo (CSV)",
                    data=fallback_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"ruta_directa_{origen_id}_{destino_id}.csv",
                    mime="text/csv"
                )

            with col2:
                st.pydeck_chart(
                    pdk.Deck(
                        layers=[edges_layer, nodes_layer, fallback_layer],
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
        "Cada nodo es una parada (incluye las nuevas con coordenadas como Municipalidad, Gobernación,"
        " Banrural, Downtown, CUSAM, etc.). "
        "Cada línea es una conexión urbana. "
        "Las etiquetas enseñan distancia estimada y minutos bus/bici."
    )
    st.graphviz_chart(dot_src, use_container_width=True)
