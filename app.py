def build_graph_graphviz(nodos_df: pd.DataFrame,
                          aristas_df: pd.DataFrame,
                          hex_nodes: str,
                          hex_edges: str,
                          dirigido_flag: bool):
    """
    Genera un grafo lógico (no mapa) combinando:
    - TODOS los nodos/aristas reales del CSV
    - MÁS los nodos importantes fijos de la ciudad (Parque Central, Megapaca, etc.)
      con conexiones predefinidas para que se vean en la red.

    Cada arista lleva label con:
      distancia_km
      min bus
      min bici
    """

    # 1. NODOS BASE FIJOS (aseguramos que salgan en el grafo aunque no estén en nodos.csv)
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

    # 2. ARISTAS EXTRA (para que el grafo tenga forma de red y no queden nodos sueltos)
    # Para estas, inventamos distancia=1.0 km si no hay una real.
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

    # 3. Partimos armando una lista de nodos unificados:
    #    - todos los nombres del CSV
    #    - más los extras que definimos arriba
    nombres_csv = [str(x) for x in nodos_df["nombre"].tolist()]
    nodos_unificados = sorted(set(nombres_csv + nodos_extra))

    # 4. Aristas unificadas:
    #    - todas las reales del CSV
    #    - más las extra (con distancias sintéticas si no existen reales)
    aristas_unificadas = []

    # a) primero las reales del CSV/aristas_df
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

    # b) luego metemos las extra manuales
    #    si ya existe conexión entre ese par (aunque sea al revés), no la repetimos
    ya = set()
    for e in aristas_unificadas:
        key = tuple(sorted([e["o"], e["d"]]))
        ya.add(key)

    for a, b in conexiones_extra:
        key2 = tuple(sorted([a, b]))
        if key2 in ya and not dirigido_flag:
            continue
        ya.add(key2)

        dist_km = 1.0  # estimado fijo para que aparezcan datos
        bus_min = tiempo_por_dist(dist_km, VEL_BUS_KMH)   # ~2.0 min
        bici_min = tiempo_por_dist(dist_km, VEL_BICI_KMH) # ~4.0 min

        aristas_unificadas.append({
            "o": a,
            "d": b,
            "dist_km": dist_km,
            "bus_min": bus_min,
            "bici_min": bici_min,
            "is_extra": True,
        })

    # 5. Construimos el DOT para Graphviz
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

    # Declarar todos los nodos unificados
    for name in nodos_unificados:
        safe_name = name.replace('"', '\\"')
        dot_lines.append(f'  "{safe_name}";')

    # Declarar aristas con labels distancia / bus / bici
    usados_dot = set()
    for e in aristas_unificadas:
        o_name = e["o"].replace('"', '\\"')
        d_name = e["d"].replace('"', '\\"')

        # si el grafo no es dirigido, evitamos duplicar
        key_d = tuple(sorted([o_name, d_name]))
        if not dirigido_flag:
            if key_d in usados_dot:
                continue
            usados_dot.add(key_d)

        edge_label = f'{e["dist_km"]:.2f} km / {e["bus_min"]:.1f}m bus / {e["bici_min"]:.1f}m bici'

        dot_lines.append(
            f'  "{o_name}" -- "{d_name}" [label="{edge_label}"];'
        )

    dot_lines.append("}")
    return "\n".join(dot_lines)


