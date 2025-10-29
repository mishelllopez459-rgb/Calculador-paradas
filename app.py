<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>San Marcos — Route Optimizer</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  body { font-family: Arial, sans-serif; margin: 10px; }
  #map { height: 500px; margin-top: 10px; }
  select, button { margin: 5px; padding: 5px; }
</style>
</head>
<body>
<h2>San Marcos — Route Optimizer</h2>

<label for="origen">Origen:</label>
<select id="origen"></select>

<label for="destino">Destino:</label>
<select id="destino"></select>

<label for="modo">Modo:</label>
<select id="modo">
  <option value="auto">Auto (40 km/h)</option>
  <option value="bus">Bus (25 km/h)</option>
  <option value="peaton">Peatón (5 km/h)</option>
</select>

<button id="calcular">Calcular</button>

<p id="resultado"></p>

<div id="map"></div>

<script>
  var map = L.map('map').setView([14.965, -91.79], 14);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 18,
    attribution: '© OpenStreetMap contributors'
  }).addTo(map);

  // Todos los nodos (anteriores + nuevos)
  const nodos = [
    { nombre: "Cancha Sintética Golazo", lat: 14.9675, lon: -91.7940 },
    { nombre: "Bazar Chino", lat: 14.962, lon: -91.786 },
    { nombre: "Aldea San Rafael Soche", lat: 14.9645, lon: -91.7935 },
    { nombre: "Catedral", lat: 14.961, lon: -91.786 },
    { nombre: "Centro de Salud", lat: 14.962, lon: -91.7885 },
    { nombre: "INTECAP San Marcos", lat: 14.963, lon: -91.7885 },
    { nombre: "Iglesia Candelero de Oro", lat: 14.9625, lon: -91.788 },
    { nombre: "Megapaca", lat: 14.961, lon: -91.7905 },
    { nombre: "Parque Central", lat: 14.9645, lon: -91.793 },
    { nombre: "Pollo Campero", lat: 14.967, lon: -91.791 },
    { nombre: "SAT San Marcos", lat: 14.969, lon: -91.7875 },
    { nombre: "Terminal Central", lat: 14.9655, lon: -91.7938 },
    { nombre: "Terminal de Buses", lat: 14.962, lon: -91.788 },
    { nombre: "Universidad San Carlos", lat: 14.9712, lon: -91.7815 },
    // Nodos nuevos
    { nombre: "Gobernación San Marcos", lat: 14.966084, lon: -91.794452 },
    { nombre: "DOWNTOWN CAFE Y DISCOTECA", lat: 14.964449, lon: -91.794986 },
    { nombre: "Municipalidad de San Marcos", lat: 14.964601, lon: -91.793954 },
    { nombre: "Centro universitario CUSAM", lat: 14.965126, lon: -91.799426 },
    { nombre: "CLICOLOR", lat: 14.966102, lon: -91.799505 },
    { nombre: "Fundap microcredito", lat: 14.962564, lon: -91.799215 },
    { nombre: "ACREDICOM R. L", lat: 14.966154, lon: -91.793269 },
    { nombre: "Banrural San Marcos", lat: 14.965649, lon: -91.795858 },
    { nombre: "Hotel y Restaurante Santa Barbara", lat: 14.963654, lon: -91.796771 },
    { nombre: "Salon Terracota", lat: 14.966691, lon: -91.797232 },
    { nombre: "Contraloría General de Cuentas", lat: 14.966359, lon: -91.797010 },
    { nombre: "Centro medico de especialidades", lat: 14.964610, lon: -91.797402 },
    { nombre: "Cementerio San Marcos", lat: 14.964338, lon: -91.800597 },
    { nombre: "Dominós pizza SM", lat: 14.963560, lon: -91.791470 },
    { nombre: "Ministerio de Ambiente", lat: 14.963989, lon: -91.793348 },
    { nombre: "Ministerio público de la Mujer", lat: 14.968695, lon: -91.798307 },
    { nombre: "Guzgado de primera instancia", lat: 14.969025, lon: -91.797968 },
    { nombre: "Edificio Tribunales", lat: 14.964409, lon: -91.794494 }
  ];

  // Aristas (conecta cada nuevo nodo con el nodo más cercano para simplicidad)
  const aristas = [
    { desde: "Cancha Sintética Golazo", hasta: "Pollo Campero" },
    { desde: "Cancha Sintética Golazo", hasta: "SAT San Marcos" },
    { desde: "Pollo Campero", hasta: "Aldea San Rafael Soche" },
    { desde: "Aldea San Rafael Soche", hasta: "Parque Central" },
    { desde: "Parque Central", hasta: "INTECAP San Marcos" },
    { desde: "INTECAP San Marcos", hasta: "Centro de Salud" },
    { desde: "Centro de Salud", hasta: "Iglesia Candelero de Oro" },
    { desde: "Catedral", hasta: "Bazar Chino" },
    { desde: "Terminal Central", hasta: "Universidad San Carlos" },
    { desde: "SAT San Marcos", hasta: "Universidad San Carlos" },
    { desde: "Terminal de Buses", hasta: "INTECAP San Marcos" },
    // Conexiones de nodos nuevos
    { desde: "Gobernación San Marcos", hasta: "Cancha Sintética Golazo" },
    { desde: "DOWNTOWN CAFE Y DISCOTECA", hasta: "Municipalidad de San Marcos" },
    { desde: "Municipalidad de San Marcos", hasta: "Parque Central" },
    { desde: "Centro universitario CUSAM", hasta: "CLICOLOR" },
    { desde: "CLICOLOR", hasta: "Centro universitario CUSAM" },
    { desde: "Fundap microcredito", hasta: "Centro medico de especialidades" },
    { desde: "ACREDICOM R. L", hasta: "Gobernación San Marcos" },
    { desde: "Banrural San Marcos", hasta: "Pollo Campero" },
    { desde: "Hotel y Restaurante Santa Barbara", hasta: "Fundap microcredito" },
    { desde: "Salon Terracota", hasta: "Contraloría General de Cuentas" },
    { desde: "Contraloría General de Cuentas", hasta: "Salon Terracota" },
    { desde: "Centro medico de especialidades", hasta: "DOWNTOWN CAFE Y DISCOTECA" },
    { desde: "Cementerio San Marcos", hasta: "Centro universitario CUSAM" },
    { desde: "Dominós pizza SM", hasta: "Centro de Salud" },
    { desde: "Ministerio de Ambiente", hasta: "INTECAP San Marcos" },
    { desde: "Ministerio público de la Mujer", hasta: "Guzgado de primera instancia" },
    { desde: "Guzgado de primera instancia", hasta: "Universidad San Carlos" },
    { desde: "Edificio Tribunales", hasta: "Municipalidad de San Marcos" }
  ];

  // Función distancia Haversine
  function calcularDistancia(lat1, lon1, lat2, lon2) {
    const R = 6371;
    const dLat = (lat2-lat1)*Math.PI/180;
    const dLon = (lon2-lon1)*Math.PI/180;
    const a = Math.sin(dLat/2)**2 + Math.cos(lat1*Math.PI/180)*Math.cos(lat2*Math.PI/180)*Math.sin(dLon/2)**2;
    const c = 2*Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
    return R*c;
  }

  // Construir grafo con pesos
  const grafo = {};
  nodos.forEach(n => grafo[n.nombre]=[]);
  aristas.forEach(a=>{
    const n1=nodos.find(n=>n.nombre===a.desde);
    const n2=nodos.find(n=>n.nombre===a.hasta);
    const dist=calcularDistancia(n1.lat,n1.lon,n2.lat,n2.lon);
    grafo[a.desde].push({nodo:a.hasta,peso:dist});
    grafo[a.hasta].push({nodo:a.desde,peso:dist});
    L.polyline([[n1.lat,n1.lon],[n2.lat,n2.lon]],{color:"lightblue",weight:2}).addTo(map);
  });

  // Marcadores
  nodos.forEach(n=>{
    L.circleMarker([n.lat,n.lon],{radius:6,color:"blue",fillColor:"cyan",fillOpacity:0.8})
      .addTo(map).bindPopup(`<b>${n.nombre}</b>`);
  });

  // Llena selects
  const origenSelect=document.getElementById("origen");
  const destinoSelect=document.getElementById("destino");
  nodos.forEach(n=>{
    const opt1=document.createElement("option"); opt1.value=n.nombre; opt1.text=n.nombre;
    const opt2=opt1.cloneNode(true);
    origenSelect.appendChild(opt1);
    destinoSelect.appendChild(opt2);
  });

  // Dijkstra
  function dijkstra(grafo,inicio,fin){
    const dist={}; const prev={}; const Q=new Set();
    for(let nodo in grafo){ dist[nodo]=Infinity; prev[nodo]=null; Q.add(nodo);}
    dist[inicio]=0;
    while(Q.size>0){
      let u=Array.from(Q).reduce((a,b)=>dist[a]<dist[b]?a:b);
      Q.delete(u);
      if(u===fin) break;
      grafo[u].forEach(v=>{
        const alt=dist[u]+v.peso;
        if(alt<dist[v.nodo]){dist[v.nodo]=alt; prev[v.nodo]=u;}
      });
    }
    const path=[]; let u=fin;
    if(prev[u]||u===inicio){while(u){path.unshift(u); u=prev[u];}}
    return path;
  }

  document.getElementById("calcular").onclick=function(){
    const origen=origenSelect.value;
    const destino=destinoSelect.value;
    const modo=document.getElementById("modo").value;
    if(!origen||!destino){alert("Selecciona origen y destino");return;}
    // Borra rutas anteriores
    map.eachLayer(layer=>{if(layer instanceof L.Polyline&&!layer._url) map.removeLayer(layer);});
    // Re-dibuja aristas
    aristas.forEach(a=>{
      const n1=nodos.find(n=>n.nombre===a.desde);
      const n2=nodos.find(n=>n.nombre===a.hasta);
      L.polyline([[n1.lat,n1.lon],[n2.lat,n2.lon]],{color:"lightblue",weight:2}).addTo(map);
    });
    const ruta=dijkstra(grafo,origen,destino);
    if(ruta.length===0){alert("No hay ruta");return;}
    // Dibuja ruta resaltada
    for(let i=0;i<ruta.length-1;i++){
      const n1=nodos.find(n=>n.nombre===ruta[i]);
      const n2=nodos.find(n=>n.nombre===ruta[i+1]);
      L.polyline([[n1.lat,n1.lon],[n2.lat,n2.lon]],{color:"red",weight:4}).addTo(map);
    }
    // Distancia y tiempo
    let distanciaTotal=0;
    for(let i=0;i<ruta.length-1;i++){
      const n1=nodos.find(n=>n.nombre===ruta[i]);
      const n2=nodos.find(n=>n.nombre===ruta[i+1]);
      distanciaTotal+=calcularDistancia(n1.lat,n1.lon,n2.lat,n2.lon);
    }
    let velocidad;
    if(modo==="auto") velocidad=40;
    else if(modo==="bus") velocidad=25;
    else velocidad=5;
    const tiempo=distanciaTotal/velocidad;
    document.getElementById("resultado").innerHTML=
      `Ruta: ${ruta.join(" → ")} <br>Distancia: ${distanciaTotal.toFixed(2)} km <br>Tiempo estimado: ${tiempo.toFixed(2)} h`;
  };

</script>
</body>
</html>
