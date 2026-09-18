# Informe del Proyecto: Travel Planner Multidestino

## Objetivo general

Este proyecto implementa un sistema de planificacion de viajes multidestino. La idea principal es permitir que un usuario elija varias ciudades, un medio de transporte y un criterio de optimizacion, para luego comparar su ruta elegida con una ruta optimizada automaticamente.

El sistema combina algoritmos clasicos, API web, interfaz grafica, cache, procesamiento asincronico de reservas, procesamiento por lotes y recomendaciones con IA.

## Estructura principal

El codigo importante esta dentro de:

```text
travel_planner/
  app/
    api/
    booking/
    caches/
    core/
    data/
    ui/
    ai/
    tests/
```

Ademas, en la raiz del proyecto hay archivos de documentacion, pruebas sueltas antiguas y el informe final.

## Modulos principales

### Core

La carpeta `app/core` contiene la logica algoritmica del sistema.

`graph.py` define un grafo dirigido de ciudades y rutas. Cada ruta tiene destino, costo, tiempo y tipo de transporte. Usa el algoritmo de Dijkstra para encontrar el camino mas corto entre dos ciudades, ya sea optimizando por costo o por tiempo. Tambien permite filtrar por transporte, por ejemplo solo auto, tren o avion.

`tsp_dp.py` implementa el problema del viajante usando programacion dinamica con bitmasking y memoizacion. Sirve para calcular el orden optimo para visitar varias ciudades. El algoritmo usado es una variante de Held-Karp, adecuada para una cantidad moderada de ciudades.

`tsp_genetic.py` resuelve el mismo problema del viajante pero con un algoritmo genetico (heuristico): seleccion por torneo, Order Crossover, mutacion por intercambio y elitismo. A diferencia de Held-Karp, que es exacto pero crece como O(n^2 * 2^n), el genetico escala a muchas mas ciudades a cambio de no garantizar el optimo. La aplicacion usa Held-Karp para pocas ciudades y el genetico cuando la cantidad crece.

`itinerary_validator.py` contiene reglas de validacion de itinerarios: presupuesto maximo, duracion maxima, cantidad de segmentos, ciudades requeridas, ciudades prohibidas, tipos de transporte permitidos, escalas y consistencia de horarios.

### Data

`app/data/routes_fixed.py` genera automaticamente rutas entre ciudades europeas. Usa coordenadas aproximadas y distancia Haversine para calcular distancias. A partir de eso genera costo y tiempo para auto, tren y avion.

Esta informacion alimenta tanto al grafo de rutas como a las matrices usadas por el TSP.

### API

`app/api/server.py` expone una API con FastAPI.

Endpoints principales:

```text
GET  /health
GET  /routes/matrix
POST /routes/shortest
GET  /routes/compare
POST /routes/optimize-multi
POST /itinerary/plan
POST /reservations
POST /reservations/batch
GET  /reservations/user/{user_id}
DELETE /reservations/{reservation_id}
GET  /stats
```

La API se encarga de conectar los datos, los algoritmos, el cache y las reservas. Por ejemplo, cuando la interfaz pide una matriz de costos para tren, la API responde con las ciudades y la matriz correspondiente. Cuando la interfaz pide optimizar varias ciudades, la API llama al solver TSP.

Tambien usa un `lifespan` de FastAPI para iniciar un loop de procesamiento de lotes en background.

### UI

`app/ui/streamlit_app.py` es la interfaz grafica hecha con Streamlit.

Desde ahi el usuario puede:

- Elegir tipo de transporte.
- Elegir si optimiza por costo o por tiempo.
- Seleccionar multiples ciudades.
- Calcular y comparar rutas.
- Elegir una ruta para reservar.
- Crear reservas individuales o en lote.
- Ver reservas existentes.
- Ver estadisticas del sistema.

La interfaz consume la API local en:

```text
http://localhost:8000
```

Por eso, para usar la interfaz correctamente, primero debe estar levantada la API.

### Booking

La carpeta `app/booking` maneja reservas.

`reservations.py` define `ReservationManager`, que crea, procesa, consulta y cancela reservas. El procesamiento es asincronico con `asyncio`, simulando operaciones de confirmacion con proveedores y envio de notificaciones.

Estados posibles de una reserva:

```text
pending
processing
confirmed
failed
cancelled
```

`batching.py` contiene un procesador por lotes. Permite agrupar muchas reservas y procesarlas juntas. Esto sirve para simular una carga masiva de reservas sin bloquear la respuesta inmediata del servidor.

### Caches

La carpeta `app/caches` contiene dos tipos de cache.

`lru_cache.py` implementa un cache LRU local. Guarda resultados recientes y elimina el elemento menos usado cuando supera la capacidad. Tambien incluye una variante con TTL y un decorador para cachear funciones.

`redis_cache.py` implementa un cache distribuido usando Redis. Esta pensado para escenarios donde varias instancias de la aplicacion comparten cache.

`cache_backend.py` es un selector de backend: segun la variable de entorno `CACHE_BACKEND` intenta usar Redis y, si no esta disponible (paquete ausente o servidor caido), cae automaticamente al cache LRU en memoria. Asi la API funciona igual con o sin Redis corriendo. La API usa este selector, por lo que por defecto guarda rutas y optimizaciones repetidas en el LRU local y puede escalar a Redis sin tocar codigo.

### AI

`app/ai/gemini_recommendations.py` integra recomendaciones con Gemini. La idea es generar sugerencias de lugares o destinos relacionados con las ciudades seleccionadas.

Esta parte depende de configuracion externa de API key, por lo que puede no estar activa si no se configuran las credenciales correspondientes.

### Tests

La carpeta `app/tests` contiene pruebas con Pytest.

Actualmente cubren:

- Dijkstra y rutas cortas.
- TSP exacto (Held-Karp) con tour cerrado y abierto.
- TSP con algoritmo genetico: permutacion valida, determinismo con semilla fija,
  convergencia al optimo de Held-Karp, monotonia del elitismo y casos borde.
- Validaciones de itinerarios.
- Cache LRU y TTL.
- Reservas asincronicas.
- Procesamiento batch.
- Endpoints principales de FastAPI.

Para ejecutar:

```powershell
cd D:\pef2\FinalPEF\travel_planner
pytest
```

El resultado esperado actualmente es:

```text
33 passed, 1 skipped
```

## Flujo esperado de uso

1. Se levanta la API FastAPI.
2. Se levanta la interfaz Streamlit.
3. El usuario entra a Ruta Multidestino.
4. Selecciona transporte: auto, avion o tren.
5. Selecciona criterio: costo o tiempo.
6. Elige varias ciudades.
7. Presiona Calcular costos.
8. La UI pide matrices a la API.
9. La API usa los datos de rutas fijas.
10. Se calcula la ruta del usuario y la ruta optimizada.
11. El usuario elige una ruta.
12. Puede crear una reserva individual o un lote de reservas.
13. Las reservas quedan visibles en Mis Reservas.

## Como correr el proyecto

### Comandos utilizados

Estos son los comandos usados para configurar la clave de IA, instalar dependencias, ejecutar pruebas, abrir el reporte de cobertura y levantar la API.

Por seguridad, en el informe no se deja escrita la clave real de Google. En su lugar se usa `TU_API_KEY` y cada persona debe reemplazarlo por su propia clave antes de ejecutar el proyecto.

```powershell
cd D:\pef2\FinalPEF\travel_planner

$env:GOOGLE_API_KEY="TU_API_KEY"

.\install_and_run.bat

pytest

start htmlcov\index.html

uvicorn app.api.server:app --reload
```

El comando `pytest` tambien genera el reporte HTML de cobertura porque `pytest.ini` tiene configurado `--cov-report=html`. Por eso, despues de ejecutar las pruebas, se puede abrir:

```powershell
start htmlcov\index.html
```

Primero activar el entorno virtual:

```powershell
cd D:\pef2\FinalPEF\travel_planner
.\.venv\Scripts\activate
```

Levantar la API:

```powershell
uvicorn app.api.server:app --reload
```

Abrir documentacion de la API:

```text
http://127.0.0.1:8000/docs
```

En otra terminal, levantar la interfaz:

```powershell
cd D:\pef2\FinalPEF\travel_planner
.\.venv\Scripts\activate
streamlit run app\ui\streamlit_app.py
```

Abrir la interfaz:

```text
http://localhost:8501
```

## Dependencias usadas

Principales librerias:

- FastAPI: API REST.
- Uvicorn: servidor para FastAPI.
- Streamlit: interfaz grafica.
- Pytest: tests.
- Pytest-cov: reporte de cobertura.
- Pytest-asyncio: soporte para pruebas asincronicas.
- Requests: comunicacion HTTP desde la UI.
- Pandas: visualizacion de datos en estadisticas.
- Redis: cache distribuido opcional.
- Google GenAI: recomendaciones con IA.

## Consideraciones importantes

La API debe estar corriendo antes de abrir la interfaz Streamlit, porque la UI consulta endpoints locales.

La parte de Redis no es obligatoria para el funcionamiento principal. Esta implementada como soporte de cache distribuido, pero el flujo principal usa cache local.

La parte de IA puede requerir configuracion adicional de credenciales. Si no esta configurada, las recomendaciones pueden no funcionar, pero el calculo de rutas y reservas sigue siendo independiente.

La cobertura total puede verse baja porque tambien se mide la interfaz Streamlit completa, Redis y Gemini, que son partes mas dificiles de testear sin mocks o servicios externos. Lo mas importante para validar el funcionamiento actual es que la suite de tests pase correctamente.

## Resumen corto

El proyecto es una aplicacion de planificacion de viajes multidestino. Usa rutas generadas entre ciudades europeas, Dijkstra para caminos cortos, TSP con programacion dinamica para optimizar multiples destinos, FastAPI para exponer servicios, Streamlit para la interfaz, cache para acelerar consultas repetidas y procesamiento asincronico/batch para manejar reservas.

El flujo principal deberia permitir seleccionar ciudades una sola vez, calcular rutas, comparar costos, elegir una opcion y crear una reserva.

## Cambios recientes

Esta seccion documenta las mejoras aplicadas al proyecto siguiendo el plan de mejoras
(`PLAN_MEJORAS.md`), organizado en fases atomicas.

### Fase 0 - Seguridad

Se elimino la clave de API de Gemini que estaba escrita en un comentario del codigo
fuente (`app/ai/gemini_recommendations.py`). A partir de ahora la clave se toma
unicamente de la variable de entorno `GOOGLE_API_KEY`, que cada integrante configura en
un archivo `.env` local. Se agrego `.env` al `.gitignore` para que nunca se suba al
repositorio. Como accion complementaria del equipo, la clave anterior debe revocarse en
Google AI Studio y reemplazarse por una nueva: dado que la credencial vieja quedo en el
historial de commits, revocarla es lo que la neutraliza definitivamente.

### Fase 1 - Higiene del repositorio

Se limpio el repositorio y se integro Redis de forma real:

- **Redis integrado con fallback.** Se agrego `app/caches/cache_backend.py`, un selector
  que usa Redis cuando esta configurado (`CACHE_BACKEND=redis`) y cae al LRU en memoria si
  Redis no esta disponible. La API pasa a usar este selector en lugar de instanciar el LRU
  directamente. Con esto el cache distribuido deja de ser codigo muerto sin dejar de
  funcionar cuando no hay un servidor Redis levantado.
- **Codigo muerto eliminado.** Se quito `start_batch_loop()` de `server.py` (nunca se
  llamaba; el arranque real usa `batch_loop()` en el `lifespan`), el modelo
  `AIRecommendation` y los campos `recommendations` de las respuestas (nunca se poblaban),
  y el directorio huerfano `app/ml/`.
- **Ignorados de git.** Se agregaron `.env`, `**/.venv/`, `.coverage` y `htmlcov/` al
  `.gitignore`, y se quito `.coverage` del control de versiones. Los entornos virtuales ya
  no se versionan.
- **README actualizado.** El arbol de directorios del README ahora refleja la estructura
  real: incluye `core/tsp_genetic.py`, `data/`, `ai/gemini_recommendations.py` y
  `caches/cache_backend.py`, y ya no menciona el inexistente `ml/recommender.py`.

### Fase 2 - Cobertura de tests del algoritmo genetico

Se agrego `app/tests/test_tsp_genetic.py`, que antes no existia. El algoritmo genetico
pasaba de tener 0% de cobertura a 94%. Los tests verifican:

- **Permutacion valida:** el tour visita las 5 ciudades una sola vez y cierra en el origen.
- **Determinismo:** con la misma semilla, dos corridas dan el mismo costo y la misma ruta.
- **Calidad:** en 5 ciudades el genetico alcanza el optimo exacto que calcula Held-Karp.
- **Elitismo:** el mejor costo por generacion nunca empeora (monotono no creciente).
- **Casos borde:** una ciudad, dos ciudades y una matriz con ciudad inalcanzable (costo
  infinito) sin que el algoritmo se rompa.

Estos tests son la red de seguridad para cualquier refactor futuro del genetico (por
ejemplo, mover la seleccion de algoritmo al backend). La suite completa pasa de
`26 passed` a `33 passed, 1 skipped`.

### Fase 4.B - Laboratorio de comparacion de algoritmos

Se agrego una pagina nueva "Laboratorio" a la interfaz (`app/ui/streamlit_app.py`) que
compara empiricamente el algoritmo exacto (Held-Karp) contra el heuristico (genetico).
Es la demostracion central del tema de la materia: mostrar donde el exacto deja de
escalar.

La pagina no depende de la API: arma las matrices de costos directamente desde
`routes_fixed` (ciudades europeas reales) para un subconjunto aleatorio de `n` ciudades,
reproducible con una semilla. Tiene dos bloques:

- **Comparacion puntual:** para un `n` elegido, corre ambos algoritmos y muestra una tabla
  con costo encontrado, tiempo de ejecucion y gap de optimalidad del genetico respecto al
  optimo exacto. Held-Karp corre solo hasta n=12; por encima se informa que es inviable
  (2^n estados) y solo se ejecuta el genetico.
- **Benchmark completo:** corre el rango n=4..14 y grafica el tiempo con eje Y logaritmico
  (Altair). Como Held-Karp no puede correr mas alla de n=12, se agrega una extrapolacion
  de su tiempo usando su complejidad teorica t = C * n^2 * 2^n (con C calibrado con el
  ultimo punto medido), lo que permite estimar el punto de cruce donde el exacto pasaria a
  ser mas lento que el heuristico (alrededor de n=16 con los parametros por defecto).

Hallazgo pedagogico: el genetico tiene un costo fijo alto (poblacion x generaciones), asi
que para pocas ciudades Held-Karp es mas rapido ademas de exacto. La ventaja del heuristico
no es la velocidad para n chico, sino que sigue siendo viable cuando el exacto ya no puede
correr por la explosion combinatoria.
