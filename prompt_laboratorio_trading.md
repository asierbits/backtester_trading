# PROMPT: LABORATORIO DE ESTRATEGIAS DE TRADING (DINERO FICTICIO)

## 1. ROL Y OBJETIVO

Actúa como un ingeniero senior de software y un quant (analista cuantitativo) con experiencia en backtesting riguroso. Vas a construir, de principio a fin, una aplicación local llamada **"Laboratorio"**: un laboratorio de estrategias de trading con dinero 100 % ficticio.

La app debe:
1. Descargar datos históricos de mercado (cripto primero, acciones después).
2. Probar miles de estrategias automáticamente con un backtester realista.
3. Detectar y penalizar el sobreajuste (overfitting) y la suerte estadística.
4. Usar un panel de agentes de IA (API de Claude) que critican cada estrategia finalista.
5. Mostrar todo en un panel web con ranking, gráficas y un veredicto por estrategia: **APROBADA / SOSPECHOSA / DESCARTADA**, con el motivo.

El objetivo NO es fabricar una estrategia que "gane mucho" en el pasado. El objetivo es **descubrir cuáles son humo y cuáles sobreviven a una validación seria**. Un programa que diga "esta estrategia es humo" vale más que uno que enseñe gráficas bonitas.

No se opera con dinero real en ningún momento. No hay conexión con brokers para ejecutar órdenes. Es solo para aprender y analizar.

## 2. CONTEXTO DEL USUARIO

- Soy emprendedor y dirijo una agencia de IA. Quiero que esta app sea el primer módulo de una aplicación personal más grande que irá creciendo por módulos (otros módulos futuros: agente de voz, prospección, segundo cerebro). Por eso la arquitectura debe ser **modular**, con un panel principal con menú lateral donde "Trading" sea el primer módulo.
- Uso el ordenador en Windows o Mac (haz el código compatible con ambos). Sé programar lo básico, así que quiero instrucciones de instalación claras y un único comando para arrancarlo.
- Idioma de la interfaz, textos y comentarios del código: **español**.

## 3. STACK TÉCNICO

- **Python 3.11+**.
- **Datos:** `pandas`, `numpy`, `pyarrow` (guardar datos en Parquet).
- **Descarga de datos:** `ccxt` (Binance como fuente principal de cripto, sin clave de API para datos públicos) y `yfinance` (acciones/ETF, como módulo opcional).
- **Backtesting:** implementa tu propio motor vectorizado con `numpy`/`pandas` (para tener control total de comisiones, slippage y sesgos). Puedes usar `numba` para acelerar. No dependas de una librería de backtesting cerrada.
- **Indicadores:** implementa los indicadores tú mismo con pandas/numpy (SMA, EMA, RSI, ATR, Bandas de Bollinger, MACD, ruptura de canales Donchian) o usa `pandas-ta` si está disponible. Verifica que no introducen sesgo de futuro.
- **Base de datos:** SQLite con `sqlite3` o `SQLAlchemy`, para guardar experimentos, resultados y críticas.
- **Panel web:** **Streamlit** con navegación multipágina, gráficas con **Plotly**.
- **Agentes de IA:** SDK oficial `anthropic` de Python. La clave se lee de un archivo `.env` (`ANTHROPIC_API_KEY`) con `python-dotenv`. Nunca escribas la clave en el código ni la subas a ningún sitio. El nombre del modelo debe ser configurable en un archivo de configuración.
- **Configuración:** un archivo `config.yaml` con todos los parámetros (activos, fechas, capital inicial, comisiones, slippage, número de estrategias, etc.).
- **Pruebas:** `pytest`.
- **Calidad:** tipado con type hints, docstrings en español, `ruff` para formato.

## 4. ARQUITECTURA DE CARPETAS

```
laboratorio/
├── README.md                  # Instalación y uso en español, paso a paso
├── requirements.txt
├── .env.example               # ANTHROPIC_API_KEY=pon_tu_clave_aqui
├── config.yaml
├── instalar.sh / instalar.bat # Crea el entorno virtual e instala todo
├── arrancar.sh / arrancar.bat # Lanza el panel con un solo comando
├── app.py                     # Entrada de Streamlit (menú lateral por módulos)
├── core/
│   ├── config.py              # Carga de config.yaml y .env
│   ├── db.py                  # SQLite: esquema y acceso
│   └── logging_setup.py
├── modulos/
│   └── trading/
│       ├── datos/
│       │   ├── descarga.py    # ccxt/yfinance → Parquet local, con caché y reanudación
│       │   └── limpieza.py    # Huecos, duplicados, outliers, zonas horarias
│       ├── estrategias/
│       │   ├── base.py        # Clase base Estrategia: parámetros → señales
│       │   ├── cruce_medias.py
│       │   ├── rsi_reversion.py
│       │   ├── ruptura_donchian.py
│       │   ├── bollinger.py
│       │   ├── macd.py
│       │   └── generador.py   # Crea miles de variaciones de parámetros
│       ├── backtest/
│       │   ├── motor.py       # Simulación vectorizada realista
│       │   ├── costes.py      # Comisiones y slippage
│       │   └── metricas.py    # Todas las métricas
│       ├── validacion/
│       │   ├── train_test.py
│       │   ├── walk_forward.py
│       │   ├── monte_carlo.py
│       │   ├── multiples_pruebas.py  # Corrección por probar miles de estrategias
│       │   └── veredicto.py   # Reglas que deciden APROBADA/SOSPECHOSA/DESCARTADA
│       ├── agentes/
│       │   ├── panel.py       # Orquesta los agentes críticos
│       │   ├── prompts.py     # Prompts de cada agente
│       │   └── cliente.py     # Llamadas a la API con reintentos y control de coste
│       ├── paper/
│       │   └── paper_trading.py  # (Fase 3) Paper trading en tiempo real
│       └── paginas/           # Páginas de Streamlit del módulo
├── data/                      # Parquet y SQLite (en .gitignore)
├── informes/                  # Informes exportados en HTML/PDF
└── tests/
```

## 5. DATOS HISTÓRICOS

- Activos por defecto (configurables): BTC/USDT, ETH/USDT, SOL/USDT, BNB/USDT, XRP/USDT.
- Temporalidades: 1h y 1d (configurable, con opción de 4h).
- Periodo por defecto: desde 2020-01-01 hasta hoy.
- La descarga debe: paginar correctamente, respetar los límites de la API (rate limit), reanudar si se corta, guardar en Parquet por activo y temporalidad, y mostrar una barra de progreso.
- Limpieza: detectar y reportar velas faltantes, duplicados, precios a cero o negativos, volumen cero y saltos anómalos. Nunca rellenar huecos con datos inventados sin avisarlo en el informe.
- Tener un **modo de datos de ejemplo** (datos sintéticos realistas generados con un proceso estocástico) para poder probar la app sin conexión.
- Debe poder **importar un CSV propio** con columnas `timestamp, open, high, low, close, volume`.

## 6. ESTRATEGIAS

Cada estrategia es una clase con parámetros y un método que devuelve una serie de posiciones objetivo (por ejemplo: 1 = largo, 0 = fuera de mercado; en una fase posterior también -1 = corto).

Estrategias iniciales y su espacio de parámetros (ejemplos, configurable):
1. **Cruce de medias** (SMA y EMA): media rápida 5–50, media lenta 20–250.
2. **RSI reversión a la media**: periodo 5–30, umbral de compra 15–40, umbral de venta 60–85.
3. **Ruptura de canal Donchian**: ventana de entrada 10–100, ventana de salida 5–50.
4. **Bandas de Bollinger**: periodo 10–50, desviaciones 1.5–3.
5. **MACD**: parámetros rápida/lenta/señal.
6. **Combinadas**: filtro de tendencia (precio sobre SMA 200) + señal de entrada de las anteriores.

El **generador** crea miles de combinaciones con búsqueda en rejilla o aleatoria, y **guarda el número total de estrategias probadas**, porque es necesario para corregir por múltiples pruebas más adelante.

Incluye también estrategias **de control**:
- **Comprar y mantener (buy & hold)**.
- **Estrategia aleatoria** (entra y sale al azar con la misma frecuencia de operaciones). Sirve para saber qué resultado da la pura suerte.

## 7. MOTOR DE BACKTEST (REGLAS ESTRICTAS)

Esto es lo más importante. El motor debe ser **honesto**:

1. **Sin sesgo de futuro (lookahead bias):** una señal calculada con la vela de cierre del momento *t* solo puede ejecutarse en la apertura (o cierre) de la vela *t+1*. Añade un test automático que verifique que cambiar datos futuros no altera las decisiones pasadas.
2. **Comisiones:** configurables (por defecto 0,1 % por operación, estilo Binance spot), aplicadas en cada entrada y salida.
3. **Slippage (deslizamiento):** configurable (por defecto 0,05 %), y opcionalmente dependiente de la volatilidad y del volumen.
4. **Sin apalancamiento** en la versión inicial. Capital inicial ficticio: 1.000 €.
5. **Sin reinversión irreal:** tamaño de posición del 100 % del capital en la versión simple; opción de sizing por volatilidad en fases posteriores.
6. **Cálculo correcto de equity** (curva de capital) incluyendo operaciones abiertas (mark-to-market).
7. **Registro de cada operación:** fecha de entrada y salida, precios, comisión, resultado, duración.
8. **Rendimiento:** debe poder probar al menos **5.000 estrategias sobre 3 años de datos horarios de 5 activos en pocos minutos** en un portátil normal (vectoriza y usa numba, y paraleliza con `multiprocessing` o `joblib`).

### Métricas a calcular por estrategia
- Rentabilidad total y anualizada (CAGR).
- Volatilidad anualizada.
- **Sharpe** y **Sortino**.
- **Caída máxima (max drawdown)** y duración de la peor caída.
- **Calmar** (CAGR / max drawdown).
- Número de operaciones, % de operaciones ganadoras, ratio ganancia/pérdida media, profit factor, expectativa por operación.
- Exposición al mercado (% del tiempo invertido).
- Mayor racha de pérdidas.
- Comparativa con buy & hold en el mismo periodo.
- Rentabilidad neta tras costes vs bruta (para ver cuánto se comen las comisiones).

## 8. VALIDACIÓN ANTI-HUMO (EL CORAZÓN DEL PROYECTO)

Para cada estrategia finalista aplica, y guarda los resultados de:

1. **Train/Test:** divide cronológicamente (por ejemplo 60 % entrenamiento, 40 % test). Las estrategias se seleccionan con el entrenamiento y se evalúan en el test, que nunca se toca durante la selección. Reporta la **degradación** (cuánto empeora el rendimiento fuera de muestra).
2. **Walk-forward:** ventanas deslizantes (optimiza en una ventana, valida en la siguiente, avanza, repite). Reporta la curva de capital encadenada solo con tramos fuera de muestra y el porcentaje de ventanas rentables.
3. **Monte Carlo:**
   - Barajar/remuestrear el orden de las operaciones (bootstrap) miles de veces para obtener un intervalo de confianza de rentabilidad y de caída máxima.
   - Probar con pequeñas perturbaciones en los parámetros para comprobar **robustez** (una buena estrategia no debe hundirse si cambias la media de 20 a 22).
4. **Mapa de calor de parámetros:** para cada familia de estrategias, visualizar el rendimiento en función de dos parámetros. Las zonas estables (mesetas) son buena señal; los picos aislados son sobreajuste.
5. **Corrección por múltiples pruebas:** dado que se probaron N estrategias, calcula el **Deflated Sharpe Ratio** o, como mínimo, una corrección de Bonferroni/Benjamini-Hochberg y un test de "reality check" sencillo (comparar el mejor Sharpe contra la distribución de los mejores Sharpe obtenidos con estrategias aleatorias/barajadas). Explica el resultado en lenguaje claro.
6. **Comparación con la estrategia aleatoria y con buy & hold.**
7. **Número mínimo de operaciones:** descartar automáticamente las estrategias con pocas operaciones (por ejemplo < 30) por falta de significancia estadística.
8. **Coste de ejecución:** repetir el backtest con comisiones y slippage x2 y comprobar si la estrategia sigue en pie.
9. **Análisis por regímenes:** rendimiento en mercado alcista, bajista y lateral, para detectar estrategias que solo funcionaron porque el mercado subía.

### Veredicto automático (reglas configurables)
- **APROBADA:** supera train/test con degradación aceptable, es rentable en la mayoría de ventanas walk-forward, es robusta a cambios de parámetros, supera a buy & hold ajustado por riesgo (Sharpe o Calmar) y aguanta el doble de costes y la corrección por múltiples pruebas.
- **SOSPECHOSA:** cumple algunas pero no todas las condiciones.
- **DESCARTADA:** falla en fuera de muestra, tiene pocas operaciones, depende de un pico de parámetros o desaparece con costes realistas.

Cada veredicto incluye una **lista de motivos** legible ("Descartada: el Sharpe cae de 1,8 en entrenamiento a 0,1 en test").

## 9. PANEL DE AGENTES CRÍTICOS (CLAUDE API)

Cada estrategia finalista (por ejemplo el top 10–20 según el entrenamiento) pasa por un panel de agentes. Cada agente recibe un **resumen estructurado en JSON** con las métricas, resultados de validación y gráficas resumidas (nunca datos crudos enormes), y devuelve una respuesta **en JSON con esquema fijo**:

```
{
  "agente": "...",
  "puntuacion": 0-10,
  "preocupaciones": ["...", "..."],
  "puntos_fuertes": ["...", "..."],
  "recomendacion": "aprobar | dudar | descartar",
  "explicacion": "texto breve en español"
}
```

Agentes:
1. **El Escéptico:** ¿funcionó solo porque el mercado subía? ¿Es un resultado de una sola racha o de un solo periodo? ¿Hay sospechas de sobreajuste?
2. **El Gestor de Riesgo:** ¿la caída máxima es soportable? ¿Una mala racha de N operaciones seguidas te arruinaría? ¿El tamaño de posición es razonable?
3. **El Estadístico:** ¿hay suficientes operaciones? ¿Es significativa estadísticamente? ¿Cómo quedó tras la corrección por múltiples pruebas?
4. **El Realista de Ejecución:** ¿comisiones, slippage, liquidez, huecos de precio, fallos de la API, impuestos? ¿Es operable en la vida real con un capital pequeño?
5. **El Abogado del Diablo:** intenta construir la mejor explicación posible de por qué la estrategia podría ser un espejismo, y propone tests adicionales para romperla.
6. **El Árbitro:** lee las cinco críticas y el veredicto automático, resuelve contradicciones y redacta el **veredicto final** en un párrafo, indicando qué prueba adicional haría falta.

Requisitos del panel:
- Ejecución en paralelo con control de concurrencia y **reintentos con espera exponencial**.
- **Caché:** no repetir llamadas idénticas (hash del resumen + versión del prompt).
- **Control de coste:** límite configurable de llamadas y tokens por experimento, y estimación de coste antes de lanzar.
- **Validación del JSON** devuelto; si falla, reintentar y registrar el error.
- Los agentes **no pueden aprobar** una estrategia que el veredicto automático haya descartado; solo pueden endurecer el veredicto, nunca relajarlo. La estadística manda, la IA critica.
- Los prompts de los agentes viven en `prompts.py`, versionados y fáciles de editar.

## 10. PANEL WEB (STREAMLIT)

Páginas del módulo Trading:
1. **Inicio:** resumen de los últimos experimentos, estado de los datos y accesos rápidos.
2. **Datos:** descargar/actualizar activos, ver calidad de datos, importar CSV, gráfica de velas.
3. **Nuevo experimento:** elegir activos, temporalidad, periodo, familias de estrategias, rangos de parámetros, costes, capital inicial y número máximo de estrategias. Botón "Lanzar" con barra de progreso y tiempo estimado.
4. **Ranking:** tabla ordenable y filtrable con todas las estrategias, sus métricas y el veredicto (con colores). Columnas para train vs test.
5. **Detalle de estrategia:** curva de capital vs buy & hold, drawdown, operaciones sobre el gráfico de precio, distribución de resultados por operación, mapa de calor de parámetros, walk-forward, Monte Carlo (abanico de trayectorias) y las críticas de los agentes.
6. **Laboratorio de humo:** panel resumen del tipo "Probé 4.000 estrategias: el 97 % eran humo, X sobrevivieron". Incluye la comparación con estrategias aleatorias.
7. **Informes:** exportar un informe completo de un experimento en HTML (y PDF si es viable).
8. **Ajustes:** claves, modelo de IA, costes por defecto, límites de gasto.

Diseño: limpio, oscuro o claro seleccionable, gráficas interactivas con Plotly, sin saturar. Todo en español.

## 11. BASE DE DATOS (SQLite)

Tablas mínimas: `experimentos`, `estrategias` (familia, parámetros JSON, hash), `resultados` (métricas por tramo: train, test, walk-forward), `operaciones`, `validaciones`, `criticas_agentes`, `veredictos`, `datos_meta` (qué datos hay descargados y hasta qué fecha). Guarda la **configuración exacta** de cada experimento y la **semilla aleatoria** para que sea **reproducible**.

## 12. FASES DE ENTREGA

**Fase 1 (versión mínima funcional, prioridad máxima):**
- Estructura del proyecto, instalación y arranque con un comando.
- Descarga de datos de 5 criptos (diarios y horarios) + modo de datos de ejemplo.
- 3 estrategias (cruce de medias, RSI, Donchian) con generador de variaciones.
- Motor de backtest honesto con costes, métricas y tests anti-sesgo de futuro.
- Ranking y detalle de estrategia en Streamlit.
- Train/test y comparación con buy & hold y con estrategia aleatoria.
- Un único agente crítico sobre el top 10.

**Fase 2:**
- Walk-forward, Monte Carlo, mapa de calor, corrección por múltiples pruebas, análisis por regímenes y doble coste.
- Panel completo de agentes + árbitro, caché y control de coste.
- Bollinger, MACD y estrategias combinadas.
- Informes exportables.
- Soporte de acciones con `yfinance`.

**Fase 3:**
- Paper trading en tiempo real: seguir las estrategias APROBADAS sobre precios actuales con dinero ficticio, registrando cada operación simulada y comparando con lo que predijo el backtest.
- Alertas (Telegram o email) cuando una estrategia paper genera señal.
- Estrategias cortas (posiciones -1), sizing por volatilidad y optimización de cartera entre estrategias aprobadas.

## 13. REGLAS DE TRABAJO PARA TI (IMPORTANTE)

1. **Empieza por un plan breve** (arquitectura y orden de trabajo), y luego construye por fases. No intentes hacerlo todo de golpe.
2. **Escribe código real y completo, sin dejar "TODO" ni funciones vacías.** Si algo no se puede implementar, dímelo claramente.
3. **Prueba lo que construyas.** Ejecuta los tests, y si no tienes acceso a internet para descargar datos reales, usa el modo de datos sintéticos para verificar que todo el flujo funciona, y **dímelo explícitamente**.
4. **Sé honesto con los resultados.** Si con datos de ejemplo una estrategia "gana", recuérdame que son datos sintéticos. Nunca presentes resultados pasados como una promesa de rentabilidad futura. No des consejo financiero.
5. **Verifica tu trabajo:** revisa el motor con un caso simple calculado a mano (por ejemplo, una compra y una venta con comisión conocida) y confirma que coincide.
6. **Seguridad:** la clave de API solo va en `.env`, que debe estar en `.gitignore`. No registres claves en los logs.
7. **Documenta:** el `README.md` debe explicar, para alguien con poca experiencia, cómo instalar, cómo descargar datos, cómo lanzar un experimento, cómo interpretar los veredictos y cómo añadir una estrategia nueva.
8. **Al terminar cada fase**, resume en pocas líneas qué hay hecho, qué probaste, qué limitaciones quedan y cuál es el siguiente paso recomendado.
9. **Si una decisión es ambigua y cambia mucho el resultado** (por ejemplo cripto vs acciones, o tipo de corrección estadística), pregúntame antes de seguir; si no, decide con criterio, anótalo y continúa.

## 14. CRITERIOS DE ÉXITO

- Con un solo comando arranca el panel y puedo lanzar un experimento con datos reales.
- El sistema prueba como mínimo 5.000 estrategias en pocos minutos.
- Los resultados son reproducibles (misma semilla, mismos resultados).
- El programa descarta de forma convincente la mayoría de estrategias por sobreajuste, y puedo ver **por qué** cada una fue aprobada o descartada.
- El panel de agentes añade críticas útiles y específicas, no frases genéricas.
- El código es modular: puedo añadir otro módulo (por ejemplo "voz" o "prospección") al menú lateral sin reescribir nada del trading.

Empieza ahora con el plan de la Fase 1.
