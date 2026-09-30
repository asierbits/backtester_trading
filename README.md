# Laboratorio · Módulo Trading

Laboratorio de estrategias de trading con **dinero 100 % ficticio**. Descarga datos de
mercado, prueba miles de estrategias con un backtester honesto, **detecta el humo**
(sobreajuste y suerte) y pasa las finalistas por un panel de agentes de IA que las
critica. Cada estrategia acaba con un veredicto —**APROBADA / SOSPECHOSA / DESCARTADA**—
y la lista de motivos.

> El objetivo no es encontrar una estrategia que "gane mucho" en el pasado, sino saber
> cuáles son humo. Nada de esto es consejo financiero y no se conecta con ningún broker.

---

## 1. Instalación (una sola vez)

Necesitas **Python 3.11 o superior** ([python.org](https://www.python.org/downloads/);
en Windows marca *"Add Python to PATH"* al instalarlo).

| Sistema | Instalar | Arrancar |
|---|---|---|
| **Mac / Linux** | `./instalar.sh` | `./arrancar.sh` |
| **Windows** | doble clic en `instalar.bat` | doble clic en `arrancar.bat` |

`arrancar` abre el panel en el navegador (`http://localhost:8501`). Para pararlo, pulsa
`Ctrl+C` en la terminal (o cierra la ventana en Windows).

**Clave de Anthropic (opcional, solo para los agentes):** copia `.env.example` como `.env`
(el instalador lo hace) y pon tu clave en `ANTHROPIC_API_KEY=...`, o pégala en la página
**Ajustes**. La clave solo vive en `.env`, que está en `.gitignore`: nunca se sube a GitHub
ni aparece en los logs.

## 2. Primeros pasos (5 minutos, sin conexión)

1. Arranca el panel y ve a **Trading → Nuevo experimento**.
2. Deja la fuente en **sintetico** (datos de ejemplo generados al azar) y pulsa
   **Lanzar experimento**.
3. Mira el **Laboratorio de humo** ("Probé 30.000 estrategias: el 99 % eran humo…"), el
   **Ranking** y el **Detalle** de las finalistas.

> Con datos sintéticos, cualquier estrategia que "gane" lo hace por azar: sirven para
> aprender a usar la herramienta, no para sacar conclusiones de mercado.

## 3. Descargar datos reales

Página **Datos → Descargar**:

- **Binance** (cripto, sin clave): BTC, ETH, SOL, BNB y XRP frente a USDT, velas de 1h y
  1d desde 2020 (configurable). Pagina, respeta el límite de la API, **reanuda** si se
  corta y guarda Parquet en `data/mercado/binance/`.
- **Yahoo Finance** (acciones/ETF): los tickers de `datos.acciones` en `config.yaml`.
  Yahoo solo da velas horarias de los últimos 730 días.
- **CSV propio**: columnas `timestamp, open, high, low, close, volume`.

La pestaña **Calidad** informa de velas faltantes, duplicados, precios imposibles,
volumen cero y saltos anómalos. **Los huecos nunca se rellenan con datos inventados.**

También desde la terminal:

```bash
.venv/bin/python -m modulos.trading.datos.descarga --fuente binance      # Mac/Linux
.venv\Scripts\python -m modulos.trading.datos.descarga --fuente binance  # Windows
```

## 4. Lanzar un experimento

En **Nuevo experimento** eliges activos, temporalidad, periodo, familias de estrategias,
rangos de parámetros, costes, capital, semilla y número máximo de combinaciones. Verás
cuántas estrategias se probarán, el tiempo estimado y el coste estimado de los agentes
antes de pulsar **Lanzar**.

Desde la terminal (usa `config.yaml`):

```bash
.venv/bin/python -m modulos.trading.experimento --fuente binance --temporalidad 1h --max 5000
```

Rendimiento medido en un portátil de 4 núcleos: **25.000 backtests (5.000 combinaciones ×
5 activos) sobre ~6,7 años de velas horarias (59.000 velas por activo) en ~30 segundos**.

### Qué hace por dentro

1. Divide cada activo en **entrenamiento (60 %)** y **test (40 %, lo más reciente)**.
2. Genera las combinaciones (rejilla o muestra aleatoria reproducible) y **guarda cuántas
   se probaron**, porque hace falta para corregir por múltiples pruebas.
3. Backtest de todas en train y test, en paralelo.
4. Controles: **comprar y mantener** y **estrategias aleatorias** con la misma frecuencia de
   operaciones (la vara de medir de la suerte).
5. Elige las finalistas **solo con el entrenamiento** (el test no se toca).
6. Validación completa de las finalistas (sección 5) y veredicto.
7. (Opcional) Panel de agentes de IA.

Todo queda en `data/laboratorio.sqlite` con la **configuración exacta y la semilla**: la
misma semilla da los mismos resultados.

## 5. Cómo interpretar los veredictos

| Veredicto | Significa |
|---|---|
| ✅ **APROBADA** | Superó **todas** las pruebas. Aun así puede fallar en el futuro: el siguiente paso es el paper trading. |
| ⚠️ **SOSPECHOSA** | Pasó algunas pruebas pero no todas. No hay pruebas suficientes de que la ventaja sea real. |
| ⛔ **DESCARTADA** | Falla fuera de muestra, tiene pocas operaciones, no se distingue del azar, depende de un pico de parámetros o desaparece con costes más altos. |

Las pruebas que se aplican a cada finalista:

| Prueba | Pregunta que responde |
|---|---|
| Train / test | ¿Cuánto se degrada fuera de muestra? (el Sharpe de test debe conservar ≥ 50 %) |
| Walk-forward | Re-optimizando en una ventana y validando en la siguiente, ¿cuántas ventanas son rentables? |
| Monte Carlo | Remuestreando las operaciones miles de veces, ¿qué rango de resultados y de caídas cabe esperar? |
| Robustez | Si cambias los parámetros un ~10 % (la media de 20 a 22), ¿se hunde? |
| Mapa de calor | ¿Está en una meseta estable o en un pico aislado? |
| Múltiples pruebas | Tras probar N estrategias, ¿supera al mejor resultado que daría la suerte? (Deflated Sharpe Ratio + Benjamini-Hochberg + reality check) |
| Aleatorias | ¿Supera al percentil 95 de las estrategias aleatorias en test? |
| Comprar y mantener | ¿Es mejor ajustado por riesgo (Sharpe o Calmar)? |
| Costes x2 | ¿Sigue en pie con el doble de comisiones y slippage? |
| Regímenes | ¿Solo gana cuando el mercado sube? |
| Nº de operaciones | ¿Hay suficientes (≥ 30 en train, ≥ 10 en test) para que signifique algo? |

Cada veredicto trae motivos legibles, por ejemplo: *"✗ Falla fuera de muestra: el Sharpe
pasa de 1,65 en entrenamiento a -0,27 en test."* Los umbrales se cambian en `config.yaml`
(secciones `veredicto` y `validacion`) o en **Ajustes**.

Las estrategias que no son finalistas reciben un veredicto básico (pocas operaciones,
azar, fallo en test) y como mucho pueden ser SOSPECHOSAS. Desde su **Detalle** puedes
pulsar **Validar a fondo** para pasarles todas las pruebas.

## 6. El panel de agentes críticos

Cada finalista pasa por cinco agentes (El Escéptico, El Gestor de Riesgo, El Estadístico,
El Realista de Ejecución y El Abogado del Diablo) y un **Árbitro** que resuelve
contradicciones y redacta el veredicto final. Reciben un resumen estructurado (métricas,
validaciones, veredicto automático), nunca datos crudos, y responden en JSON con esquema fijo.

- **La estadística manda**: los agentes solo pueden **endurecer** el veredicto, nunca relajarlo.
- Llamadas en paralelo con límite de concurrencia, **reintentos con espera exponencial** y
  validación del JSON (si falla, se reintenta y se registra).
- **Caché**: una llamada idéntica (mismo resumen, prompt, modelo y esfuerzo) no se repite ni se paga.
- **Control de coste**: límite de llamadas y de tokens por experimento y estimación del coste
  antes de lanzar (en **Nuevo experimento** y en **Ranking**).
- Modelo, esfuerzo y límites en **Ajustes** o en `config.yaml → agentes`. Por defecto usa
  `claude-opus-5-5` con el modelo de respaldo del servidor activado (si el modelo rechaza una
  petición, la API la repite con otro modelo); se desactiva con `usar_fallbacks: false`.
- Los prompts están en `modulos/trading/agentes/prompts.py`, con versión (`VERSION_PROMPTS`).

Sin clave, todo funciona igual salvo este panel.

## 7. Paper trading, alertas y cartera

- Desde el **Detalle** de una estrategia (idealmente APROBADA): **Seguir en paper trading**.
- En **Paper trading** la actualizas: descarga las velas cerradas de Binance, recalcula
  señales con el mismo motor y registra cada operación simulada. Compara lo simulado con lo
  que predijo el backtest.
- Bucle automático en una terminal aparte: `python -m modulos.trading.paper.paper_trading --bucle`.
- **Alertas** por Telegram o email: rellena `TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID` o las
  variables `SMTP_*` en `.env` y actívalas en Ajustes.
- **Cartera**: combina varias estrategias con pesos iguales, inverso de la volatilidad o mínima
  varianza (pesos calculados en train, evaluados en test).
- **Cortos** (-1) y **sizing por volatilidad**: se activan en **Nuevo experimento** o en `config.yaml → backtest`.

## 8. Añadir una estrategia nueva

1. Crea `modulos/trading/estrategias/mi_estrategia.py`:

   ```python
   import numpy as np
   from modulos.trading.estrategias.base import Estrategia, registrar
   from modulos.trading.estrategias.indicadores import CacheIndicadores, maquina_estados


   @registrar
   class MiEstrategia(Estrategia):
       familia = "mi_estrategia"
       nombre_legible = "Mi estrategia"
       descripcion = "Compra cuando el cierre supera la SMA y vende cuando la pierde."

       @classmethod
       def valida(cls, p):  # descarta combinaciones sin sentido
           return p["periodo"] > 2

       def _posiciones(self, c: CacheIndicadores, cortos: bool) -> np.ndarray:
           media = c.media("sma", self.parametros["periodo"])
           return maquina_estados(c.close > media, c.close < media)  # 1 = largo, 0 = fuera
   ```

   Usa solo datos hasta el cierre de cada vela: el motor ya ejecuta en la vela siguiente.
2. Impórtala en `modulos/trading/estrategias/__init__.py`.
3. Añade su espacio de parámetros en `config.yaml`:
   ```yaml
   parametros:
     mi_estrategia:
       periodo: [5, 100, 1]      # [mín, máx, paso]
   ```
4. Añádela a `experimento.familias` (o elígela en **Nuevo experimento**).
5. Añade un caso en `tests/test_sin_sesgo_futuro.py` y ejecuta los tests.

## 9. Añadir un módulo nuevo (voz, prospección, segundo cerebro…)

Crea `modulos/<nombre>/modulo.py` con:

```python
import streamlit as st

NOMBRE = "Voz"
ORDEN = 20
DESCRIPCION = "Agente de voz."


def paginas():
    return [st.Page("modulos/voz/paginas/inicio.py", title="Inicio", url_path="voz")]
```

`app.py` descubre los módulos solo: aparecerá en el menú lateral sin tocar nada del trading.
`core/` (configuración, base de datos y logs) es compartido por todos los módulos.

## 10. Estructura

```
laboratorio/
├── app.py                     # Entrada de Streamlit (menú lateral por módulos)
├── config.yaml                # Todos los parámetros
├── instalar.* / arrancar.*    # Instalación y arranque con un comando
├── core/                      # Config, SQLite y logs (compartido)
├── modulos/
│   ├── registro.py            # Descubre los módulos
│   └── trading/
│       ├── datos/             # Descarga (Binance, Yahoo, CSV, sintéticos) y limpieza
│       ├── estrategias/       # Indicadores, familias, controles y generador
│       ├── backtest/          # Motor (numba), costes y métricas
│       ├── validacion/        # Train/test, walk-forward, Monte Carlo, múltiples pruebas, regímenes, veredicto
│       ├── agentes/           # Prompts, cliente de la API y panel
│       ├── paper/             # Paper trading, alertas y cartera
│       ├── paginas/           # Páginas de Streamlit
│       ├── experimento.py     # Orquestación
│       ├── analisis.py        # Cálculos para el detalle (curvas, mapas de calor)
│       ├── graficas.py        # Gráficas Plotly (tema claro/oscuro)
│       └── informes.py        # Informe HTML
├── data/                      # Parquet y SQLite (no se suben a git)
├── informes/                  # Informes exportados
└── tests/                     # pytest
```

## 11. Tests y calidad

```bash
.venv/bin/python -m pytest      # 95 tests
.venv/bin/ruff check .          # estilo
```

Entre otros, comprueban:
- **Sin sesgo de futuro**: cambiar los datos futuros no altera ninguna señal ni la curva
  de capital pasada (todas las familias, con y sin cortos, y los tres modos de slippage).
- **Motor contra cálculo a mano**: una compra y una venta con comisión conocida, slippage, cortos.
- **Reproducibilidad**: dos experimentos con la misma semilla dan resultados idénticos.
- **Agentes** con una API simulada: reintentos ante JSON inválido, caché, presupuesto y que
  nunca relajan un veredicto.

## 12. Decisiones tomadas y limitaciones

- **Cripto primero** (Binance, 24/7). Las acciones funcionan con Yahoo, pero su calendario
  bursátil hace que en velas horarias el informe de calidad cuente como "faltantes" las
  horas con el mercado cerrado.
- **Corrección por múltiples pruebas**: se usan a la vez Deflated Sharpe Ratio, Benjamini-Hochberg
  y un reality check contra el "mejor de N" aleatorias. Para aprobar se exige DSR ≥ 95 % y
  significancia BH. Es exigente a propósito; el reality check asume pruebas independientes,
  así que es conservador (las estrategias reales están correlacionadas).
- **Ejecución**: la señal del cierre de t se ejecuta en la apertura de t+1 (configurable a
  cierre de t+1). Sin apalancamiento; el tamaño se fija al entrar.
- **Monte Carlo** remuestrea las operaciones de **test** (fuera de muestra): con pocas
  operaciones el intervalo es muy ancho, y eso también es información.
- **Sin impuestos, financiación de cortos ni comisiones de retirada**: el agente Realista
  de Ejecución los menciona, pero el motor no los modela.
- **PDF**: el informe se exporta en HTML con estilos de impresión; para PDF usa
  "Imprimir → Guardar como PDF" del navegador.
- **Paper trading** necesita conexión a Binance; sin conexión puede "reproducir" el último
  10 % de los datos guardados como demostración.
