# Web scraping con Python

Material de la unidad de web scraping de *Ciencia de datos aplicada* (ITBA).

```
web-scraping/
├── data/              páginas HTML de práctica (se sirven desde GitHub Pages)
├── selenium-code/     ejemplos con requests, BeautifulSoup y Selenium
├── playwright-code/   ejemplos con Playwright y extracción asistida por un LLM
└── requirements.txt   librerías necesarias
```

Las páginas de `data/` están publicadas en
`https://jumafernandez.github.io/itba-data-science/web-scraping/data/` y los scripts las usan por esa URL.

## 1. Preparar el entorno (una sola vez)

Se necesita Python 3.10 o superior. Desde la raíz del repositorio:

```bash
python3 -m venv .venv
source .venv/bin/activate            # en Windows: .venv\Scripts\activate
pip install -r web-scraping/requirements.txt
playwright install chromium
```

El último comando descarga el navegador Chromium que usa Playwright. Si se trabaja en
Visual Studio Code, abrir la carpeta del repositorio y elegir el intérprete `.venv`
cuando lo pregunte.

Cada vez que se abra una terminal nueva hay que volver a activar el entorno:

```bash
source .venv/bin/activate
```

## 2. Scripts de Playwright

Se ejecutan desde la raíz del repositorio con el entorno activado. Los cuatro abren una
ventana de Chromium y muestran el resultado en la terminal.

| Script | Qué hace |
|---|---|
| `playwright-01-primer-ejemplo.py` | Abre la tabla de tiendas, extrae el `h1` y lista las marcas. |
| `playwright-02-interacciones-formulario.py` | Completa el formulario de inicio de sesión, hace clic en *Ingresar* y lee la respuesta. |
| `playwright-03-resolucion-tp.py` | Resuelve el formulario del TP: lista los elementos con sus atributos, completa todos los campos y lee la respuesta. |
| `playwright-04-esperas-automaticas.py` | Inicia sesión en una página que tarda 3 segundos en mostrar el saldo. No hay esperas programadas: Playwright espera solo a que aparezca el elemento. |

```bash
python web-scraping/playwright-code/playwright-01-primer-ejemplo.py
python web-scraping/playwright-code/playwright-02-interacciones-formulario.py
python web-scraping/playwright-code/playwright-03-resolucion-tp.py
python web-scraping/playwright-code/playwright-04-esperas-automaticas.py
```

Todos usan la API síncrona (`sync_playwright`) y el bloque `with` que se ve en las diapositivas.
`headless=False` muestra la ventana y `slow_mo=500` frena cada acción medio segundo para poder seguirla.

## 3. Extracción asistida por un LLM (`playwright-05-mcp-openai.py`)

Este script no controla el navegador directamente: le pide a un modelo de OpenAI que lo haga
a través de un servidor MCP de Playwright. Hacen falta **tres terminales** y una clave de OpenAI.

**Terminal 1: servidor MCP.** Requiere Node.js. Queda escuchando en el puerto 8931.

```bash
npx @playwright/mcp@latest --port 8931
```

**Terminal 2: túnel público con ngrok.** OpenAI no puede llegar a `localhost`, por eso hay que
exponer el puerto. Requiere una cuenta gratuita de ngrok con el authtoken configurado.

```bash
ngrok http 8931 --host-header=localhost:8931
```

Copiar la URL que aparece en la línea *Forwarding* y pegarla en la variable `SERVER_URL`
del script, agregando `/mcp` al final. Si la cuenta tiene un dominio fijo reservado, se puede
usar `--domain <dominio>` y la URL no cambia entre corridas.

**Terminal 3: el script.** Con el entorno activado y la clave en la variable de entorno:

```bash
export OPENAI_API_KEY=sk-...
python web-scraping/playwright-code/playwright-05-mcp-openai.py
```

Se abre un Chrome que navega solo hasta la página pedida. En la terminal de ngrok se ven las
llamadas `POST /mcp` que hace OpenAI. Al terminar, el script imprime la respuesta del modelo
en JSON y la misma información como DataFrame.

Nunca escribir la clave dentro del script si el archivo se va a subir a un repositorio:
OpenAI detecta las claves publicadas y las revoca.

### Cómo está armado el pedido

- `tools` declara el servidor MCP con su URL y limita las herramientas a `browser_navigate` y `browser_snapshot`.
- El prompt está escrito en lenguaje natural, sin nombrar herramientas ni parámetros. Eso solo funciona bien con
  modelos recientes (`gpt-6-astra`): los anteriores, como `gpt-4.1`, tienden a pedir snapshots parciales
  (`target` o `depth`), no llegan a ver el contenido y devuelven una lista vacía.
- `text={"format": {"type": "json_object"}}` obliga a que la respuesta sea JSON válido para que `json.loads` no falle.

## 4. Scripts de Selenium

Están en `selenium-code/` y corresponden a las diapositivas anteriores. Usan la misma carpeta `data/`.
Selenium descarga el driver de Chrome automáticamente, pero necesita Google Chrome instalado.

```bash
python web-scraping/selenium-code/ppt2-interacciones-selenium.py
```

## Problemas frecuentes

- **`ModuleNotFoundError: No module named 'playwright'`**: el entorno no está activado o no se instalaron las librerías (sección 1).
- **`Executable doesn't exist`** al lanzar el navegador: falta correr `playwright install chromium`.
- **ngrok responde con una página de advertencia**: el script ya envía el header `ngrok-skip-browser-warning`; verificar que la URL termine en `/mcp`.
- **El modelo devuelve `{"noticias": []}`**: no leyó la página completa. Suele pasar con modelos anteriores a `gpt-6`; si no se puede cambiar el modelo, agregar al prompt que llame a `browser_snapshot` sin argumentos.
- **El DataFrame sale vacío o `json.loads` falla**: imprimir `response.output_text` para ver qué devolvió el modelo.
