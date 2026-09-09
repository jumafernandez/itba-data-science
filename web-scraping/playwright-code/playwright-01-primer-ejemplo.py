from playwright.sync_api import sync_playwright

# 1. Inicializamos el entorno de Playwright
with sync_playwright() as p:

    # 2. Abrimos Chromium (headless=False para ver la ventana) y creamos una página (pestaña)
    browser = p.chromium.launch(headless=False, slow_mo=500)
    page = browser.new_page()

    # 3. Navegamos a un sitio web (hay que pasar el protocolo completo: https://)
    URL = "https://jumafernandez.github.io/itba-data-science/web-scraping/data/ppt1-example-01.html"
    page.goto(URL)

    # 4. Localizamos elementos en el DOM y extraemos la información visible
    titulo = page.locator("h1").inner_text()
    print("Título de la página (h1):", titulo)

    # Todas las celdas <td> con clase "marca" (nombres de las tiendas)
    tiendas = page.locator("td.marca").all_inner_texts()
    print("Cantidad de tiendas:", len(tiendas))
    for tienda in tiendas:
        print("-", tienda)

    # 5. Cerramos el navegador
    browser.close()
