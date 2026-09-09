from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False, slow_mo=500)
    page = browser.new_page()

    URL = "https://jumafernandez.github.io/itba-data-science/web-scraping/data/formulario.html"
    page.goto(URL)

    # Localizamos los input por su id y escribimos en ellos
    page.locator("#email").fill("estudiante@example.com")
    page.locator("#passwd").fill("888888")

    # Borramos lo escrito y lo volvemos a cargar
    page.locator("#email").clear()
    page.locator("#passwd").clear()
    page.locator("#email").fill("estudiante@example.com")
    page.locator("#passwd").fill("888888")

    # Localizamos el botón de tipo submit y hacemos clic
    page.locator("button[type='submit']").click()

    # Después del clic la página muestra un div que estaba oculto.
    # No hace falta programar una espera: Playwright espera solo a que aparezca (auto-waiting).
    resultado = page.locator("#contenidoMostrado").inner_text()
    print("La página respondió:")
    print(resultado)

    page.wait_for_timeout(3000)  # 3 segundos para ver el resultado en pantalla
    browser.close()
