from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False, slow_mo=500)
    page = browser.new_page()

    URL = "https://jumafernandez.github.io/itba-data-science/web-scraping/data/formulario-saldo.html"
    page.goto(URL)

    # Completamos el formulario y hacemos clic en "Ingresar"
    page.locator("#email").fill("usuario@ejemplo.com")
    page.locator("#passwd").fill("888888")
    page.locator("button[type='submit']").click()

    # La página tarda 3 segundos en mostrar el saldo: el elemento #saldo no existe hasta entonces.
    # No programamos ninguna espera. Playwright espera solo a que aparezca (auto-waiting).
    saldo = page.locator("#saldo").inner_text()
    print("Saldo actual:", saldo)

    page.wait_for_timeout(3000)
    browser.close()
