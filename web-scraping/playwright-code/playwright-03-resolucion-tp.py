from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False, slow_mo=500)
    page = browser.new_page()

    URL = "https://jumafernandez.github.io/itba-data-science/web-scraping/data/formulario-tp.html"
    page.goto(URL)

    # Consigna 1: listar los elementos del formulario y el valor de sus atributos
    elementos = page.locator("#formulario input, #formulario select, #formulario button").all()
    for elemento in elementos:
        print("id:", elemento.get_attribute("id"),
              "| name:", elemento.get_attribute("name"),
              "| type:", elemento.get_attribute("type"),
              "| placeholder:", elemento.get_attribute("placeholder"))

    # Consigna 2: interactuar con todos los elementos y presionar "Ingresar"
    page.locator("#email").fill("correo@example.com")
    page.locator("#passwd").fill("contraseña123")
    page.locator("#recordar").check()
    page.locator("#pais").select_option(label="Brasil")
    page.locator("button[type='submit']").click()

    # El div con el resultado aparece recién después del clic.
    # En Selenium hacía falta un WebDriverWait; acá Playwright espera solo (auto-waiting).
    print("La página respondió:")
    print(page.locator("#contenidoMostrado").inner_text())

    page.wait_for_timeout(3000)
    browser.close()
