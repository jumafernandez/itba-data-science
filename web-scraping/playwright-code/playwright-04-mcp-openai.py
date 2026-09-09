# Antes de ejecutar este script:
#   1) Levantar el servidor MCP de Playwright:
#        npx @playwright/mcp@latest --port 8931
#   2) Exponerlo con ngrok (OpenAI necesita una URL pública):
#        ngrok http 8931 --host-header=localhost:8931
#   3) Copiar la URL que muestra ngrok en SERVER_URL (agregando /mcp al final)
#   4) Pegar la clave de OpenAI en el parámetro api_key (solo para la demo: nunca subir una clave a GitHub)

from openai import OpenAI
import pandas as pd
import json

#client = OpenAI(api_key="PEGAR-ACA-TU-CLAVE-DE-OPENAI")
client = OpenAI()   # lee la clave de la variable de entorno OPENAI_API_KEY

SERVER_URL = "https://nonadept-pearlene-yawnful.ngrok-free.dev/mcp"

prompt = """
Abrí https://www.itba.edu.ar/ y listá los títulos de las noticias de la sección "¿Qué está pasando en el ITBA?".
Respondé SOLO en JSON con esta estructura:
{
  "noticias": [
    {"titulo": "string"}
  ]
}
"""

response = client.responses.create(
    model="gpt-6-astra",
    tools=[
        {
            "type": "mcp",
            "server_label": "playwright",
            "server_url": SERVER_URL,
            "headers": {"ngrok-skip-browser-warning": "1"},
            "require_approval": "never",
            "allowed_tools": ["browser_navigate", "browser_snapshot"],
        }
    ],
    input=prompt,
    text={"format": {"type": "json_object"}},   # obliga a que la respuesta sea JSON válido
)

print("Respuesta del modelo:")
print(response.output_text)

data = json.loads(response.output_text)
df = pd.DataFrame(data["noticias"])
print(df)
