"""Capturas del dashboard, una por pestaña principal, para los documentos de la fase 4.

Levanta la app con Streamlit en un puerto libre, la abre en Chrome sin interfaz por el
protocolo de DevTools (`choreographer`, que llega con kaleido, así que no hay que instalar nada)
y, para cada pestaña principal, hace clic en ella, espera a que Streamlit termine de dibujar y
guarda la página completa en `figuras/fase-4/tablero-<pestaña>.png`. Los nombres de las pestañas
salen de la propia página, en su orden. Una captura simple de Chrome sin interfaz no sirve:
no puede cambiar de pestaña.

Uso, desde la raíz del repositorio: uv run python -m saber11.capturas
Necesita Chrome o Chromium; la variable BROWSER_PATH elige otro ejecutable.
"""

from __future__ import annotations

import asyncio
import base64
import json
import socket
import subprocess
import sys
import time
import unicodedata
import urllib.request
from pathlib import Path

import choreographer

from saber11.datos import RAIZ_PROYECTO

APP = RAIZ_PROYECTO / "dashboard" / "app.py"
DESTINO = RAIZ_PROYECTO / "figuras" / "fase-4"
ANCHO, ALTO = 1400, 900
ESPERA_MAXIMA_S = 120
# Las pestañas principales son los botones de la primera lista de pestañas de la página.
BOTONES_JS = "[...document.querySelector('[role=\"tablist\"]').querySelectorAll('[role=\"tab\"]')]"
PESTANAS_JS = BOTONES_JS + ".map(b => b.innerText.trim())"
LISTA_JS = (
    "document.querySelectorAll('[role=\"tablist\"]').length > 0 && !document.querySelector('[data-testid=\"stStatusWidget\"]')"
    " && [...document.images].every(i => i.complete)"
)
ALTO_JS = "Math.max(document.querySelector('[data-testid=\"stMainBlockContainer\"]').scrollHeight + 40, %d)" % ALTO


def nombre_de_archivo(pestana: str) -> str:
    """`tablero-pruebas-de-hipotesis.png` para «Pruebas de hipótesis»."""
    sin_tildes = unicodedata.normalize("NFKD", pestana).encode("ascii", "ignore").decode()
    return f"tablero-{'-'.join(sin_tildes.lower().split())}.png"


def puerto_libre() -> int:
    with socket.socket() as conexion:
        conexion.bind(("127.0.0.1", 0))
        return conexion.getsockname()[1]


def levantar_app(puerto: int) -> subprocess.Popen:
    """Streamlit sin interfaz en `puerto`; vuelve cuando el servidor responde a su prueba de salud."""
    proceso = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(APP), "--server.headless", "true", "--server.port", str(puerto),
         "--server.fileWatcherType", "none", "--browser.gatherUsageStats", "false", "--client.toolbarMode", "viewer"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    limite = time.monotonic() + ESPERA_MAXIMA_S
    while time.monotonic() < limite:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{puerto}/_stcore/health", timeout=2) as respuesta:
                if respuesta.read() == b"ok":
                    return proceso
        except OSError:
            time.sleep(0.5)
    proceso.terminate()
    raise TimeoutError("Streamlit no respondió a tiempo")


async def evaluar(pestana: choreographer.Tab, expresion: str) -> object:
    respuesta = await pestana.send_command("Runtime.evaluate", params={"expression": expresion, "returnByValue": True})
    return respuesta["result"]["result"].get("value")


async def esperar_dibujo(pestana: choreographer.Tab, reposo_s: float = 3.0) -> None:
    """Espera a que la app termine de correr y a que carguen las imágenes; luego deja que Plotly dibuje."""
    limite = time.monotonic() + ESPERA_MAXIMA_S
    while not await evaluar(pestana, LISTA_JS):
        if time.monotonic() > limite:
            raise TimeoutError("La app no terminó de dibujar")
        await asyncio.sleep(0.5)
    await asyncio.sleep(reposo_s)


async def capturar_pestanas(url: str, destino: Path) -> list[Path]:
    destino.mkdir(parents=True, exist_ok=True)
    escritas = []
    async with choreographer.Browser(headless=True) as navegador:
        pestana = await navegador.create_tab(url)  # el tamaño se fija por emulación: Chrome solo lo acepta al crear ventanas
        await pestana.send_command("Emulation.setDeviceMetricsOverride", params={"width": ANCHO, "height": ALTO, "deviceScaleFactor": 1, "mobile": False})
        await esperar_dibujo(pestana, reposo_s=5.0)
        for nombre in await evaluar(pestana, PESTANAS_JS):
            await evaluar(pestana, BOTONES_JS + f".find(b => b.innerText.trim() === {json.dumps(nombre)}).click()")
            await pestana.send_command("Emulation.setDeviceMetricsOverride", params={"width": ANCHO, "height": ALTO, "deviceScaleFactor": 1, "mobile": False})
            await esperar_dibujo(pestana)
            alto = int(await evaluar(pestana, ALTO_JS))
            await pestana.send_command("Emulation.setDeviceMetricsOverride", params={"width": ANCHO, "height": alto, "deviceScaleFactor": 1, "mobile": False})
            await esperar_dibujo(pestana)
            imagen = await pestana.send_command("Page.captureScreenshot", params={"format": "png"})
            ruta = destino / nombre_de_archivo(nombre)
            ruta.write_bytes(base64.b64decode(imagen["result"]["data"]))
            escritas.append(ruta)
    return escritas


def main() -> None:
    puerto = puerto_libre()
    servidor = levantar_app(puerto)
    try:
        escritas = asyncio.run(capturar_pestanas(f"http://127.0.0.1:{puerto}", DESTINO))
    finally:
        servidor.terminate()
        servidor.wait(timeout=30)
    for ruta in escritas:
        print(ruta.relative_to(RAIZ_PROYECTO))


if __name__ == "__main__":
    main()
