"""Perceptrón multicapa en PyTorch para `PUNT_GLOBAL`, determinista y con parada temprana.

Recibe las entradas ya transformadas por el `Pipeline` ajustado en entrenamiento. Dos capas
ocultas de 128 y 64 con ReLU y abandono de 0,1, Adam con tasa de 1e-3, pérdida MSE, lotes de
4.096 y hasta 200 épocas con parada temprana (paciencia 10) en validación; al terminar se
restauran los pesos de la mejor época. El objetivo se estandariza con la media y la
desviación de entrenamiento, así que las pérdidas están en unidades de esa desviación al
cuadrado y las predicciones vuelven a la escala del puntaje.

Determinismo: `configurar_determinismo()` va antes de iniciar CUDA, porque sin
`CUBLAS_WORKSPACE_CONFIG` cuBLAS falla en modo determinista; la semilla se fija justo antes
de crear la red, para que nada de lo que corra antes en el cuaderno cambie su inicialización.
"""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

SEMILLA = 42
OCULTAS = (128, 64)
ABANDONO = 0.1
TASA = 1e-3
LOTE = 4096
EPOCAS = 200
PACIENCIA = 10
VARIABLE_CUBLAS, VALOR_CUBLAS = "CUBLAS_WORKSPACE_CONFIG", ":4096:8"


def configurar_determinismo() -> None:
    """Fija la variable de cuBLAS (si no estaba) y exige algoritmos deterministas."""
    os.environ.setdefault(VARIABLE_CUBLAS, VALOR_CUBLAS)
    torch.use_deterministic_algorithms(True)


def dispositivo() -> torch.device:
    """La GPU con más memoria si hay alguna; si no, CPU, con más tiempo y cifras que pueden diferir.

    En la máquina del proyecto hay dos GPU y torch numera primero la de menos memoria, así
    que `cuda` a secas no sería la RTX 5080. XGBoost recibe el mismo índice con
    `device=str(dispositivo())`. Va después de `configurar_determinismo()`, porque consultar
    las GPU inicia CUDA.
    """
    if not torch.cuda.is_available():
        return torch.device("cpu")
    indice = max(range(torch.cuda.device_count()), key=lambda i: torch.cuda.get_device_properties(i).total_memory)
    return torch.device("cuda", indice)


class Red(nn.Module):
    """Capas lineales con ReLU y abandono, y una salida lineal."""

    def __init__(self, entradas: int, ocultas: tuple[int, ...] = OCULTAS, abandono: float = ABANDONO) -> None:
        super().__init__()
        capas: list[nn.Module] = []
        for anterior, siguiente in zip((entradas, *ocultas[:-1]), ocultas, strict=True):
            capas += [nn.Linear(anterior, siguiente), nn.ReLU(), nn.Dropout(abandono)]
        self.capas = nn.Sequential(*capas, nn.Linear(ocultas[-1], 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.capas(x).squeeze(-1)


class ParadaTemprana:
    """Lleva la mejor pérdida de validación y avisa cuando pasan `paciencia` épocas sin mejorarla."""

    def __init__(self, paciencia: int = PACIENCIA) -> None:
        self.paciencia = paciencia
        self.mejor = float("inf")
        self.epoca_mejor = 0
        self.sin_mejora = 0

    def registrar(self, epoca: int, perdida: float) -> bool:
        """Registra la pérdida de la época; devuelve True si hay que detenerse."""
        if perdida < self.mejor:
            self.mejor, self.epoca_mejor, self.sin_mejora = perdida, epoca, 0
            return False
        self.sin_mejora += 1
        return self.sin_mejora >= self.paciencia


@dataclass(frozen=True)
class Entrenamiento:
    """Pesos de la mejor época (en CPU), curvas de pérdida y escala del objetivo."""

    estado: dict[str, torch.Tensor]
    perdidas_entrenamiento: tuple[float, ...]
    perdidas_validacion: tuple[float, ...]
    epoca_mejor: int
    entradas: int
    media: float
    escala: float

    @property
    def epocas_corridas(self) -> int:
        return len(self.perdidas_validacion)


def _tensor(valores: np.ndarray, en: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.asarray(valores, dtype=np.float32), device=en)


def _perdida_validacion(red: Red, X: torch.Tensor, y: torch.Tensor, lote: int) -> float:
    red.eval()
    with torch.no_grad():
        suma = sum(float(nn.functional.mse_loss(red(X[i : i + lote]), y[i : i + lote], reduction="sum")) for i in range(0, len(X), lote))
    return suma / len(X)


def _entrenar_en(
    X_entrenamiento: np.ndarray,
    y_entrenamiento: np.ndarray,
    X_validacion: np.ndarray,
    y_validacion: np.ndarray,
    en: torch.device,
    semilla: int,
    epocas: int,
    paciencia: int,
    lote: int,
    tasa: float,
) -> Entrenamiento:
    media, escala = float(np.mean(y_entrenamiento)), float(np.std(y_entrenamiento))
    X_e, X_v = _tensor(X_entrenamiento, en), _tensor(X_validacion, en)
    y_e, y_v = _tensor((np.asarray(y_entrenamiento) - media) / escala, en), _tensor((np.asarray(y_validacion) - media) / escala, en)
    torch.manual_seed(semilla)
    red = Red(X_e.shape[1]).to(en)
    optimizador = torch.optim.Adam(red.parameters(), lr=tasa)
    orden = torch.Generator().manual_seed(semilla)
    parada = ParadaTemprana(paciencia)
    perdidas_e: list[float] = []
    perdidas_v: list[float] = []
    mejor_estado: dict[str, torch.Tensor] = {}
    for epoca in range(1, epocas + 1):
        red.train()
        suma = 0.0
        for indices in torch.randperm(len(X_e), generator=orden).split(lote):
            indices = indices.to(en)
            optimizador.zero_grad()
            perdida = nn.functional.mse_loss(red(X_e[indices]), y_e[indices])
            perdida.backward()
            optimizador.step()
            suma += perdida.detach().item() * len(indices)
        perdidas_e.append(suma / len(X_e))
        perdidas_v.append(_perdida_validacion(red, X_v, y_v, lote))
        detener = parada.registrar(epoca, perdidas_v[-1])
        if parada.epoca_mejor == epoca:
            mejor_estado = {nombre: valor.detach().cpu().clone() for nombre, valor in red.state_dict().items()}
        if detener:
            break
    return Entrenamiento(mejor_estado, tuple(perdidas_e), tuple(perdidas_v), parada.epoca_mejor, X_e.shape[1], media, escala)


def entrenar(
    X_entrenamiento: np.ndarray,
    y_entrenamiento: np.ndarray,
    X_validacion: np.ndarray,
    y_validacion: np.ndarray,
    en: torch.device | None = None,
    semilla: int = SEMILLA,
    epocas: int = EPOCAS,
    paciencia: int = PACIENCIA,
    lote: int = LOTE,
    tasa: float = TASA,
) -> Entrenamiento:
    """Entrena la red y devuelve los pesos de la mejor época de validación, ya en CPU.

    La red, el optimizador y los tensores viven dentro de `_entrenar_en`; al volver ya no
    queda ninguna referencia a ellos y se libera la memoria de la GPU.
    """
    en = en or dispositivo()
    resultado = _entrenar_en(X_entrenamiento, y_entrenamiento, X_validacion, y_validacion, en, semilla, epocas, paciencia, lote, tasa)
    gc.collect()
    if en.type == "cuda":
        torch.cuda.empty_cache()
    return resultado


def predecir(entrenamiento: Entrenamiento, X: np.ndarray, en: torch.device | None = None, lote: int = LOTE) -> np.ndarray:
    """Predicciones en la escala del puntaje con los pesos de la mejor época."""
    en = en or dispositivo()
    red = Red(entrenamiento.entradas).to(en)
    red.load_state_dict(entrenamiento.estado)
    red.eval()
    entradas = _tensor(X, en)
    with torch.no_grad():
        salida = torch.cat([red(entradas[i : i + lote]) for i in range(0, len(entradas), lote)]).cpu().numpy()
    return salida.astype(float) * entrenamiento.escala + entrenamiento.media
