"""Pruebas de saber11.mlp en CPU: determinismo con la semilla y corte de la parada temprana."""

import numpy as np
import pytest
import torch

from saber11 import mlp

CPU = torch.device("cpu")


@pytest.fixture
def datos() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(2)
    X = rng.normal(size=(600, 6)).astype(np.float32)
    y = 250 + 20 * X[:, 0] - 10 * X[:, 1] + rng.normal(0, 5, 600)
    return X[:500], y[:500], X[500:], y[500:]


def test_dos_entrenamientos_con_la_misma_semilla_dan_los_mismos_pesos(datos):
    mlp.configurar_determinismo()
    primero = mlp.entrenar(*datos, en=CPU, epocas=4, lote=64)
    torch.rand(100)  # consumir el generador entre los dos no debe cambiar nada
    segundo = mlp.entrenar(*datos, en=CPU, epocas=4, lote=64)
    assert primero.estado.keys() == segundo.estado.keys()
    assert all(torch.equal(primero.estado[k], segundo.estado[k]) for k in primero.estado)
    assert primero.perdidas_validacion == segundo.perdidas_validacion
    otro = mlp.entrenar(*datos, en=CPU, epocas=4, lote=64, semilla=7)
    assert not all(torch.equal(primero.estado[k], otro.estado[k]) for k in primero.estado)


def test_la_parada_temprana_corta_al_cumplir_la_paciencia():
    parada = mlp.ParadaTemprana(paciencia=3)
    perdidas = [5.0, 4.0, 3.0, 3.5, 3.0, 3.2, 1.0]
    detenciones = [parada.registrar(epoca, perdida) for epoca, perdida in enumerate(perdidas, start=1)]
    assert detenciones == [False, False, False, False, False, True, False]
    assert parada.epoca_mejor == 7


def test_sin_aprendizaje_el_entrenamiento_se_detiene_en_la_paciencia(datos):
    resultado = mlp.entrenar(*datos, en=CPU, epocas=50, paciencia=4, lote=64, tasa=0.0)
    assert resultado.epoca_mejor == 1
    assert resultado.epocas_corridas == 1 + 4
    assert len(resultado.perdidas_entrenamiento) == resultado.epocas_corridas


def test_predice_en_la_escala_del_puntaje_con_los_pesos_de_la_mejor_epoca(datos):
    X_e, y_e, X_v, y_v = datos
    resultado = mlp.entrenar(X_e, y_e, X_v, y_v, en=CPU, epocas=30, lote=64, tasa=1e-2)
    prediccion = mlp.predecir(resultado, X_v, en=CPU)
    assert prediccion.shape == (len(X_v),)
    rmse = np.sqrt(np.mean((prediccion - y_v) ** 2))
    assert rmse < np.std(y_v)
    assert np.mean((prediccion - y_v) ** 2) / resultado.escala**2 == pytest.approx(min(resultado.perdidas_validacion), rel=1e-4)
