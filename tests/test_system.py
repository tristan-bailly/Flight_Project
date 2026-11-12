"""
Suite de tests unitaires et d'intégration du système ATC AI.
"""

import os
import sys
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    NM_TO_M,
    FT_TO_M,
    SEPARATION_HORIZ_M,
    SEPARATION_VERT_M,
    NMAC_HORIZ_M,
    NMAC_VERT_M
)
from src.simulation.aircraft import Aircraft
from src.simulation.encounters import compute_cpa, generate_encounter
from src.simulation.atc_env import RealisticATCEnv
from src.agent.dqn import D3QNAgent
from src.api.app import app

def test_icao_constants():
    """Vérifie le respect strict des normes OACI."""
    assert SEPARATION_HORIZ_M == 5.0 * 1852.0
    assert abs(SEPARATION_VERT_M - 304.8) < 1e-3
    assert NMAC_HORIZ_M < SEPARATION_HORIZ_M
    assert NMAC_VERT_M < SEPARATION_VERT_M

def test_aircraft_kinematics():
    """Vérifie la dynamique de vol de l'avion (3°/s de virage max, intégration)."""
    ac = Aircraft("TEST01", x=0.0, y=0.0, z=10000.0, speed=200.0, heading_deg=0.0)
    assert ac.vx == 0.0
    assert ac.vy == 200.0

    # Virage de 30° : en 2s avec taux max 3°/s, il ne doit tourner que de 6°
    ac.set_heading_target(30.0)
    ac.step(2.0)
    assert abs(ac.heading_deg - 6.0) < 1e-4

def test_cpa_computation():
    """Vérifie le calcul du point de rapprochement maximal (CPA)."""
    # Deux avions face-à-face à 200 m/s séparés de 20 000 m
    ac1 = Aircraft("AC1", x=0.0, y=0.0, z=10000.0, speed=200.0, heading_deg=90.0)
    ac2 = Aircraft("AC2", x=20000.0, y=0.0, z=10000.0, speed=200.0, heading_deg=270.0)

    t_cpa, d_h, d_v = compute_cpa(ac1, ac2)
    # Vitesse relative = 400 m/s -> t_cpa = 20000 / 400 = 50 s
    assert abs(t_cpa - 50.0) < 1e-2
    assert abs(d_h - 0.0) < 1e-2
    assert abs(d_v - 0.0) < 1e-2

def test_realistic_atc_env():
    """Vérifie le fonctionnement de l'environnement Gymnasium."""
    env = RealisticATCEnv(scenario_type="head_on")
    obs, info = env.reset(seed=123)

    assert obs.shape == (10,)
    assert env.action_space.n == 9

    # Exécution d'une action
    next_obs, r, term, trunc, info = env.step(1)
    assert next_obs.shape == (10,)
    assert isinstance(r, float)
    assert "dist_horiz_nm" in info

def test_d3qn_agent():
    """Vérifie le réseau de neurones et la prise de décision."""
    agent = D3QNAgent(state_dim=10, action_dim=9, device="cpu")
    dummy_state = np.zeros(10, dtype=np.float32)

    action = agent.select_action(dummy_state, evaluate=True)
    assert 0 <= action < 9

def test_api_endpoint():
    """Vérifie l'API FastAPI de prédiction d'advisory."""
    client = TestClient(app)
    health_resp = client.get("/health")
    assert health_resp.status_code == 200

    payload = {
        "ownship": {
            "callsign": "AFR123",
            "x_m": 0.0,
            "y_m": 0.0,
            "altitude_m": 10000.0,
            "speed_ms": 230.0,
            "heading_deg": 90.0,
            "vertical_rate_ms": 0.0
        },
        "intruder": {
            "callsign": "DLH456",
            "x_m": 25000.0,
            "y_m": 0.0,
            "altitude_m": 10000.0,
            "speed_ms": 230.0,
            "heading_deg": 270.0,
            "vertical_rate_ms": 0.0
        }
    }
    resp = client.post("/predict-advisory", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["conflict_detected"] is True
    assert "recommended_action" in data

