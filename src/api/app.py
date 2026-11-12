"""
Service API FastAPI pour le déploiement opérationnel du système ATC D3QN.
Reçoit les données de vol en direct (ou via OpenSky), détecte les conflits et
émet des recommandations de manœuvres immédiates (Advisories).
"""

import os
import sys
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import (
    NM_TO_M,
    FT_TO_M,
    KTS_TO_MS,
    SEPARATION_HORIZ_M,
    SEPARATION_VERT_M
)
from src.simulation.aircraft import Aircraft
from src.simulation.encounters import compute_cpa
from src.agent.dqn import D3QNAgent
from src.simulation.atc_env import RealisticATCEnv

app = FastAPI(
    title="Air Traffic Control Collision Avoidance API (D3QN)",
    description="Microservice de détection et résolution automatique de conflits aériens en route (OACI).",
    version="1.0.0"
)

MODEL_PATH = "models/d3qn_atc_latest.pt"

# Chargement immédiat du modèle au chargement du module
agent = D3QNAgent(state_dim=10, action_dim=9, device="cpu")
if os.path.exists(MODEL_PATH):
    agent.load(MODEL_PATH)
    print(f"Modèle ATC D3QN chargé depuis {MODEL_PATH}")
else:
    print(f"Attention : Modèle non trouvé dans {MODEL_PATH}.")

class FlightState(BaseModel):
    callsign: str = Field(..., example="AFR123")
    x_m: float = Field(..., description="Position Est en mètres (repère local)", example=0.0)
    y_m: float = Field(..., description="Position Nord en mètres", example=0.0)
    altitude_m: float = Field(..., description="Altitude géométrique en mètres", example=10000.0)
    speed_ms: float = Field(..., description="Vitesse sol en m/s (~450 kts = 230 m/s)", example=230.0)
    heading_deg: float = Field(..., description="Cap en degrés (0-360)", example=90.0)
    vertical_rate_ms: float = Field(0.0, description="Vitesse verticale en m/s", example=0.0)

class ConflictQuery(BaseModel):
    ownship: FlightState
    intruder: FlightState

class AdvisoryResponse(BaseModel):
    conflict_detected: bool
    time_to_cpa_s: float
    cpa_horizontal_dist_nm: float
    cpa_vertical_dist_ft: float
    current_horizontal_dist_nm: float
    current_vertical_dist_ft: float
    recommended_action: str
    action_code: int
    rationale: str

ACTION_EXPLANATIONS = {
    0: "Maintenir cap, vitesse et niveau actuels (Aucun conflit immédiat)",
    1: "Effectuer un virage à droite de +15° pour créer un espacement latéral",
    2: "Effectuer un virage à gauche de -15° pour créer un espacement latéral",
    3: "Virage d'évitement franc à droite de +30°",
    4: "Virage d'évitement franc à gauche de -30°",
    5: "Monter immédiatement de 1 000 ft (Taux de montée +1500 ft/min)",
    6: "Descendre immédiatement de 1 000 ft (Taux de descente -1500 ft/min)",
    7: "Augmenter la vitesse sol de +20 kts pour franchir l'intersection en premier",
    8: "Réduire la vitesse sol de -20 kts pour céder le passage",
}

@app.get("/health")
def health_check():
    return {"status": "operational", "model_loaded": agent is not None}

@app.post("/predict-advisory", response_model=AdvisoryResponse)
def predict_advisory(query: ConflictQuery):
    if agent is None:
        raise HTTPException(status_code=500, detail="Modèle non disponible.")

    own = Aircraft(
        callsign=query.ownship.callsign,
        x=query.ownship.x_m,
        y=query.ownship.y_m,
        z=query.ownship.altitude_m,
        speed=query.ownship.speed_ms,
        heading_deg=query.ownship.heading_deg,
        vertical_rate=query.ownship.vertical_rate_ms
    )

    intru = Aircraft(
        callsign=query.intruder.callsign,
        x=query.intruder.x_m,
        y=query.intruder.y_m,
        z=query.intruder.altitude_m,
        speed=query.intruder.speed_ms,
        heading_deg=query.intruder.heading_deg,
        vertical_rate=query.intruder.vertical_rate_ms
    )

    # 1. Calcul géométrique du CPA
    t_cpa, d_cpa_h, d_cpa_v = compute_cpa(own, intru)
    curr_dx = intru.x - own.x
    curr_dy = intru.y - own.y
    curr_dz = intru.z - own.z
    curr_dist_h = float(np.hypot(curr_dx, curr_dy))
    curr_dist_v = float(abs(curr_dz))

    # Conflit si le CPA prévu est dans moins de 3 minutes ET sous les minima OACI (5 NM / 1000 ft)
    conflict = (t_cpa > 0) and (t_cpa < 180.0) and (d_cpa_h < SEPARATION_HORIZ_M) and (d_cpa_v < SEPARATION_VERT_M)

    # 2. Construction de l'observation pour l'agent
    dvx = intru.vx - own.vx
    dvy = intru.vy - own.vy
    dvz = intru.vz - own.vz

    obs = np.array([
        curr_dx / RealisticATCEnv.NORM_DIST_HORIZ,
        curr_dy / RealisticATCEnv.NORM_DIST_HORIZ,
        curr_dz / RealisticATCEnv.NORM_DIST_VERT,
        dvx / RealisticATCEnv.NORM_SPEED,
        dvy / RealisticATCEnv.NORM_SPEED,
        dvz / RealisticATCEnv.NORM_V_RATE,
        curr_dist_h / SEPARATION_HORIZ_M,
        curr_dist_v / SEPARATION_VERT_M,
        t_cpa / RealisticATCEnv.NORM_TIME,
        own.speed / RealisticATCEnv.NORM_SPEED
    ], dtype=np.float32)

    # 3. Décision du modèle
    if conflict:
        action = agent.select_action(obs, evaluate=True)
    else:
        action = 0  # MAINTAIN

    action_name = RealisticATCEnv.ACTION_NAMES[action]
    rationale = ACTION_EXPLANATIONS.get(action, "Ordre ATC généré.")

    return AdvisoryResponse(
        conflict_detected=conflict,
        time_to_cpa_s=round(t_cpa, 1),
        cpa_horizontal_dist_nm=round(d_cpa_h / NM_TO_M, 2),
        cpa_vertical_dist_ft=round(d_cpa_v / FT_TO_M, 0),
        current_horizontal_dist_nm=round(curr_dist_h / NM_TO_M, 2),
        current_vertical_dist_ft=round(curr_dist_v / FT_TO_M, 0),
        recommended_action=action_name,
        action_code=action,
        rationale=rationale
    )
