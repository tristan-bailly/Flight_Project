"""
Module d'ingestion en temps réel du flux ADS-B OpenSky Network.
Détecte automatiquement les paires d'avions en conflit prédictif dans le ciel réel
et interroge l'agent D3QN pour émettre des résolutions en direct.
"""

import json
import os
import sys
import numpy as np
import pandas as pd
from typing import List, Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import (
    NM_TO_M,
    FT_TO_M,
    SEPARATION_HORIZ_M,
    SEPARATION_VERT_M
)
from src.simulation.aircraft import Aircraft
from src.simulation.encounters import compute_cpa
from src.agent.dqn import D3QNAgent
from src.simulation.atc_env import RealisticATCEnv

def load_opensky_credentials(creds_path: str = "credentials.json"):
    if os.path.exists(creds_path):
        with open(creds_path, "r") as f:
            creds = json.load(f)
            os.environ["OPENSKY_CLIENT_ID"] = creds.get("clientId", "")
            os.environ["OPENSKY_CLIENT_SECRET"] = creds.get("clientSecret", "")
        return True
    return False

def scan_live_conflicts(bbox_europe: bool = True, max_pairs: int = 10) -> List[Dict[str, Any]]:
    """
    Récupère le trafic live via OpenSky, projette les trajectoires et détecte les conflits imminents.
    """
    try:
        from pyopensky.rest import REST
    except ImportError:
        print("pyopensky n'est pas installé.")
        return []

    load_opensky_credentials()
    rest = REST()

    # Récupération du snapshot live
    print("Interrogation de l'API OpenSky Network...")
    df = rest.states()
    if df is None or len(df) == 0:
        print("Aucun vol reçu de l'API.")
        return []

    # Filtrage des vols en l'air et avec coordonnées valides
    df = df[df["onground"] == False].dropna(subset=["latitude", "longitude", "altitude", "groundspeed", "track"])
    
    if bbox_europe:
        # Zone Europe
        df = df[
            (df["latitude"] >= 35.0) & (df["latitude"] <= 65.0) &
            (df["longitude"] >= -10.0) & (df["longitude"] <= 30.0)
        ]

    print(f"Vols en vol filtrés en Europe : {len(df)}")
    if len(df) < 2:
        return []

    # Projection locale plane (tangent plane approché)
    mean_lat = df["latitude"].mean()
    m_per_deg_lat = 111139.0
    m_per_deg_lon = 111139.0 * np.cos(np.radians(mean_lat))

    df["x_m"] = df["longitude"] * m_per_deg_lon
    df["y_m"] = df["latitude"] * m_per_deg_lat
    df["z_m"] = df["altitude"]

    # Chargement de l'agent D3QN
    agent = D3QNAgent(state_dim=10, action_dim=9, device="cpu")
    model_path = "models/d3qn_atc_latest.pt"
    if os.path.exists(model_path):
        agent.load(model_path)

    # Détection par KDTree spatial
    from scipy.spatial import KDTree
    coords_2d = df[["x_m", "y_m"]].values
    tree = KDTree(coords_2d)

    # Recherche des voisins à moins de 25 NM (~46 km)
    radius_search_m = 25.0 * NM_TO_M
    pairs = tree.query_pairs(radius_search_m)
    print(f"Paires d'aéronefs à proximité (< 25 NM) : {len(pairs)}")

    conflicts = []
    df_list = df.to_dict("records")

    for idx1, idx2 in pairs:
        flight1 = df_list[idx1]
        flight2 = df_list[idx2]

        v_rate1 = flight1.get("vertical_rate", 0.0) or 0.0
        v_rate2 = flight2.get("vertical_rate", 0.0) or 0.0

        ac1 = Aircraft(
            callsign=flight1.get("callsign", flight1.get("icao24", "AC1")),
            x=flight1["x_m"], y=flight1["y_m"], z=flight1["z_m"],
            speed=flight1["groundspeed"], heading_deg=flight1["track"],
            vertical_rate=v_rate1
        )
        ac2 = Aircraft(
            callsign=flight2.get("callsign", flight2.get("icao24", "AC2")),
            x=flight2["x_m"], y=flight2["y_m"], z=flight2["z_m"],
            speed=flight2["groundspeed"], heading_deg=flight2["track"],
            vertical_rate=v_rate2
        )

        t_cpa, d_cpa_h, d_cpa_v = compute_cpa(ac1, ac2)

        # Conflit si CPA dans moins de 3 minutes ET séparation violée au CPA
        if (0.0 < t_cpa <= 180.0) and (d_cpa_h < SEPARATION_HORIZ_M) and (d_cpa_v < SEPARATION_VERT_M):
            # Construction de l'observation pour l'agent D3QN
            dx = ac2.x - ac1.x
            dy = ac2.y - ac1.y
            dz = ac2.z - ac1.z
            dvx = ac2.vx - ac1.vx
            dvy = ac2.vy - ac1.vy
            dvz = ac2.vz - ac1.vz
            curr_dist_h = float(np.hypot(dx, dy))
            curr_dist_v = float(abs(dz))

            obs = np.array([
                dx / RealisticATCEnv.NORM_DIST_HORIZ,
                dy / RealisticATCEnv.NORM_DIST_HORIZ,
                dz / RealisticATCEnv.NORM_DIST_VERT,
                dvx / RealisticATCEnv.NORM_SPEED,
                dvy / RealisticATCEnv.NORM_SPEED,
                dvz / RealisticATCEnv.NORM_V_RATE,
                curr_dist_h / SEPARATION_HORIZ_M,
                curr_dist_v / SEPARATION_VERT_M,
                t_cpa / RealisticATCEnv.NORM_TIME,
                ac1.speed / RealisticATCEnv.NORM_SPEED
            ], dtype=np.float32)

            action_idx = agent.select_action(obs, evaluate=True)
            action_name = RealisticATCEnv.ACTION_NAMES[action_idx]

            conflicts.append({
                "flight_1": ac1.callsign.strip(),
                "flight_2": ac2.callsign.strip(),
                "current_dist_nm": round(curr_dist_h / NM_TO_M, 2),
                "current_alt_diff_ft": round(curr_dist_v / FT_TO_M, 0),
                "time_to_cpa_s": round(t_cpa, 0),
                "cpa_dist_nm": round(d_cpa_h / NM_TO_M, 2),
                "cpa_alt_diff_ft": round(d_cpa_v / FT_TO_M, 0),
                "advisory_for_ac1": action_name
            })

            if len(conflicts) >= max_pairs:
                break

    return conflicts

if __name__ == "__main__":
    print("--- Scan des conflits en direct via OpenSky ---")
    detected = scan_live_conflicts()
    if detected:
        print(f"\n🚨 {len(detected)} Conflit(s) détecté(s) dans le ciel en direct :")
        for c in detected:
            print(f"✈️ {c['flight_1']} vs {c['flight_2']} | CPA dans {c['time_to_cpa_s']}s à {c['cpa_dist_nm']} NM | Ordre IA : {c['advisory_for_ac1']}")
    else:
        print("Aucun conflit imminent détecté dans le secteur analysé.")

