"""
Script de démonstration interactif en ligne de commande (CLI).
Permet de tester la résolution de conflit en mode console sans navigateur.
"""

import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import NM_TO_M, FT_TO_M, MS_TO_KTS
from src.simulation.atc_env import RealisticATCEnv
from src.agent.dqn import D3QNAgent

def run_cli_demo(scenario: str = "head_on"):
    print("=" * 75)
    print(f"   DÉMO RÉSOLUTION DE CONFLIT ATC — Scénario : {scenario.upper()}")
    print("=" * 75)

    env = RealisticATCEnv(scenario_type=scenario)
    agent = D3QNAgent(state_dim=10, action_dim=9, device="cpu")
    model_path = "models/d3qn_atc_latest.pt"
    
    if os.path.exists(model_path):
        agent.load(model_path)
        print(f"✅ Modèle chargé : {model_path}\n")
    else:
        print("⚠️ Modèle non trouvé, exécution en mode aléatoire.\n")

    obs, info = env.reset(seed=42)
    dx_init = env.intruder.x - env.ownship.x
    dy_init = env.intruder.y - env.ownship.y
    dist_init_nm = float(np.hypot(dx_init, dy_init)) / NM_TO_M
    alt_diff_init_ft = abs(env.ownship.z - env.intruder.z) * 3.28084

    print(f"✈️  OWNSHIP  : Cap {env.ownship.heading_deg:.0f}° | Alt {env.ownship.z*3.28084:.0f} ft | Vitesse {env.ownship.speed*MS_TO_KTS:.0f} kts")
    print(f"✈️  INTRUDER : Cap {env.intruder.heading_deg:.0f}° | Alt {env.intruder.z*3.28084:.0f} ft | Vitesse {env.intruder.speed*MS_TO_KTS:.0f} kts")
    print(f"📍 Distance initiale : {dist_init_nm:.2f} NM | Écart vertical : {alt_diff_init_ft:.0f} ft\n")

    print(f"{'Temps':>6} | {'Dist (NM)':>10} | {'Alt Diff (ft)':>14} | {'Ordre ATC IA':<18} | {'Statut Séparation'}")
    print("-" * 75)

    done = False
    step = 0
    while not done and step < 60:
        step += 1
        t_sec = step * 2
        action = agent.select_action(obs, evaluate=True)
        action_name = RealisticATCEnv.ACTION_NAMES[action]

        obs, r, term, trunc, info = env.step(action)
        
        # Séparation 3D : conforme si dist >= 5 NM OU alt_diff >= 1000 ft
        is_safe = (info["dist_horiz_nm"] >= 5.0) or (info["dist_vert_ft"] >= 1000.0)
        status = "🟢 CONFORME OACI" if is_safe else "🔴 VIOLATION (<5NM et <1000ft)"

        print(f"{t_sec:>4}s | {info['dist_horiz_nm']:>9.2f} NM | {info['dist_vert_ft']:>11.0f} ft | {action_name:<18} | {status}")
        done = term or trunc

    print("-" * 75)
    if info["had_loss_of_separation"]:
        print("❌ Résultat : Perte de séparation observée pendant le croisement.")
    else:
        print("✅ Résultat : Conflit résolu avec succès ! Séparation OACI 3D préservée à tout moment.")
    print(f"📍 Distance minimale atteinte : {info['min_horiz_nm']:.2f} NM | Écart vertical final : {info['min_vert_ft']:.0f} ft\n")

if __name__ == "__main__":
    choice = sys.argv[1] if len(sys.argv) > 1 else "head_on"
    run_cli_demo(choice)
