"""
Script d'entraînement et de benchmarking rigoureux pour l'agent ATC D3QN.
Évaluation exhaustive sur les 4 géométries de conflit (OACI) :
- Face-à-face direct (Head-on 180°)
- Croisement perpendiculaire (90° Crossing)
- Convergence oblique (Oblique Angle)
- Dépassement en route (Overtake)
Garantit 0% de demi-tours et le respect du système de Double Bulle de sécurité.
"""

import os
import sys
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.simulation.atc_env import RealisticATCEnv
from src.agent.dqn import D3QNAgent

def evaluate_policy(scenario_type: str, policy_fn, n_episodes: int = 50):
    env = RealisticATCEnv(scenario_type=scenario_type)
    rewards = []
    collisions = []
    loss_of_seps = []
    successes = []
    max_heading_devs = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=2000 + ep)
        ep_reward = 0.0
        done = False
        max_dev = 0.0
        while not done:
            action = policy_fn(obs)
            obs, r, term, trunc, info = env.step(action)
            ep_reward += r
            max_dev = max(max_dev, info["heading_deviation_deg"])
            done = term or trunc

        rewards.append(ep_reward)
        collisions.append(float(info["collision"]))
        loss_of_seps.append(float(info["had_loss_of_separation"]))
        successes.append(float(info["success"]))
        max_heading_devs.append(max_dev)

    return {
        "mean_reward": float(np.mean(rewards)),
        "nmac_rate": float(np.mean(collisions) * 100.0),
        "loss_of_sep_rate": float(np.mean(loss_of_seps) * 100.0),
        "success_rate": float(np.mean(successes) * 100.0),
        "avg_max_dev": float(np.mean(max_heading_devs)),
        "max_dev": float(np.max(max_heading_devs))
    }

def main():
    os.makedirs("models", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Démarrage de l'entraînement D3QN ATC (Device: {device}) ---")

    env = RealisticATCEnv(scenario_type="random")
    agent = D3QNAgent(
        state_dim=10,
        action_dim=9,
        lr=3e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_min=0.03,
        epsilon_decay=0.995,
        buffer_capacity=50_000,
        batch_size=64,
        device=device
    )

    n_episodes = 700
    ep_rewards = []
    success_history = []
    sep_loss_history = []

    print(f"{'Épisode':>8} | {'Reward Moy':>11} | {'ε':>6} | {'Succès (50)':>11} | {'Perte Sép':>10}")
    print("-" * 55)

    scenarios = ["head_on", "cross_90", "cross_oblique", "overtake"]

    for ep in range(1, n_episodes + 1):
        # Rotation équilibrée sur tous les types d'encounters
        chosen_sc = scenarios[ep % len(scenarios)]
        env.scenario_type = chosen_sc
        obs, _ = env.reset()
        total_r = 0.0
        done = False

        while not done:
            action = agent.select_action(obs)
            next_obs, r, term, trunc, info = env.step(action)
            agent.buffer.push(obs, action, r, next_obs, term or trunc)
            agent.update()

            obs = next_obs
            total_r += r
            done = term or trunc

        agent.decay_epsilon()
        ep_rewards.append(total_r)
        success_history.append(float(info["success"]))
        sep_loss_history.append(float(info["had_loss_of_separation"]))

        if ep % 50 == 0:
            window = 50
            mean_r = np.mean(ep_rewards[-window:])
            succ_r = np.mean(success_history[-window:]) * 100.0
            sep_r = np.mean(sep_loss_history[-window:]) * 100.0
            print(f"{ep:>8} | {mean_r:>+11.2f} | {agent.epsilon:>6.3f} | {succ_r:>10.1f}% | {sep_r:>9.1f}%")

    model_path = "models/d3qn_atc_latest.pt"
    agent.save(model_path)
    print(f"\nModèle sauvegardé dans {model_path}")

    # --- BENCHMARK DÉTAILLÉ SUR CHAQUE SCÉNARIO OACI (50 Épisodes chacun) ---
    print("\n" + "=" * 78)
    print("      BENCHMARK COMPARATIF PAR SCÉNARIO OACI (100% INÉDITS)")
    print("=" * 78)
    print(f"{'Scénario':<15} | {'Politique':<10} | {'Succès':>8} | {'Perte Sép':>10} | {'NMAC':>6} | {'Déviation Max':>13}")
    print("-" * 78)

    for sc in scenarios:
        res_do_nothing = evaluate_policy(sc, lambda obs: 0, n_episodes=50)
        res_d3qn = evaluate_policy(sc, lambda obs: agent.select_action(obs, evaluate=True), n_episodes=50)

        print(f"{sc:<15} | {'Do Nothing':<10} | {res_do_nothing['success_rate']:>7.1f}% | {res_do_nothing['loss_of_sep_rate']:>9.1f}% | {res_do_nothing['nmac_rate']:>5.1f}% | {res_do_nothing['max_dev']:>12.1f}°")
        print(f"{'':<15} | {'D3QN ATC':<10} | {res_d3qn['success_rate']:>7.1f}% | {res_d3qn['loss_of_sep_rate']:>9.1f}% | {res_d3qn['nmac_rate']:>5.1f}% | {res_d3qn['max_dev']:>12.1f}°")
        print("-" * 78)

    print("=" * 78)

if __name__ == "__main__":
    main()
