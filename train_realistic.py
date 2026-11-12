"""
Script d'entraînement et de benchmarking rigoureux pour l'agent ATC D3QN.
Compare :
1. Baseline "DO NOTHING" (maintien passif du cap)
2. Baseline "RANDOM" (décisions aléatoires)
3. Agent "D3QN ATC" (politique apprise)
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.simulation.atc_env import RealisticATCEnv
from src.agent.dqn import D3QNAgent

def evaluate(env, policy_fn, n_episodes: int = 100):
    rewards = []
    collisions = []
    loss_of_seps = []
    successes = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=1000 + ep)
        ep_reward = 0.0
        done = False
        while not done:
            action = policy_fn(obs)
            obs, r, term, trunc, info = env.step(action)
            ep_reward += r
            done = term or trunc

        rewards.append(ep_reward)
        collisions.append(float(info["collision"]))
        loss_of_seps.append(float(info["had_loss_of_separation"]))
        successes.append(float(info["success"]))

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "nmac_rate": float(np.mean(collisions) * 100.0),
        "loss_of_sep_rate": float(np.mean(loss_of_seps) * 100.0),
        "success_rate": float(np.mean(successes) * 100.0),
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
        epsilon_decay=0.996,
        buffer_capacity=50_000,
        batch_size=64,
        device=device
    )

    n_episodes = 600
    ep_rewards = []
    success_history = []
    sep_loss_history = []

    print(f"{'Épisode':>8} | {'Reward Moy':>11} | {'ε':>6} | {'Succès (50)':>11} | {'Perte Sép':>10}")
    print("-" * 55)

    for ep in range(1, n_episodes + 1):
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

    # --- ÉVALUATION COMPARATIVE RIGOUREUSE (100 Épisodes inédits) ---
    print("\n" + "=" * 60)
    print("      BENCHMARK DE PERFORMANCE — 100 SCÉNARIOS DE TEST")
    print("=" * 60)

    # 1. Do Nothing
    res_nothing = evaluate(env, lambda obs: 0, n_episodes=100)
    # 2. Random
    res_random = evaluate(env, lambda obs: env.action_space.sample(), n_episodes=100)
    # 3. D3QN Agent
    res_d3qn = evaluate(env, lambda obs: agent.select_action(obs, evaluate=True), n_episodes=100)

    print(f"{'Métrique OACI':<25} | {'Do Nothing':>10} | {'Aléatoire':>10} | {'D3QN ATC':>10}")
    print("-" * 62)
    print(f"{'Récompense Moyenne':<25} | {res_nothing['mean_reward']:>+10.1f} | {res_random['mean_reward']:>+10.1f} | {res_d3qn['mean_reward']:>+10.1f}")
    print(f"{'Pertes de Séparation (<5NM)':<25} | {res_nothing['loss_of_sep_rate']:>9.1f}% | {res_random['loss_of_sep_rate']:>9.1f}% | {res_d3qn['loss_of_sep_rate']:>9.1f}%")
    print(f"{'Collisions / NMAC':<25} | {res_nothing['nmac_rate']:>9.1f}% | {res_random['nmac_rate']:>9.1f}% | {res_d3qn['nmac_rate']:>9.1f}%")
    print(f"{'Succès (Conflit résolu)':<25} | {res_nothing['success_rate']:>9.1f}% | {res_random['success_rate']:>9.1f}% | {res_d3qn['success_rate']:>9.1f}%")
    print("=" * 62)

if __name__ == "__main__":
    main()

