"""
Environnement Gymnasium d'Air Traffic Control (ATC) en unités physiques réelles.
Conforme aux normes de séparation OACI (5 NM horizontal, 1 000 ft vertical).
Optimisation métier :
- Priorité à l'étagement vertical (économie de carburant et respect des horaires)
- Pénalité de déviation de cap (interdiction des demi-tours aberrants)
- Pénalité de temps perdu
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.config import (
    NM_TO_M,
    FT_TO_M,
    KTS_TO_MS,
    SEPARATION_HORIZ_M,
    SEPARATION_VERT_M,
    NMAC_HORIZ_M,
    NMAC_VERT_M,
    SAFE_HORIZ_M,
    SAFE_VERT_M,
    SIM_DT_S,
    MAX_EPISODE_STEPS
)
from src.simulation.aircraft import Aircraft
from src.simulation.encounters import generate_encounter, compute_cpa

class RealisticATCEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    NORM_DIST_HORIZ = 10.0 * NM_TO_M
    NORM_DIST_VERT = 2000.0 * FT_TO_M
    NORM_SPEED = 500.0
    NORM_V_RATE = 20.0
    NORM_TIME = 120.0

    ACTION_NAMES = [
        "MAINTAIN",
        "TURN_RIGHT_15",
        "TURN_LEFT_15",
        "TURN_RIGHT_30",
        "TURN_LEFT_30",
        "CLIMB_1000FT",
        "DESCEND_1000FT",
        "SPEED_UP_20KTS",
        "SPEED_DOWN_20KTS"
    ]

    def __init__(self, scenario_type: str = "random", dt: float = SIM_DT_S, max_steps: int = MAX_EPISODE_STEPS):
        super(RealisticATCEnv, self).__init__()
        
        self.scenario_type = scenario_type
        self.dt = dt
        self.max_steps = max_steps
        
        self.action_space = spaces.Discrete(9)
        
        # 10 dimensions d'observation
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(10,), dtype=np.float32
        )
        
        self.ownship: Aircraft = None
        self.intruder: Aircraft = None
        self.current_step = 0
        self.nominal_heading_deg = 0.0
        self.nominal_altitude_m = 0.0
        self.prev_horiz_dist = 0.0
        self.prev_vert_dist = 0.0
        self.min_horiz_observed = float("inf")
        self.min_vert_observed = float("inf")
        self.had_loss_of_separation = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)

        self.ownship, self.intruder = generate_encounter(self.scenario_type, rng=rng)
        
        self.current_step = 0
        self.had_loss_of_separation = False
        
        # Mémorisation du plan de vol nominal (destination)
        self.nominal_heading_deg = float(self.ownship.heading_deg)
        self.nominal_altitude_m = float(self.ownship.z)
        
        dx = self.intruder.x - self.ownship.x
        dy = self.intruder.y - self.ownship.y
        dz = self.intruder.z - self.ownship.z
        self.prev_horiz_dist = float(np.hypot(dx, dy))
        self.prev_vert_dist = float(abs(dz))
        self.min_horiz_observed = self.prev_horiz_dist
        self.min_vert_observed = self.prev_vert_dist

        obs = self._get_observation()
        info = {
            "ownship": self.ownship.to_dict(),
            "intruder": self.intruder.to_dict()
        }
        return obs, info

    def _get_observation(self) -> np.ndarray:
        dx = self.intruder.x - self.ownship.x
        dy = self.intruder.y - self.ownship.y
        dz = self.intruder.z - self.ownship.z
        
        dvx = self.intruder.vx - self.ownship.vx
        dvy = self.intruder.vy - self.ownship.vy
        dvz = self.intruder.vz - self.ownship.vz

        dist_h = float(np.hypot(dx, dy))
        dist_v = float(abs(dz))
        t_cpa, _, _ = compute_cpa(self.ownship, self.intruder)

        obs = np.array([
            dx / self.NORM_DIST_HORIZ,
            dy / self.NORM_DIST_HORIZ,
            dz / self.NORM_DIST_VERT,
            dvx / self.NORM_SPEED,
            dvy / self.NORM_SPEED,
            dvz / self.NORM_V_RATE,
            dist_h / SEPARATION_HORIZ_M,
            dist_v / SEPARATION_VERT_M,
            t_cpa / self.NORM_TIME,
            self.ownship.speed / self.NORM_SPEED
        ], dtype=np.float32)

        return np.clip(obs, -10.0, 10.0)

    def step(self, action: int):
        # 1. Application des ordres ATC
        if action == 1:    # Turn Right 15°
            self.ownship.set_heading_target(self.ownship.heading_deg + 15.0)
        elif action == 2:  # Turn Left 15°
            self.ownship.set_heading_target(self.ownship.heading_deg - 15.0)
        elif action == 3:  # Turn Right 30°
            self.ownship.set_heading_target(self.ownship.heading_deg + 30.0)
        elif action == 4:  # Turn Left 30°
            self.ownship.set_heading_target(self.ownship.heading_deg - 30.0)
        elif action == 5:  # Climb +1000 ft (Vertical separation)
            self.ownship.change_altitude_target(+1000.0 * FT_TO_M)
        elif action == 6:  # Descend -1000 ft (Vertical separation)
            self.ownship.change_altitude_target(-1000.0 * FT_TO_M)
        elif action == 7:  # Speed Up +20 kts
            self.ownship.change_speed_target(+20.0 * KTS_TO_MS)
        elif action == 8:  # Speed Down -20 kts
            self.ownship.change_speed_target(-20.0 * KTS_TO_MS)

        # 2. Intégration cinématique
        self.ownship.step(self.dt)
        self.intruder.step(self.dt)

        # 3. Calculs géométriques
        dx = self.intruder.x - self.ownship.x
        dy = self.intruder.y - self.ownship.y
        dz = self.intruder.z - self.ownship.z
        curr_dist_h = float(np.hypot(dx, dy))
        curr_dist_v = float(abs(dz))

        self.min_horiz_observed = min(self.min_horiz_observed, curr_dist_h)
        self.min_vert_observed = min(self.min_vert_observed, curr_dist_v)

        t_cpa, _, _ = compute_cpa(self.ownship, self.intruder)

        # Événements OACI
        is_nmac = (curr_dist_h < NMAC_HORIZ_M) and (curr_dist_v < NMAC_VERT_M)
        is_loss_of_sep = (curr_dist_h < SEPARATION_HORIZ_M) and (curr_dist_v < SEPARATION_VERT_M)
        if is_loss_of_sep:
            self.had_loss_of_separation = True

        terminated = False
        collision = False
        success = False
        reward = 0.0

        # === 1. PÉNALITÉS DE SÉCURITÉ ABSOLUE ===
        if is_nmac:
            reward = -300.0
            terminated = True
            collision = True
        elif is_loss_of_sep:
            penetration_h = (SEPARATION_HORIZ_M - curr_dist_h) / SEPARATION_HORIZ_M
            penetration_v = (SEPARATION_VERT_M - curr_dist_v) / SEPARATION_VERT_M
            reward = -15.0 * (penetration_h + penetration_v) - 3.0
        else:
            # Condition de succès propre
            # Conflit désamorcé : CPA passé ET (distance > 7 NM OU séparation verticale > 1000 ft)
            is_cleared = (t_cpa <= 0.0) and (curr_dist_h >= SAFE_HORIZ_M or curr_dist_v >= SEPARATION_VERT_M)
            if is_cleared:
                if not self.had_loss_of_separation:
                    reward = +150.0
                    # BONUS SPÉCIAL : Si l'agent a résolu par l'altitude ou léger virage sans faire demi-tour
                    heading_err = abs((self.ownship.heading_deg - self.nominal_heading_deg + 180.0) % 360.0 - 180.0)
                    if heading_err < 35.0:
                        reward += 50.0  # Bonus vol direct / économie carburant
                    success = True
                else:
                    reward = -30.0
                terminated = True

        # === 2. PÉNALITÉS MÉTIER (CARBURANT & PLAN DE VOL) ===
        # A. Pénalité d'écart au cap nominal vers la destination (empêche les demi-tours absurdes)
        heading_deviation = abs((self.ownship.heading_deg - self.nominal_heading_deg + 180.0) % 360.0 - 180.0)
        # Faire demi-tour (180°) coûte -1.8 par seconde ! Un léger virage de 15° ne coûte que -0.15.
        fuel_delay_penalty = - (heading_deviation / 180.0) * 1.8
        reward += fuel_delay_penalty

        # B. Coût d'action
        if action != 0:
            reward -= 0.1

        self.prev_horiz_dist = curr_dist_h
        self.prev_vert_dist = curr_dist_v
        self.current_step += 1
        truncated = self.current_step >= self.max_steps

        obs = self._get_observation()
        info = {
            "collision": collision,
            "success": success,
            "had_loss_of_separation": self.had_loss_of_separation,
            "dist_horiz_nm": curr_dist_h / NM_TO_M,
            "dist_vert_ft": curr_dist_v / FT_TO_M,
            "t_cpa_s": t_cpa,
            "min_horiz_nm": self.min_horiz_observed / NM_TO_M,
            "min_vert_ft": self.min_vert_observed / FT_TO_M,
            "heading_deviation_deg": heading_deviation,
            "action_name": self.ACTION_NAMES[action]
        }

        return obs, reward, terminated, truncated, info
