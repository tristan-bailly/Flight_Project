"""
Environnement Gymnasium d'Air Traffic Control (ATC) en unités physiques réelles.
Conforme aux normes de séparation OACI et intégrant le système de Double Bulle :
1. Bulle Légale Infranchissable (5 NM / 1 000 ft)
2. Bulle d'Anticipation et de Précaution Métier (8 NM / 1 500 ft)
Protection stricte d'enveloppe de vol interdisant les demi-tours (déviation max ≤ 45°).
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
    PRECAUTION_HORIZ_M,
    PRECAUTION_VERT_M,
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

    NORM_DIST_HORIZ = 15.0 * NM_TO_M
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
        self.nominal_speed_ms = 0.0
        self.prev_horiz_dist = 0.0
        self.prev_vert_dist = 0.0
        self.min_horiz_observed = float("inf")
        self.min_vert_observed = float("inf")
        self.had_loss_of_separation = False
        self.entered_caution_zone = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)

        self.ownship, self.intruder = generate_encounter(self.scenario_type, rng=rng)
        
        self.current_step = 0
        self.had_loss_of_separation = False
        self.entered_caution_zone = False
        
        self.nominal_heading_deg = float(self.ownship.heading_deg)
        self.nominal_altitude_m = float(self.ownship.z)
        self.nominal_speed_ms = float(self.ownship.speed)
        
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
            "intruder": self.intruder.to_dict(),
            "dist_horiz_nm": self.prev_horiz_dist / NM_TO_M,
            "dist_vert_ft": self.prev_vert_dist / FT_TO_M,
            "heading_deviation_deg": 0.0,
            "collision": False,
            "success": False,
            "had_loss_of_separation": False,
            "entered_caution_zone": False,
            "caution_active": (self.prev_horiz_dist < PRECAUTION_HORIZ_M and self.prev_vert_dist < PRECAUTION_VERT_M)
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

        # Déviation relative de cap par rapport au plan de vol nominal (normalisée sur 45°)
        heading_dev = (self.ownship.heading_deg - self.nominal_heading_deg + 180.0) % 360.0 - 180.0
        norm_heading_dev = heading_dev / 45.0

        obs = np.array([
            dx / self.NORM_DIST_HORIZ,
            dy / self.NORM_DIST_HORIZ,
            dz / self.NORM_DIST_VERT,
            dvx / self.NORM_SPEED,
            dvy / self.NORM_SPEED,
            dvz / self.NORM_V_RATE,
            dist_h / SEPARATION_HORIZ_M,     # Ratio par rapport aux 5 NM (<1.0 = infraction)
            dist_v / SEPARATION_VERT_M,      # Ratio par rapport aux 1000 ft
            t_cpa / self.NORM_TIME,
            norm_heading_dev                 # Écart relatif de cap (borné [-1, 1])
        ], dtype=np.float32)

        return np.clip(obs, -10.0, 10.0)

    def step(self, action: int):
        # 1. Application des ordres ATC tactiques réalistes (bornés par rapport au plan de vol)
        if action == 0:    # MAINTAIN / RESUME NOMINAL
            self.ownship.set_heading_target(self.nominal_heading_deg, self.nominal_heading_deg)
            self.ownship.set_altitude_target(self.nominal_altitude_m)
            self.ownship.set_speed_target(self.nominal_speed_ms)
        elif action == 1:  # Turn Right 15°
            self.ownship.set_heading_target(self.nominal_heading_deg + 15.0, self.nominal_heading_deg)
        elif action == 2:  # Turn Left 15°
            self.ownship.set_heading_target(self.nominal_heading_deg - 15.0, self.nominal_heading_deg)
        elif action == 3:  # Turn Right 35° (vecteur d'évitement franc)
            self.ownship.set_heading_target(self.nominal_heading_deg + 35.0, self.nominal_heading_deg)
        elif action == 4:  # Turn Left 35° (vecteur d'évitement franc)
            self.ownship.set_heading_target(self.nominal_heading_deg - 35.0, self.nominal_heading_deg)
        elif action == 5:  # Climb +1050 ft (séparation verticale garantie > 1 000 ft)
            self.ownship.set_altitude_target(self.nominal_altitude_m + 1050.0 * FT_TO_M)
        elif action == 6:  # Descend -1050 ft (séparation verticale garantie > 1 000 ft)
            self.ownship.set_altitude_target(self.nominal_altitude_m - 1050.0 * FT_TO_M)
        elif action == 7:  # Speed Up +20 kts
            self.ownship.set_speed_target(self.nominal_speed_ms + 20.0 * KTS_TO_MS)
        elif action == 8:  # Speed Down -20 kts
            self.ownship.set_speed_target(self.nominal_speed_ms - 20.0 * KTS_TO_MS)

        # 2. Intégration cinématique réaliste
        self.ownship.step(self.dt)
        self.intruder.step(self.dt)

        # 3. Calculs géométriques et métriques OACI
        dx = self.intruder.x - self.ownship.x
        dy = self.intruder.y - self.ownship.y
        dz = self.intruder.z - self.ownship.z
        curr_dist_h = float(np.hypot(dx, dy))
        curr_dist_v = float(abs(dz))

        self.min_horiz_observed = min(self.min_horiz_observed, curr_dist_h)
        self.min_vert_observed = min(self.min_vert_observed, curr_dist_v)

        t_cpa, _, _ = compute_cpa(self.ownship, self.intruder)

        # Événements de sécurité OACI (Double Bulle)
        # Note : marge epsilon de 1.0 m pour éviter les faux positifs d'arrondi à 1000 ft
        is_nmac = (curr_dist_h < NMAC_HORIZ_M) and (curr_dist_v < NMAC_VERT_M)
        is_loss_of_sep = (curr_dist_h < SEPARATION_HORIZ_M) and (curr_dist_v < (SEPARATION_VERT_M - 1.0))
        is_caution = (curr_dist_h < PRECAUTION_HORIZ_M) and (curr_dist_v < (PRECAUTION_VERT_M - 1.0))

        if is_loss_of_sep:
            self.had_loss_of_separation = True
        if is_caution:
            self.entered_caution_zone = True

        terminated = False
        collision = False
        success = False
        reward = 0.0

        # === 1. PÉNALITÉS DE SÉCURITÉ DOUBLE BULLE ===
        if is_nmac:
            reward = -350.0
            terminated = True
            collision = True

        elif is_loss_of_sep:
            # Pénétration dans la bulle rouge des 5 NM : faute lourde
            penetration_h = (SEPARATION_HORIZ_M - curr_dist_h) / SEPARATION_HORIZ_M
            penetration_v = max(0.0, (SEPARATION_VERT_M - curr_dist_v) / SEPARATION_VERT_M)
            reward = -25.0 * (penetration_h + penetration_v) - 5.0

        elif is_caution:
            # Pénétration dans la bulle d'anticipation (entre 5 NM et 8 NM)
            # Incitation progressive à manœuvrer tôt pour ne jamais toucher les 5 NM
            penetration_buffer = (PRECAUTION_HORIZ_M - curr_dist_h) / (PRECAUTION_HORIZ_M - SEPARATION_HORIZ_M)
            reward = -2.5 * penetration_buffer

        else:
            # Hors de la bulle d'anticipation (> 8 NM) ou CPA passé
            is_cleared = (t_cpa <= 0.0) and (curr_dist_h >= PRECAUTION_HORIZ_M or curr_dist_v >= (SEPARATION_VERT_M - 1.0))
            if is_cleared:
                if not self.had_loss_of_separation:
                    reward = +150.0
                    # Bonus Marge de précaution préservée (l'avion n'a jamais franchi 6.0 NM)
                    if self.min_horiz_observed >= 6.0 * NM_TO_M or self.min_vert_observed >= 1000.0 * FT_TO_M:
                        reward += 40.0

                    heading_err = abs((self.ownship.heading_deg - self.nominal_heading_deg + 180.0) % 360.0 - 180.0)
                    if heading_err < 15.0:
                        reward += 30.0  # Bonus reprise de cap direct
                    success = True
                else:
                    reward = -30.0
                terminated = True

        # === 2. PÉNALITÉ DE DÉVIATION DE ROUTE & COÛT DE MANŒUVRE ===
        heading_deviation = abs((self.ownship.heading_deg - self.nominal_heading_deg + 180.0) % 360.0 - 180.0)
        fuel_delay_penalty = - (heading_deviation / 45.0) * 0.4
        reward += fuel_delay_penalty

        if action != 0:
            reward -= 0.05

        self.prev_horiz_dist = curr_dist_h
        self.prev_vert_dist = curr_dist_v
        self.current_step += 1
        truncated = self.current_step >= self.max_steps

        obs = self._get_observation()
        info = {
            "collision": collision,
            "success": success,
            "had_loss_of_separation": self.had_loss_of_separation,
            "entered_caution_zone": self.entered_caution_zone,
            "caution_active": is_caution,
            "dist_horiz_nm": curr_dist_h / NM_TO_M,
            "dist_vert_ft": curr_dist_v / FT_TO_M,
            "t_cpa_s": t_cpa,
            "min_horiz_nm": self.min_horiz_observed / NM_TO_M,
            "min_vert_ft": self.min_vert_observed / FT_TO_M,
            "heading_deviation_deg": heading_deviation,
            "action_name": self.ACTION_NAMES[action]
        }

        return obs, reward, terminated, truncated, info
