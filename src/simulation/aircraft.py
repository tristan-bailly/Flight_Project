"""
Modèle cinématique 3D d'un aéronef avec gestion complète des 3 leviers ATC :
1. Cap (Heading)
2. Altitude / Taux vertical (Climb / Descend)
3. Vitesse sol (Speed control : accélération / décélération)
Comprend une protection d'enveloppe de vol (Flight Envelope Protection) interdisant
les déviations extrêmes de cap (> 45°) afin d'exclure tout demi-tour irréaliste.
"""

import numpy as np
from src.config import (
    MAX_TURN_RATE_DEG_S,
    STANDARD_CLIMB_RATE_MS,
    STANDARD_DESC_RATE_MS,
    CRUISE_SPEED_MS,
    CRUISE_SPEED_MIN_MS,
    CRUISE_SPEED_MAX_MS,
    FT_TO_M,
    KTS_TO_MS
)

class Aircraft:
    def __init__(
        self,
        callsign: str,
        x: float,
        y: float,
        z: float,
        speed: float = CRUISE_SPEED_MS,
        heading_deg: float = 0.0,
        vertical_rate: float = 0.0,
    ):
        self.callsign = callsign
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.speed = float(speed)
        self.heading_deg = float(heading_deg % 360.0)
        self.vertical_rate = float(vertical_rate)
        
        # Cibles de commande de l'autopilote
        self.target_heading_deg = self.heading_deg
        self.target_speed = self.speed
        self.target_altitude = self.z
        self.target_vertical_rate = self.vertical_rate

    @property
    def vx(self) -> float:
        rad = np.radians(self.heading_deg)
        return self.speed * np.sin(rad)

    @property
    def vy(self) -> float:
        rad = np.radians(self.heading_deg)
        return self.speed * np.cos(rad)

    @property
    def vz(self) -> float:
        return self.vertical_rate

    # --- Commandes ATC & Autopilote ---
    def set_heading_target(self, target_deg: float, nominal_deg: float = None, max_deviation_deg: float = 45.0):
        """
        Virage vers un nouveau cap (degrés).
        Si nominal_deg est fourni, applique une protection stricte d'enveloppe de vol (max_deviation_deg).
        """
        target = float(target_deg % 360.0)
        if nominal_deg is not None:
            dev = (target - nominal_deg + 180.0) % 360.0 - 180.0
            dev_clamped = np.clip(dev, -max_deviation_deg, max_deviation_deg)
            target = float((nominal_deg + dev_clamped) % 360.0)
        self.target_heading_deg = target

    def set_altitude_target(self, target_alt_m: float):
        """Consigne absolue d'altitude cible (mètres)."""
        self.target_altitude = float(target_alt_m)
        if self.target_altitude > self.z + 1.0:
            self.target_vertical_rate = STANDARD_CLIMB_RATE_MS
        elif self.target_altitude < self.z - 1.0:
            self.target_vertical_rate = STANDARD_DESC_RATE_MS
        else:
            self.target_vertical_rate = 0.0

    def set_speed_target(self, target_speed_ms: float):
        """Consigne absolue de vitesse cible (m/s)."""
        new_speed = np.clip(target_speed_ms, CRUISE_SPEED_MIN_MS, CRUISE_SPEED_MAX_MS)
        self.target_speed = float(new_speed)

    def change_altitude_target(self, delta_alt_m: float):
        """Ordre de changement d'altitude (+/- delta en mètres, ex: +/- 1000 ft)."""
        self.set_altitude_target(self.target_altitude + float(delta_alt_m))

    def change_speed_target(self, delta_speed_ms: float):
        """Ordre de consigne de vitesse (+/- delta en m/s, ex: +/- 20 kts)."""
        self.set_speed_target(self.speed + delta_speed_ms)

    def step(self, dt: float):
        """
        Intégration cinématique sur un intervalle dt (secondes).
        Limites physiques réalistes : virage 3°/s, accélération longitudinale 1 m/s², accélération verticale 1.5 m/s².
        """
        # 1. Pilotage du cap (3°/s max)
        diff_h = (self.target_heading_deg - self.heading_deg + 180.0) % 360.0 - 180.0
        max_turn = MAX_TURN_RATE_DEG_S * dt
        if abs(diff_h) <= max_turn:
            self.heading_deg = self.target_heading_deg
        else:
            self.heading_deg = (self.heading_deg + np.sign(diff_h) * max_turn) % 360.0

        # 2. Pilotage de la vitesse sol (accélération réaliste de transport civil : ~1.0 m/s²)
        diff_s = self.target_speed - self.speed
        max_accel = 1.0 * dt
        if abs(diff_s) <= max_accel:
            self.speed = self.target_speed
        else:
            self.speed += np.sign(diff_s) * max_accel

        # 3. Pilotage de l'altitude : palier atteint à ±15 m
        alt_err = self.target_altitude - self.z
        if abs(alt_err) < 15.0:
            self.target_vertical_rate = 0.0
        elif alt_err > 0 and self.target_vertical_rate <= 0:
            self.target_vertical_rate = STANDARD_CLIMB_RATE_MS
        elif alt_err < 0 and self.target_vertical_rate >= 0:
            self.target_vertical_rate = STANDARD_DESC_RATE_MS

        # Pilotage du taux de montée (accélération verticale douce)
        diff_v = self.target_vertical_rate - self.vertical_rate
        max_v_accel = 1.5 * dt
        if abs(diff_v) <= max_v_accel:
            self.vertical_rate = self.target_vertical_rate
        else:
            self.vertical_rate += np.sign(diff_v) * max_v_accel

        # 4. Intégration de la position
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.z += self.vz * dt

    def to_dict(self):
        return {
            "callsign": self.callsign,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "speed": self.speed,
            "heading_deg": self.heading_deg,
            "vertical_rate": self.vertical_rate,
            "target_heading_deg": self.target_heading_deg,
            "target_altitude": self.target_altitude,
            "target_speed": self.target_speed
        }
