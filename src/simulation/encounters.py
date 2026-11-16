"""
Générateur de géométries de rencontres et conflits aéronautiques (Encounter Generator).
Garantit que TOUS les scénarios démarrent en amont du conflit (distance initiale > 8 NM),
permettant une résolution réaliste et fluide conforme aux normes OACI.
"""

import numpy as np
from src.config import (
    NM_TO_M,
    FT_TO_M,
    CRUISE_SPEED_MIN_MS,
    CRUISE_SPEED_MAX_MS,
    SEPARATION_HORIZ_M,
    SEPARATION_VERT_M
)
from src.simulation.aircraft import Aircraft

def compute_cpa(own: Aircraft, intru: Aircraft):
    """
    Calcule le Closest Point of Approach (CPA) entre deux aéronefs si trajectoires rectilignes.
    Retourne :
      t_cpa : temps restant en secondes jusqu'au point de rapprochement maximal
      d_cpa_horiz : distance horizontale au CPA (mètres)
      d_cpa_vert : distance verticale au CPA (mètres)
    """
    dx = intru.x - own.x
    dy = intru.y - own.y
    dz = intru.z - own.z
    
    dvx = intru.vx - own.vx
    dvy = intru.vy - own.vy
    dvz = intru.vz - own.vz
    
    rel_speed_sq = dvx**2 + dvy**2
    if rel_speed_sq < 1e-4:
        t_cpa = 0.0
    else:
        t_cpa = -(dx * dvx + dy * dvy) / rel_speed_sq
        t_cpa = max(0.0, t_cpa)

    # Positions au CPA
    cpa_dx = dx + dvx * t_cpa
    cpa_dy = dy + dvy * t_cpa
    cpa_dz = dz + dvz * t_cpa

    d_cpa_horiz = float(np.hypot(cpa_dx, cpa_dy))
    d_cpa_vert = float(abs(cpa_dz))

    return t_cpa, d_cpa_horiz, d_cpa_vert

def generate_encounter(scenario_type: str = "random", rng: np.random.Generator = None):
    """
    Génère un couple (Ownship, Intruder) engagé dans une situation de conflit prédictif réaliste.
    - Ownship : démarre à l'origine avec cap et vitesse de croisière standards
    - Intruder : positionné à une distance initiale garantie (hors zone de précaution 8 NM)
      avec une trajectoire convergeant vers un CPA critique (0 à 2.5 NM et < 600 ft).
    """
    if rng is None:
        rng = np.random.default_rng()

    types = ["head_on", "cross_90", "cross_oblique", "overtake"]
    if scenario_type == "random" or scenario_type not in types:
        scenario_type = rng.choice(types)

    # Altitude de croisière standard (FL300 à FL380, ex: 10 000 mètres)
    base_alt = rng.uniform(9500.0, 11500.0)
    
    # Ownship
    v_own = rng.uniform(CRUISE_SPEED_MIN_MS, CRUISE_SPEED_MAX_MS)
    heading_own = rng.uniform(0.0, 360.0)
    ownship = Aircraft(
        callsign="OWNSHIP",
        x=0.0,
        y=0.0,
        z=base_alt,
        speed=v_own,
        heading_deg=heading_own,
        vertical_rate=0.0
    )

    # Intruder speed & heading according to encounter geometry
    if scenario_type == "head_on":
        # Face-à-face : cap opposé à 180° ± 5°
        heading_intru = (heading_own + 180.0 + rng.uniform(-5.0, 5.0)) % 360.0
        v_intru = rng.uniform(CRUISE_SPEED_MIN_MS, CRUISE_SPEED_MAX_MS)
        # Distance initiale pour head-on : 18 à 22 NM (donne ~80s d'anticipation avec v_rel ~900 kts)
        target_init_dist = rng.uniform(18.0, 22.0) * NM_TO_M

    elif scenario_type == "cross_90":
        # Croisement orthogonal ± 90°
        side = rng.choice([-90.0, 90.0])
        heading_intru = (heading_own + side + rng.uniform(-5.0, 5.0)) % 360.0
        v_intru = rng.uniform(CRUISE_SPEED_MIN_MS, CRUISE_SPEED_MAX_MS)
        target_init_dist = rng.uniform(14.0, 17.0) * NM_TO_M

    elif scenario_type == "cross_oblique":
        # Croisement oblique (angles 40-70° ou 110-140°)
        angle = rng.choice([rng.uniform(40.0, 70.0), rng.uniform(110.0, 140.0)])
        side = rng.choice([-1.0, 1.0])
        heading_intru = (heading_own + side * angle) % 360.0
        v_intru = rng.uniform(CRUISE_SPEED_MIN_MS, CRUISE_SPEED_MAX_MS)
        target_init_dist = rng.uniform(14.0, 17.0) * NM_TO_M

    elif scenario_type == "overtake":
        # Dépassement sur le même axe avec surplus de vitesse
        heading_intru = (heading_own + rng.uniform(-3.0, 3.0)) % 360.0
        # Intruder plus rapide de 60 à 90 m/s (~120-180 kts)
        v_intru = v_own + rng.uniform(60.0, 90.0)
        # Distance initiale pour dépassement : 9.5 à 11.5 NM (juste en amont des 8 NM)
        target_init_dist = rng.uniform(9.5, 11.5) * NM_TO_M

    # Vitesses vectorielles
    rad_int = np.radians(heading_intru)
    vx_int = v_intru * np.sin(rad_int)
    vy_int = v_intru * np.cos(rad_int)
    vz_int = rng.choice([0.0, rng.uniform(-1.5, 1.5)])  # palier ou très faible taux

    dvx = vx_int - ownship.vx
    dvy = vy_int - ownship.vy
    v_rel = np.hypot(dvx, dvy)
    if v_rel < 1e-2:
        v_rel = 10.0
        dvx = 10.0

    # Déviation nominale au CPA (miss distance entre 0 et 2.5 NM horizontalement, < 500 ft verticalement)
    miss_distance = rng.uniform(0.0, 2.5 * NM_TO_M)
    perp_x = -dvy / v_rel
    perp_y = dvx / v_rel
    side_sign = rng.choice([-1.0, 1.0])
    miss_x = perp_x * miss_distance * side_sign
    miss_y = perp_y * miss_distance * side_sign
    miss_dz = rng.uniform(-400.0 * FT_TO_M, 400.0 * FT_TO_M)

    # Calcul du temps jusqu'au CPA tel que la distance initiale soit exactement target_init_dist
    t_cpa = np.sqrt(max(0.0, target_init_dist**2 - miss_distance**2)) / v_rel

    # Position initiale relative
    dx_init = miss_x - dvx * t_cpa
    dy_init = miss_y - dvy * t_cpa
    dz_init = miss_dz - (vz_int - ownship.vz) * t_cpa

    intruder = Aircraft(
        callsign="INTRUDER",
        x=ownship.x + dx_init,
        y=ownship.y + dy_init,
        z=ownship.z + dz_init,
        speed=v_intru,
        heading_deg=heading_intru,
        vertical_rate=vz_int
    )

    return ownship, intruder
