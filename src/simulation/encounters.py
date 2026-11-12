"""
Générateur de géométries de rencontres et conflits aéronautiques (Encounter Generator).
Permet de générer des scénarios d'évitement rigoureux et paramétriques pour l'entraînement RL.
"""

import numpy as np
from src.config import (
    NM_TO_M,
    FT_TO_M,
    CRUISE_SPEED_MS,
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
    
    # Produit scalaire position relative . vitesse relative
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
    Génère un couple (Ownship, Intruder) engagé dans une situation de conflit prédictif.
    - Ownship : démarre centré ou proche de l'origine
    - Intruder : placé de telle sorte qu'un conflit (viol de 5 NM et 1000 ft) se produira dans 60 à 150 s.
    """
    if rng is None:
        rng = np.random.default_rng()

    types = ["head_on", "cross_90", "cross_oblique", "overtake"]
    if scenario_type == "random" or scenario_type not in types:
        scenario_type = rng.choice(types)

    # Temps cible jusqu'au CPA (1 à 2.5 minutes)
    t_cpa = rng.uniform(60.0, 150.0)

    # Altitude de croisière standard (FL300 à FL380, ex: 10 000 mètres)
    base_alt = rng.uniform(9000.0, 11500.0)
    
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

    # Intruder speed
    v_intru = rng.uniform(CRUISE_SPEED_MIN_MS, CRUISE_SPEED_MAX_MS)
    
    # Calcule les caps relatifs selon le scénario
    if scenario_type == "head_on":
        # Face-à-face : cap opposé à 180° ± 5°
        heading_intru = (heading_own + 180.0 + rng.uniform(-5.0, 5.0)) % 360.0
    elif scenario_type == "cross_90":
        # Croisement orthogonal ± 90°
        side = rng.choice([-90.0, 90.0])
        heading_intru = (heading_own + side + rng.uniform(-10.0, 10.0)) % 360.0
    elif scenario_type == "cross_oblique":
        # Croisement oblique
        angle = rng.choice([rng.uniform(30.0, 75.0), rng.uniform(105.0, 160.0)])
        side = rng.choice([-1.0, 1.0])
        heading_intru = (heading_own + side * angle) % 360.0
    elif scenario_type == "overtake":
        # Dépassement sur même cap
        heading_intru = (heading_own + rng.uniform(-5.0, 5.0)) % 360.0
        # Vitesse plus rapide pour dépasser
        v_intru = v_own + rng.uniform(30.0, 60.0)

    # Calcul de la position initiale de l'intruder pour provoquer une collision ou quasi-collision au CPA
    # Déviation aléatoire au CPA (entre 0 et 3 NM horizontal, et 0 et 600 ft vertical -> Conflit garanti si pas d'action)
    miss_distance_horiz = rng.uniform(0.0, 3.0 * NM_TO_M)
    miss_angle = rng.uniform(0.0, 2.0 * np.pi)
    miss_x = miss_distance_horiz * np.cos(miss_angle)
    miss_y = miss_distance_horiz * np.sin(miss_angle)
    
    miss_dz = rng.uniform(-600.0 * FT_TO_M, 600.0 * FT_TO_M)

    # Vitesse vectorielle de l'intruder
    rad_int = np.radians(heading_intru)
    vx_int = v_intru * np.sin(rad_int)
    vy_int = v_intru * np.cos(rad_int)
    vz_int = rng.choice([0.0, rng.uniform(-2.0, 2.0)])  # palier ou légère montée/descente

    # Position de départ telle que dans t_cpa, la distance soit (miss_x, miss_y, miss_dz)
    # dx_cpa = dx_init + (vx_int - vx_own) * t_cpa = miss_x
    dx_init = miss_x - (vx_int - ownship.vx) * t_cpa
    dy_init = miss_y - (vy_int - ownship.vy) * t_cpa
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

