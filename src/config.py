"""
Constantes aéronautiques standard (OACI / ICAO & Eurocontrol).
Toutes les grandeurs de base dans la simulation sont en unités SI (mètres, secondes, m/s).
Des constantes de conversion sont fournies pour l'affichage en unités aéro (NM, ft, kts).
"""

# --- Facteurs de conversion ---
NM_TO_M = 1852.0          # 1 Nautical Mile = 1852 mètres
M_TO_NM = 1.0 / NM_TO_M

FT_TO_M = 0.3048          # 1 Pied = 0.3048 mètres
M_TO_FT = 1.0 / FT_TO_M

KTS_TO_MS = 0.514444      # 1 Noeud = 0.5144 m/s
MS_TO_KTS = 1.0 / KTS_TO_MS

FPM_TO_MS = 0.00508       # 1 ft/min = 0.00508 m/s
MS_TO_FPM = 1.0 / FPM_TO_MS

# --- Normes de séparation OACI en route (En-Route Separation) ---
SEPARATION_HORIZ_M = 5.0 * NM_TO_M       # 5 NM = 9 260 m
SEPARATION_VERT_M = 1000.0 * FT_TO_M     # 1 000 ft = 304.8 m

# Seuil de collision physique imminente / NMAC (Near Mid-Air Collision)
# Défini standard FAA/OACI : 500 ft horizontal (~150m) et 100 ft vertical (~30m)
NMAC_HORIZ_M = 500.0 * FT_TO_M           # ~152 m
NMAC_VERT_M = 100.0 * FT_TO_M            # ~30.5 m

# Distance de confort / Résolution garantie
SAFE_HORIZ_M = 7.0 * NM_TO_M             # 7 NM (au-delà de la norme pour marge)
SAFE_VERT_M = 1500.0 * FT_TO_M           # 1 500 ft

# --- Performances standard d'un avion civil de ligne (type A320 / B737) ---
CRUISE_SPEED_MS = 230.0                  # ~450 kts = Mach 0.78 (~830 km/h)
CRUISE_SPEED_MIN_MS = 180.0              # ~350 kts
CRUISE_SPEED_MAX_MS = 260.0              # ~500 kts

STANDARD_CLIMB_RATE_MS = 1500.0 * FPM_TO_MS  # +1 500 ft/min ≈ 7.62 m/s
STANDARD_DESC_RATE_MS = -1500.0 * FPM_TO_MS # -1 500 ft/min ≈ -7.62 m/s

# Taux de virage standard (Standard Rate Turn = 3°/seconde)
MAX_TURN_RATE_DEG_S = 3.0

# --- Pas de simulation ---
SIM_DT_S = 2.0                           # 1 pas de temps = 2 secondes de vol réel
MAX_EPISODE_STEPS = 120                  # 120 steps * 2s = 240s = 4 minutes de vol

