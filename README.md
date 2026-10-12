# Flight_Project

Description des 17 colonnes du jeu de données
1. Identifiants et origine
icao24 (string) : L'adresse hexadécimale unique à 24 bits du transpondeur de l'appareil (l'équivalent de la plaque d'immatriculation de l'avion).
callsign (string) : Le numéro de vol (ex: AFR123 pour Air France).
origin_country (string) : Le pays d'immatriculation de l'appareil.
2. Position spatio-temporelle (3D + Temps)
time_position (int/float) : Horodatage Unix (en secondes) de la dernière mise à jour de la position.
last_contact (int/float) : Horodatage Unix de la dernière émission reçue du transpondeur.
longitude (float) : Longitude en degrés (WGS-84).
latitude (float) : Latitude en degrés (WGS-84).
baro_altitude (float) : Altitude barométrique en mètres.
geo_altitude (float) : Altitude géométrique (GPS) en mètres.
3. Cinématique et dynamique du vol
velocity (float) : Vitesse sol (ground speed) en mètres par seconde (m/s). Multiplie par 3,6 pour l'obtenir en km/h.
true_track (float) : Cap réel de l'avion en degrés par rapport au Nord vrai (0  à 360 ∘).
vertical_rate (float) : Vitesse verticale en m/s (>0 = montée, <0 = descente, 0 = vol en palier).
4. État technique et capteurs
on_ground (bool) : True si l'avion est au sol, False s'il est en air.
squawk (string) : Code transpondeur à 4 chiffres attribué par le contrôle aérien (ex: 7700 pour une urgence).
spi (bool) : Indicateur d'état spécial (Special Purpose Indicator).
position_source (int) : Origine de la mesure (0 = ADS-B, 1 = ASTERIX, 2 = MLAT, 3 = FLARM).
sensors (list) : Identifiants des récepteurs au sol ayant capté le signal (souvent nul).