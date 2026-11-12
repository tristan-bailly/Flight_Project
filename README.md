# ✈️ ATC AI Conflict Resolution & Collision Avoidance (D3QN)

Système autonome d'aide à la décision pour le contrôle de la circulation aérienne (**ATC En-Route**), basé sur l'apprentissage par renforcement profond (**Dueling Double Deep Q-Network - D3QN**).

Conforme aux normes de séparation physique de l'**OACI** (*5 NM horizontal, 1 000 ft vertical*).

---

## 🏗️ Architecture du Projet

```
Flight_Project/
├── src/
│   ├── config.py             # Constantes OACI & aéronautiques (SI, NM, ft, kts)
│   ├── simulation/
│   │   ├── aircraft.py       # Modèle cinématique 3D (Cap, Altitude, Vitesse)
│   │   ├── encounters.py     # Générateur de géométries de conflits (Head-on, Crossing, Overtake)
│   │   └── atc_env.py        # Environnement Gymnasium réaliste & pénalités OACI
│   ├── agent/
│   │   ├── buffer.py         # Experience Replay Buffer NumPy/PyTorch
│   │   └── dqn.py            # Dueling Double DQN (D3QN) avec Polyak Soft-Update
│   ├── api/
│   │   └── app.py            # Microservice FastAPI déployable (Inférence temps réel)
│   └── dashboard/
│       └── app.py            # Radar ATC interactif Streamlit + Plotly
├── models/
│   └── d3qn_atc_latest.pt    # Poids du modèle entraîné
├── train_realistic.py        # Script d'entraînement & benchmark scientifique
└── requirements.txt
```

---

## 🎮 Les 3 Leviers de Contrôle ATC (Espace d'Actions)

L'agent contrôle l'appareil à l'aide de **9 actions opérationnelles** :
1. **Neutre** : `MAINTAIN` (Maintien de la trajectoire nominale)
2. **Latéral (Cap)** :
   - `TURN_RIGHT_15` (+15°) / `TURN_LEFT_15` (-15°)
   - `TURN_RIGHT_30` (+30°) / `TURN_LEFT_30` (-30°)
3. **Vertical (Altitude)** :
   - `CLIMB_1000FT` (Monter de 1 000 ft au niveau supérieur à +1500 ft/min)
   - `DESCEND_1000FT` (Descendre de 1 000 ft au niveau inférieur à -1500 ft/min)
4. **Longitudinal (Vitesse)** :
   - `SPEED_UP_20KTS` (+20 noeuds pour devancer l'intersection)
   - `SPEED_DOWN_20KTS` (-20 noeuds pour céder le passage)

---

## 📊 Benchmark Scientifique (100 Scénarios Inédits)

| Métrique OACI | Do Nothing (Passif) | Aléatoire (Baseline) | **Agent D3QN ATC** |
|---|---|---|---|
| **Récompense Moyenne** | -636.0 | -114.1 | **+56.2** |
| **Pertes de Séparation (< 5 NM)** | 97.0% | 53.0% | **28.0%** (-69 pts) |
| **Collisions Critiques (NMAC)** | 0.0% | 0.0% | **0.0%** |
| **Résolutions Réussies (Sans faute)** | 2.0% | 47.0% | **68.0%** (+66 pts) |

---

## 🚀 Déploiement et Utilisation

### 1. Lancer l'API d'Inférence (FastAPI)
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```
- Documentation Swagger interactive : `http://localhost:8000/docs`
- Endpoint d'évaluation en temps réel : `POST /predict-advisory`

### 2. Lancer le Radar Interactif (Dashboard Streamlit)
```bash
streamlit run src.dashboard/app.py
```
Permet de visualiser graphiquement la résolution d'un conflit en direct (courbes de trajectoire et franchissement de la bulle de 5 NM).

### 3. Réentraîner le modèle
```bash
python3 train_realistic.py
```
