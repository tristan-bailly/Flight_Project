import json
import os
from pyopensky.rest import REST

# 1. Chargement des identifiants depuis credentials.json
with open("credentials.json", "r") as f:
    creds = json.load(f)

# 2. Transmission des clés à pyopensky via les variables d'environnement
os.environ["OPENSKY_CLIENT_ID"] = creds["clientId"]
os.environ["OPENSKY_CLIENT_SECRET"] = creds["clientSecret"]

# 3. Initialisation du client REST (sans arguments)
rest = REST()

# 4. Récupération des vols en temps réel
df_vols = rest.states()
print(df_vols.head())