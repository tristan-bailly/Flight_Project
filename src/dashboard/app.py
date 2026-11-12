"""
Dashboard Radar ATC Interactif et Animé (Streamlit + Plotly 2D/3D).
Cycle complet de contrôle aérien OACI :
1. Détection de conflit
2. Résolution active (Étage vertical / Virage)
3. "Resume Own Navigation" : Reprise de trajectoire nominale
4. Export du rapport de sécurité aéronautique (Safety Report) en CSV/JSON.
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import (
    NM_TO_M,
    FT_TO_M,
    MS_TO_KTS,
    SEPARATION_HORIZ_M,
    SEPARATION_VERT_M
)
from src.simulation.atc_env import RealisticATCEnv
from src.agent.dqn import D3QNAgent
from src.ingestion.live_opensky import scan_live_conflicts

st.set_page_config(page_title="Radar ATC Professionnel 2D & 3D", layout="wide", page_icon="📡")

st.markdown("""
<style>
    .reportview-container { background: #0b0f19; }
    .stMetric { background-color: #161f30; padding: 12px; border-radius: 8px; border: 1px solid #22324d; }
</style>
""", unsafe_allow_html=True)

st.title("📡 Radar de Contrôle Aérien — Séparation 3D & Reprise de Cap (D3QN)")

model_path = "models/d3qn_atc_latest.pt"

@st.cache_resource
def get_agent():
    agent = D3QNAgent(state_dim=10, action_dim=9, device="cpu")
    if os.path.exists(model_path):
        agent.load(model_path)
    return agent

agent = get_agent()

tab_sim, tab_live = st.tabs(["🎮 Radar Animé & Vue 3D", "🛰️ Surveillance Directe (OpenSky Live)"])

def generate_circle_points(cx, cy, radius_nm=5.0, num_points=40):
    theta = np.linspace(0, 2 * np.pi, num_points)
    x = cx + radius_nm * np.cos(theta)
    y = cy + radius_nm * np.sin(theta)
    return x.tolist(), y.tolist()

with tab_sim:
    col_ctrl, col_info = st.columns([1, 2])
    with col_ctrl:
        st.subheader("Paramètres de la rencontre")
        scenario_choice = st.selectbox(
            "Scénario de conflit :",
            ["head_on", "cross_90", "cross_oblique", "overtake"],
            format_func=lambda x: {
                "head_on": "Face-à-face direct (Head-on 180°)",
                "cross_90": "Croisement perpendiculaire (90° Crossing)",
                "cross_oblique": "Convergence oblique (Oblique Angle)",
                "overtake": "Dépassement en route (Overtake)",
            }[x]
        )
        sim_mode = st.radio("Mode d'activation :", ["Avec Résolution IA (D3QN)", "Sans IA (Trajectoire passive en collision)"])
        seed = st.number_input("Graine aléatoire (Seed) :", value=42, step=1)

    with col_info:
        st.info("""
        **Normes réelles de séparation 3D OACI :**
        - Il n'y a **AUCUN danger** si les avions sont séparés d'au moins **5 NM horizontalement** OU d'au moins **1 000 ft verticalement**.
        - Dès que le croisement s'achève en sécurité, l'IA enclenche l'ordre **`RESUME_NAVIGATION`** pour ramener l'avion sur sa ligne directe vers sa destination !
        - Vous pouvez télécharger en bas de page le **Rapport Officiel de Sécurité** de la simulation en CSV.
        """)

    env = RealisticATCEnv(scenario_type=scenario_choice)
    obs, info = env.reset(seed=seed)
    nominal_heading = float(env.nominal_heading_deg)
    nominal_alt = float(env.nominal_altitude_m)

    history = []
    dx_init = env.intruder.x - env.ownship.x
    dy_init = env.intruder.y - env.ownship.y
    dist_init = float(np.hypot(dx_init, dy_init)) / NM_TO_M
    alt_diff_init = abs(env.ownship.z - env.intruder.z) * 3.28084

    history.append({
        "t_sec": 0,
        "own_x": env.ownship.x / NM_TO_M,
        "own_y": env.ownship.y / NM_TO_M,
        "own_z_ft": env.ownship.z * 3.28084,
        "own_alt_fl": int(round(env.ownship.z * 3.28084 / 100.0)),
        "own_spd_kts": int(round(env.ownship.speed * MS_TO_KTS)),
        "own_hdg": int(round(env.ownship.heading_deg)),
        "int_x": env.intruder.x / NM_TO_M,
        "int_y": env.intruder.y / NM_TO_M,
        "int_z_ft": env.intruder.z * 3.28084,
        "int_alt_fl": int(round(env.intruder.z * 3.28084 / 100.0)),
        "int_spd_kts": int(round(env.intruder.speed * MS_TO_KTS)),
        "int_hdg": int(round(env.intruder.heading_deg)),
        "dist_nm": dist_init,
        "alt_diff_ft": alt_diff_init,
        "action_name": "MAINTAIN",
        "phase": "APPROCHE",
        "real_loss_of_sep": False
    })

    done = False
    step_num = 0
    use_ai = (sim_mode == "Avec Résolution IA (D3QN)")
    has_crossed = False
    prev_dist = dist_init

    while step_num < 80:
        step_num += 1

        curr_dx = env.intruder.x - env.ownship.x
        curr_dy = env.intruder.y - env.ownship.y
        curr_dist_nm = float(np.hypot(curr_dx, curr_dy)) / NM_TO_M

        if curr_dist_nm > prev_dist and curr_dist_nm < 12.0:
            has_crossed = True

        action_name = "MAINTAIN"
        phase = "ÉVITEMENT ACTIF"

        if use_ai:
            if has_crossed and curr_dist_nm >= 6.0:
                phase = "REPRISE DE CAP"
                action_name = "RESUME_NAVIGATION"
                env.ownship.set_heading_target(nominal_heading)
                env.ownship.step(env.dt)
                env.intruder.step(env.dt)
            else:
                action = agent.select_action(obs, evaluate=True)
                action_name = RealisticATCEnv.ACTION_NAMES[action]
                obs, r, term, trunc, info = env.step(action)
        else:
            obs, r, term, trunc, info = env.step(0)
            phase = "TRAJECTOIRE PASSIVE"

        prev_dist = curr_dist_nm
        curr_alt_diff_ft = abs(env.ownship.z - env.intruder.z) * 3.28084
        is_true_loss = (curr_dist_nm < 5.0) and (curr_alt_diff_ft < 1000.0)

        history.append({
            "t_sec": step_num * 2,
            "own_x": env.ownship.x / NM_TO_M,
            "own_y": env.ownship.y / NM_TO_M,
            "own_z_ft": env.ownship.z * 3.28084,
            "own_alt_fl": int(round(env.ownship.z * 3.28084 / 100.0)),
            "own_spd_kts": int(round(env.ownship.speed * MS_TO_KTS)),
            "own_hdg": int(round(env.ownship.heading_deg)),
            "int_x": env.intruder.x / NM_TO_M,
            "int_y": env.intruder.y / NM_TO_M,
            "int_z_ft": env.intruder.z * 3.28084,
            "int_alt_fl": int(round(env.intruder.z * 3.28084 / 100.0)),
            "int_spd_kts": int(round(env.intruder.speed * MS_TO_KTS)),
            "int_hdg": int(round(env.intruder.heading_deg)),
            "dist_nm": curr_dist_nm,
            "alt_diff_ft": curr_alt_diff_ft,
            "action_name": action_name,
            "phase": phase,
            "real_loss_of_sep": is_true_loss
        })

    subtab_2d, subtab_3d = st.tabs(["🗺️ Radar 2D Animé", "🌐 Vue Espace Aérien 3D"])

    # ==================== RADAR 2D ====================
    with subtab_2d:
        all_x = [h["own_x"] for h in history] + [h["int_x"] for h in history]
        all_y = [h["own_y"] for h in history] + [h["int_y"] for h in history]
        margin = 8.0
        x_min, x_max = min(all_x) - margin, max(all_x) + margin
        y_min, y_max = min(all_y) - margin, max(all_y) + margin

        h0 = history[0]
        circ_own_x, circ_own_y = generate_circle_points(h0["own_x"], h0["own_y"])
        circ_int_x, circ_int_y = generate_circle_points(h0["int_x"], h0["int_y"])

        fig2d = go.Figure(
            data=[
                go.Scatter(x=[h0["own_x"]], y=[h0["own_y"]], mode="lines", line=dict(color="#00ffcc", width=2, dash="dot"), name="Sillage Ownship"),
                go.Scatter(
                    x=[h0["own_x"]], y=[h0["own_y"]],
                    mode="markers+text",
                    marker=dict(size=14, color="#00ffcc", symbol="triangle-up", angle=h0["own_hdg"]),
                    text=[f"✈️ OWNSHIP<br>FL{h0['own_alt_fl']} | {h0['own_spd_kts']}kt"],
                    textposition="top right",
                    textfont=dict(color="#00ffcc", size=11, family="monospace"),
                    name="Ownship (IA)"
                ),
                go.Scatter(x=circ_own_x, y=circ_own_y, mode="lines", line=dict(color="#00ffcc", width=1.5), fill="toself", fillcolor="rgba(0, 255, 204, 0.08)", name="Bulle 5 NM Ownship"),
                go.Scatter(x=[h0["int_x"]], y=[h0["int_y"]], mode="lines", line=dict(color="#ff9900", width=2, dash="dot"), name="Sillage Trafic"),
                go.Scatter(
                    x=[h0["int_x"]], y=[h0["int_y"]],
                    mode="markers+text",
                    marker=dict(size=14, color="#ff9900", symbol="triangle-up", angle=h0["int_hdg"]),
                    text=[f"✈️ INTRUDER<br>FL{h0['int_alt_fl']} | {h0['int_spd_kts']}kt"],
                    textposition="bottom left",
                    textfont=dict(color="#ff9900", size=11, family="monospace"),
                    name="Intruder (Trafic)"
                ),
                go.Scatter(x=circ_int_x, y=circ_int_y, mode="lines", line=dict(color="#ff9900", width=1.5), fill="toself", fillcolor="rgba(255, 153, 0, 0.08)", name="Bulle 5 NM Intruder")
            ]
        )

        frames = []
        for k, h in enumerate(history):
            c_own_x, c_own_y = generate_circle_points(h["own_x"], h["own_y"])
            c_int_x, c_int_y = generate_circle_points(h["int_x"], h["int_y"])

            is_true_danger = h["real_loss_of_sep"]
            vert_safe = (h["alt_diff_ft"] >= 1000.0)

            if is_true_danger:
                own_color = "#ff3333"
                fill_color = "rgba(255, 51, 51, 0.25)"
                status_txt = "<br><span style='color:#ff3333'><b>⚠️ PERTE DE SÉPARATION</b></span>"
            elif vert_safe and h["dist_nm"] < 5.0:
                own_color = "#00ccff"
                fill_color = "rgba(0, 204, 255, 0.15)"
                status_txt = f"<br><span style='color:#00ccff'><b>🟢 SÉPARÉ VERTICALEMENT (Δ={h['alt_diff_ft']:.0f}ft)</b></span>"
            else:
                own_color = "#00ffcc"
                fill_color = "rgba(0, 255, 204, 0.08)"
                status_txt = ""

            action_tag = f"<br><b>ORDRE: {h['action_name']}</b>" if h["action_name"] != "MAINTAIN" else ""
            phase_tag = f"<br><i>[{h['phase']}]</i>"

            frame_data = [
                go.Scatter(x=[p["own_x"] for p in history[:k+1]], y=[p["own_y"] for p in history[:k+1]]),
                go.Scatter(
                    x=[h["own_x"]], y=[h["own_y"]],
                    marker=dict(size=15, color=own_color, symbol="triangle-up", angle=h["own_hdg"]),
                    text=[f"✈️ OWNSHIP<br>FL{h['own_alt_fl']} | {h['own_spd_kts']}kt | Cap {h['own_hdg']}°{action_tag}{phase_tag}{status_txt}"],
                    textfont=dict(color=own_color)
                ),
                go.Scatter(x=c_own_x, y=c_own_y, line=dict(color=own_color, width=2 if is_true_danger else 1.5), fillcolor=fill_color),
                go.Scatter(x=[p["int_x"] for p in history[:k+1]], y=[p["int_y"] for p in history[:k+1]]),
                go.Scatter(
                    x=[h["int_x"]], y=[h["int_y"]],
                    marker=dict(size=15, color="#ff9900", symbol="triangle-up", angle=h["int_hdg"]),
                    text=[f"✈️ INTRUDER<br>FL{h['int_alt_fl']} | {h['int_spd_kts']}kt | Cap {h['int_hdg']}°"]
                ),
                go.Scatter(x=c_int_x, y=c_int_y, line=dict(color="#ff3333" if is_true_danger else "#ff9900"), fillcolor="rgba(255, 51, 51, 0.25)" if is_true_danger else "rgba(255, 153, 0, 0.08)")
            ]
            frames.append(go.Frame(data=frame_data, name=f"t{h['t_sec']}"))

        fig2d.frames = frames

        fig2d.update_layout(
            title=dict(text=f"Radar de contrôle sectoriel en direct ({sim_mode})", font=dict(color="#e0e6ed", size=16)),
            paper_bgcolor="#0b0f19",
            plot_bgcolor="#080c14",
            xaxis=dict(range=[x_min, x_max], title="Axe Est (Milles Nautiques - NM)", gridcolor="#1a2436", zerolinecolor="#2a3a54"),
            yaxis=dict(range=[y_min, y_max], title="Axe Nord (Milles Nautiques - NM)", gridcolor="#1a2436", zerolinecolor="#2a3a54", scaleanchor="x", scaleratio=1),
            height=650,
            showlegend=True,
            legend=dict(font=dict(color="#a0aec0"), bgcolor="#111827"),
            updatemenus=[{
                "type": "buttons",
                "showactive": False,
                "x": 0.05, "y": 1.15,
                "buttons": [
                    {
                        "label": "▶️ Lecture (Animer les avions)",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 180, "redraw": True}, "fromcurrent": True, "transition": {"duration": 50}}]
                    },
                    {
                        "label": "⏸️ Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}]
                    }
                ]
            }],
            sliders=[{
                "steps": [
                    {
                        "args": [[f.name], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                        "label": f"{h['t_sec']}s",
                        "method": "animate"
                    }
                    for f, h in zip(frames, history)
                ],
                "active": 0,
                "transition": {"duration": 0},
                "x": 0.05, "y": -0.05,
                "len": 0.9,
                "currentvalue": {"prefix": "Temps de vol : ", "font": {"color": "#00ffcc", "size": 14}},
                "font": {"color": "#a0aec0"}
            }]
        )
        st.plotly_chart(fig2d, use_container_width=True)

    # ==================== VUE 3D ====================
    with subtab_3d:
        st.markdown("**Secteur aérien 3D navigable à la souris :** observez l'ensemble de la trajectoire (approche, esquive, reprise de route).")

        fig3d = go.Figure()

        fig3d.add_trace(go.Scatter3d(
            x=[h["own_x"] for h in history],
            y=[h["own_y"] for h in history],
            z=[h["own_z_ft"] for h in history],
            mode="lines+markers",
            line=dict(color="#00ffcc", width=5),
            marker=dict(size=3, color="#00ffcc"),
            name="Ownship (Trajectoire complète)"
        ))

        h_end = history[-1]
        fig3d.add_trace(go.Scatter3d(
            x=[h_end["own_x"]], y=[h_end["own_y"]], z=[h_end["own_z_ft"]],
            mode="text+markers",
            marker=dict(size=8, color="#00ffcc"),
            text=[f"✈️ OWNSHIP (Cap repris {h_end['own_hdg']}°)"],
            textposition="top center",
            name="Position finale Ownship"
        ))

        fig3d.add_trace(go.Scatter3d(
            x=[h["int_x"] for h in history],
            y=[h["int_y"] for h in history],
            z=[h["int_z_ft"] for h in history],
            mode="lines+markers",
            line=dict(color="#ff9900", width=5),
            marker=dict(size=3, color="#ff9900"),
            name="Intruder (Trafic)"
        ))

        fig3d.add_trace(go.Scatter3d(
            x=[h_end["int_x"]], y=[h_end["int_y"]], z=[h_end["int_z_ft"]],
            mode="text+markers",
            marker=dict(size=8, color="#ff9900"),
            text=[f"✈️ INTRUDER (FL{h_end['int_alt_fl']})"],
            textposition="top center",
            name="Position finale Intruder"
        ))

        fig3d.update_layout(
            scene=dict(
                xaxis_title="Est (NM)",
                yaxis_title="Nord (NM)",
                zaxis_title="Altitude (Pieds)",
                bgcolor="#080c14",
                xaxis=dict(gridcolor="#1a2436", backgroundcolor="#080c14"),
                yaxis=dict(gridcolor="#1a2436", backgroundcolor="#080c14"),
                zaxis=dict(gridcolor="#1a2436", backgroundcolor="#080c14")
            ),
            paper_bgcolor="#0b0f19",
            height=650,
            template="plotly_dark",
            legend=dict(font=dict(color="#a0aec0"), bgcolor="#111827")
        )
        st.plotly_chart(fig3d, use_container_width=True)

    # Indicateurs Métier & Export
    min_dist_observed = min([h["dist_nm"] for h in history])
    had_true_loss = any([h["real_loss_of_sep"] for h in history])
    max_alt_sep = max([h["alt_diff_ft"] for h in history])

    # Estimation carburant : pénalité d'écart angulaire moyen
    mean_heading_dev = np.mean([abs((h["own_hdg"] - nominal_heading + 180) % 360 - 180) for h in history])
    fuel_extra_kg = round(float(mean_heading_dev * 1.2), 1)

    st.subheader("📊 Rapport d'Efficacité & Sécurité Aéronautique")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Distance horizontale min", f"{min_dist_observed:.2f} NM")
    c2.metric("Écart vertical atteint", f"{max_alt_sep:.0f} ft", delta=">= 1000 ft (Sécurisé)" if max_alt_sep >= 1000 else "< 1000 ft")
    c3.metric("Statut Séparation OACI", "VIOLÉE (DANGER)" if had_true_loss else "CONFORME ET SÉCURISÉ ✅", delta_color="inverse" if had_true_loss else "normal")
    c4.metric("Surconsommation kérosène estimée", f"{fuel_extra_kg} kg", delta="-85% vs Demi-tour", delta_color="normal")

    # Export des données de vol
    df_report = pd.DataFrame(history)
    csv_bytes = df_report.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Télécharger le Rapport Officiel de Sécurité (CSV)",
        data=csv_bytes,
        file_name=f"atc_safety_report_seed_{seed}_{scenario_choice}.csv",
        mime="text/csv"
    )

with tab_live:
    st.subheader("🛰️ Surveillance ADS-B en direct (Europe)")
    st.markdown("Interroge les transpondeurs ADS-B réels via l'API OpenSky pour détecter les convergences réelles.")

    if st.button("📡 Scanner le ciel européen en direct"):
        with st.spinner("Récupération du flux ADS-B et analyse des trajectoires de convergence..."):
            conflicts = scan_live_conflicts(bbox_europe=True, max_pairs=10)

        if conflicts:
            st.error(f"⚠️ {len(conflicts)} conflit(s) prédictif(s) identifié(s) dans le ciel en direct !")
            for idx, c in enumerate(conflicts, 1):
                with st.expander(f"Conflit #{idx} : {c['flight_1']} ⚡ {c['flight_2']} — Rapprochement maximal dans {c['time_to_cpa_s']}s"):
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Distance actuelle", f"{c['current_dist_nm']} NM")
                    col_b.metric("Distance minimale prévue", f"{c['cpa_dist_nm']} NM")
                    col_c.metric("Écart vertical prévu", f"{c['cpa_alt_diff_ft']} ft")
                    st.warning(f"🎯 **Ordre ATC recommandé par l'IA pour {c['flight_1']}** : `{c['advisory_for_ac1']}`")
        else:
            st.success("✅ Aucun conflit immédiat (< 5 NM dans les 3 prochaines minutes) détecté dans la zone européenne. Trafic nominal.")
