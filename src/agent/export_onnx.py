"""
Script d'export du modèle D3QN PyTorch vers le standard industriel ONNX.
Permet un déploiement embarqué / microservice ultra-rapide (< 1 ms d'inférence)
sans dépendre de PyTorch.
"""

import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.agent.dqn import D3QNAgent

def export_to_onnx(model_path: str = "models/d3qn_atc_latest.pt", onnx_output_path: str = "models/d3qn_atc.onnx"):
    agent = D3QNAgent(state_dim=10, action_dim=9, device="cpu")
    if not os.path.exists(model_path):
        print(f"Erreur : Modèle PyTorch {model_path} introuvable.")
        return False

    agent.load(model_path)
    model = agent.online_net
    model.eval()

    # Tenseur factice d'entrée (Batch size 1, 10 caractéristiques d'observation)
    dummy_input = torch.randn(1, 10, dtype=torch.float32)

    print(f"Export en cours vers {onnx_output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["observation_vector"],
        output_names=["q_values"],
        dynamic_axes={
            "observation_vector": {0: "batch_size"},
            "q_values": {0: "batch_size"}
        }
    )

    if os.path.exists(onnx_output_path):
        size_kb = os.path.getsize(onnx_output_path) / 1024.0
        print(f"✅ Modèle ONNX exporté avec succès ! Taille : {size_kb:.1f} KB")
        return True
    return False

if __name__ == "__main__":
    export_to_onnx()

