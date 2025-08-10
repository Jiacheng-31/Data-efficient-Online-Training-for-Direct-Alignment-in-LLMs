#!/usr/bin/env python3
import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    parser = argparse.ArgumentParser(
        description="Convert a DPO policy checkpoint into a HuggingFace-compatible model directory"
    )
    parser.add_argument(
        "--base_model", required=True, type=str,
        help="Original model name or local path for the base LM (e.g., qwen2.5-0.5B)"
    )
    parser.add_argument(
        "--policy_ckpt", required=True, type=str,
        help="Path to the DPO policy checkpoint file (policy.pt)"
    )
    parser.add_argument(
        "--out_dir", required=True, type=str,
        help="Output directory where the converted model and tokenizer will be saved"
    )
    args = parser.parse_args()

    base_model = args.base_model
    ckpt_path = args.policy_ckpt
    out_dir = args.out_dir

    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading tokenizer and base model from '{base_model}'...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True)

    print(f"Loading DPO checkpoint from '{ckpt_path}'...")
    bundle = torch.load(ckpt_path, map_location="cpu")
    # Extract the actual state_dict
    if isinstance(bundle, dict) and "state" in bundle and isinstance(bundle["state"], dict):
        state_dict = bundle["state"]
    else:
        state_dict = bundle

    print("Applying state_dict to model...")
    model.load_state_dict(state_dict, strict=True)

    print(f"Saving converted model to '{out_dir}'...")
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    print("✅ Conversion complete. You can now load the model with `from_pretrained(out_dir)`.")

if __name__ == "__main__":
    main()
