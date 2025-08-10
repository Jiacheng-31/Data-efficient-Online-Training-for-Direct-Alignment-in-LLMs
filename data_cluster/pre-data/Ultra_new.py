#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate one chosen/rejected pair per record using:
- highest scored response as "chosen"
- lowest scored response as "rejected"
Skip if completions < 2 or if best and worst scores are equal.
"""
import json
import argparse

def generate_extreme_pair(input_json: str, output_jsonl: str):
    # Load JSON (supports single object or list)
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    records = data if isinstance(data, list) else [data]

    written = 0
    with open(output_jsonl, 'w', encoding='utf-8') as fout:
        for record in records:
            instruction = record.get("instruction", "")
            comps = record.get("completions", [])

            # Skip if <2 completions
            if len(comps) < 2:
                continue

            # Sort completions by score descending
            sorted_comps = sorted(
                comps,
                key=lambda x: x.get("fine-grained_score", float('-inf')),
                reverse=True
            )

            top = sorted_comps[0]
            bottom = sorted_comps[-1]

            # Skip if scores are equal
            if top.get("fine-grained_score") == bottom.get("fine-grained_score"):
                continue

            # Build prompt+response strings
            prefix = f"\n\nHuman: {instruction}"
            chosen_str = f"{prefix}\n\nAssistant: {top['response']}"
            rejected_str = f"{prefix}\n\nAssistant: {bottom['response']}"

            fout.write(json.dumps({
                "chosen": chosen_str,
                "rejected": rejected_str
            }, ensure_ascii=False) + "\n")

            written += 1

    print(f"Wrote {written} extreme-score pairs to {output_jsonl}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create one chosen/rejected pair per instruction from highest and lowest scored completions"
    )
    parser.add_argument("input_json", help="Path to input JSON file")
    parser.add_argument("output_jsonl", help="Path to output JSONL file for chosen/rejected pairs")
    args = parser.parse_args()
    generate_extreme_pair(args.input_json, args.output_jsonl)
