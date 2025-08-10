#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sort completions by fine-grained_score and output statistics for sanity checks.
"""
import json
import argparse
import statistics

def transform_records(records):
    transformed = []
    for record in records:
        instruction = record.get("instruction", "")
        comps = record.get("completions", [])
        # extract desired fields
        items = []
        for comp in comps:
            items.append({
                "model": comp.get("model"),
                "fine-grained_score": comp.get("fine-grained_score"),
                "response": comp.get("response"),
            })
        # sort by score descending
        items_sorted = sorted(
            items,
            key=lambda x: x["fine-grained_score"] if x["fine-grained_score"] is not None else float('-inf'),
            reverse=True
        )
        transformed.append({
            "instruction": instruction,
            "completions": items_sorted
        })
    # if only one, return single object
    return transformed[0] if len(transformed) == 1 else transformed


def print_stats(transformed):
    # Determine if single record or list
    records = transformed if isinstance(transformed, list) else [transformed]
    total = len(records)
    comp_lengths = [len(r.get("completions", [])) for r in records]

    models_missing = 0
    scores_missing = 0
    responses_missing = 0
    for r in records:
        for comp in r.get("completions", []):
            if comp.get("model") is None:
                models_missing += 1
            if comp.get("fine-grained_score") is None:
                scores_missing += 1
            if comp.get("response") is None:
                responses_missing += 1

    print("=== Statistics ===")
    print(f"Total instructions processed: {total}")
    print(f"Completions per instruction: min={min(comp_lengths)}, max={max(comp_lengths)}, avg={statistics.mean(comp_lengths):.2f}")
    print(f"Total completions: {sum(comp_lengths)}")
    print(f"Missing fields: model={models_missing}, score={scores_missing}, response={responses_missing}")
    print("==================")


def main(input_path, output_path):
    # load JSON array
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    result = transform_records(data)

    # print statistics for sanity checks
    print_stats(result)

    # write output
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Transformed data saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sort completions by fine-grained_score and output stats")
    parser.add_argument("input_json", help="Path to input JSON file")
    parser.add_argument("output_json", help="Path to output JSON file")
    args = parser.parse_args()
    main(args.input_json, args.output_json)
