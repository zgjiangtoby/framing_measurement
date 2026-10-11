"""Prepare two answer orders and a label-blind technical sample for FrameNews."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CODE = Path(__file__).resolve().parent
FRAMES = [
    "Economic: financial costs, benefits, prices, or economic consequences.",
    "Capacity and resources: the availability or lack of resources and capacity to act.",
    "Morality: religious or ethical judgments of right and wrong.",
    "Fairness and equality: the distribution of rights, responsibilities, or resources.",
    "Legality, constitutionality, and jurisdiction: legal authority, rights, or constitutional issues.",
    "Policy prescription and evaluation: what policies should be adopted and whether they work.",
    "Crime and punishment: rule breaking, enforcement, sanctions, or punishment.",
    "Security and defense: threats, protection, defense, or national security.",
    "Health and safety: physical or mental health, safety, or health risks.",
    "Quality of life: everyday well-being, comfort, or the quality of community life.",
    "Cultural identity: traditions, identity, belonging, or shared cultural values.",
    "Public opinion: public attitudes, polling, or people's expressed views.",
    "Political: political competition, parties, officials, strategy, or political consequences.",
    "External regulation and reputation: international relations, external rules, or standing abroad.",
    "Other: a primary frame not covered by the other categories.",
]
LABELS = list("ABCDEFGHIJKLMNO")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(20261009)
    summary = {}
    with (args.output/"inputs.jsonl").open("x") as out, (args.output/"pilot.jsonl").open("x") as pilot, (args.output/"items.jsonl").open("x") as items:
        for language, name in [("en", "mfc_stratified_sample.csv"), ("pt", "frames_recsys_test.csv")]:
            frame = pd.read_csv(CODE/"data/framenews/data"/name)
            assert frame.prompt_id.is_unique and frame.text.is_unique and frame.text.notna().all()
            selected = set(rng.choice(len(frame), size=8, replace=False).tolist())
            truncated = 0
            for index, row in frame.iterrows():
                words = str(row.text).split()
                article = " ".join(words[:768])
                truncated += len(words) > 768
                item_id = f"{language}:{index}"
                item = {"item_id": item_id, "language": language, "source_index": int(index),
                        "prompt_id": str(row.prompt_id), "original_words": len(words),
                        "retained_words": min(768, len(words)), "technical_pilot": index in selected}
                items.write(json.dumps(item)+"\n")
                for order, codes in [(0,list(range(1,16))), (1,list(range(15,0,-1)))]:
                    options = "\n".join(f"{label}. {FRAMES[code-1]}" for label, code in zip(LABELS,codes))
                    prompt = ("Read the news article in its original language. Identify its PRIMARY policy frame: "
                              "the single perspective most emphasized by the article as a whole. Use the definitions below.\n\n"
                              +options+"\n\nARTICLE:\n"+article+
                              "\n\nReturn only the one letter for the primary frame.")
                    record = {"id": f"{item_id}:o{order}", "prompt": prompt, "labels": LABELS,
                              "item_id": item_id, "order": order, "category_codes": codes}
                    line = json.dumps(record, ensure_ascii=False)+"\n"
                    out.write(line)
                    if index in selected:
                        pilot.write(line)
            summary[language] = {"articles": len(frame), "words_truncated": truncated, "pilot_articles": len(selected)}
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
