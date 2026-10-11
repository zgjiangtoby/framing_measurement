"""Remove the task-heading policy word from both prompt conditions."""

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
from prepare_policy_criterion import ENDING, RULE, SOURCE_PDF_SHA256

OLD_HEAD = ("Read the news article in its original language. Identify its PRIMARY policy frame: "
            "the single perspective most emphasized by the article as a whole. Use the definitions below.")
NEW_HEAD = OLD_HEAD.replace("PRIMARY policy frame", "PRIMARY frame")
RULE_INSERT = "\n\nPRIMARY-FRAME CRITERION:\n" + RULE
CONDITIONS = ("policy_compact", "policy_criterion", "neutral_compact", "neutral_criterion")
EXPECTED_IDS = {f"{language}:{i}:o{order}" for language, size in (("en", 6214), ("pt", 300))
                for i in range(size) for order in (0, 1)}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_bytes().splitlines() if line.strip()]


def without_task_word(row):
    assert row["prompt"].startswith(OLD_HEAD + "\n\n")
    result = dict(row)
    result["prompt"] = NEW_HEAD + row["prompt"][len(OLD_HEAD):]
    assert len(row["prompt"]) - len(result["prompt"]) == len("policy ")
    return result


def read_pair(compact, criterion):
    rows = {"policy_compact": read_rows(compact/"inputs.jsonl"),
            "policy_criterion": read_rows(criterion/"inputs.jsonl")}
    pilot_ids = []
    for condition, directory in (("policy_compact", compact), ("policy_criterion", criterion)):
        full = rows[condition]
        assert len(full) == len({r["id"] for r in full}) == 13028
        assert {r["id"] for r in full} == EXPECTED_IDS
        pilots = read_rows(directory/"pilot.jsonl")
        ids = [r["id"] for r in pilots]
        assert len(ids) == len(set(ids)) == 32
        assert [r for r in full if r["id"] in set(ids)] == pilots
        pilot_ids.append(ids)
        for row in full:
            assert set(row) == {"id", "prompt", "labels", "item_id", "order", "category_codes"}
            assert row["labels"] == list("ABCDEFGHIJKLMNO")
            assert row["order"] in (0, 1)
            assert row["category_codes"] == (list(range(1, 16)) if row["order"] == 0 else list(range(15, 0, -1)))
            assert row["id"] == f'{row["item_id"]}:o{row["order"]}'
            assert row["prompt"].startswith(OLD_HEAD+"\n\n") and row["prompt"].endswith(ENDING)
    assert pilot_ids[0] == pilot_ids[1]
    assert (compact/"items.jsonl").read_bytes() == (criterion/"items.jsonl").read_bytes()
    items = read_rows(compact/"items.jsonl")
    assert len(items) == 6514
    for old, rule in zip(rows["policy_compact"], rows["policy_criterion"]):
        expected = dict(old)
        expected["prompt"] = old["prompt"][:-len(ENDING)] + RULE_INSERT + ENDING
        assert expected == rule
    manifest = json.loads((criterion/"summary.json").read_text())
    assert manifest["criterion_version"] == "source_faithful_v2" and manifest["criterion"] == RULE
    assert manifest["source_pdf_sha256"] == SOURCE_PDF_SHA256
    assert manifest["base_input_sha256"] == sha(compact/"inputs.jsonl")
    assert manifest["input_sha256"] == sha(criterion/"inputs.jsonl")
    return rows, pilot_ids[0]


def validate_four(compact, criterion, neutral_compact, neutral_criterion):
    rows, pilots = read_pair(compact, criterion)
    item_bytes = (compact/"items.jsonl").read_bytes()
    for old, new, directory in (("policy_compact", "neutral_compact", neutral_compact),
                                ("policy_criterion", "neutral_criterion", neutral_criterion)):
        updated = read_rows(directory/"inputs.jsonl")
        assert updated == [without_task_word(row) for row in rows[old]]
        assert read_rows(directory/"pilot.jsonl") == [r for r in updated if r["id"] in set(pilots)]
        assert (directory/"items.jsonl").read_bytes() == item_bytes
        manifest = json.loads((directory/"summary.json").read_text())
        assert manifest["complete"] and manifest["post_result"] and not manifest["gold_written"]
        assert manifest["condition"] == new and manifest["intervention"] == "delete_only_task_head_policy_word"
        assert manifest["input_sha256"] == sha(directory/"inputs.jsonl")
        assert manifest["base_input_sha256"] == sha((compact if old == "policy_compact" else criterion)/"inputs.jsonl")
        assert manifest["items_sha256"] == sha(compact/"items.jsonl")
        rows[new] = updated
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact", type=Path, required=True)
    parser.add_argument("--criterion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not any((args.output/name).exists() for name in ("neutral_compact", "neutral_criterion"))
    rows, pilot_ids = read_pair(args.compact, args.criterion)
    args.output.mkdir(parents=True, exist_ok=True)
    for old, new, base in (("policy_compact", "neutral_compact", args.compact),
                           ("policy_criterion", "neutral_criterion", args.criterion)):
        directory = args.output/new
        directory.mkdir()
        updated = [without_task_word(row) for row in rows[old]]
        for name, selected in (("inputs", updated), ("pilot", [r for r in updated if r["id"] in set(pilot_ids)])):
            (directory/f"{name}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in selected))
        (directory/"items.jsonl").write_bytes((base/"items.jsonl").read_bytes())
        manifest = {"complete": True, "post_result": True, "gold_written": False,
                    "condition": new, "base_condition": old, "rows": 13028, "pilot_rows": 32,
                    "intervention": "delete_only_task_head_policy_word", "old_head": OLD_HEAD, "new_head": NEW_HEAD,
                    "base_input_sha256": sha(base/"inputs.jsonl"), "input_sha256": sha(directory/"inputs.jsonl"),
                    "items_sha256": sha(directory/"items.jsonl"), "source_pdf_sha256": SOURCE_PDF_SHA256}
        (directory/"summary.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
    validate_four(args.compact, args.criterion, args.output/"neutral_compact", args.output/"neutral_criterion")
    print(json.dumps({"complete": True, "new_conditions": 2, "rows_per_condition": 13028,
                      "pilot_rows_per_condition": 32, "gold_written": False}), flush=True)


if __name__ == "__main__":
    main()
