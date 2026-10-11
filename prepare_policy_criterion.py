"""Append one published primary-frame criterion without changing articles or answer options."""

import argparse
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE_URL = "https://aclanthology.org/2025.starsem-1.7.pdf"
SOURCE_PDF_SHA256 = "7756efcb1286043bae6a304cd8477aae988828a27f85ab7a1552c421536392ae"
RULE = (
    "Use Policy prescription and evaluation as the primary frame only when policy itself is the article's "
    "main emphasis. This includes what is or is not currently allowed, what should or should not be done, "
    "whether hypothetical or existing policies work, and proposals for policy solutions. A passing mention "
    "of a law, regulation, or political issue does not by itself make policy the primary frame."
)
ENDING = "\n\nReturn only the one letter for the primary frame."


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=HERE/"work/inputs/policy_compact")
    parser.add_argument("--source-pdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    assert sha(args.source_pdf.read_bytes()) == SOURCE_PDF_SHA256
    base_bytes = (args.base/"inputs.jsonl").read_bytes()
    rows = [json.loads(line) for line in base_bytes.splitlines()]
    pilot_bytes = (args.base/"pilot.jsonl").read_bytes()
    pilots = [json.loads(line) for line in pilot_bytes.splitlines()]
    pilot_ids = {row["id"] for row in pilots}
    expected = {f"{language}:{i}:o{order}" for language, size in (("en", 6214), ("pt", 300))
                for i in range(size) for order in (0, 1)}
    assert len(rows) == 13028 and {row["id"] for row in rows} == expected
    assert len(pilots) == len(pilot_ids) == 32 and pilot_ids < expected
    assert {row["id"]: row for row in rows if row["id"] in pilot_ids} == {row["id"]: row for row in pilots}
    output_rows, pilot_rows = [], []
    for row in rows:
        assert row["prompt"].endswith(ENDING)
        assert row["category_codes"] == (list(range(1, 16)) if row["order"] == 0 else list(range(15, 0, -1)))
        assert row["labels"] == list("ABCDEFGHIJKLMNO")
        updated = dict(row)
        updated["prompt"] = row["prompt"][:-len(ENDING)]+"\n\nPRIMARY-FRAME CRITERION:\n"+RULE+ENDING
        encoded = (json.dumps(updated, ensure_ascii=False)+"\n").encode()
        output_rows.append(encoded)
        if row["id"] in pilot_ids:
            pilot_rows.append(encoded)
    args.output.mkdir(parents=True)
    full = b"".join(output_rows)
    (args.output/"inputs.jsonl").write_bytes(full)
    (args.output/"pilot.jsonl").write_bytes(b"".join(pilot_rows))
    items = (args.base/"items.jsonl").read_bytes()
    (args.output/"items.jsonl").write_bytes(items)
    summary = {"rows": len(rows), "pilot_rows": len(pilots), "gold_written": False,
               "intervention": "one fixed source-derived primary-frame criterion; all original text/options/IDs/orders retained",
               "criterion": RULE, "criterion_version": "source_faithful_v2", "source_url": SOURCE_URL, "source_section": "Appendix A.6, Policy prescription and evaluation",
               "source_pdf_sha256": SOURCE_PDF_SHA256, "base_input_sha256": sha(base_bytes),
               "input_sha256": sha(full), "items_sha256": sha(items), "base_pilot_sha256": sha(pilot_bytes),
               "post_result": True, "complete": True}
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    print(json.dumps(summary, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
