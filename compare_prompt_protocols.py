"""Compare compact and criterion prompts using the same sampled labels."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import norm

from analyze_budget import MODELS, PERSPECTIVES, rmse_comparison
from prepare_policy_criterion import RULE, SOURCE_PDF_SHA256


EXPECTED_MODELS = {"Qwen3-8B", "EuroLLM-1.7B-Instruct"}
MODEL_REVISIONS = {"Qwen3-8B": "b968826d9c46dd6066d109eabc6255188de91218",
                   "EuroLLM-1.7B-Instruct": "a25c7fa65fc2a644e6270b8940dbe295b51da681"}
MODEL_BATCHES = {"Qwen3-8B": 8, "EuroLLM-1.7B-Instruct": 32}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_runs(paths):
    result = {}
    for path in paths:
        run = json.loads((path/"run.json").read_text())
        assert run["status"] == "complete" and run["repetitions"] == 5000 and run["seed"] == 20261009
        model = run["model"]
        assert model in EXPECTED_MODELS and model not in result
        result[model] = (path, run)
    assert set(result) == EXPECTED_MODELS
    return result


def paired_comparison(new, old):
    assert new.shape == old.shape == (5000,)
    assert np.isfinite(new).all() and np.isfinite(old).all()
    result = rmse_comparison(new, old)
    squared = new**2-old**2
    result.update(paired_mse_difference=float(squared.mean()),
                  paired_mcse_mse_difference=float(squared.std(ddof=1)/np.sqrt(len(squared))),
                  criterion_rmse=float(np.sqrt(np.mean(new**2))),
                  compact_rmse=float(np.sqrt(np.mean(old**2))))
    critical = norm.ppf(1-.05/(2*4))
    result.update(simultaneous_mc_low=result["rmse_difference"]-critical*result["paired_mcse_rmse_difference"],
                  simultaneous_mc_high=result["rmse_difference"]+critical*result["paired_mcse_rmse_difference"])
    return result


def score_provenance_and_frequencies(run, inputs, require_hash):
    score_path = Path(run["source_scores"])
    metadata = json.loads(score_path.with_suffix(".metadata.json").read_text())
    assert metadata["status"] == "complete" and metadata["rows"] == 13028 and not metadata["gold_used"]
    assert metadata["dtype"] == "float32"
    assert run["model"] in EXPECTED_MODELS
    assert Path(metadata["model_path"]).name == MODEL_REVISIONS[run["model"]], "Unexpected model revision"
    assert metadata["batch_size"] == MODEL_BATCHES[run["model"]], "Unexpected batch size"
    assert Path(metadata["input"]).resolve() == (inputs/"inputs.jsonl").resolve()
    if require_hash:
        assert metadata["input_sha256"] == sha(inputs/"inputs.jsonl")
    elif "input_sha256" in metadata:
        assert metadata["input_sha256"] == sha(inputs/"inputs.jsonl")
    rows = [json.loads(line) for line in score_path.open() if line.strip()]
    by_id = {row["id"]: row for row in rows}
    expected = {f"{language}:{i}:o{order}" for language, size in (("en", 6214), ("pt", 300))
                for i in range(size) for order in (0, 1)}
    assert len(rows) == len(by_id) == len(expected) and set(by_id) == expected
    frequencies = []
    for language, size in (("en", 6214), ("pt", 300)):
        p = []
        for i in range(size):
            orders = []
            for order in (0, 1):
                row = by_id[f"{language}:{i}:o{order}"]
                assert row["labels"] == list("ABCDEFGHIJKLMNO")
                values = np.asarray(row["answer_log_probabilities"], dtype=float)
                assert values.shape == (15,) and np.isfinite(values).all()
                q = np.exp(values-logsumexp(values))
                orders.append(q if order == 0 else q[::-1])
            p.append(np.mean(orders, axis=0))
        p = np.asarray(p)
        hard = p.argmax(axis=1)
        for code in range(15):
            count = int((hard == code).sum())
            frequencies.append({"language": language, "category_code": code+1, "articles": size,
                "hard_primary_count": count, "hard_primary_share": count/size,
                "mean_soft_probability": float(p[:, code].mean())})
    return {"score_sha256": sha(score_path), "metadata_sha256": sha(score_path.with_suffix(".metadata.json")),
            "revision_from_metadata_path": Path(metadata["model_path"]).name, "batch_size": metadata["batch_size"],
            "input_hash_recorded": "input_sha256" in metadata}, frequencies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, action="append", required=True)
    parser.add_argument("--criterion", type=Path, action="append", required=True)
    parser.add_argument("--compact-inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    originals, criteria = read_runs(args.original), read_runs(args.criterion)
    contrasts, measurements, orders, frequencies, provenance = [], [], [], [], []
    for model in sorted(EXPECTED_MODELS):
        old_path, old_run = originals[model]
        new_path, new_run = criteria[model]
        assert old_run["source_inputs"] == new_run["source_inputs"]
        old_inputs = args.compact_inputs
        if "scoring_directory" in old_run:
            assert Path(old_run["scoring_directory"]).resolve() == old_inputs.resolve()
        new_inputs = Path(new_run["scoring_directory"])
        manifest = json.loads((new_inputs/"summary.json").read_text())
        assert manifest["post_result"] and manifest["complete"] and not manifest["gold_written"]
        assert manifest["criterion"] == RULE and manifest["source_pdf_sha256"] == SOURCE_PDF_SHA256
        assert manifest["criterion_version"] == "source_faithful_v2"
        assert manifest["base_input_sha256"] == sha(old_inputs/"inputs.jsonl")
        assert manifest["input_sha256"] == sha(new_inputs/"inputs.jsonl")
        assert manifest["items_sha256"] == sha(old_inputs/"items.jsonl") == sha(new_inputs/"items.jsonl")
        score_sources = {}
        for protocol, run, inputs in (("compact", old_run, old_inputs), ("criterion", new_run, new_inputs)):
            source, counts = score_provenance_and_frequencies(run, inputs, protocol == "criterion")
            score_sources[protocol] = source
            frequencies.extend({"checkpoint": model, "protocol": protocol, **row} for row in counts)
        with np.load(old_path/"private_sampling_indices.npz") as old, np.load(new_path/"private_sampling_indices.npz") as new:
            assert set(old.files) == set(new.files) == {"en", "pt"}
            for language in old.files:
                assert np.array_equal(old[language], new[language])
        with np.load(old_path/"private_mc_errors.npz") as old, np.load(new_path/"private_mc_errors.npz") as new:
            assert set(old.files) == set(new.files)
            controls = [key for key in old.files if key.endswith("_direct") or any(f"_{name}_" in key for name in MODELS)]
            assert controls
            for key in controls:
                assert np.array_equal(old[key], new[key]), (model, key, "unchanged control did not reproduce")
            for perspective in PERSPECTIVES:
                for budget in (25, 50, 100):
                    for representation in ("llm_soft", "llm_hard"):
                        key = f"{perspective}_{budget}_{representation}_crossfit_standardized"
                        contrasts.append({"checkpoint": model, "pt_field": perspective, "budget": budget,
                            "representation": representation, "primary_followup_family": budget == 50 and representation == "llm_soft",
                            **paired_comparison(new[key], old[key])})
        for protocol, directory in (("compact", old_path), ("criterion", new_path)):
            measurement = pd.read_csv(directory/"measurement.csv")
            assert len(measurement) == 16
            measurements.extend(measurement.assign(checkpoint=model, protocol=protocol).to_dict("records"))
            order = pd.read_csv(directory/"answer_order_diagnostics.csv")
            assert len(order) == 6514 and order.groupby("language").size().to_dict() == {"en": 6214, "pt": 300}
            assert not order.duplicated(["language", "source_index"]).any()
            for language, part in order.groupby("language"):
                orders.append({"checkpoint": model, "protocol": protocol, "language": language,
                    **{key: float(part[key].mean()) for key in
                       ("order_total_variation", "order_top_disagreement", "answer_probability_mass")}})
        provenance.append({"model": model, "original": str(old_path), "criterion": str(new_path),
            "original_run_sha256": sha(old_path/"run.json"), "criterion_run_sha256": sha(new_path/"run.json"),
            "compact_prompt_directory": str(old_inputs),
            "compact_directory_recorded_in_original_run": "scoring_directory" in old_run,
            "score_sources": score_sources,
            "sampling_indices_exactly_equal": True, "unchanged_control_errors_exactly_equal": True,
            "unchanged_control_arrays": len(controls)})
    primary = [row for row in contrasts if row["primary_followup_family"]]
    assert len(primary) == 4 and len(contrasts) == 24
    args.output.mkdir(parents=True)
    pd.DataFrame(contrasts).to_csv(args.output/"comparisons.csv", index=False)
    pd.DataFrame(measurements).to_csv(args.output/"measurement.csv", index=False)
    pd.DataFrame(orders).to_csv(args.output/"answer_order.csv", index=False)
    pd.DataFrame(frequencies).to_csv(args.output/"primary_frame_frequencies.csv", index=False)
    (args.output/"summary.json").write_text(json.dumps({"complete": True, "post_result_sensitivity": True,
        "primary_family_size": 4, "primary_followup_comparisons": primary, "repetitions": 5000,
        "scope": "fixed-corpus Monte Carlo integration uncertainty; no new-population or post-selection guarantee",
        "original_primary_family_replaced": False, "provenance": provenance}, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"complete": True, "comparisons": len(contrasts), "primary_family_size": 4}), flush=True)


if __name__ == "__main__":
    main()
