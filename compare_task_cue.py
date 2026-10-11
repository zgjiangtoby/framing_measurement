"""Compare task wording and criterion conditions across both checkpoints."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
from analyze_budget import MODELS, PERSPECTIVES, estimates, load_release, make_design, read_probabilities
from compare_prompt_protocols import EXPECTED_MODELS, score_provenance_and_frequencies, sha
from prepare_task_cue import CONDITIONS, validate_four

FAMILY_SIZE = 44
REPETITIONS = 5000
METHOD = "crossfit_standardized"
PROMPT_CONTRASTS = {
    "remove_task_word_without_criterion": {"neutral_compact": 1., "policy_compact": -1.},
    "remove_task_word_with_criterion": {"neutral_criterion": 1., "policy_criterion": -1.},
    "add_criterion_with_task_word": {"policy_criterion": 1., "policy_compact": -1.},
    "add_criterion_without_task_word": {"neutral_criterion": 1., "neutral_compact": -1.},
    "criterion_by_task_word_interaction": {"neutral_criterion": 1., "neutral_compact": -1.,
                                            "policy_criterion": -1., "policy_compact": 1.},
}


def rmse_contrast(errors, weights):
    """Delta method for a fixed linear contrast of paired RMSE estimates."""
    assert set(errors) == set(weights) and len(errors) in (2, 4)
    assert all(np.asarray(value).shape == (REPETITIONS,) and np.isfinite(value).all()
               for value in errors.values())
    assert all(np.isfinite(value) for value in weights.values())
    assert abs(sum(weights.values())) < 1e-12
    mse = {key: float(np.mean(value**2)) for key, value in errors.items()}
    rmse = {key: float(np.sqrt(value)) for key, value in mse.items()}
    assert all(value > 0 for value in rmse.values()), "zero RMSE makes the delta-method derivative undefined"
    influence = sum(weights[key]*(errors[key]**2-mse[key])/(2*rmse[key]) for key in errors)
    squared_contrast = sum(weights[key]*errors[key]**2 for key in errors)
    estimate = sum(weights[key]*rmse[key] for key in errors)
    se = float(np.std(influence, ddof=1)/np.sqrt(REPETITIONS))
    return {"rmse_contrast": float(estimate), "paired_mcse_rmse_contrast": se,
            "mse_contrast": float(np.mean(squared_contrast)),
            "paired_mcse_mse_contrast": float(np.std(squared_contrast, ddof=1)/np.sqrt(REPETITIONS)),
            "component_rmse_json": json.dumps(rmse, sort_keys=True),
            "weights_json": json.dumps(weights, sort_keys=True)}


def budget_row(model, target, budget, name, errors, weights, kind):
    result = rmse_contrast(errors, weights)
    primary = budget == 50
    critical = float(norm.ppf(1-.05/(2*FAMILY_SIZE)))
    result.update(checkpoint=model, pt_field=target, budget=budget, contrast=name,
                  kind=kind, method=METHOD, primary_family=primary,
                  primary_family_size=FAMILY_SIZE if primary else None,
                  simultaneous_mc_low=(result["rmse_contrast"]-critical*result["paired_mcse_rmse_contrast"]) if primary else None,
                  simultaneous_mc_high=(result["rmse_contrast"]+critical*result["paired_mcse_rmse_contrast"]) if primary else None)
    return result


def measurement_rows(probabilities, diagnostics, data):
    """Exact finite-release summaries, no article-sampling confidence intervals."""
    metrics, confusion = [], []
    for target, language, field in (("EN", "en", "primary_code"),
                                    ("PT_J1", "pt", PERSPECTIVES[0]),
                                    ("PT_J2", "pt", PERSPECTIVES[1])):
        p = probabilities[language]
        gold = data[language][field].to_numpy(dtype=int)-1
        hard = p.argmax(axis=1)
        counts = np.zeros((15, 15), dtype=np.int64)
        np.add.at(counts, (gold, hard), 1)
        true_support, predicted_support = counts.sum(axis=1), counts.sum(axis=0)
        diagonal = np.diag(counts)
        denominator = true_support+predicted_support
        f1 = np.divide(2*diagonal, denominator, out=np.zeros(15, dtype=float), where=denominator > 0)
        recall = np.divide(diagonal, true_support, out=np.zeros(15, dtype=float), where=true_support > 0)
        q = p[:, 0]-p[:, 3]
        y = (gold == 0).astype(float)-(gold == 3).astype(float)
        covariance = float(np.cov(y, q, ddof=1)[0, 1])
        y_variance, q_variance = float(y.var(ddof=1)), float(q.var(ddof=1))
        order = diagnostics[diagnostics.language == language]
        row = {"target": target, "language": language, "gold_field": field, "articles": len(gold),
               "accuracy": float(diagonal.sum()/len(gold)), "macro_f1_15_classes": float(f1.mean()),
               "macro_recall_gold_supported": float(recall[true_support > 0].mean()),
               "gold_supported_classes": int((true_support > 0).sum()),
               "majority_reference_accuracy": float(true_support.max()/len(gold)),
               "brier": float(((p-np.eye(15)[gold])**2).sum(axis=1).mean()),
               "gold_nll": float(-np.log(np.maximum(p[np.arange(len(gold)), gold], 1e-12)).mean()),
               "nll_floor": 1e-12, "signed_gold_variance": y_variance, "signed_q_variance": q_variance,
               "signed_covariance": covariance,
               "signed_correlation": covariance/np.sqrt(y_variance*q_variance) if y_variance*q_variance > 0 else None,
               "signed_mean_error": float(q.mean()-y.mean()),
               "policy_hard_share": float((hard == 5).mean()), "policy_mean_probability": float(p[:, 5].mean()),
               "gold_policy_share": float((gold == 5).mean()),
               **{key: float(order[key].mean()) for key in
                  ("order_total_variation", "order_top_disagreement", "answer_probability_mass")}}
        metrics.append(row)
        confusion.extend({"target": target, "language": language, "gold_category_code": i+1,
                          "predicted_category_code": j+1, "count": int(counts[i, j])}
                         for i in range(15) for j in range(15))
    return metrics, confusion


def verify_error_archive(probabilities, data, samples, archived_errors):
    """Reconstruct paired Monte Carlo arrays from this condition's score file.

    Use the same deterministic samples and estimators, then compare every saved array.
    """
    names = ["llm_soft", "llm_hard", *MODELS]
    auxiliaries = {}
    for language, frame in data.items():
        p = probabilities[language]
        predictions = {"llm_soft": p, "llm_hard": np.eye(15)[p.argmax(axis=1)],
                       **{name: np.eye(15)[frame[name].to_numpy(dtype=int)-1] for name in MODELS}}
        auxiliaries[language] = np.stack([predictions[name][:, 0]-predictions[name][:, 3]
                                          for name in names], axis=1)
    rng = np.random.default_rng(20261009)
    designs = {language: make_design(auxiliaries[language], REPETITIONS, rng) for language in data}
    assert set(samples) == set(designs)
    assert all(np.array_equal(samples[language], designs[language]["samples"]) for language in data), \
        "MC sample archive does not reproduce the frozen source-order design"
    methods = ("direct", "fixed_one", "pilot_fifth", "crossfit", "crossfit_standardized")
    expected_keys = {f"{target}_{budget}_{name}_{method}" for target in PERSPECTIVES
                     for budget in (25, 50, 100) for name in names for method in methods}
    assert set(archived_errors) == expected_keys
    maximum = 0.
    for target in PERSPECTIVES:
        signed_gold = {}
        for language, frame in data.items():
            labels = frame["primary_code" if language == "en" else target].to_numpy(dtype=int)
            signed_gold[language] = (labels == 1).astype(float)-(labels == 4).astype(float)
        truth = signed_gold["en"].mean()-signed_gold["pt"].mean()
        for budget in (25, 50, 100):
            for column, name in enumerate(names):
                current = {language: estimates(signed_gold[language], auxiliaries[language][:, column],
                                               designs[language], column, budget)[0] for language in data}
                for method in methods:
                    key = f"{target}_{budget}_{name}_{method}"
                    expected = current["en"][method]-current["pt"][method]-truth
                    observed = archived_errors[key]
                    assert observed.shape == expected.shape == (REPETITIONS,)
                    assert np.isfinite(observed).all() and np.isfinite(expected).all()
                    difference = float(np.max(np.abs(observed-expected)))
                    assert difference < 1e-10, ("MC errors do not reproduce their source scores", key, difference)
                    maximum = max(maximum, difference)
    return {"mc_error_arrays_reproduced_from_scores": len(expected_keys),
            "mc_error_max_abs_reproduction_difference": maximum,
            "sampling_indices_reproduced_from_seed_and_source_order": True,
            "archive_reproduction_tolerance": 1e-10}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=HERE/"work")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    root = args.work_dir.resolve()
    registry = {
        "inputs": {condition: str(root/"inputs"/condition) for condition in CONDITIONS},
        "analyses": [{"checkpoint": model, "condition": condition,
                      "analysis": str(root/"analysis"/(tag+"_"+condition))}
                     for model, tag in (("Qwen3-8B", "qwen"), ("EuroLLM-1.7B-Instruct", "euro"))
                     for condition in CONDITIONS],
    }
    directories = {key: Path(registry["inputs"][key]) for key in CONDITIONS}
    assert set(registry["inputs"]) == set(CONDITIONS)
    validate_four(*(directories[key] for key in CONDITIONS))
    entries = {(entry["checkpoint"], entry["condition"]): entry for entry in registry["analyses"]}
    expected = {(model, condition) for model in EXPECTED_MODELS for condition in CONDITIONS}
    assert len(registry["analyses"]) == len(entries) == 8 and set(entries) == expected
    data, sources = load_release(HERE/"data/framenews/data")
    budgets, measurements, confusion, frequencies, provenance, exact_contrasts = [], [], [], [], [], []
    errors, measurement_lookup = {}, {}
    sample_reference, control_reference = None, None
    for model in sorted(EXPECTED_MODELS):
        for condition in CONDITIONS:
            entry = entries[(model, condition)]
            directory, inputs = Path(entry["analysis"]), directories[condition]
            run = json.loads((directory/"run.json").read_text())
            assert run["status"] == "complete" and run["model"] == model
            assert run["seed"] == 20261009 and run["repetitions"] == REPETITIONS
            assert run["source_inputs"] == sources
            if "scoring_directory" in run:
                assert Path(run["scoring_directory"]).resolve() == inputs.resolve()
            else:
                assert condition == "policy_compact", "only the original compact archive may lack this field"
            is_new = condition.startswith("neutral_")
            source, freq = score_provenance_and_frequencies(run, inputs, require_hash=is_new or condition == "policy_criterion")
            metadata = json.loads(Path(run["source_scores"]).with_suffix(".metadata.json").read_text())
            batch_difference = metadata["batch_probability_max_abs_difference"]
            assert np.isfinite(batch_difference) and 0 <= batch_difference < 1e-4
            full = metadata.get("full_forward_check")
            if full is None:
                assert model == "EuroLLM-1.7B-Instruct" and condition == "policy_compact"
            else:
                assert full["passed"] and full["tolerance"] == 1e-4
                difference = full["full_vocabulary_max_abs_logit_difference"]
                assert np.isfinite(difference) and 0 <= difference < 1e-4
                assert len(full["source_row_indices"]) > 0
            p, order = read_probabilities(Path(run["source_scores"]), data, inputs)
            rows, conf = measurement_rows(p, order, data)
            archived_measurement = pd.read_csv(directory/"measurement.csv")
            for row in rows:
                field = PERSPECTIVES[0] if row["target"] == "EN" else row["gold_field"]
                archived = archived_measurement[(archived_measurement.model == "llm_soft") &
                    (archived_measurement.language == row["language"]) & (archived_measurement.pt_field == field)]
                assert len(archived) == 1
                for current, old in (("accuracy", "accuracy"), ("brier", "brier"), ("gold_nll", "gold_nll"),
                                     ("signed_covariance", "z_q_covariance"), ("signed_mean_error", "raw_contrast_error")):
                    assert abs(row[current]-float(archived.iloc[0][old])) < 1e-10, (model, condition, current)
                measurement_lookup[(model, condition, row["target"])] = row
            measurements.extend({"checkpoint": model, "condition": condition, **row} for row in rows)
            confusion.extend({"checkpoint": model, "condition": condition, **row} for row in conf)
            frequencies.extend({"checkpoint": model, "condition": condition, **row} for row in freq)
            with np.load(directory/"private_sampling_indices.npz") as archive:
                current = {key: archive[key] for key in archive.files}
            assert set(current) == {"en", "pt"}
            assert all(value.shape == (REPETITIONS, 2, 50) for value in current.values())
            if sample_reference is None:
                sample_reference = current
            else:
                assert all(np.array_equal(current[key], sample_reference[key]) for key in current)
            with np.load(directory/"private_mc_errors.npz") as archive:
                e = {key: archive[key] for key in archive.files}
            assert all(value.shape == (REPETITIONS,) and np.isfinite(value).all() for value in e.values())
            reproduced = verify_error_archive(p, data, current, e)
            controls = {key: value for key, value in e.items()
                        if key.endswith("_direct") or any(f"_{name}_" in key for name in MODELS)}
            assert controls
            if control_reference is None:
                control_reference = controls
            else:
                assert set(controls) == set(control_reference)
                assert all(np.array_equal(value, control_reference[key]) for key, value in controls.items())
            errors[(model, condition)] = e
            provenance.append({"checkpoint": model, "condition": condition, "analysis": str(directory),
                "run_sha256": sha(directory/"run.json"), "current_input_sha256": sha(inputs/"inputs.jsonl"),
                "scoring_directory_recorded_in_original_run": "scoring_directory" in run,
                "full_forward_recorded_in_original_metadata": full is not None,
                "sampling_indices_exactly_equal": True, "control_error_arrays_exactly_equal": True,
                "control_arrays": len(controls), **reproduced, **source})
    assert set(MODELS) == {"multi_e5", "xlm_roberta"}
    for model in sorted(EXPECTED_MODELS):
        for target in PERSPECTIVES:
            for budget in (25, 50, 100):
                soft_key = f"{target}_{budget}_llm_soft_{METHOD}"
                for name, weights in PROMPT_CONTRASTS.items():
                    arrays = {condition: errors[(model, condition)][soft_key] for condition in weights}
                    budgets.append(budget_row(model, target, budget, name, arrays, weights, "prompt"))
                for condition in ("neutral_compact", "neutral_criterion"):
                    for baseline in ("llm_hard", *MODELS):
                        arrays = {"soft": errors[(model, condition)][soft_key],
                                  baseline: errors[(model, condition)][f"{target}_{budget}_{baseline}_{METHOD}"]}
                        budgets.append(budget_row(model, target, budget, f"{condition}:soft_minus_{baseline}",
                                                  arrays, {"soft": 1., baseline: -1.}, "matched_representation"))
        for target in ("EN", "PT_J1", "PT_J2"):
            for name, weights in PROMPT_CONTRASTS.items():
                row = {"checkpoint": model, "target": target, "contrast": name,
                       "scope": "exact finite-release difference; no article-sampling uncertainty"}
                for metric in ("accuracy", "macro_f1_15_classes", "macro_recall_gold_supported", "brier", "gold_nll",
                               "signed_q_variance", "signed_covariance", "signed_correlation", "signed_mean_error",
                               "policy_hard_share", "policy_mean_probability", "order_total_variation",
                               "order_top_disagreement", "answer_probability_mass"):
                    values = [measurement_lookup[(model, condition, target)][metric] for condition in weights]
                    row[metric] = (sum(weight*value for weight, value in zip(weights.values(), values))
                                   if all(value is not None for value in values) else None)
                exact_contrasts.append(row)
    primary = [row for row in budgets if row["primary_family"]]
    assert len(primary) == FAMILY_SIZE and len(budgets) == 3*FAMILY_SIZE
    assert len({(row["checkpoint"], row["pt_field"], row["contrast"]) for row in primary}) == FAMILY_SIZE
    assert (len(measurements), len(confusion), len(frequencies), len(exact_contrasts)) == (24, 5400, 240, 30)
    args.output.mkdir(parents=True)
    for name, rows in (("budget_comparisons", budgets), ("measurement", measurements), ("confusion_counts", confusion),
                       ("frame_frequencies", frequencies), ("measurement_contrasts", exact_contrasts)):
        pd.DataFrame(rows).to_csv(args.output/f"{name}.csv", index=False)
    summary = {"complete": True, "post_result": True, "registry": registry,
               "repetitions": REPETITIONS, "primary_family_size": FAMILY_SIZE, "primary_budget": 50,
               "primary_comparisons": primary, "uncertainty": "paired delta-method Monte Carlo integration only",
               "simultaneous_method": "Bonferroni normal intervals over all 44 prespecified n=50 contrasts",
               "old_primary_families_replaced": False, "all_eight_conditions_retained": True,
               "historical_label_assignments_seen_before_design": 6814,
               "historical_unique_articles_seen_before_design": 6514,
               "new_human_labels": 0, "provenance": provenance}
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"complete": True, "primary_family_size": FAMILY_SIZE, "budget_rows": len(budgets),
                      "measurement_rows": len(measurements), "confusion_rows": len(confusion)}), flush=True)


if __name__ == "__main__":
    main()
