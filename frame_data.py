"""Read the aligned FrameNews articles, human codes and encoder predictions."""

import pandas as pd


MODELS = ("multi_e5", "xlm_roberta")


PERSPECTIVES = ("primary_code", "primary_code1")


def contrast(codes):
    return (codes == 1).astype(float) - (codes == 4).astype(float)


def load_release(root):
    data, inputs = {}, []
    names = {
        "en": ("mfc_stratified_sample.csv", "mfc_predictions.csv", ["primary_code"]),
        "pt": ("frames_recsys_test.csv", "npr_predictions.csv", list(PERSPECTIVES)),
    }
    for language, (labels, predictions, columns) in names.items():
        human = pd.read_csv(root / labels, usecols=["prompt_id", *columns], dtype={"prompt_id": str})
        pred = pd.read_csv(root / predictions, index_col=0)
        if human.prompt_id.isna().any() or human.prompt_id.duplicated().any():
            raise ValueError(f"{labels}: missing or repeated article IDs")
        if len(human) != len(pred) or not pred.index.equals(pd.RangeIndex(len(human))):
            raise ValueError(f"{predictions}: source row alignment does not hold")
        for column in columns:
            if not human[column].isin(range(1, 16)).all():
                raise ValueError(f"{labels}: incomplete/invalid human codes in {column}")
        for model in MODELS:
            if not pred[model].isin(range(1, 16)).all():
                raise ValueError(f"{predictions}: incomplete/invalid predictions in {model}")
            human[model] = pred[model].to_numpy()
        data[language] = human
        inputs.append({"language": language, "labels": labels, "predictions": predictions,
                       "N": len(human), "human_fields": columns,
                       "alignment": "original row order, as in the source notebook; no independent prediction ID"})
    return data, inputs
