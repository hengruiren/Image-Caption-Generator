import numpy as np


def evaluate_predictions(predictions):
    try:
        import evaluate

        bleu = evaluate.load("bleu")
        meteor = evaluate.load("meteor")
        pred_texts = [x["prediction"] for x in predictions]
        ref_texts = [x["references"] for x in predictions]
        bleu_score = bleu.compute(predictions=pred_texts, references=ref_texts, max_order=4)["bleu"]
        meteor_score = meteor.compute(predictions=pred_texts, references=ref_texts)["meteor"]
    except Exception as exc:
        print("BLEU/METEOR evaluation failed:", repr(exc))
        bleu_score = np.nan
        meteor_score = np.nan

    cider_score = np.nan
    try:
        from pycocoevalcap.cider.cider import Cider

        gts = {i: x["references"] for i, x in enumerate(predictions)}
        res = {i: [x["prediction"]] for i, x in enumerate(predictions)}
        cider_score, _ = Cider().compute_score(gts, res)
    except Exception as exc:
        print("CIDEr evaluation skipped or failed:", repr(exc))

    return {"BLEU-4": bleu_score, "METEOR": meteor_score, "CIDEr": cider_score}

