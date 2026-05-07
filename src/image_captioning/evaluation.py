from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider


def evaluate(predictions, ref_map):
    gts = {k: v for k, v in ref_map.items() if k in predictions}
    res = {k: [predictions[k]] for k in gts}

    scores = {}

    bleu_scorer = Bleu(4)
    bleu_score, _ = bleu_scorer.compute_score(gts, res)
    scores["BLEU-1"] = round(bleu_score[0], 4)
    scores["BLEU-4"] = round(bleu_score[3], 4)

    meteor_scorer = Meteor()
    meteor_score, _ = meteor_scorer.compute_score(gts, res)
    scores["METEOR"] = round(meteor_score, 4)

    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(gts, res)
    scores["CIDEr"] = round(cider_score, 4)

    return scores
