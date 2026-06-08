"""QA evaluation metrics: EM, punctuation-preserving EM, Substring EM, F1."""

from evaluation.utils import exact_match, exact_match_keep_punc, substring_match, f1_score


def evaluate_qa(gold_list: list, pred_dict: dict, metrics: list = None) -> dict:
    """Evaluate QA predictions.

    Args:
        gold_list: list of gold dicts with 'gold_answers' and 'input_index'
        pred_dict: dict mapping input_index -> prediction string
        metrics:   list of metric names, e.g. ['em', 'em_test', 'f1', 'substring_em']

    Returns:
        dict of metric_name -> score
    """
    if metrics is None:
        metrics = ["em", "f1", "substring_em"]

    em_scores = []
    em_test_scores = []
    f1_scores = []
    sub_em_scores = []
    total = 0
    missing = 0

    for gold_item in gold_list:
        idx = gold_item.get("input_index", gold_item.get("id"))
        gold_ans = gold_item.get("gold_answers", gold_item.get("answer", ""))

        if isinstance(gold_ans, list):
            gold_ans_list = gold_ans
        else:
            gold_ans_list = [gold_ans]

        if idx not in pred_dict:
            missing += 1
            continue

        pred = pred_dict[idx]
        total += 1

        best_em = max(exact_match(pred, ga) for ga in gold_ans_list)
        best_em_test = max(exact_match_keep_punc(pred, ga) for ga in gold_ans_list)
        best_sub_em = max(substring_match(ga, pred) for ga in gold_ans_list)
        best_f1 = max(f1_score(pred, ga) for ga in gold_ans_list)

        em_scores.append(float(best_em))
        em_test_scores.append(float(best_em_test))
        sub_em_scores.append(float(best_sub_em))
        f1_scores.append(best_f1)

    results = {}
    if total == 0:
        return {m: 0.0 for m in metrics}

    if "em" in metrics:
        results["em"] = sum(em_scores) / total
    if "em_test" in metrics:
        results["em_test"] = sum(em_test_scores) / total
    if "substring_em" in metrics:
        results["substring_em"] = sum(sub_em_scores) / total
    if "f1" in metrics:
        results["f1"] = sum(f1_scores) / total
    results["total"] = total
    results["missing"] = missing

    return results
