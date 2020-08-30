def update_sp(metrics, sp_gold, sp_pred):
    assert len(sp_gold) == len(sp_pred)
    tp, fp, fn = 0, 0, 0
    
    for p in sp_pred:
        if p in sp_gold:
            tp += 1
        else:
            fp += 1
    for g in sp_gold:
        if g not in sp_pred:
            fn += 1
    
    precision = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    em = 1.0 if fp + fn == 0 else 0.0
    
    metrics['sp_em'] += em
    metrics['sp_f1'] += f1
    metrics['sp_prec'] += precision
    metrics['sp_recall'] += recall
    
    return precision, recall, f1


def eval_sp(indices_golds, indices_preds):
    metrics = {'sp_em': 0, 'sp_prec': 0, 'sp_recall': 0, 'sp_f1': 0}

    assert len(indices_golds) != 0
    assert len(indices_golds) == len(indices_preds)
    
    for sp_gold, sp_pred in zip(indices_golds, indices_preds):
        update_sp(metrics, sp_gold, sp_pred)
    
    N = len(indices_golds)
    for k in metrics.keys():
        metrics[k] /= N
        metrics[k] = round(metrics[k], 3)
    print(metrics)
    return metrics