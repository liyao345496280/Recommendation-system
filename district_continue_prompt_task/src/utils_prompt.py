from sklearn.metrics import roc_auc_score
import numpy as np
from tqdm.auto import tqdm


def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def evaluate(predicts, truths):
    aucs = []
    mrrs = []
    ndcg5s = []
    ndcg10s = []

    for pre, tru in zip(predicts, truths):
        if len(set(tru))==1:
            print("只有1类，我们忽略")
            continue
        y_true = np.array(tru, dtype='float32')
        y_score = 1.0 / np.array(pre, dtype='float32')
        auc = roc_auc_score(y_true, y_score)
        mrr = mrr_score(y_true, y_score)
        ndcg5 = ndcg_score(y_true, y_score, 5)
        ndcg10 = ndcg_score(y_true, y_score, 10)

        aucs.append(auc)
        mrrs.append(mrr)
        ndcg5s.append(ndcg5)
        ndcg10s.append(ndcg10)
    return np.mean(aucs), np.mean(mrrs), np.mean(ndcg5s), np.mean(ndcg10s)




def impressions(val_scores,val_impids,val_labels):
    world_size=1
    impressions = {}  # {1: {'score': [], 'lab': []}}
    for i in range(world_size):
        scores,imp_id, labs = val_scores[i], val_impids[i],val_labels[i]
        scores = scores.cpu().numpy().tolist()
        imp_id = imp_id.cpu().numpy().tolist()
        labs = labs.cpu().numpy().tolist()
        for j in range(len(scores)):
            sco, imp, lab = scores[j], imp_id[j], labs[j]
            if imp not in impressions:
                impressions[imp] = {'score': [], 'lab': []}
                impressions[imp]['score'].append(sco)
                impressions[imp]['lab'].append(lab)
            else:
                impressions[imp]['score'].append(sco)
                impressions[imp]['lab'].append(lab)

    predicts, truths = [], []
    for imp in impressions:
        sims, labs = impressions[imp]['score'], impressions[imp]['lab']
        sl_zip = sorted(zip(sims, labs), key=lambda x: x[0], reverse=True)
        sort_sims, sort_labs = zip(*sl_zip)
        predicts.append(list(range(1, len(sort_labs) + 1, 1)))
        truths.append(sort_labs)
    return predicts, truths

