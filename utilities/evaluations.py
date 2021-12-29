from utilities.data_factory import *
from sklearn.metrics import *


def evaluate_classifications(input_y_labels, input_predictions):
    classes = list(sorted(set(input_y_labels)))
    cm = confusion_matrix(input_y_labels, input_predictions, labels=classes)
    res_dict = {}
    for i in range(len(classes)):
        od = {'tp': cm[i][i], 'fp': cm.T[i].sum() - cm[i][i], 'fn': cm[i].sum() - cm[i][i]}
        od['tn'] = cm.sum() - sum(od.values())
        od['label_count'] = od['tp'] + od['fn']
        od['precision'] = od['tp'] / (od['tp'] + od['fp'])
        od['recall'] = od['tp'] / (od['tp'] + od['fn'])
        od['f1_score'] = 2 * od['precision'] * od['recall'] / (od['precision'] + od['recall'])
        res_dict[classes[i]] = od
    res_dict['summary'] = {
        'accuracy':            sum(input_y_labels == input_predictions) / len(input_y_labels),
        'macro_avg_recall':    sum([i['recall'] for i in res_dict.values()]) / len(classes),
        'macro_avg_precision': sum([i['precision'] for i in res_dict.values()]) / len(classes),
        'weighted_recall':     sum([i['recall'] * i['label_count'] for i in res_dict.values()]) / len(input_y_labels),
        'weighted_precision':  sum([i['precision'] * i['label_count'] for i in res_dict.values()]) / len(input_y_labels)
    }
    return res_dict


def evaluate_regressions(input_y_labels, input_predictions):
    res_dict = {
        'summary': {
            'r2': r2_score(input_y_labels, input_predictions),
            'explained_variance_score': explained_variance_score(input_y_labels, input_predictions),
            'mse': mean_squared_error(input_y_labels, input_predictions)
        }
    }
    return res_dict


def evaluate_regression_extreme_predictions(input_y_labels, input_predictions, input_rets):
    target_rets = input_rets
    up_ret, dn_ret = max(target_rets), min(target_rets)
    df = pd.DataFrame({'preds': input_predictions, 'labels': input_y_labels})
    up_df = df.loc[df['preds'] >= up_ret, :].reset_index(drop=True)
    dn_df = df.loc[df['preds'] <= dn_ret, :].reset_index(drop=True)
    res_dict = {
        'summary': {
            'up_size': up_df.shape[0],
            'dn_size': dn_df.shape[0],
            'up_correct_size': up_df.loc[up_df['labels'] > 0, :].shape[0],
            'dn_correct_size': dn_df.loc[dn_df['labels'] < 0, :].shape[0]
        }
    }
    res_dict['summary']['up_accuracy'] = res_dict['summary']['up_correct_size'] / max(1, res_dict['summary']['up_size'])
    res_dict['summary']['dn_accuracy'] = res_dict['summary']['dn_correct_size'] / max(1, res_dict['summary']['dn_size'])
    return res_dict


def evaluate_stability(input_y_labels, input_predictions, input_date_list, input_type='classification'):
    df = pd.DataFrame({'label': input_y_labels, 'pred': input_predictions, 'date': input_date_list})
    gb = df.groupby('date', as_index=False)

    def helper_daily_evaluate(one_df):
        if input_type == 'classification':
            res_dict = evaluate_classifications(input_y_labels=one_df['label'],
                                                input_predictions=one_df['pred'])
        else:
            res_dict = evaluate_regressions(input_y_labels=one_df['label'],
                                            input_predictions=one_df['pred'])
        return pd.Series(res_dict['summary'])

    res_df = gb.apply(helper_daily_evaluate)
    res_df = res_df.sort_values(by='date').reset_index(drop=True)
    return res_df


