import sys
sys.path.append('..')
from utilities.evaluations import *
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker


def analysis_pipeline(input_run_prefix,
                      input_file_name,
                      input_job_name,
                      input_model,
                      input_label_col,
                      input_up_pct=85,
                      input_dn_pct=15):
    kwargs = {'input_run_prefix': input_run_prefix,
              'input_file_name': input_file_name,
              'input_job_name': input_job_name,
              'input_model': input_model,
              'input_label_col': input_label_col}
    if 'Regression' in input_model:
        log_info('Working on regression problem')
        if 'static' in input_job_name:
            log_info('Working on static training problem')
            analysis_func = runner_reg_static_res_analysis_helper
            plot_func = runner_reg_static_res_plotter
        else:
            log_info('Working on rolling training problem')
            analysis_func = runner_reg_rolling_res_analysis_helper
            kwargs['input_up_pct'] = input_up_pct
            kwargs['input_dn_pct'] = input_dn_pct
            plot_func = runner_reg_rolling_res_plotter
    else:
        log_info('Working on classification problem')
        if 'static' in input_job_name:
            log_info('Working on static training problem')
            analysis_func = runner_cls_static_res_analysis_helper
            plot_func = runner_cls_static_res_plotter
        else:
            log_info('Working on rolling training problem')
            analysis_func = runner_cls_rolling_res_analysis_helper
            plot_func = runner_cls_rolling_res_plotter

    res = analysis_func(**kwargs)
    kwargs['input_res'] = res
    if 'input_up_pct' in kwargs:
        kwargs.pop('input_up_pct')
        kwargs.pop('input_dn_pct')
    plot_func(**kwargs)
    return None


def runner_cls_static_res_analysis_helper(input_run_prefix,
                                          input_file_name,
                                          input_job_name,
                                          input_model,
                                          input_label_col):
    log_info(f'Running runner_cls_static_res_analysis_helper on '
             f'{input_run_prefix} - {input_file_name} - {input_job_name} - {input_model} - {input_label_col}')
    target_pkl = get_res_pkl_name(input_file_name, input_run_prefix, input_model, input_job_name)
    one_res_dict = read_pkl_helper(os.path.join(RUN_RESULT_DIR, target_pkl))
    one_dict = one_res_dict['job_result'][input_label_col]
    res_dict = {'accuracy': one_dict['summary']['accuracy'],
                'recall': one_dict['summary']['weighted_recall'],
                'precision': one_dict['summary']['weighted_precision'],
                'train_stab': one_dict['train_stab'],
                'test_stab': one_dict['test_stab']}
    return res_dict


def runner_cls_static_res_plotter(input_run_prefix,
                                  input_file_name,
                                  input_model,
                                  input_job_name,
                                  input_label_col,
                                  input_res):
    """ Placeholder, no specific graphs to be plotted for static classification problems"""
    log_info('No graphs to be plotted for static classification problem')
    return None


def runner_cls_rolling_res_analysis_helper(input_run_prefix,
                                           input_file_name,
                                           input_job_name,
                                           input_model,
                                           input_label_col):
    log_info(f'Running runner_cls_rolling_res_analysis_helper on '
             f'{input_run_prefix} - {input_file_name} - {input_job_name} - {input_model} - {input_label_col}')
    target_pkl = get_res_pkl_name(input_file_name, input_run_prefix, input_model, input_job_name)
    one_res_dict = read_pkl_helper(os.path.join(RUN_RESULT_DIR, target_pkl))
    one_dict = one_res_dict['job_result']
    all_dates = sorted(list(one_dict.keys()))
    accuracy_list = [one_dict[i][input_label_col]['train']['summary']['accuracy'] for i in all_dates]
    recall_list = [one_dict[i][input_label_col]['test']['summary']['recall'] for i in all_dates]
    precision_list = [one_dict[i][input_label_col]['test']['summary']['precision'] for i in all_dates]

    # R2 of each day aggregation
    accuracy_df = pd.DataFrame({'date': all_dates, 'r2': accuracy_list})
    recall_df = pd.DataFrame({'date': all_dates, 'r2': recall_list})
    precision_df = pd.DataFrame({'date': all_dates, 'r2': precision_list})

    return {
        'accuracy_df': accuracy_df,
        'recall_df': recall_df,
        'precision_df': precision_df,
        'price_df': one_res_dict['price_df']
    }


def runner_cls_rolling_res_plotter(input_run_prefix,
                                   input_file_name,
                                   input_model,
                                   input_job_name,
                                   input_label_col,
                                   input_res):
    fig, axs = plt.subplots(4, 1, figsize=(30, 25))
    target_savefig_name = get_res_pkl_name(input_file_name, input_run_prefix, input_model, input_job_name).replace('pkl', 'png')

    # Price plot
    axs[0].plot(input_res['price_df']['date'], input_res['price_df']['price'], color='blue')
    axs[0].set_title(f'{input_file_name} - {input_model} - {input_job_name} - {input_label_col} - Price', fontsize=18)
    axs[0].grid(True)
    locator = matplotlib.ticker.MaxNLocator(prune='both', nbins=5)
    axs[0].xaxis.set_major_locator(locator)

    # Recall plot
    axs[1].plot(input_res['recall_df']['date'], input_res['recall_df']['price'], color='blue')
    axs[1].set_title(f'{input_file_name} - {input_model} - {input_job_name} - {input_label_col} - Weighted Recall', fontsize=18)
    axs[1].grid(True)
    locator = matplotlib.ticker.MaxNLocator(prune='both', nbins=5)
    axs[1].xaxis.set_major_locator(locator)

    # Precision plot
    axs[2].plot(input_res['precision_df']['date'], input_res['precision_df']['price'], color='blue')
    axs[2].set_title(f'{input_file_name} - {input_model} - {input_job_name} - {input_label_col} - Weighted Precision', fontsize=18)
    axs[2].grid(True)
    locator = matplotlib.ticker.MaxNLocator(prune='both', nbins=5)
    axs[2].xaxis.set_major_locator(locator)

    # Accuracy plot
    axs[3].plot(input_res['accuracy_df']['date'], input_res['accuracy_df']['price'], color='blue')
    axs[3].set_title(f'{input_file_name} - {input_model} - {input_job_name} - {input_label_col} - Weighted Accuracy', fontsize=18)
    axs[3].grid(True)
    locator = matplotlib.ticker.MaxNLocator(prune='both', nbins=5)
    axs[3].xaxis.set_major_locator(locator)

    plt.tight_layout()
    plt.savefig(os.path.join(RES_IMAGE_DIR, target_savefig_name), facecolor='white', transparent=False)
    return


def runner_reg_static_res_analysis_helper(input_run_prefix,
                                          input_file_name,
                                          input_job_name,
                                          input_model,
                                          input_label_col):
    log_info(f'Running runner_reg_static_res_analysis_helper on '
             f'{input_run_prefix} - {input_file_name} - {input_job_name} - {input_model} - {input_label_col}')
    target_pkl = get_res_pkl_name(input_file_name, input_run_prefix, input_model, input_job_name)
    one_res_dict = read_pkl_helper(os.path.join(RUN_RESULT_DIR, target_pkl))
    one_dict = one_res_dict['job_result'][input_label_col]
    res_dict = {'train_r2': one_dict['train']['summary']['r2'],
                'test_r2': one_dict['test']['summary']['r2'],
                'preds_labels_df': one_dict['df'],
                'train_r2_stab': one_dict['train_stab'],
                'test_r2_stab': one_dict['test_stab']}
    return res_dict


def runner_reg_static_res_plotter(input_run_prefix,
                                  input_file_name,
                                  input_model,
                                  input_job_name,
                                  input_label_col,
                                  input_res):
    fig, axs = plt.subplots(3, 1, figsize=(30, 25))
    target_savefig_name = get_res_pkl_name(input_file_name, input_run_prefix, input_model, input_job_name).replace('pkl', 'png')

    # Scatter plot of testing set
    axs[0].scatter(input_res['preds_labels_df']['preds'], input_res['preds_labels_df']['labels'])
    axs[0].set_title(f'{input_file_name} - {input_model} - {input_job_name} - {input_label_col}'
                     f' - R2: {input_res["train_r2"]:.3f} (test_r2), {input_res["test_r2"]:.3f} (Test)', fontsize=18)
    axs[0].set_xlim(xmin=-100, xmax=100)
    axs[0].grid(True)

    # Train stability
    axs[1].plot(input_res['train_r2_stab']['date'], input_res['train_r2_stab']['r2'], color='blue')
    axs[1].set_title(f'{input_file_name} - {input_model} - {input_job_name} - {input_label_col}'
                     f' - Train R2 Stability', fontsize=18)
    axs[1].grid(True)
    locator = matplotlib.ticker.MaxNLocator(prune='both', nbins=5)
    axs[1].xaxis.set_major_locator(locator)
    axs[1].set_ylim(ymin=-0.2)

    # Test stability
    axs[2].plot(input_res['test_r2_stab']['date'], input_res['test_r2_stab']['r2'], color='blue')
    axs[2].set_title(f'{input_file_name} - {input_model} - {input_job_name} - {input_label_col}'
                     f' - Test R2 Stability', fontsize=18)
    axs[2].grid(True)
    locator = matplotlib.ticker.MaxNLocator(prune='both', nbins=5)
    axs[2].xaxis.set_major_locator(locator)
    axs[2].set_ylim(ymin=-0.2)

    plt.tight_layout()
    plt.savefig(os.path.join(RES_IMAGE_DIR, target_savefig_name), facecolor='white', transparent=False)
    return


def runner_reg_rolling_res_analysis_helper(input_run_prefix,
                                           input_file_name,
                                           input_job_name,
                                           input_model,
                                           input_label_col,
                                           input_up_pct,
                                           input_dn_pct):
    log_info(f'Running runner_reg_rolling_res_analysis_helper on '
             f'{input_run_prefix} - {input_file_name} - {input_job_name} - {input_model} - {input_label_col}')
    target_pkl = get_res_pkl_name(input_file_name, input_run_prefix, input_model, input_job_name)
    one_res_dict = read_pkl_helper(os.path.join(RUN_RESULT_DIR, target_pkl))
    one_dict = one_res_dict['job_result']
    all_dates = sorted(list(one_dict.keys()))
    train_r2_list = [one_dict[i][input_label_col]['train']['summary']['r2'] for i in all_dates]
    test_r2_list = [one_dict[i][input_label_col]['test']['summary']['r2'] for i in all_dates]

    # R2 of each day aggregation
    train_r2_df = pd.DataFrame({'date': all_dates, 'r2': train_r2_list})
    train_r2_df = train_r2_df.loc[train_r2_df['r2'] > -3, :].reset_index(drop=True)
    test_r2_df = pd.DataFrame({'date': all_dates, 'r2': test_r2_list})
    test_r2_df = test_r2_df.loc[test_r2_df['r2'] > -3, :].reset_index(drop=True)

    # preds and labels aggregation
    df_list = [v[input_label_col]['df'] for k, v in one_dict.items()]
    preds_labels_df = pd.concat(df_list).reset_index(drop=True)

    # extreme market prediction accuracy
    df_pcts_dict = {k: [v[input_label_col]['df'], v[input_label_col]['extreme_market_rets']] for k, v in
                    one_dict.items()}
    extreme_preds_dict = {k: evaluate_regression_extreme_predictions(input_predictions=v[0]['preds'],
                                                                     input_y_labels=v[0]['labels'],
                                                                     input_rets=[v[1][input_dn_pct],
                                                                                 v[1][input_up_pct]])['summary'] for
                          k, v in df_pcts_dict.items()}
    extreme_preds_df = pd.DataFrame(extreme_preds_dict).T.reset_index()
    extreme_preds_df.columns = ['date'] + extreme_preds_df.columns.tolist()[1:]

    return {
        'train_r2_df': train_r2_df,
        'test_r2_df': test_r2_df,
        'preds_labels_df': preds_labels_df,
        'extreme_preds_df': extreme_preds_df,
        'price_df': one_res_dict['price_df']
    }


def runner_reg_rolling_res_plotter(input_run_prefix,
                                   input_file_name,
                                   input_model, 
                                   input_job_name, 
                                   input_label_col,
                                   input_res):
    fig, axs = plt.subplots(5, 1, figsize=(30, 25))
    target_savefig_name = get_res_pkl_name(input_file_name, input_run_prefix, input_model, input_job_name).replace('pkl', 'png')

    # R2 stability plot
    axs[0].plot(input_res['test_r2_df']['date'], input_res['test_r2_df']['r2'], color='red', label='Test')
    axs[0].plot(input_res['train_r2_df']['date'], input_res['train_r2_df']['r2'], color='blue', label='Train')
    axs[0].legend(fontsize=18)
    axs[0].set_title(f'{input_file_name} - {input_model} - {input_job_name} - {input_label_col} - r2 stab',
                     fontsize=18)
    axs[0].grid(True)
    locator = matplotlib.ticker.MaxNLocator(prune='both', nbins=5)
    axs[0].xaxis.set_major_locator(locator)
    axs[0].set_ylim(ymin=-0.2)

    # Price plot for comparison
    one_price_df = input_res['price_df']
    one_price_df = one_price_df.loc[one_price_df['date'] >= input_res['test_r2_df']['date'].min(), :].reset_index(drop=True)
    axs[1].plot(one_price_df['date'], one_price_df['price'], color='blue')
    axs[1].set_title(f'{input_file_name} - {input_model} - {input_job_name} - Price Plot', fontsize=18)
    axs[1].grid(True)
    locator = matplotlib.ticker.MaxNLocator(prune='both', nbins=5)
    axs[1].xaxis.set_major_locator(locator)

    # Extreme predictions plot - Upside
    l1 = axs[2].plot(input_res['extreme_preds_df']['date'], input_res['extreme_preds_df']['up_accuracy'], color='blue',
                     label='Upside Accuracy')
    axs[2].set_ylim([0, 1])
    axs2_sub = axs[2].twinx()
    l2 = axs2_sub.plot(input_res['extreme_preds_df']['date'], input_res['extreme_preds_df']['up_size'], color='red',
                       label='Upside Prediction Count')
    axs2_sub.plot(input_res['extreme_preds_df']['date'], [0] * len(input_res['extreme_preds_df']), color='black')
    axs[2].set_title(
        f'{input_file_name} - {input_model} - {input_job_name} - {input_label_col} - Extreme Movement Upside Prediction Accuracy',
        fontsize=18)
    axs[2].legend(l1 + l2, [l.get_label() for l in l1 + l2], fontsize=10)
    axs[2].grid(True)
    locator = matplotlib.ticker.MaxNLocator(prune='both', nbins=5)
    axs[2].xaxis.set_major_locator(locator)

    # Extreme predictions plot - Downside
    l3 = axs[3].plot(input_res['extreme_preds_df']['date'], input_res['extreme_preds_df']['dn_accuracy'], color='blue',
                     label='Downside Accuracy')
    axs[3].set_ylim([0, 1])
    axs3_sub = axs[3].twinx()
    l4 = axs3_sub.plot(input_res['extreme_preds_df']['date'], input_res['extreme_preds_df']['dn_size'], color='red',
                       label='Upside Prediction Count')
    axs3_sub.plot(input_res['extreme_preds_df']['date'], [0] * len(input_res['extreme_preds_df']), color='black')
    axs[3].set_title(
        f'{input_file_name} - {input_model} - {input_job_name} - {input_label_col} - Extreme Movement Downside Prediction Accuracy',
        fontsize=18)
    axs[3].legend(l3 + l4, [l.get_label() for l in l3 + l4], fontsize=10)
    axs[3].grid(True)
    locator = matplotlib.ticker.MaxNLocator(prune='both', nbins=5)
    axs[3].xaxis.set_major_locator(locator)

    # Overall scatter plot
    reg_eval_res = evaluate_regressions(input_y_labels=input_res['preds_labels_df']['labels'],
                                        input_predictions=input_res['preds_labels_df']['preds'])
    pl_df = input_res['preds_labels_df']
    dn_limit, up_limit = [np.percentile(pl_df['preds'], i) for i in [0.003, 99.997]]
    pl_df = pl_df.loc[(pl_df['preds'] > dn_limit) & (pl_df['preds'] < up_limit), :].reset_index(drop=True)
    axs[4].scatter(x=pl_df['preds'],
                   y=pl_df['labels'])
    m = LinearRegression(fit_intercept=True)
    r = m.fit(pl_df['preds'].to_numpy().reshape(-1, 1),
              pl_df['labels'].to_numpy().reshape(-1, 1))
    plot_x = [i(pl_df['preds']) for i in [min, max]]
    plot_y = [i * r.coef_[0][0] + r.intercept_[0] for i in plot_x]
    axs[4].plot(plot_x, plot_y, color='red', label=f'intercept={r.intercept_[0]:.3f}, slope={r.coef_[0][0]:.3f}')
    axs[4].set_xlabel('Predictions')
    axs[4].set_ylabel('Labels')
    axs[4].legend(fontsize=18)
    axs[4].set_title(
        f'{input_file_name} - {input_model} - {input_job_name} - {input_label_col} - Pred-Label Plot - r2 = {reg_eval_res["summary"]["r2"]:.3f}',
        fontsize=18)
    axs[4].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(RES_IMAGE_DIR, target_savefig_name), facecolor='white', transparent=False)
    return


def get_res_pkl_name(input_file_name, input_run_prefix, input_model, input_job_name):
    target_file_code = input_file_name.split('.')[0].split('_')[1]
    target_pkl = f'{input_run_prefix}_{target_file_code}_{input_model}_{input_job_name}.pkl'
    return target_pkl
