from utilities.models import *
from utilities.evaluations import *
from utilities.pca_helper import *


def one_model_runner(input_model_name,
                     input_model_config,
                     input_data_factory,
                     input_job_configs,
                     input_save_res_dict,
                     input_file_prefix):
    """ Run one model with specified jobs

    :param input_model_name:
    :param input_model_config:
    :param input_data_factory:
    :param input_job_configs:
    :param input_save_res_dict:
    :param input_file_prefix:
    :return:
    """
    log_header(f'Start running {input_model_name} on {len(input_job_configs)} jobs')
    all_res_dict = {}
    for job in input_job_configs:

        # Job configs
        start_time = timeit.default_timer()
        mn = input_model_name
        model_class, model_params = [input_model_config[i] for i in ['model_class', 'model_params']]
        oj, job_config = [job.get(i) for i in ['job_name', 'job_config']]
        d_fac = input_data_factory
        model_type = input_model_config['model_type']
        scaler_type = 'MinMax' if model_type == 'classification' else 'Standard' if model_type == 'regression' else None
        assert scaler_type in ['MinMax', 'Standard']
        log_info(f'{mn} - Running {oj}')

        # Construct dict for saving results
        one_save_res_dict = input_save_res_dict.copy()

        # Static training
        if job_config['training_type'] == 'static':

            # Prepare data first
            raw_features = get_raw_features()
            d_fac.split(input_split_ratio=job_config['split_ratio'])
            train_df = input_data_factory.get_train_df()
            test_df = input_data_factory.get_test_df()

            one_res = helper_pipeline(input_model_class=model_class,
                                      input_model_params=model_params,
                                      input_model_type=model_type,
                                      input_train_df=train_df,
                                      input_test_df=test_df,
                                      input_features=raw_features,
                                      input_pca_n_components=job_config['n_components'],
                                      input_scaler_type=scaler_type,
                                      input_labels_col=job_config['label_cols'],
                                      input_extreme_market_pcts=job_config['extreme_market_pcts'],
                                      input_label_pcts=input_model_config.get('label_pcts'),
                                      input_eval_stab=True,
                                      input_prefix=f'{mn} - {oj}')

            one_save_res_dict['job_name'] = oj
            one_save_res_dict['job_config'] = job_config.copy()
            one_save_res_dict['job_result'] = one_res

        elif job_config['training_type'] in ['dynamic', 'rolling']:

            # Prepare data first
            raw_features = input_data_factory.feature_cols
            if job_config['training_type'] == 'rolling':
                d_fac.split(input_rolling_train_days=job_config['rolling_days'])
            else:
                d_fac.split(input_split_ratio=job_config['starting_ratio'],
                            input_bool_full_day=True)

            # Start looping over dates
            one_save_res_dict['job_result'] = {}
            count = 0
            bool_first_train = True
            bool_moving_window = True if job_config['training_type'] == 'rolling' else False
            while 1:

                # Step one day first
                if not bool_first_train:
                    log_info(f'{mn} - {oj} - Stepping for the #{count} time')

                    step_res = d_fac.step_one_day(input_bool_moving_window=bool_moving_window)
                    if not step_res:
                        log_info(f'{mn} - {oj} - Stepped to last day already')
                        break
                    count += 1
                if bool_first_train:
                    bool_first_train = False
                log_info(f'{mn} - {oj} - Running #{count + 1} time on {len(d_fac.train_df)} / {len(d_fac.test_df)} train/test data')

                # Prepare one running data
                one_train_df = input_data_factory.get_train_df()
                one_test_df = input_data_factory.get_test_df()

                # Limit prediction window to the next day only
                one_test_df = one_test_df.loc[one_test_df['date'] == one_test_df['date'].min(), :].reset_index()
                target_date = one_test_df['date'].min()

                one_res = helper_pipeline(input_model_class=model_class,
                                          input_model_params=model_params,
                                          input_model_type=model_type,
                                          input_train_df=one_train_df,
                                          input_test_df=one_test_df,
                                          input_features=raw_features,
                                          input_pca_n_components=job_config['n_components'],
                                          input_scaler_type=scaler_type,
                                          input_labels_col=job_config['label_cols'],
                                          input_extreme_market_pcts=job_config['extreme_market_pcts'],
                                          input_label_pcts=input_model_config.get('label_pcts'),
                                          input_eval_stab=False,
                                          input_prefix=f'{mn} - {oj} - {target_date}')

                one_save_res_dict['job_result'][target_date] = one_res

            one_save_res_dict['job_name'] = oj
            one_save_res_dict['job_config'] = job_config.copy()

        else:
            log_error(f'{input_model_name} - Invalid training_type')
            raise Exception('InvalidTrainingType')

        log_info(f'{mn} - Finished {oj} in {timeit.default_timer() - start_time:.2f}s')
        file_name = f'{input_file_prefix}_{oj}'
        save_path = os.path.join(RUN_RESULT_DIR, file_name)
        save_pkl_helper(one_save_res_dict, save_path)
        log_info(f'{mn} - Saved at {save_path}')

    log_header(f'Finished all jobs of {input_model_name}')
    return all_res_dict


def helper_pipeline(input_model_class,
                    input_model_params,
                    input_model_type,
                    input_train_df,
                    input_test_df,
                    input_features,
                    input_pca_n_components,
                    input_scaler_type,
                    input_labels_col,
                    input_label_pcts,
                    input_extreme_market_pcts,
                    input_eval_stab,
                    input_prefix):
    """ Pipeline for running one training job, say, rolling on one day

    :param input_model_class:
    :param input_model_params:
    :param input_model_type:
    :param input_train_df:
    :param input_test_df:
    :param input_features:
    :param input_pca_n_components:
    :param input_scaler_type:
    :param input_labels_col:
    :param input_label_pcts:
    :param input_extreme_market_pcts:
    :param input_eval_stab:
    :param input_prefix:
    :return:
    """
    log_info(f'{input_prefix} - Starting one pipeline')
    one_rd = {}

    scaler = get_scaler(input_x_df=input_train_df[input_features],
                        input_scaler_type=input_scaler_type)

    # PCA if needed
    pca_helper = None
    if input_pca_n_components > 0:
        log_info(f'{input_prefix} Fitting to PCA for {input_pca_n_components} components')
        pca_helper = PCAHelper(input_x_df=scaler.transform(input_train_df[input_features]),
                               input_components_count=input_pca_n_components)
        one_rd['explained_var_ratio'] = pca_helper.explained_var_ratio

    # Classify labels if needed
    labels_col = input_labels_col

    evaluate_func = evaluate_classifications if input_model_type == 'classification' else evaluate_regressions
    for one_label in labels_col:
        # Getting only label col we are interested about
        tmp_train_df = input_train_df.dropna(subset=[one_label]).reset_index(drop=True)
        tmp_test_df = input_test_df.dropna(subset=[one_label]).reset_index(drop=True)

        train_x = scaler.transform(tmp_train_df[input_features])
        test_x = scaler.transform(tmp_test_df[input_features])
        if input_pca_n_components > 0:
            log_info(f'{input_prefix} Transforming train and test x to {input_pca_n_components} components')
            train_x = pca_helper.transform(train_x)
            test_x = pca_helper.transform(test_x)

        if input_model_type == 'classification':
            log_info(f'{input_prefix} - Classifying {one_label} - {str(input_label_pcts)}')
            trim_val = get_trimming_values(input_y_series=tmp_train_df[one_label], input_pcts=input_label_pcts)
            train_labels = get_classifications(input_y_ls=tmp_train_df[one_label], input_trim_val=trim_val)
            test_labels = get_classifications(input_y_ls=tmp_test_df[one_label], input_trim_val=trim_val)
        else:
            train_labels = tmp_train_df[one_label]
            test_labels = tmp_test_df[one_label]

        # Train and evaluate model (res_dict to save results)
        log_info(f'{input_prefix} - Start training on {one_label}')
        one_rd[one_label] = {}
        target_model = Model(input_train_x_df=train_x,
                             input_train_y_series=train_labels,
                             input_model=input_model_class(**input_model_params))
        target_model.train()
        train_preds = target_model.predict(train_x)
        test_preds = target_model.predict(test_x)
        one_rd[one_label]['train'] = evaluate_func(input_y_labels=train_labels,
                                                   input_predictions=train_preds)
        one_rd[one_label]['train_size'] = len(train_labels)
        one_rd[one_label]['test'] = evaluate_func(input_y_labels=test_labels,
                                                  input_predictions=test_preds)
        one_rd[one_label]['test_size'] = len(test_labels)
        if input_model_type == 'regression':
            one_rd[one_label]['extreme_market_rets'] = {i: np.percentile(train_labels, i) for i in input_extreme_market_pcts}
        if input_eval_stab:
            one_rd[one_label]['train_stab'] = evaluate_stability(input_y_labels=train_labels,
                                                                 input_predictions=train_preds,
                                                                 input_date_list=tmp_train_df['date'].tolist(),
                                                                 input_type=input_model_type)
            one_rd[one_label]['test_stab'] = evaluate_stability(input_y_labels=test_labels,
                                                                input_predictions=test_preds,
                                                                input_date_list=tmp_test_df['date'].tolist(),
                                                                input_type=input_model_type)
        one_rd[one_label]['df'] = pd.DataFrame({'preds': test_preds, 'labels': test_labels})
    log_info(f'{input_prefix} - Finished one pipeline')
    return one_rd
