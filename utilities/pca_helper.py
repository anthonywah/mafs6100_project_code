from sklearn.decomposition import PCA
from utilities.data_factory import *
from utilities.helper_functions import *


class PCAHelper:
    def __init__(self, input_x_df, input_components_count):
        self.pca = PCA(n_components=input_components_count)
        self.n = input_components_count
        self.pca.fit(input_x_df)
        self.explained_var_ratio = calc_pca_explained_var(self.pca, self.n)

    def transform(self, input_x_df):
        return pd.DataFrame(self.pca.transform(input_x_df))


def calc_pca_explained_var(input_pca, input_n_components):
    count = 1
    explained_variance_sum = 0
    while count < input_n_components:
        explained_variance_sum += input_pca.explained_variance_ratio_[count - 1]
        count += 1
    log_info(f'First {count} PCs -> {explained_variance_sum * 100:.2f}% of Var')
    return explained_variance_sum


def get_pca_components_count(input_features_df, input_thres=0.97):
    pca = PCA()
    pca.fit(input_features_df)
    count = 1
    explained_variance_sum = 0
    while explained_variance_sum < input_thres:
        explained_variance_sum += pca.explained_variance_ratio_[count - 1]
        count += 1
    log_info(f'The first {count} components explained {explained_variance_sum * 100:.2f}% of the variance')
    return count


def get_pca_explained_var(input_features_df, input_n_components=20):
    pca = PCA()
    pca.fit(input_features_df)
    count = 1
    explained_variance_sum = 0
    while count < input_n_components:
        explained_variance_sum += pca.explained_variance_ratio_[count - 1]
        count += 1
    log_info(f'The first {count} components explained {explained_variance_sum * 100:.2f}% of the variance')
    return count


def get_pca_components_stability(input_df, input_interval, input_thres):
    """ Get rolling # of PC to explain input_thres % of variance

    :param input_df:
    :param input_interval:
    :param input_thres:
    :return:
    """
    all_days_list = sorted(input_df['date'].unique().tolist())
    features_col = get_raw_features()
    count = 0
    res_list = []
    while count < len(all_days_list):
        if count + input_interval > len(all_days_list):
            one_list = all_days_list[count:]
        else:
            one_list = all_days_list[count:count + input_interval]
        count += input_interval
        if not one_list:
            break
        one_features_df = input_df.loc[input_df['date'].isin(one_list), features_col].reset_index(drop=True)
        scaler = get_scaler(one_features_df, input_scaler_type='Standard')
        one_res = get_pca_components_count(scaler.transform(one_features_df), input_thres=input_thres)
        res_list.append({'start_date': one_list[0], 'end_date': one_list[-1], 'n_components': one_res})
        log_info(f'Got result from {one_list[0]} to {one_list[-1]}')
    return pd.DataFrame(res_list)


def get_pca_explained_var_stability(input_df, input_interval, input_n_components):
    """ Get explained variance of each PC with regards to # of PC specified

    :param input_df:
    :param input_interval:
    :param input_n_components:
    :return:
    """
    all_days_list = sorted(input_df['date'].unique().tolist())
    features_col = get_raw_features()
    count = 0
    res_list = []
    while count < len(all_days_list):
        if count + input_interval > len(all_days_list):
            one_list = all_days_list[count:]
        else:
            one_list = all_days_list[count:count + input_interval]
        count += input_interval
        if not one_list:
            break
        train_date_list = one_list[:-1]
        test_date_list = [one_list[-1]]
        f_df = input_df.loc[input_df['date'].isin(one_list), ['date'] + features_col].reset_index(drop=True)
        one_train_features_df = f_df.loc[f_df['date'].isin(train_date_list), features_col].reset_index(drop=True)
        one_test_features_df = f_df.loc[f_df['date'].isin(test_date_list), features_col].reset_index(drop=True)
        scaler = get_scaler(one_train_features_df, input_scaler_type='Standard')

        pca_model = PCA(n_components=input_n_components, svd_solver='auto')
        pca_model = pca_model.fit(scaler.transform(one_train_features_df))
        test_exp_var = explained_variance(scaler.transform(one_test_features_df), pca_model)

        one_res = {f'PC{i + 1}': test_exp_var[i] for i in range(len(input_n_components))}
        one_res['total'] = sum(test_exp_var)
        one_res['train_start_date'] = train_date_list[0]
        one_res['train_end_date'] = train_date_list[-1]
        one_res['test_date'] = test_date_list[0]
        res_list.append(one_res)
        log_info(f'Got result from {one_res["train_start_date"]} - {one_res["test_date"]}')
    return pd.DataFrame(res_list)


def explained_variance(X, model):
    result = np.zeros(model.n_components)
    for i in range(model.n_components):
        x_trans = model.transform(X)
        x_trans_i = np.zeros_like(x_trans)
        x_trans_i[:, i] = x_trans[:, i]
        x_approx_i = model.inverse_transform(x_trans_i)
        result[i] = 1 - (np.linalg.norm(x_approx_i - X) /
                          np.linalg.norm(X - model.mean_)) ** 2
    return result
