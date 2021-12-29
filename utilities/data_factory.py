from utilities.data_access import *
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np


SPOT_FUTURE_MAPPING = {
    'step1_0050.csv.pklz': 'step1_NYF1.csv',
    'step1_2330.csv.pklz': 'step1_CDF1.csv',
    'step1_2603.csv.pklz': 'step1_CZF1.csv',
}


class DataFactory:
    def __init__(self,
                 input_file_name='step1_0050.csv.pklz',
                 input_get_futures=False,
                 input_split_ratio=0.6,
                 input_rolling_train_days=None,
                 input_min_update_count=100):
        log_info(f'Reading {input_file_name}')
        self.df = read_one_file(input_file_name=input_file_name)
        self.feature_cols = get_raw_features()
        self.labels_col = get_raw_labels()
        if input_get_futures:
            log_info(f'Reading {SPOT_FUTURE_MAPPING[input_file_name]}')
            fut_df = read_one_file(input_file_name=SPOT_FUTURE_MAPPING[input_file_name])
            fut_features_col = get_futures_features()
            fut_df.rename(columns={i: f'f_{i}' for i in fut_features_col}, inplace=True)
            fut_df.rename(columns={i: f'f_{i}' for i in self.labels_col}, inplace=True)
            fut_features_col = [f'f_{i}' for i in fut_features_col]
            fut_labels_col = [f'f_{i}' for i in self.labels_col]
            self.feature_cols += fut_features_col
            self.labels_col += fut_labels_col
            orig_len, orig_width = self.df.shape[0], self.df.shape[1]
            self.df = pd.merge(self.df, fut_df[['date', 'secondlyMark'] + fut_features_col + fut_labels_col],
                               on=['date', 'secondlyMark'], how='outer').reset_index(drop=True)
            log_info(f'Got new df with len {orig_len} -> {self.df.shape[0]} and columns {orig_width} -> {self.df.shape[1]}')
        count_df = self.df.groupby('date').count()[['time']]
        exc_date_list = count_df.loc[count_df['time'] < input_min_update_count, :].index.tolist()
        if exc_date_list:
            log_info(f'Filtering out dates {", ".join(exc_date_list)} due to < {input_min_update_count} updates')
            orig_len = self.df.shape[0]
            self.df = self.df.loc[~self.df['date'].isin(exc_date_list), :].reset_index(drop=True)
            log_info(f'Original {orig_len} rows -> {self.df.shape[0]} rows remain')
        self.df = self.df.sort_values('secondlyMark').reset_index(drop=True)
        self.df.loc[:, self.labels_col] *= 10000
        log_info(f'Converting NaN to 0')
        self.df.loc[:, self.feature_cols] = self.df[self.feature_cols].fillna(0.0)
        self.split_ratio, self.train_df, self.test_df = 0.0, pd.DataFrame(), pd.DataFrame()
        self.rolling_train_days = input_rolling_train_days
        log_info(f'Splitting df to train and test df')
        self.split(input_split_ratio=input_split_ratio,
                   input_rolling_train_days=input_rolling_train_days)
        log_info(f'{input_file_name} train-test data ready')

    def split(self, input_split_ratio=0.6, input_rolling_train_days=None):
        self.split_ratio = input_split_ratio
        if not input_rolling_train_days:
            split_i = int(len(self.df) * input_split_ratio)
            orig_ind = split_i
            split_i = self.df.loc[self.df['date'] == self.df.loc[split_i:, 'date'].unique().tolist()[1], 'date'].index[0]
            log_info(f'split_i changed from {orig_ind} to {split_i} to fit full day data')
            self.train_df = self.df.loc[:split_i - 1, :]
            self.test_df = self.df.loc[split_i:, :]
        else:
            all_dates = self.df['date'].sort_values().unique().tolist()
            train_days = all_dates[:input_rolling_train_days]
            test_days = all_dates[input_rolling_train_days:]
            self.train_df = self.df.loc[self.df['date'].isin(train_days), :]
            self.test_df = self.df.loc[self.df['date'].isin(test_days), :]
        return

    def step_one_day(self, input_bool_moving_window=False):
        log_info('Start stepping one day')
        input_x_day = 1
        orig_train_df_len = len(self.train_df)
        orig_test_df_len = len(self.test_df)
        orig_test_date_list = self.test_df['date'].sort_values().unique().tolist()
        append_date_list = orig_test_date_list[:input_x_day]
        remain_date_list = orig_test_date_list[input_x_day:]
        if not remain_date_list:
            log_info('No more dates to step forward')
            return False
        orig_train_last_date = self.train_df['date'].max()
        self.train_df = self.train_df.append(self.test_df.loc[self.test_df['date'].isin(append_date_list), :])
        if input_bool_moving_window:
            orig_train_first_date = self.train_df['date'].min()
            self.train_df = self.train_df.loc[self.train_df['date'] != orig_train_first_date, :].reset_index(drop=True)
            new_train_first_date = self.train_df['date'].min()
            log_info(f'Moving train_df first date from {orig_train_first_date} to {new_train_first_date}')
        self.train_df = self.train_df.sort_values('secondlyMark').reset_index(drop=True)
        train_df_last_date = self.train_df['date'].max()
        orig_test_first_date = self.test_df['date'].min()
        self.test_df = self.test_df.loc[self.test_df['date'].isin(remain_date_list), :].reset_index(drop=True)
        self.test_df = self.test_df.sort_values('secondlyMark').reset_index(drop=True)
        test_df_first_date = self.test_df['date'].min()
        log_info(f'Step finished: train_df last date {orig_train_last_date} -> {train_df_last_date}, '
                 f'len {orig_train_df_len} -> {len(self.train_df)}; '
                 f'test_df first_date {orig_test_first_date} -> {test_df_first_date}, '
                 f'len {orig_test_df_len} -> {len(self.test_df)}')
        return True

    def get_train_df(self):
        return self.train_df.reset_index(drop=True)

    def get_test_df(self):
        return self.test_df.reset_index(drop=True)

    def get_price_df(self):
        tmp_df = self.df[['date', 'BP1', 'SP1']].copy()
        tmp_df.loc[:, 'price'] = (tmp_df['BP1'] + tmp_df['SP1']) / 2
        res = tmp_df[['date', 'price']].groupby('date').last().reset_index()
        log_info('Price df ready')
        return res


def get_scaler(input_x_df, input_scaler_type='MinMax'):
    assert input_scaler_type in ['MinMax', 'Standard']
    if input_scaler_type == 'MinMax':
        scaler = MinMaxScaler()
        scaler.fit(input_x_df)
    elif input_scaler_type == 'Standard':
        scaler = StandardScaler()
        scaler.fit(input_x_df)
    else:
        scaler = None
        log_error(f'Invalid scaler_type {input_scaler_type}')
    return scaler


def get_classifications(input_y_ls, input_pcts=None, input_trim_val=None):
    """ Get classifications from series of labels, input_pct should be pcts

    :param input_y_ls:
    :param input_pcts:
    :param input_trim_val: Adopt if provided
    :return:
    """
    tmp = np.array(input_y_ls)
    if input_pcts is None:
        input_pcts = [0, 25, 75, 100]
    trim_val = input_trim_val
    if trim_val is None:
        trim_val = get_trimming_values(input_y_ls, input_pcts)
    cond_list = []
    choice_list = []
    for i in range(len(trim_val) - 1):
        pct_str = f'{input_pcts[i]}-{input_pcts[i + 1]}'
        if i == 0:
            cond_list.append(tmp < trim_val[i + 1])
            choice_list.append([f'{pct_str}|<{trim_val[i + 1]:.2f}'] * len(input_y_ls))
        elif i == len(trim_val) - 2:
            cond_list.append(trim_val[i] <= tmp)
            choice_list.append([f'{pct_str}|>={trim_val[i]:.2f}'] * len(input_y_ls))
        else:
            cond_list.append((trim_val[i] <= tmp) & (tmp < trim_val[i + 1]))
            choice_list.append([f'{pct_str}|{trim_val[i]:.2f}-{trim_val[i + 1]:.2f}'] * len(input_y_ls))
    return np.select(cond_list, choice_list).tolist()


def get_trimming_values(input_y_series, input_pcts):
    """ Get trimming values from series of labels

    :param input_y_series:
    :param input_pcts:
    :return:
    """
    return np.percentile(input_y_series, input_pcts)
