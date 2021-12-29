from utilities.helper_functions import *
import pandas as pd


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_DATA_DIR = os.path.join(PROJECT_DIR, 'input_data')

RUN_RESULT_DIR = os.path.join(PROJECT_DIR, 'run_results')

RES_IMAGE_DIR = os.path.join(RUN_RESULT_DIR, 'res_images')

FILE_LIST = [
    'step1_0050.csv.pklz',
    'step1_2330.csv.pklz',
    'step1_2603.csv.pklz'
]


def read_one_file(input_file_name='step1_0050.csv.pklz'):
    """ Read one file from ./input_data

    :param input_file_name:
    :return:
    """
    target_path = os.path.join(INPUT_DATA_DIR, input_file_name)
    if 'pklz' in input_file_name:
        df = pd.read_pickle(target_path, compression='gzip').reset_index(drop=True)
    else:
        df = pd.read_pickle(target_path).reset_index(drop=True)
    log_info(f'Got {target_path} with {len(df)} rows and {df.shape[1]} columns')
    return df


def read_all_files():
    """ Read all files available in ./input_data and return a dict

    :return:
    """
    file_list = listdir_fullpath('input_data')
    res_df_list = pool_run_func(read_one_file, file_list)
    res_dict = {os.path.basename(file_list[i]): res_df_list[i] for i in range(len(file_list))}
    return res_dict


def get_raw_features():
    """ All features columns

    :return:
    """
    return read_pkl_helper(os.path.join(INPUT_DATA_DIR, 'raw_features.pkl'))


def get_futures_features():
    """ All futures features columns

    :return:
    """
    return read_pkl_helper(os.path.join(INPUT_DATA_DIR, 'futures_features.pkl'))


def get_raw_labels():
    """ All labels columns, same for both futures and spot

    :return:
    """
    return read_pkl_helper(os.path.join(INPUT_DATA_DIR, 'raw_labels.pkl'))


if __name__ == '__main__':
    print(PROJECT_DIR)

