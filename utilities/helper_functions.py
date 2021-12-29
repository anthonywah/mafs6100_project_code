import datetime
import os
import _pickle as cp
import json
import pandas as pd
import logging
from pytz import timezone, utc
import multiprocessing as mp
import multiprocessing.pool
from contextlib import closing


class InfoColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    WHITE = '\033[97m'
    NOTHING = ''


HEADER = 9

FORMAT_STR = "%(asctime)s.%(msecs)03d | %(name)-8s | {}%(levelname)-8s | %(filename)-24s:%(lineno)5s{} | %(message)s"


def custom_tz(*args):
    utc_dt = utc.localize(datetime.datetime.utcnow())
    my_tz = timezone('Asia/Hong_Kong')
    converted = utc_dt.astimezone(my_tz)
    return converted.timetuple()


class MyFormatter(logging.Formatter):
    def __init__(self, input_format_dict):
        super().__init__()
        self.format_dict = input_format_dict
        self.format_str = FORMAT_STR.format('', '')
        self.date_fmt = '%Y-%m-%d %H:%M:%S'

    def format(self, record):
        log_fmt = self.format_dict.get(record.levelno, self.format_str)
        formatter = logging.Formatter(log_fmt, self.date_fmt)
        return formatter.format(record)


def header(self, message, *args, **kwargs):
    self._log(HEADER, message, args, **kwargs)


logging.addLevelName(HEADER, "HEADER")
logging.Logger.header = header

loggers = {}


def get_logger(name):
    global loggers

    color_formats = {
        HEADER: InfoColors.HEADER + FORMAT_STR.format('', '') + InfoColors.ENDC,
        logging.DEBUG: FORMAT_STR.format(InfoColors.WHITE, InfoColors.ENDC),
        logging.INFO: FORMAT_STR.format(InfoColors.NOTHING, InfoColors.NOTHING),
        logging.WARNING: FORMAT_STR.format(InfoColors.WARNING, InfoColors.ENDC),
        logging.ERROR: FORMAT_STR.format(InfoColors.FAIL, InfoColors.ENDC),
        logging.CRITICAL: FORMAT_STR.format(InfoColors.FAIL, InfoColors.ENDC),
    }

    if loggers.get(name):
        return loggers.get(name)
    else:
        logger = logging.getLogger(name)
        logging.Formatter.converter = custom_tz
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setFormatter(MyFormatter(color_formats))
        logger.addHandler(ch)
        logger.propagate = False
        loggers[name] = logger
        return logger


helper_logger = get_logger('helper')

log_info = helper_logger.info
log_error = helper_logger.error
log_critical = helper_logger.critical
log_debug = helper_logger.debug
log_header = helper_logger.header


def read_csv_helper(input_dir, raise_error=True, log=False):
    try:
        df = pd.read_csv(input_dir)
        if log:
            log_info(f'Successfully read {os.path.basename(input_dir)}')
    except Exception as e:
        df = None
        if log:
            log_info(f'Failed at reading {os.path.basename(input_dir)}')
        if raise_error:
            raise e
    return df


def read_json_helper(input_dir, raise_error=True, log=False):
    try:
        with open(input_dir, 'rb') as f:
            res_dict = json.load(f)
        if log:
            log_info(f'Successfully read {os.path.basename(input_dir)}')
    except Exception as e:
        res_dict = None
        if log:
            log_info(f'Failed at reading {os.path.basename(input_dir)}')
        if raise_error:
            raise e
    return res_dict


def save_pkl_helper(pkl, save_path):
    with open(save_path, 'wb') as f:
        cp.dump(pkl, f)
    return


def read_pkl_helper(input_dir, raise_error=True, log=False):
    try:
        with open(input_dir, 'rb') as f:
            res = cp.load(f)
        if log:
            log_info(f'Successfully read {os.path.basename(input_dir)}')
    except Exception as e:
        res = None
        if log:
            log_info(f'Failed at reading {os.path.basename(input_dir)}')
        if raise_error:
            raise e
    return res


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


def print_dict(input_dict):
    for k, v in input_dict.items():
        print(f'{k:<30}: {v}')


def get_cur_time_str():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def pool_run_func(input_func, input_arg_list):
    with mp.Pool(32) as p:
        res = p.starmap(func=input_func,
                        iterable=input_arg_list)
        p.terminate()
    return res

