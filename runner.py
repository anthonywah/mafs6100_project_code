from main.runner_func import *
from main.runner_eval import *
from runner_configs.job_rc import *
from runner_configs.model_rc import *
from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()
    aa = parser.add_argument
    aa('--prefix',      dest='prefix',      action='store',         type=str,
       help='Runner prefix for saving job result')
    aa('--stock_code',  dest='stock_code',  action='store',         type=str,
       help='Stock code to be run')
    aa('--get_futures', dest='get_futures', action='store_true',
       help='True if get futures data for training also')
    aa('--model_name',  dest='model_name',  action='store',         type=str,
       help='Model to be trained, see readme for full list')
    aa('--job_name',    dest='job_name',    action='store',         type=str,
       help='Job to be run, see readme for full list')

    args = parser.parse_args()
    log_info('Running on the below arguments')
    print(args.__dict__)
    run_prefix = args.prefix
    one_file = f'step1_{args.stock_code}.csv.pklz'
    assert one_file in FILE_LIST
    bool_get_futures = args.get_futures
    one_model_name = args.model_name
    assert one_model_name in DEFAULT_MODEL_CONFIGS.keys()
    one_job_name = args.job_name
    assert one_job_name in [i['job_name'] for i in DEFAULT_JOB_CONFIGS]

    # Start running
    d = DataFactory(input_file_name=one_file, input_get_futures=False)
    file_code = one_file.split('.')[0].split('_')[1]
    price_df = d.get_price_df()

    starting = timeit.default_timer()
    file_prefix = f'{run_prefix}_{file_code}_{one_model_name}'
    log_header(f'Start running {file_prefix}')
    one_model_config = DEFAULT_MODEL_CONFIGS[one_model_name]
    res = {'file_code': file_code,
           'price_df': price_df,
           'model_name': one_model_name,
           'model_config': one_model_config}
    jobs = [i for i in DEFAULT_JOB_CONFIGS if i['job_name'] == one_job_name]
    one_model_runner(input_model_name=one_model_name,
                     input_model_config=one_model_config,
                     input_data_factory=d,
                     input_job_configs=jobs,
                     input_save_res_dict=res,
                     input_file_prefix=file_prefix)
    log_header(f'Finished running {file_prefix} in {timeit.default_timer() - starting:.2f}s')
