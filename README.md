# MAFS6100 Short-term Alpha Forecasting via ML/DL Techniques

First of all please cd to the project directory and run `pip install -r requirements.txt` to install all required packages.

## Path and Structure

The directories in this project can be summarized as below:

### input_data

Input data should be stored in path specified in `utilities.data_access.INPUT_DATA_DIR`. 

Also, in the data folders, the following should be present:

- stock files (step1_0050.csv.pklz, step1_2330.csv.pklz, step1_2603.csv.pklz with the same format)
- futures data (step1_CDF1.csv, step1_CZF1.csv, step_1_NYF1.csv)
- raw_features.pkl, which is a list of all features column name in stock data file
- raw_labels.pkl, which is a list of all labels column name in stock data file
- futures_features.pkl, which is a list of all features column name in futures data file

If there are updates to stock files stored in the `INPUT_DATA_DIR` (e.g. more stocks), please make appropriate changes to `utilities.data_access.FILE_LIST`.


### utilities

Scripts in this folder are mostly helper functions useful for analysis. For instance, `utilities.data_access` and `utilities.data_factory` are helper functions for accessing the data.

One thing to note is that `utilites.models` listed some model wrappers for complex model structure, e.g. LSTM.


### run_results

Results from training will be stored in this folder as pickle files. `runner_result_analysis.ipynb` is the main notebook used to analysis the files in this directory

Also, `run_results.res_images` stores the graphs saved from teh analysis.


### runner_configs

Configurations and parameters on models and jobs are stored in this directory. `runner_configs.job_rc` stores all the jobs available for running, which can be modified.

`runner_configs.model_rc` also stored different models available for training, and also can be modified.

One thing to note is that in the usage section, when you specify the job and model for training, they MUST be present in the above scripts. 


## Usage

The main runner script is runner.py, which is used to run static or rolling training of different models with specified parameters and configurations.

After running this scripts, results will be stored in `run_results` and can be visualised via `runner_result_analysis.ipynb`

The key parameters are detailed below (can be found in `runner.py` as well):

```bash
python runner.py --prefix [prefix] --stock_code [stock_code] --get_futures --model_name [model_name] --job_name [job_name]
```
where:

prefix:
- tag for storing the training result

stock_code:
- stock to be analyzed, need to be one from existing stock files (for instance, 0050, 2330, 2603)

get_futures:
- specify if need to include futures data in the training as well

model_name:
- target model to analyze, must be present in `runner_configs.model_rc.DEFAULT_MODEL_CONFIGS`

job_name:
- target job to run, must be present in `runner_configs.job_rc.DEFAULT_JOB_CONFIGS`


For instance you can run the following:

```bash
python runner.py --prefix w_fut --stock_code 0050 --get_futures --model_name LinearRegression --job_name static_training_raw_features
```


## Result Analysis

Open `runner_result_analysis.ipynb` and you can see the configurations required in the second cell. The configuration are the same as one required in `runner.py`.

After defining the configurations, run up to the third cell and you will see the graphs in the notebook. Also the graph will be saved in `run_results/res_images`


## Miscellaneous

- `features_engineering.ipynb` and `label_analytics.ipynb` are used for visualizing the graphs attached in the report in the corresponding sections.
- In `bokai_scripts` it stores the models constructed by Bokai CAO. LSTM and MLP are incorporated into `utilities.models` but CNN-LSTM is too complicated to be extracted. Also, Bokai mentioned to send the scripts directly so I have just included that in this directory.

