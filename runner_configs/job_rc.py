
DEFAULT_JOB_CONFIGS = [
    {
        'job_name': 'static_training_raw_features',
        'job_config': {
            'training_type': 'static',
            'label_cols': ['del_M_60', 'del_M_300', 'del_M_900'],
            'n_components': 0,
            'split_ratio': 0.6,
            'extreme_market_pcts': [5, 10, 15, 20, 25, 30, 70, 75, 80, 85, 90, 95]
        }
    },
    {
        'job_name': 'static_training_20_pca_features',
        'job_config': {
            'training_type': 'static',
            'label_cols': ['del_M_60', 'del_M_300', 'del_M_900'],
            'n_components': 20,
            'split_ratio': 0.6,
            'extreme_market_pcts': [5, 10, 15, 20, 25, 30, 70, 75, 80, 85, 90, 95]
        }
    },
    {
        'job_name': 'static_training_50_pca_features',
        'job_config': {
            'training_type': 'static',
            'label_cols': ['del_M_60', 'del_M_300', 'del_M_900'],
            'n_components': 50,
            'split_ratio': 0.6,
            'extreme_market_pcts': [5, 10, 15, 20, 25, 30, 70, 75, 80, 85, 90, 95]
        }
    },
    {
        'job_name': 'dynamic_training_raw_features',
        'job_config': {
            'training_type': 'dynamic',
            'label_cols': ['del_M_60', 'del_M_300', 'del_M_900'],
            'n_components': 0,
            'starting_ratio': 0.1,
            'extreme_market_pcts': [5, 10, 15, 20, 25, 30, 70, 75, 80, 85, 90, 95]
        }
    },
    {
        'job_name': 'dynamic_training_pca_features',
        'job_config': {
            'training_type': 'dynamic',
            'label_cols': ['del_M_60', 'del_M_300', 'del_M_900'],
            'n_components': 0,
            'starting_ratio': 0.1,
            'extreme_market_pcts': [5, 10, 15, 20, 25, 30, 70, 75, 80, 85, 90, 95]
        }
    },
    {
        'job_name': 'rolling_10_training_raw_features',
        'job_config': {
            'training_type': 'rolling',
            'label_cols': ['del_M_60', 'del_M_300', 'del_M_900'],
            'n_components': 0,
            'rolling_days': 10,
            'extreme_market_pcts': [5, 10, 15, 20, 25, 30, 70, 75, 80, 85, 90, 95]
        }
    },
    {
        'job_name': 'rolling_50_training_raw_features',
        'job_config': {
            'training_type': 'rolling',
            'label_cols': ['del_M_60', 'del_M_300', 'del_M_900'],
            'n_components': 0,
            'rolling_days': 50,
            'extreme_market_pcts': [5, 10, 15, 20, 25, 30, 70, 75, 80, 85, 90, 95]
        }
    },
    {
        'job_name': 'rolling_90_training_raw_features',
        'job_config': {
            'training_type': 'rolling',
            'label_cols': ['del_M_60', 'del_M_300', 'del_M_900'],
            'n_components': 0,
            'rolling_days': 90,
            'extreme_market_pcts': [5, 10, 15, 20, 25, 30, 70, 75, 80, 85, 90, 95]
        }
    },
    {
        'job_name': 'rolling_10_training_10_pca_features',
        'job_config': {
            'training_type': 'rolling',
            'label_cols': ['del_M_60', 'del_M_300', 'del_M_900'],
            'n_components': 10,
            'rolling_days': 10,
            'extreme_market_pcts': [5, 10, 15, 20, 25, 30, 70, 75, 80, 85, 90, 95]
        }
    },
    {
        'job_name': 'rolling_10_training_20_pca_features',
        'job_config': {
            'training_type': 'rolling',
            'label_cols': ['del_M_60', 'del_M_300', 'del_M_900'],
            'n_components': 20,
            'rolling_days': 10,
            'extreme_market_pcts': [5, 10, 15, 20, 25, 30, 70, 75, 80, 85, 90, 95]
        }
    },
]
