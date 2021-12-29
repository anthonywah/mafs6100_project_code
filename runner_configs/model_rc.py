import sys
sys.path.append('..')
from utilities.models import *

DEFAULT_MODEL_CONFIGS = {
    'LinearRegression': {
        'model_type': 'regression',
        'model_class': LinearRegression,
        'model_params': {},
    },
    'RidgeRegression': {
        'model_type': 'regression',
        'model_class': Ridge,
        'model_params': {'alpha': 1.0},
    },
    'LassoRegression': {
        'model_type': 'regression',
        'model_class': Lasso,
        'model_params': {'alpha': 0.05},
    },
    'LSTMRegression': {
        'model_type': 'regression',
        'model_class': LSTMWrapper,
        'model_params': {},
    },
    'MLP': {
        'model_type': 'classification',
        'model_class': MLPClassifier,
        'model_params': {'hidden_layer_sizes': (200, 100, 100),
                         'max_iter': 500,
                         'verbose': True},
        'label_pcts': [0, 25, 75, 100]
    },
    'RandomForest': {
        'model_type': 'classification',
        'model_class': RandomForestClassifierWrapper,
        'model_params': {},
        'label_pcts': [0, 30, 70, 100]
    },
    'GradientBoosting': {
        'model_type': 'classification',
        'model_class': GradientBoostingClassifierWrapper,
        'model_params': {},
        'label_pcts': [0, 30, 70, 100]
    },
}