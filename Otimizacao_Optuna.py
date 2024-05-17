# -*- coding: utf-8 -*-
"""Model.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1k06oyMON9qOFf1FtoCLToX5F4JOnnvRJ

#### Bibliotecas
"""

from datetime import datetime
from time import time
from contextlib import contextmanager
from typing import List, Union
import pathlib
import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr,kruskal
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
from tsai.all import *
import optuna
from optuna.integration import FastAIV2PruningCallback
from fastai.vision.all import *
from fastai.text.all import *
from fastai.collab import *
from fastai.tabular.all import *


os.environ["DEVICE"] = "cuda"

"""### Carregando os dados"""

set_seed(1, False)
@contextmanager
def cwd(path: str) -> None:

    """
    Context manager para mudar o diretório de trabalho.
    Mantém o diretório original após a execução do bloc
    
    o de código.
    """

    oldpwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(oldpwd)

with cwd('dados/TrainTestVal_dataset/'):
    y_labels = pd.read_csv('y_labels.csv')['localidade']
    gpp_cax_test,gpp_peru_test,gpp_santarem_test = [pd.read_csv(f'dados_datasetTeste/{x}') for x in sorted(os.listdir('dados_datasetTeste'))]
    X_test,X_train,X_val,y_test,y_train,y_val = [np.load(data) for data in sorted([x for x in os.listdir() if not(x in ['dados_datasetTeste','y_labels.csv',".ipynb_checkpoints"])])]

X, y, splits = combine_split_data(xs=[X_train, X_val, X_test], ys=[y_train, y_val, y_test])
plot_splits(splits)

tfms = [None, TSForecasting()]
get_splits_len(splits) # [1806, 408, 408] ~= 70%,15%,15%

"""# Otimização do modelo selecionado com o Optuna"""

def objective_InceptionTimePlus(trial):
    # Definir os hiperparâmetros para otimização
    nf = trial.suggest_int('nf', 16, 128)
    fc_dropout = trial.suggest_float('fc_dropout', 0.0, 0.9)
    ks = trial.suggest_int('ks', 10, 100)
    conv_dropout = trial.suggest_float('conv_dropout', 0.0, 0.9)
    sa = trial.suggest_categorical('sa', [True, False])
    se = trial.suggest_categorical('se', [True, False])
    arch_config = {
        'nf': nf,
        'fc_dropout': fc_dropout,
        'ks': ks,
        'conv_dropout': conv_dropout,
        'sa': sa,
        'se': se
    }
    learning_rate_model = trial.suggest_float("learning_rate_model", 1e-5, 1e-2, log=True)
    Huber_delta = trial.suggest_float("Huber_delta", 1, 2)
    standardize_sample = trial.suggest_categorical('by_sample', [True, False])
    standardize_var = trial.suggest_categorical('by_var', [True, False])
    arch = InceptionTimePlus  # Corrigir a arquitetura para InceptionTimePlus
    
    # Definir a instância do TSForecaster com os callbacks
    learn = TSForecaster(X, y, splits=splits, path='models', tfms=tfms,
                         batch_tfms=TSStandardize(by_sample=standardize_sample, by_var=standardize_var),
                         arch=arch, arch_config=arch_config, metrics=[rmse],
                         cbs=[
                             FastAIV2PruningCallback(trial, monitor='_rmse'),
                             SaveModel(monitor='_rmse', comp=np.less, fname='best_model',with_opt=True,verbose=True),
                         ],
                         device=device, loss_func=HuberLoss('mean', Huber_delta), seed=1)
    
    with ContextManagers([learn.no_bar()]):
        learn.fit_one_cycle(550, lr_max=learning_rate_model)
        # Carregar o melhor modelo salvo
        learn.load('best_model')
        # Obter o valor de RMSE da melhor época
        raw_preds, target, preds = learn.get_X_preds(X[splits[1]], y[splits[1]])
        #print(mean_squared_error(y_true=target,y_pred=raw_preds,squared=False))
        intermediate_value = mean_squared_error(y_true=target, y_pred=raw_preds, squared=False)
    
    # Salvar o modelo e os resultados do trial
    folder_path = "./optuna_tests/objective_InceptionTimePlus/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    file_path = os.path.join(folder_path, "{}.pickle".format(trial.number))
    if os.path.exists(file_path):
        os.remove(file_path)
    
    with open(file_path, "wb") as fout:
        pickle.dump(learn, fout)
    
    return intermediate_value

# Configurar e rodar o estudo Optuna
study_ic = run_optuna_study(objective_InceptionTimePlus,sampler= optuna.samplers.TPESampler(n_startup_trials=500,seed=1),n_trials=1500,gc_after_trial=True,direction="minimize",show_plots=False)

print(f"O Melhor modelo foi o de número {study_ic.best_trial.number}")
print("Best hyperparameters: ", study_ic.best_trial.params)
