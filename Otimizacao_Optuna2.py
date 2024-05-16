def objective_InceptionTimePlus(trial):
    
    nf = trial.suggest_int('nf', 16, 64)  # Número de filtros
    nb_filters = trial.suggest_int('nb_filters', 32, 128)  # Número de filtros de entrada
    fc_dropout = trial.suggest_float('fc_dropout', 0.0, 0.5)  # Dropout na camada fully connected
    ks = trial.suggest_int('ks', 10, 50)  # Tamanho do kernel
    conv_dropout = trial.suggest_float('conv_dropout', 0.0, 0.5)  # Dropout nas camadas convolucionais
    sa = trial.suggest_categorical('sa', [True, False])  # Self-attention
    se = trial.suggest_categorical('se', [True, False])  # Squeeze-and-Excitation
    arch_config = {
        'nf': nf,
        'nb_filters': nb_filters,
        'fc_dropout': fc_dropout,
        'ks': ks,
        'conv_dropout': conv_dropout,
        'sa': sa,
        'se': se
    }
    learning_rate_model = trial.suggest_float("learning_rate_model", 1e-5, 1e-2, log=True)  # search through all float values between 0.0 and 0.5 in log increment steps
    Huber_delta = trial.suggest_float("Huber_delta", 1, 2)
    
    standardize_sample = trial.suggest_categorical('by_sample', [True, False])
    standardize_var = trial.suggest_categorical('by_var', [True, False])
    arch = XceptionTimePlus
    learn = TSForecaster(X, y, splits=splits, path='models', tfms=tfms,
                        batch_tfms=TSStandardize(by_sample=standardize_sample, by_var=standardize_var),arch=arch,
                        arch_config= arch_config, metrics=[rmse], cbs=FastAIPruningCallback(trial), device=device,
                        loss_func=HuberLoss('mean',Huber_delta),seed=1)
    
    with ContextManagers([learn.no_bar(),learn.no_logging()]):
            learn.fit_one_cycle(550, lr_max=learning_rate_model)
            intermediate_value = learn.recorder.values[-1][-1]
    folder_path = "./optuna_tests/objective_InceptionTimePlus/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    file_path = os.path.join(folder_path, "{}.pickle".format(trial.number))
    if os.path.exists(file_path):
        os.remove(file_path)
    # Salva o novo arquivo
    with open(file_path, "wb") as fout:
        pickle.dump(learn, fout)
    return intermediate_value
study_ic = run_optuna_study(objective_InceptionTimePlus,sampler= optuna.samplers.TPESampler(n_startup_trials=500,seed=1),n_trials=1500,gc_after_trial=True,direction="minimize",show_plots=False)
print(f"O Melhor modelo foi o de número {study_ic.best_trial.number}")
