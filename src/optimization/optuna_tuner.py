import optuna
from src.optimization.objective import objective

def tune_hyperparameters(n_trials=50):
    # Study の設定
    study = optuna.create_study(
        direction="maximize",  # 目的が最大化なら "maximize", 最小化なら "minimize"
        study_name="model_optimization"
    )
    # ハイパーパラメータ探索の開始
    study.optimize(objective, n_trials=n_trials)  # 試行回数を指定

    # 最良の結果を表示
    print(f"Best value: {study.best_value}")
    print(f"Best params: {study.best_params}")
    
    # 結果の保存
    study.trials_dataframe().to_csv("output/results/optuna_results.csv", index=False)

if __name__ == "__main__":
    tune_hyperparameters()