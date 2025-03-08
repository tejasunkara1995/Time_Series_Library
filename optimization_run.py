import optuna
import os
import random

def objective(trial):
    """Objective function for Bayesian Optimization"""
    e_layers = trial.suggest_int("e_layers", 3, 8)  # Choose between 3 to 8 layers
    n_heads = trial.suggest_int("n_heads", 3, 8)  # Choose between 3 to 8 attention heads
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])  # Pick from fixed options
    patch_len = trial.suggest_int("patch_len", 6, 16, step=2)  # Choose even values between 6 to 16
    stride = trial.suggest_int("stride", 2, patch_len - 1, step=2)  # Ensure stride < patch_len and is even
    learning_rate = trial.suggest_categorical("learning_rate", [0.001, 0.0001, 0.00001])  # Discrete values

    # Ensure stride < patch_len and is even
    if stride >= patch_len or stride % 2 != 0:
        return float('inf')  # Return a bad score to avoid invalid runs

    # Define model ID
    model_id = f"trial_e{e_layers}_h{n_heads}_b{batch_size}_p{patch_len}_s{stride}_lr{learning_rate}"
    
    # Run the model with selected hyperparameters
    command = f"""
    python -u run.py --task_name long_term_forecast --is_training 1
      --root_path ./dataset/illness/ --data_path national_illness_24_plus_cdc_influenza_weekly_dataset.csv
      --model_id "{model_id}"
      --model PatchTST2 --data custom --features MS --seq_len 36 --label_len 0 --pred_len 12
      --e_layers {e_layers} --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp'
      --n_heads {n_heads} --batch_size {batch_size} --patch_len {patch_len} --stride {stride}
      --num_workers 20 --target ILITOTAL --learning_rate {learning_rate} --with_retrain 0
    """
    
    print(f"Running command:\n{command}")
    os.system(command)

    # Read the sMAPE result from a log file (assuming run.py writes sMAPE to result_log.txt)
    try:
        with open("result_log.txt", "r") as file:
            smape = float(file.readline().strip())
    except:
        smape = float('inf')  # If no valid output, assume high error

    return smape  # Lower sMAPE is better

# Run Bayesian Optimization for 50 trials
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

# Get best parameters
print("Best Hyperparameters:", study.best_params)

# Save best parameters
with open("best_params.txt", "w") as file:
    file.write(str(study.best_params))