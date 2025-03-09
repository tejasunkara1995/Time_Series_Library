import optuna
import os
import subprocess
import re

# Define log file paths
output_log_file = "optuna_output.log"
best_params_file = "best_params.txt"
error_log_file = "optuna_errors.log"

def log_message(message, log_file):
    """Writes logs to a file and prints to console"""
    with open(log_file, "a") as file:
        file.write(message + "\n")
    print(message)

def extract_mean_smape():
    """Extracts the last reported 'mean smape over horizons' from the output log"""
    smape = float('inf')  # Default to high value (if no valid sMAPE is found)
    
    if os.path.exists(output_log_file):
        with open(output_log_file, "r") as file:
            lines = file.readlines()
            for line in reversed(lines):  # Search from the end to get the latest value
                match = re.search(r"mean smape over horizons:\s*([\d.]+)", line, re.IGNORECASE)
                if match:
                    smape = float(match.group(1))
                    break  # Stop searching after finding the latest sMAPE
    
    return smape

def objective(trial):
    """Optimized Objective Function with Given Parameter Ranges"""
    e_layers = trial.suggest_int("e_layers", 2, 10)  # Encoder layers
    n_heads = trial.suggest_int("n_heads", 2, 10)  # Attention heads
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256])  # Batch sizes
    patch_len = trial.suggest_categorical("patch_len", [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24])  # Patch lengths
    stride = trial.suggest_categorical("stride", [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22])  # Even strides only
    learning_rate = trial.suggest_categorical("learning_rate", [0.001, 0.0001, 0.00001])  # Learning rates

    # Define model ID
    model_id = f"trial_e{e_layers}_h{n_heads}_b{batch_size}_p{patch_len}_s{stride}_lr{learning_rate}"

    # Formatted command with exact structure
    command = f"""python -u run.py --task_name long_term_forecast --is_training 1 \
      --root_path ./dataset/illness/ \
      --data_path national_illness_24_plus_cdc_influenza_weekly_dataset.csv \
      --model_id {model_id} \
      --model PatchTST2 \
      --data custom \
      --features MS \
      --seq_len 36 \
      --label_len 0 \
      --pred_len 12 \
      --e_layers {e_layers} \
      --d_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --n_heads {n_heads} \
      --batch_size {batch_size} \
      --patch_len {patch_len} \
      --stride {stride} \
      --num_workers 20 \
      --target ILITOTAL \
      --learning_rate {learning_rate} \
      --with_retrain 0"""

    log_message(f"Running: {model_id}", output_log_file)

    try:
        os.system(command)  # Execute model training
    except Exception as e:
        log_message(f"Error running {model_id}: {e}", error_log_file)
        return float('inf')

    # Extract the latest 'mean smape over horizons' from the log file
    smape = extract_mean_smape()
    log_message(f"Trial {trial.number}: {model_id}, Mean sMAPE: {smape}", output_log_file)

    return smape  # Lower values are better

# Optimize with parallel processing (if available)
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30, n_jobs=2)  # Reduce trials, add parallel jobs

# Save best parameters
best_params = study.best_params
with open(best_params_file, "w") as file:
    file.write(str(best_params))

log_message(f"Best parameters saved to {best_params_file}", output_log_file)