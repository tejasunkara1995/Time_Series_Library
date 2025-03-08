import optuna
import os
import subprocess

# Define log file paths
output_log_file = "optuna_output.log"  # Logs all console outputs
best_params_file = "best_params.txt"  # Stores the best hyperparameters
error_log_file = "optuna_errors.log"  # Stores error messages

# Function to log messages to a file
def log_message(message, log_file):
    with open(log_file, "a") as file:
        file.write(message + "\n")
    print(message)  # Also print to console

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
        log_message(f"Skipping invalid config: stride={stride}, patch_len={patch_len}", output_log_file)
        return float('inf')  # Return a bad score to avoid invalid runs

    # Define model ID
    model_id = f"trial_e{e_layers}_h{n_heads}_b{batch_size}_p{patch_len}_s{stride}_lr{learning_rate}"
    
    # Corrected command using single-line string formatting (NO EXTRA SPACES AFTER `\`)
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

    log_message(f"Running command: {command}", output_log_file)

    # Execute the command safely
    try:
        process = subprocess.run(command, shell=True, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        log_message(f"Output:\n{process.stdout}", output_log_file)
    except subprocess.CalledProcessError as e:
        log_message(f"Error executing command: {e.stderr}", error_log_file)
        return float('inf')  # If command fails, return a bad score

    # Read the sMAPE result from a log file (Check if file exists)
    smape = float('inf')  # Default high error if no valid output
    if os.path.exists("result_log.txt"):
        try:
            with open("result_log.txt", "r") as file:
                smape = float(file.readline().strip())
                log_message(f"Trial: {model_id}, sMAPE: {smape}", output_log_file)
        except Exception as e:
            log_message(f"Error reading sMAPE for {model_id}: {e}", error_log_file)

    return smape  # Lower sMAPE is better

# Run Bayesian Optimization for 50 trials
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

# Get best parameters
best_params = study.best_params
log_message(f"Best Hyperparameters: {best_params}", output_log_file)

# Save best parameters to a separate file
with open(best_params_file, "w") as file:
    file.write(str(best_params))

log_message(f"Best parameters saved to {best_params_file}", output_log_file)