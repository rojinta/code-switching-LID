import re
import matplotlib.pyplot as plt
import os

def parse_logs(log_file):
    """
    Parses a single log file to extract the best metrics for the corresponding mask probability.
    """
    best_metrics = {}
    f1_macro_pattern = re.compile(r"F1 Macro: (\d+\.\d+)")
    metrics_pattern = re.compile(
    r"Dev F1 Macro = (\d+\.\d+), Dev F1 Weighted = (\d+\.\d+)"
    r"Dev Precision Macro = (\d+\.\d+), Dev Recall Macro = (\d+\.\d+)"
    r"Dev Accuracy = (\d+\.\d+)"
)

    with open(log_file, 'r') as f:
        for line in f:
            # Detect the mask probability
            if "Starting training with mask probability" in line:
                current_prob = float(re.search(r"mask probability: (\d+\.\d+)", line).group(1))
            
            # Detect metrics for the current epoch
            if "Saved new best model" in line:
                f1_macro_match = f1_macro_pattern.search(line)
                if f1_macro_match:
                    f1_macro = float(f1_macro_match.group(1))
                    
                    # Extract additional metrics from the preceding line
                    prev_line = next(f, None)
                    metrics_match = metrics_pattern.search(prev_line)
                    if metrics_match:
                        f1_macro, f1_weighted, precision_macro, recall_macro, accuracy = map(float, metrics_match.groups())
                        best_metrics[current_prob] = {
                            "F1 Macro": f1_macro,
                            "F1 Weighted": f1_weighted,
                            "Precision Macro": precision_macro,
                            "Recall Macro": recall_macro,
                            "Accuracy": accuracy,
                        }
                    else:
                        print(f"Metrics not found in line: {prev_line}")
    return best_metrics


def aggregate_metrics(log_dir):
    aggregated_metrics = {}
    for log_file in os.listdir(log_dir):
        if log_file.endswith(".log"):  # Process only log files
            full_path = os.path.join(log_dir, log_file)
            file_metrics = parse_logs(full_path)
            aggregated_metrics.update(file_metrics)
    return aggregated_metrics

def plot_metrics(best_metrics, output_dir):
    probabilities = sorted(best_metrics.keys())
    metrics = ["F1 Macro", "F1 Weighted", "Precision Macro", "Recall Macro", "Accuracy"]
    
    os.makedirs(output_dir, exist_ok=True)

    for metric in metrics:
        values = [best_metrics[prob][metric] for prob in probabilities]
        plt.figure(figsize=(8, 6))
        plt.plot(probabilities, values, marker='o', label=metric)
        plt.title(f"{metric} vs. Mask Probability")
        plt.xlabel("Mask Probability")
        plt.ylabel(metric)
        plt.grid()
        plt.savefig(os.path.join(output_dir, f"{metric.replace(' ', '_').lower()}_vs_mask_probability.png"))
        plt.close()

# Paths for logs and plots
log_directory = "../logs"  # Folder containing the log files
plot_directory = "../plot"  # Folder to save the plots

# Parse logs and generate plots
best_metrics = aggregate_metrics(log_directory)
plot_metrics(best_metrics, plot_directory)
