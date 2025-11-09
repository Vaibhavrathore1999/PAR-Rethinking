import re
import matplotlib.pyplot as plt
import glob
import os

# Folder containing your log .txt files
LOG_FOLDER = "exp_result/PA100k/vit_b.adam/log"   # <-- change this path if needed

# Regex patterns for key metrics
patterns = {
    'epoch': re.compile(r"Epoch\s+(\d+)"),
    'train_loss': re.compile(r"Loss:\s*([\d.]+)"),
    'valid_loss': re.compile(r"valid losses\s+([\d.]+)"),
    'train_acc': re.compile(r"Evaluation on train set.*?Acc:\s*([\d.]+)", re.DOTALL),
    'train_f1': re.compile(r"Evaluation on train set.*?F1:\s*([\d.]+)", re.DOTALL),
    'valid_acc': re.compile(r"Evaluation on test set.*?Acc:\s*([\d.]+)", re.DOTALL),
    'valid_f1': re.compile(r"Evaluation on test set.*?F1:\s*([\d.]+)", re.DOTALL),
}

def parse_log(file_path):
    """Parse metrics from one log file."""
    with open(file_path, 'r') as f:
        text = f.read()

    data = {k: [] for k in patterns.keys()}

    # For each epoch, extract metrics in order
    for match in re.finditer(patterns['epoch'], text):
        epoch_num = int(match.group(1))
        epoch_text = text[match.start():text.find("------------------------------------------------------------", match.start())]
        data['epoch'].append(epoch_num)

        for key, pattern in patterns.items():
            if key == 'epoch':
                continue
            found = pattern.search(epoch_text)
            data[key].append(float(found.group(1)) if found else None)

    return data

def plot_metrics(data, log_file):
    """Plot losses and F1/accuracy curves and save as PNG."""
    epochs = data['epoch']
    if not epochs:
        print(f"⚠️ No epochs found in {log_file}")
        return

    plt.figure(figsize=(12, 8))
    
    # --- Loss curves ---
    plt.subplot(2, 1, 1)
    plt.plot(epochs, data['train_loss'], label='Train Loss', marker='o')
    plt.plot(epochs, data['valid_loss'], label='Validation Loss', marker='o')
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # --- Performance curves ---
    plt.subplot(2, 1, 2)
    plt.plot(epochs, data['train_f1'], label='Train F1', marker='s')
    plt.plot(epochs, data['valid_f1'], label='Validation F1', marker='s')
    plt.plot(epochs, data['train_acc'], label='Train Acc', marker='^', linestyle='--')
    plt.plot(epochs, data['valid_acc'], label='Validation Acc', marker='^', linestyle='--')
    plt.title("Performance Metrics")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.suptitle(os.path.basename(log_file), fontsize=14, y=1.02)

    # Save to same folder as log
    save_path = os.path.splitext(log_file)[0] + "_metrics.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close()

    print(f"✅ Saved plot: {save_path}")

def main():
    log_files = glob.glob(os.path.join(LOG_FOLDER, "*.txt"))
    if not log_files:
        print("⚠️ No log files found in", LOG_FOLDER)
        return
    
    for log_file in log_files:
        print(f"Processing: {os.path.basename(log_file)}")
        data = parse_log(log_file)
        plot_metrics(data, log_file)

if __name__ == "__main__":
    main()
