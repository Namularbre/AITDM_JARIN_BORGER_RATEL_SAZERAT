import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

def load_exp_data(exp_dir):
    """Its goes through the ‘seed’ subfolders and extracts the accuracies."""
    global_accs = []
    client_accs = {0: [], 1: [], 2: []} # Ajuster si vous avez plus de clients
    
    # Search for all subfolders beginning with 'seed'
    seed_dirs = [d for d in os.listdir(exp_dir) if d.startswith('seed') and os.path.isdir(os.path.join(exp_dir, d))]
    
    if not seed_dirs:
        # If there is no ‘seed’ subfolder, try reading directly from the root of the folder
        seed_dirs = [""]
        
    for seed in seed_dirs:
        json_path = os.path.join(exp_dir, seed, "results.json")
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    # Test global recovery (or ‘val’, as you prefer)
                    global_accs.append(data["test"]["accuracy"])
                    # Retrieval by customer (we use the ‘val per client’ value as in your JSON)
                    for c_id in client_accs.keys():
                        if str(c_id) in data["val_per_client"]:
                            client_accs[c_id].append(data["val_per_client"][str(c_id)]["accuracy"])
                except Exception as e:
                    print(f"Read error in {json_path}: {e}")
                    
    if not global_accs:
        return None
        
    # Calculating means and standard deviations
    results = {
        "Global": (np.mean(global_accs), np.std(global_accs)),
        "Client 0": (np.mean(client_accs[0]), np.std(client_accs[0])) if client_accs[0] else (0, 0),
        "Client 1": (np.mean(client_accs[1]), np.std(client_accs[1])) if client_accs[1] else (0, 0),
        "Client 2": (np.mean(client_accs[2]), np.std(client_accs[2])) if client_accs[2] else (0, 0),
    }
    return results

def main():
    parser = argparse.ArgumentParser(description="Generate a comparative chart based on the Flower results.")
    parser.add_argument("folders", nargs="+", help="List of experiment files to be compared (ex: outputs/experiments/fedavg)")
    args = parser.parse_args()
    
    all_data = {}
    for folder in args.folders:
        exp_name = os.path.basename(os.path.normpath(folder))
        data = load_exp_data(folder)
        if data:
            all_data[exp_name] = data
        else:
            print(f"⚠️ No valid “results.json” file found in {folder}")

    if not all_data:
        print("❌ No data to display.")
        return

    # Graph configuration
    categories = ["Global", "Client 0", "Client 1", "Client 2"]
    x = np.arange(len(categories))
    width = 0.25  # Bar width
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the bars for each experiment specified in the parameters
    for i, (exp_name, metrics) in enumerate(all_data.items()):
        means = [metrics[cat][0] for cat in categories]
        stds = [metrics[cat][1] for cat in categories]
        
        # Offset the bars so that they do not overlap
        pos = x + (i - len(all_data)/2 + 0.5) * width
        
        ax.bar(pos, means, width, yerr=stds, label=exp_name, capsize=5, alpha=0.85)

    # Style and legends
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Comparison of Overall Accuracy and Accuracy by Client (Mean ± Std)', fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(fontsize=10, loc='lower left')
    
    plt.tight_layout()
    
    # Saving and displaying
    output_fig = "comparison_results.png"
    plt.savefig(output_fig, dpi=300)
    print(f"🎉 Graph successfully saved as : '{output_fig}'")
    plt.show()

if __name__ == "__main__":
    main()