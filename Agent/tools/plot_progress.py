import os
import sys
import csv
import matplotlib.pyplot as plt
from pathlib import Path
import Agent.agent as agent

current_file = Path(__file__).resolve()
project_root = current_file.parents[2]    
progress_path = project_root / "progress.csv"  


if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


def plot_progress(progress_path):

    if not Path(progress_path).exists():
        print(f"Error: File not found at {progress_path}")
        return

    with open(progress_path, mode="r") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            print("The CSV file is empty.")
            return
        
        rows = list(reader)
        num_rows = len(rows)
        is_end_loop = getattr(agent, 'end_loop', False)
        
        if num_rows > 0 and is_end_loop:
            experiments = [int(row[0]) for row in rows]
            val_loss = [float(row[1]) for row in rows]
            
            plt.figure(figsize=(10, 6))
            plt.plot(experiments, val_loss, label="Validation Loss over Experiments", marker='o')
            plt.xlabel("Experiment")
            plt.ylabel("Loss")
            plt.title("Validation Loss")
            plt.legend()
            plt.grid()
            plt.show()
        else:
            print(f"Skipping plot. Rows: {num_rows}, agent.end_loop: {is_end_loop}")


if __name__ == "__main__":
    plot_progress(progress_path)