import csv
from pathlib import Path

import matplotlib.pyplot as plt


def plot_progress(progress_path):
    progress_path = Path(progress_path)

    if not progress_path.is_file():
        print(f"Error: File not found at {progress_path}")
        return

    experiments = []
    val_losses = []

    with progress_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)

        try:
            next(reader)  # Skip header
        except StopIteration:
            print("The CSV file is empty.")
            return

        for row in reader:
            if len(row) < 2:
                continue

            try:
                experiment = int(row[0])
                val_loss = float(row[1])
            except ValueError:
                # Skip crashed experiments or invalid rows
                continue

            experiments.append(experiment)
            val_losses.append(val_loss)

    if not experiments:
        print("No valid experiment results to plot.")
        return

    plt.figure(figsize=(10, 6))

    plt.plot(
        experiments,
        val_losses,
        marker="o",
        label="Validation Loss",
    )

    plt.xlabel("Experiment")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss over Experiments")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save plot next to progress.csv
    output_path = progress_path.with_name("progress.png")
    plt.savefig(output_path, dpi=300)

    print(f"Plot saved to: {output_path}")

    plt.close()


if __name__ == "__main__":
    from paths import get_paths

    paths = get_paths()
    plot_progress(paths["progress_path"])