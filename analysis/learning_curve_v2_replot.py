import pandas as pd
import matplotlib.pyplot as plt

# ===== Load your CSV =====
df = pd.read_csv("./xgb_18class_results/learning_curve_v2_data.csv")

# train fractions
x = df["train_fraction"]

# Each ratio r corresponds to its own columns
ratios = {
    "all": {
        "top1": df["top1_mean"],
        "top2": df["top2_mean"],
        "top3": df["top3_mean"],
    },
    "r=0.001": {
        "top1": df["r0.001_top1_mean"],
        "top2": df["r0.001_top2_mean"],
        "top3": df["r0.001_top3_mean"],
    },
    "r=0.01": {
        "top1": df["r0.01_top1_mean"],
        "top2": df["r0.01_top2_mean"],
        "top3": df["r0.01_top3_mean"],
    },
    "r=0.1": {
        "top1": df["r0.1_top1_mean"],
        "top2": df["r0.1_top2_mean"],
        "top3": df["r0.1_top3_mean"],
    },
}

# ===== 1. Plot per-ratio curves =====
for name, metrics in ratios.items():
    plt.figure(figsize=(6,4))
    plt.plot(x, metrics["top1"], marker='o', label="Top-1")
    plt.plot(x, metrics["top2"], marker='s', label="Top-2")
    plt.plot(x, metrics["top3"], marker='^', label="Top-3")
    
    plt.xlabel("train_fraction")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Performance vs Training Fraction ({name})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"plot_{name.replace('=','')}.png", dpi=200)
    plt.show()

# ===== 2. Combined plot: compare r for Top-1 / Top-2 / Top-3 =====
plt.figure(figsize=(7,5))

for name, metrics in ratios.items():
    plt.plot(x, metrics["top1"], marker='o', label=f"{name} Top-1")

for name, metrics in ratios.items():
    plt.plot(x, metrics["top2"], marker='s', label=f"{name} Top-2")

for name, metrics in ratios.items():
    plt.plot(x, metrics["top3"], marker='^', label=f"{name} Top-3")

plt.xlabel("train_fraction")
plt.ylabel("Accuracy (%)")
plt.title("Overall Top-1 / Top-2 / Top-3 vs Cache Size Ratio")
plt.legend(ncol=2)
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("plot_overall.png", dpi=200)
plt.show()
