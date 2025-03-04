import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_excel("ENB2012_data.xlsx")
os.makedirs("outputs", exist_ok=True)

numeric_describe = data.describe().T

missing_summary = data.isna().sum().to_dict()

dtypes_summary = {col: str(dtype) for col, dtype in data.dtypes.items()}

corr = data.corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of All Features and Targets")
plt.tight_layout()
heatmap_path = os.path.join("outputs", "correlation_heatmap.png")
plt.savefig(heatmap_path, dpi=300)
plt.close()

for col in data.columns:
    plt.figure()
    sns.histplot(data[col], kde=True, color="blue", edgecolor="black")
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    hist_path = os.path.join("outputs", f"{col}_hist.png")
    plt.savefig(hist_path, dpi=300)
    plt.close()

target_columns = ["Y1", "Y2"]
for target in target_columns:
    plt.figure()
    sns.boxplot(y=data[target], color="green")
    plt.title(f"Box Plot of {target}")
    plt.ylabel(target)
    plt.tight_layout()
    boxplot_path = os.path.join("outputs", f"{target}_boxplot.png")
    plt.savefig(boxplot_path, dpi=300)
    plt.close()

analysis_dict = {
    "numeric_describe": numeric_describe.to_dict(),
    "missing_values": missing_summary,
    "data_types": dtypes_summary,
    "correlation_matrix": corr.to_dict()
}

analysis_path = os.path.join("outputs", "analysis.json")
with open(analysis_path, "w", encoding="utf-8") as f:
    json.dump(analysis_dict, f, indent=4)

print("EDA completed successfully. Results saved to 'outputs/' directory.")