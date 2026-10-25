# HR Attrition Analysis - IBM HR Analytics dataset
# Works as a normal .py file, or open in VS Code / Jupyter (each "# %%" is a cell).
# Install once:  pip install pandas matplotlib seaborn
# Put "WA_Fn-UseC_-HR-Employee-Attrition.csv" (or your CSV name below) in the same folder.

# %% STEP 1: Import libraries and load data
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

CSV_FILE = "WA_Fn-UseC_-HR-Employee-Attrition.csv"   # change if your file name differs
os.makedirs("images", exist_ok=True)
sns.set_theme(style="whitegrid")

df = pd.read_csv(CSV_FILE, encoding="utf-8-sig")      # utf-8-sig removes a hidden BOM character
df.columns = df.columns.str.strip()
print("Shape:", df.shape)
print(df.head())

# %% STEP 2: Understand the data
df.info()
print("\nMissing values:\n", df.isnull().sum().sum())
print("Duplicate rows:", df.duplicated().sum())
print("\nAttrition counts:\n", df["Attrition"].value_counts())

# %% STEP 3: Clean the data
# Drop columns with the same value for everyone (no analytical value)
drop_cols = [c for c in ["EmployeeCount", "StandardHours", "Over18"] if c in df.columns]
df = df.drop(columns=drop_cols)
df = df.drop_duplicates()

# Numeric flag: 1 = left, 0 = stayed (makes averages = attrition rate)
df["AttritionFlag"] = (df["Attrition"] == "Yes").astype(int)

# Age groups
df["AgeGroup"] = pd.cut(
    df["Age"], bins=[17, 25, 35, 45, 55, 100],
    labels=["18-25", "26-35", "36-45", "46-55", "55+"]
)

# Income bands (4 equal-sized groups)
df["IncomeBand"] = pd.qcut(
    df["MonthlyIncome"], q=4, labels=["Low", "Medium", "High", "Very High"]
)

# Tenure groups
df["TenureGroup"] = pd.cut(
    df["YearsAtCompany"], bins=[-1, 2, 5, 10, 50],
    labels=["0-2 yrs", "3-5 yrs", "6-10 yrs", "10+ yrs"]
)

# Readable labels for rating columns (1-4 scale in this dataset)
sat_map = {1: "Low", 2: "Medium", 3: "High", 4: "Very High"}
for col in ["JobSatisfaction", "EnvironmentSatisfaction", "WorkLifeBalance"]:
    df[col + "Label"] = df[col].map(sat_map)

print("\nCleaned shape:", df.shape)

# %% STEP 4: Overall attrition rate
total = len(df)
left = int(df["AttritionFlag"].sum())
rate = left / total * 100
print(f"Total employees : {total}")
print(f"Employees left  : {left}")
print(f"Attrition rate  : {rate:.2f}%")

# %% STEP 5: Attrition rate by each factor
def attrition_by(col):
    """Return attrition rate (%) and headcount for each group in a column."""
    out = (
        df.groupby(col, observed=True)["AttritionFlag"]
        .agg(Employees="count", Left="sum", AttritionRate="mean")
        .reset_index()
    )
    out["AttritionRate"] = (out["AttritionRate"] * 100).round(2)
    return out.sort_values("AttritionRate", ascending=False)

factors = ["Department", "JobRole", "OverTime", "AgeGroup", "IncomeBand",
           "TenureGroup", "MaritalStatus", "Gender", "BusinessTravel",
           "JobSatisfactionLabel", "WorkLifeBalanceLabel"]

for f in factors:
    print(f"\n--- Attrition by {f} ---")
    print(attrition_by(f).to_string(index=False))

# %% STEP 6: Visualizations
def bar_chart(col, title, filename, rotate=0):
    data = attrition_by(col)
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(data=data, x=col, y="AttritionRate", color="#2f6fdb")
    ax.axhline(rate, color="red", linestyle="--", label=f"Overall {rate:.1f}%")
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=9)
    plt.title(title)
    plt.ylabel("Attrition Rate (%)")
    plt.xlabel("")
    plt.xticks(rotation=rotate, ha="right" if rotate else "center")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"images/{filename}", dpi=150)
    plt.show()

bar_chart("OverTime", "Attrition Rate by Overtime", "attrition_overtime.png")
bar_chart("Department", "Attrition Rate by Department", "attrition_department.png", rotate=15)
bar_chart("JobRole", "Attrition Rate by Job Role", "attrition_jobrole.png", rotate=40)
bar_chart("AgeGroup", "Attrition Rate by Age Group", "attrition_age.png")
bar_chart("IncomeBand", "Attrition Rate by Income Band", "attrition_income.png")
bar_chart("TenureGroup", "Attrition Rate by Tenure", "attrition_tenure.png")
bar_chart("JobSatisfactionLabel", "Attrition Rate by Job Satisfaction", "attrition_satisfaction.png")

# Overall pie chart
plt.figure(figsize=(5, 5))
df["Attrition"].value_counts().plot.pie(
    autopct="%1.1f%%", colors=["#4caf50", "#e53935"], startangle=90, ylabel="")
plt.title("Employees Stayed vs Left")
plt.tight_layout()
plt.savefig("images/attrition_overall.png", dpi=150)
plt.show()

# Monthly income distribution: stayed vs left
plt.figure(figsize=(9, 5))
sns.boxplot(data=df, x="Attrition", y="MonthlyIncome", palette=["#4caf50", "#e53935"], hue="Attrition", legend=False)
plt.title("Monthly Income: Stayed vs Left")
plt.tight_layout()
plt.savefig("images/income_boxplot.png", dpi=150)
plt.show()

# Correlation heatmap (numeric columns only)
num = df.select_dtypes(include="number").drop(columns=["EmployeeNumber"], errors="ignore")
plt.figure(figsize=(14, 10))
sns.heatmap(num.corr(), cmap="coolwarm", center=0, annot=False)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("images/correlation_heatmap.png", dpi=150)
plt.show()

print("\nTop correlations with attrition:")
print(num.corr()["AttritionFlag"].drop("AttritionFlag").sort_values().round(3))

# %% STEP 7: High-risk group (overtime + low income + short tenure)
risk = df[(df["OverTime"] == "Yes") &
          (df["IncomeBand"] == "Low") &
          (df["YearsAtCompany"] <= 3)]
print(f"\nHigh-risk group size: {len(risk)}")
print(f"High-risk attrition rate: {risk['AttritionFlag'].mean() * 100:.1f}% (overall {rate:.1f}%)")

# %% STEP 8: Export cleaned data for Power BI
df.to_csv("hr_attrition_cleaned.csv", index=False)
print("\nSaved hr_attrition_cleaned.csv -> import this file into Power BI")
