import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openai

openai.api_key = os.environ.get("eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIyZjIwMDE2MzJAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.6t-UFH93uouFPUZorlp4I3v35T1YGQ_D2myjPoAlTSE")

def load_data(filename):
    try:
        df = pd.read_csv("goodreads.csv")
        print(f"Loaded {filename} successfully!")
        return df
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

def basic_analysis(df):
    summary = df.describe(include="all")
    missing = df.isnull().sum()
    return summary, missing

def correlation_analysis(df):
    corr = df.corr(numeric_only=True)
    return corr

def detect_outliers(df):
    outliers = df[df.select_dtypes(include=['number']).apply(lambda x: (x - x.mean()).abs() > 3 * x.std())]
    return outliers

def create_charts(df, corr):
    # Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig("correlation_heatmap.png")
    plt.close()

def create_charts(df, corr):
    # Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig("correlation_heatmap.png")
    plt.close()

def ask_gpt(prompt):
    response = openai.Completion.create(
        engine="gpt-4o-mini",
        prompt=prompt,
        max_tokens=1000
    )
    return response['choices'][0]['text']

def write_readme(summary, missing, insights):
    with open("README.md", "w") as f:
        f.write("# Automated Data Analysis Report\n")
        f.write("\n### Summary Statistics\n")
        f.write(summary.to_markdown())
        f.write("\n\n### Missing Values\n")
        f.write(missing.to_markdown())
        f.write("\n\n### Insights from Analysis\n")
        f.write(insights)
        f.write("\n\n### Charts\n")
        f.write("![Correlation Heatmap](correlation_heatmap.png)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py dataset.csv")
        sys.exit(1)

    filename = sys.argv[1]
    df = load_data(filename)

    summary, missing = basic_analysis(df)
    corr = correlation_analysis(df)
    create_charts(df, corr)

    insights_prompt = f"""
    You are analyzing a dataset with the following summary statistics:
    {summary.to_string()}

    Missing values:
    {missing.to_string()}

    What important insights can you draw from this data?
    """

    insights = ask_gpt(insights_prompt)
    write_readme(summary, missing, insights)
    print("Analysis complete. Files generated.")
