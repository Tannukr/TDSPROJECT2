import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import openai

# # Load the AI Proxy Token
# API_TOKEN = os.environ.get("eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIyZjIwMDE2MzJAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.6t-UFH93uouFPUZorlp4I3v35T1YGQ_D2myjPoAlTSE")
API_TOKEN = os.environ.get("AIPROXY_TOKEN")

if not API_TOKEN:
    raise ValueError("AIPROXY_TOKEN environment variable not set.")

# Initialize OpenAI
openai.api_key = API_TOKEN

def analyze_data(filename):
    try:
        # Load the dataset using pandas
        data = pd.read_csv(filename)
        print(f"Dataset '{filename}' loaded successfully.")
        print(f"Shape of the dataset: {data.shape}")
        print(f"Columns: {data.columns.tolist()}")
        
        # Summary statistics
        summary = {
            "shape": data.shape,
            "columns": data.columns.tolist(),
            "missing_values": data.isnull().sum().to_dict(),
            "summary_stats": data.describe(include='all').to_dict()
        }
        print("Summary statistics generated.")
        
        return data, summary
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None

def ask_llm(prompt):
    try:
        response = openai.Completion.create(
            model="gpt-4o-mini",
            prompt=prompt,
            max_tokens=1000
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(f"Error with AI analysis: {e}")
        return None

def analyze_data_generic(filename):
    # Load and summarize the dataset
    data, summary = analyze_data(filename)
    if data is None:
        return
    
    # Ask LLM for insights
    llm_prompt = (
        f"The dataset '{filename}' contains the following columns: {summary['columns']}.\n"
        f"Here are some summary statistics:\n{summary['summary_stats']}.\n"
        f"Here are the missing values per column:\n{summary['missing_values']}.\n"
        "Can you suggest interesting analyses or insights based on this data?"
    )
    insights = ask_llm(llm_prompt)
    if insights:
        print(f"LLM Insights:\n{insights}")
    
    # Create visualizations
    create_visualizations(data, filename)
    
    # Generate README.md
    generate_readme(filename, summary, insights, [f"{filename}_correlation_heatmap.png"])
    
def create_visualizations(df, filename):
    try:
        # Correlation Heatmap (if numeric columns exist)
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 1:
            plt.figure(figsize=(10, 8))
            correlation_matrix = df[numeric_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
            plt.title('Correlation Heatmap')
            heatmap_filename = f"{filename}_correlation_heatmap.png"
            plt.savefig(heatmap_filename)
            plt.close()
            print(f"Correlation heatmap saved as '{heatmap_filename}'.")
    except Exception as e:
        print(f"Error generating visualizations: {e}")

def generate_readme(filename, summary, insights, images):
    try:
        # Construct the README content
        readme_content = f"# Automated Analysis Report for {filename}\n\n"
        readme_content += "## Summary\n\n"
        readme_content += f"- **Shape:** {summary['shape'][0]} rows, {summary['shape'][1]} columns\n"
        readme_content += f"- **Columns:** {', '.join(summary['columns'])}\n\n"
        readme_content += "### Missing Values\n\n"
        for col, missing in summary['missing_values'].items():
            readme_content += f"- **{col}:** {missing} missing values\n"
        readme_content += "\n### Summary Statistics\n\n"
        readme_content += "```json\n"
        readme_content += f"{summary['summary_stats']}\n"
        readme_content += "```\n\n"
        readme_content += "## Insights\n\n"
        readme_content += f"{insights}\n\n"
        readme_content += "## Visualizations\n\n"
        for img in images:
            readme_content += f"![{img}]({img})\n\n"
        
        # Save to README.md
        with open("README.md", "w") as f:
            f.write(readme_content)
        print("README.md generated successfully.")
    except Exception as e:
        print(f"Error generating README.md: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <dataset.csv>")
        sys.exit(1)

    file_path = sys.argv[1]
    analyze_data_generic(file_path)
