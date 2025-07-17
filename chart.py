import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse
import sys
import warnings

warnings.filterwarnings('ignore')
plt.style.use('default')
sns.set_palette("husl")

def load_data(file_path):
    try:
        file_ext = Path(file_path).suffix.lower()

        if file_path.startswith(('http://', 'https://')):
            import requests
            from io import StringIO
            response = requests.get(file_path)
            response.raise_for_status()
            if file_ext == '.csv' or file_path.endswith('.csv'):
                return pd.read_csv(StringIO(response.text))
            elif 'json' in file_path:
                return pd.read_json(StringIO(response.text))
        else:
            if file_ext == '.csv':
                return pd.read_csv(file_path)
            elif file_ext in ['.xlsx', '.xls']:
                return pd.read_excel(file_path)
            elif file_ext == '.json':
                return pd.read_json(file_path)
            elif file_ext == '.parquet':
                return pd.read_parquet(file_path)
            elif file_ext == '.tsv':
                return pd.read_csv(file_path, sep='\t')
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def validate_columns(df, columns):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        print(f"Error: Missing columns: {missing}")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

def setup_plot(figsize=(10, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    return fig, ax

def save_or_show(fig, output):
    if output:
        fig.savefig(output, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output}")
    else:
        plt.show()

def plot_counts(df, column, top_n=20, chart_type='auto', output=None, title=None):
    validate_columns(df, [column])
    value_counts = df[column].value_counts().head(top_n)

    if chart_type == 'auto':
        unique_vals = df[column].nunique()
        chart_type = 'line' if unique_vals > 30 and pd.api.types.is_numeric_dtype(df[column]) else 'bar'

    fig, ax = setup_plot()

    if chart_type == 'bar':
        bars = ax.bar(value_counts.index.astype(str), value_counts.values,
                      color=plt.cm.Set2(np.linspace(0, 1, len(value_counts))))
        ax.set_xticklabels(value_counts.index.astype(str), rotation=45, ha='right')
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height, f'{int(height)}', ha='center', va='bottom')
    elif chart_type == 'line':
        ax.plot(value_counts.index.astype(str), value_counts.values, marker='o')
        ax.set_xticklabels(value_counts.index.astype(str), rotation=45, ha='right')
    else:
        print(f"Unsupported chart type for counts: {chart_type}")
        return

    ax.set_xlabel(column)
    ax.set_ylabel('Count')
    
    # Use custom title if provided, otherwise use default
    plot_title = title if title else f"Distribution of {column}"
    ax.set_title(plot_title)
    
    save_or_show(fig, output)

def plot_x_vs_y(df, x, y, chart_type='auto', output=None, title=None):
    validate_columns(df, [x, y])
    data = df[[x, y]].dropna()

    if chart_type == 'auto':
        if pd.api.types.is_numeric_dtype(data[y]) and not pd.api.types.is_categorical_dtype(data[x]):
            chart_type = 'scatter'
        else:
            chart_type = 'line'

    fig, ax = setup_plot()

    if chart_type == 'scatter':
        ax.scatter(data[x], data[y], alpha=0.6)
    elif chart_type == 'line':
        ax.plot(data[x], data[y], marker='o')
    elif chart_type == 'bar':
        grouped = data.groupby(x)[y].mean().reset_index()
        ax.bar(grouped[x].astype(str), grouped[y])
        ax.set_xticklabels(grouped[x].astype(str), rotation=45, ha='right')
    else:
        print(f"Unsupported chart type: {chart_type}")
        return

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    
    # Use custom title if provided, otherwise use default
    plot_title = title if title else f"{y} vs {x}"
    ax.set_title(plot_title)
    
    save_or_show(fig, output)

def main():
    parser = argparse.ArgumentParser(description="Quick Data Visualizer")
    parser.add_argument("file_path", help="Path to your CSV, Excel, JSON, etc.")
    parser.add_argument("--column", help="Column to plot counts for (Mode 1)")
    parser.add_argument("--x", help="X-axis field (Mode 2)")
    parser.add_argument("--y", help="Y-axis field (Mode 2)")
    parser.add_argument("--chart_type", choices=["bar", "scatter", "line", "auto"], default="auto")
    parser.add_argument("--top_n", type=int, default=20)
    parser.add_argument("--output", help="Path to save plot (optional)")
    parser.add_argument("--title", help="Custom plot title")
    args = parser.parse_args()

    df = load_data(args.file_path)
    print(f"Loaded data: {df.shape[0]} rows × {df.shape[1]} columns")

    if args.column:
        plot_counts(df, args.column, args.top_n, args.chart_type, args.output, args.title)
    elif args.x and args.y:
        plot_x_vs_y(df, args.x, args.y, args.chart_type, args.output, args.title)
    else:
        print("❌ Error: Specify either --column for counts or both --x and --y for x vs y plot.")
        sys.exit(1)

if __name__ == "__main__":
    main()