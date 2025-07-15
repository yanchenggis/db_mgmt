import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
import warnings
import json
warnings.filterwarnings('ignore')

def load_data(file_path):
    """Load data from file path or URL"""
    import requests
    from io import StringIO
    
    # Check if it's a URL
    if file_path.startswith(('http://', 'https://')):
        response = requests.get(file_path)
        response.raise_for_status()
        
        if file_path.endswith('.json') or 'json' in file_path:
            json_data = response.json()
            
            # Handle different JSON structures
            if isinstance(json_data, list):
                return pd.DataFrame(json_data)
            elif isinstance(json_data, dict):
                if 'features' in json_data:
                    # GeoJSON format
                    features = json_data['features']
                    flattened = []
                    for feature in features:
                        row = {}
                        if 'properties' in feature:
                            row.update(feature['properties'])
                        if 'geometry' in feature:
                            row['geometry_type'] = feature['geometry'].get('type', 'Unknown')
                        flattened.append(row)
                    return pd.DataFrame(flattened)
                else:
                    try:
                        return pd.DataFrame([json_data])
                    except:
                        return pd.json_normalize(json_data)
        elif file_path.endswith('.csv'):
            return pd.read_csv(StringIO(response.text))
        elif file_path.endswith('.geojson'):
            return pd.DataFrame(gpd.read_file(StringIO(response.text)).drop(columns='geometry'))
    
    # Local file handling
    file_ext = Path(file_path).suffix.lower()
    
    if file_ext == '.csv':
        return pd.read_csv(file_path)
    elif file_ext in ['.xlsx', '.xls']:
        return pd.read_excel(file_path)
    elif file_ext in ['.shp', '.geojson', '.gpkg']:
        gdf = gpd.read_file(file_path)
        return pd.DataFrame(gdf.drop(columns='geometry'))
    elif file_ext == '.json':
        with open(file_path, 'r') as f:
            json_data = json.load(f)
        
        if isinstance(json_data, list):
            return pd.DataFrame(json_data)
        elif isinstance(json_data, dict):
            if 'features' in json_data:
                features = json_data['features']
                flattened = []
                for feature in features:
                    row = {}
                    if 'properties' in feature:
                        row.update(feature['properties'])
                    if 'geometry' in feature:
                        row['geometry_type'] = feature['geometry'].get('type', 'Unknown')
                    flattened.append(row)
                return pd.DataFrame(flattened)
            else:
                try:
                    return pd.DataFrame([json_data])
                except:
                    return pd.json_normalize(json_data)
    elif file_ext == '.parquet':
        return pd.read_parquet(file_path)
    elif file_ext == '.tsv':
        return pd.read_csv(file_path, sep='\t')
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")

def clean_and_export(input_file, output_file, columns=None, drop_nulls=False, null_threshold=None):
    """Clean data and export to new file"""
    
    print(f"Loading data from: {input_file}")
    df = load_data(input_file)
    print(f"Original data: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Select specific columns if provided
    if columns:
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            print(f"Warning: These columns don't exist: {missing_cols}")
        
        available_cols = [col for col in columns if col in df.columns]
        df = df[available_cols]
        print(f"Selected {len(available_cols)} columns: {available_cols}")
    
    # Drop rows with any null values
    if drop_nulls:
        df = df.dropna()
        print(f"After dropping nulls: {df.shape[0]} rows")
    
    # Drop columns with high null percentage
    if null_threshold:
        original_cols = df.columns.tolist()
        df = df.loc[:, df.isnull().mean() < (null_threshold/100)]
        dropped_cols = [col for col in original_cols if col not in df.columns]
        if dropped_cols:
            print(f"Dropped columns with >{null_threshold}% nulls: {dropped_cols}")
    
    # Export based on output file extension
    output_ext = Path(output_file).suffix.lower()
    
    if output_ext == '.csv':
        df.to_csv(output_file, index=False)
    elif output_ext in ['.xlsx', '.xls']:
        df.to_excel(output_file, index=False)
    elif output_ext == '.json':
        df.to_json(output_file, orient='records', indent=2)
    elif output_ext == '.parquet':
        df.to_parquet(output_file, index=False)
    elif output_ext == '.tsv':
        df.to_csv(output_file, sep='\t', index=False)
    else:
        print(f"Unsupported output format: {output_ext}. Saving as CSV.")
        df.to_csv(output_file.replace(output_ext, '.csv'), index=False)
    
    print(f"Cleaned data saved: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Output file: {output_file}")
    
    return df

def main():
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python clean_data.py <input_file> <output_file> [options]")
        print("\nOptions:")
        print("  --columns col1,col2,col3    Select specific columns")
        print("  --drop-nulls               Drop rows with any null values")
        print("  --null-threshold 50        Drop columns with >50% nulls")
        print("\nExamples:")
        print("  python clean_data.py data.csv clean_data.csv")
        print("  python clean_data.py data.csv clean_data.xlsx --columns name,age,city")
        print("  python clean_data.py data.csv clean_data.json --drop-nulls")
        print("  python clean_data.py https://example.com/data.csv local_clean.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # Parse options
    columns = None
    drop_nulls = False
    null_threshold = None
    
    i = 3
    while i < len(sys.argv):
        if sys.argv[i] == '--columns' and i + 1 < len(sys.argv):
            columns = sys.argv[i + 1].split(',')
            i += 2
        elif sys.argv[i] == '--drop-nulls':
            drop_nulls = True
            i += 1
        elif sys.argv[i] == '--null-threshold' and i + 1 < len(sys.argv):
            null_threshold = float(sys.argv[i + 1])
            i += 2
        else:
            i += 1
    
    try:
        clean_and_export(input_file, output_file, columns, drop_nulls, null_threshold)
        print("Done!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()