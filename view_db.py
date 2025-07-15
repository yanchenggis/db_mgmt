import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')
import requests
from io import StringIO, BytesIO


def load_data(file_path):
    """Load data based on file extension or URL"""
    import requests
    from io import StringIO
    
    # Check if it's a URL
    if file_path.startswith(('http://', 'https://')):
        response = requests.get(file_path)
        response.raise_for_status()
        
        if file_path.endswith('.json') or 'json' in file_path:
            import json
            json_data = response.json()
            
            # Handle different JSON structures
            if isinstance(json_data, list):
                # If it's a list of objects, convert directly
                return pd.DataFrame(json_data)
            elif isinstance(json_data, dict):
                # If it's a dictionary, try to flatten it
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
                    # Try to convert dict to DataFrame
                    try:
                        return pd.DataFrame([json_data])
                    except:
                        # If that fails, try to normalize
                        return pd.json_normalize(json_data)
        elif file_path.endswith('.csv'):
            return pd.read_csv(StringIO(response.text))
        elif file_path.endswith('.geojson'):
            return pd.DataFrame(gpd.read_file(StringIO(response.text)).drop(columns='geometry'))
    
    # Original local file handling
    file_ext = Path(file_path).suffix.lower()
    
    if file_ext == '.csv':
        return pd.read_csv(file_path)
    elif file_ext in ['.xlsx', '.xls']:
        return pd.read_excel(file_path)
    elif file_ext in ['.shp', '.geojson', '.gpkg']:
        gdf = gpd.read_file(file_path)
        return pd.DataFrame(gdf.drop(columns='geometry'))
    elif file_ext == '.json':
        # Better JSON handling for local files too
        with open(file_path, 'r') as f:
            json_data = json.load(f)
        
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
    elif file_ext == '.parquet':
        return pd.read_parquet(file_path)
    elif file_ext == '.tsv':
        return pd.read_csv(file_path, sep='\t')
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")

def analyze_data(df):
    """Analyze dataframe and categorize fields"""
    
    analysis = {
        'total_columns': len(df.columns),
        'total_rows': len(df),
        'unique_fields': [],
        'categorical_fields': [],
        'high_null_fields': []
    }
    
    for col in df.columns:
        null_count = df[col].isnull().sum()
        null_percentage = (null_count / len(df)) * 100
        unique_count = df[col].nunique()
        unique_ratio = unique_count / len(df)
        
        col_info = {
            'name': col,
            'dtype': str(df[col].dtype),
            'null_count': null_count,
            'null_percentage': null_percentage,
            'unique_count': unique_count
        }
        
        # High null fields (>20%)
        if null_percentage > 20:
            analysis['high_null_fields'].append(col_info.copy())
        
        # Categorize fields
        if df[col].dtype in ['int64', 'float64'] or unique_ratio > 0.5:
            # Unique/Numerical fields
            if df[col].dtype in ['int64', 'float64']:
                non_null_data = df[col].dropna()
                if len(non_null_data) > 0:
                    col_info.update({
                        'min': non_null_data.min(),
                        'max': non_null_data.max(),
                        'mean': non_null_data.mean(),
                        'median': non_null_data.median()
                    })
            analysis['unique_fields'].append(col_info)
        else:
            # Categorical fields
            value_counts = df[col].value_counts(dropna=False)
            col_info['value_counts'] = value_counts.to_dict()
            analysis['categorical_fields'].append(col_info)
    
    return analysis

def print_analysis(analysis, file_path):
    """Print analysis results to console"""
    
    print(f"\n{'='*60}")
    print(f"DATA PROFILE: {Path(file_path).name}")
    print(f"{'='*60}")
    
    # Summary
    print(f"Columns: {analysis['total_columns']}")
    print(f"Rows: {analysis['total_rows']}")
    print(f"Unique fields: {len(analysis['unique_fields'])}")
    print(f"Categorical fields: {len(analysis['categorical_fields'])}")
    print(f"High null fields (>20%): {len(analysis['high_null_fields'])}")
    
    # Unique/Numerical Fields
    if analysis['unique_fields']:
        print(f"\n{'='*60}")
        print("UNIQUE/NUMERICAL FIELDS")
        print(f"{'='*60}")
        
        for field in analysis['unique_fields']:
            print(f"\n{field['name']} ({field['dtype']})")
            print(f"  Unique values: {field['unique_count']}")
            print(f"  Null count: {field['null_count']} ({field['null_percentage']:.1f}%)")
            
            if 'min' in field:
                print(f"  Min: {field['min']:.2f}" if isinstance(field['min'], float) else f"  Min: {field['min']}")
                print(f"  Max: {field['max']:.2f}" if isinstance(field['max'], float) else f"  Max: {field['max']}")
                print(f"  Mean: {field['mean']:.2f}")
                print(f"  Median: {field['median']:.2f}" if isinstance(field['median'], float) else f"  Median: {field['median']}")
    
    # Categorical Fields
    if analysis['categorical_fields']:
        print(f"\n{'='*60}")
        print("CATEGORICAL FIELDS")
        print(f"{'='*60}")
        
        for field in analysis['categorical_fields']:
            print(f"\n{field['name']} ({field['dtype']})")
            print(f"  Null count: {field['null_count']} ({field['null_percentage']:.1f}%)")
            print("  Value counts:")
            
            for value, count in field['value_counts'].items():
                value_str = str(value) if pd.notna(value) else 'null'
                print(f"    {value_str}: {count}")
    
    # High Null Fields
    if analysis['high_null_fields']:
        print(f"\n{'='*60}")
        print("FIELDS WITH >20% NULL VALUES")
        print(f"{'='*60}")
        
        for field in analysis['high_null_fields']:
            print(f"{field['name']} ({field['dtype']}): {field['null_count']} nulls ({field['null_percentage']:.1f}%)")

def profile_data(file_path):
    """Main function to profile data"""
    
    print(f"Loading data from: {file_path}")
    
    # Load data
    df = load_data(file_path)
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Analyze data
    analysis = analyze_data(df)
    
    # Print results
    print_analysis(analysis, file_path)
    
    return analysis

# Usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python script.py <file_path>")
        print("Supported: .csv, .xlsx, .xls, .shp, .geojson, .gpkg, .json, .parquet, .tsv")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    try:
        analysis = profile_data(file_path)
        print(f"\nDone!")
    except Exception as e:
        print(f"Error: {e}")