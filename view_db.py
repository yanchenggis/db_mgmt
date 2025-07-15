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
    """Load data based on file extension or URL, return DataFrame and spatial info"""
    import requests
    from io import StringIO
    
    spatial_info = {
        'is_spatial': False,
        'crs': None,
        'geometry_column': None,
        'geometry_types': None
    }
    
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
                return pd.DataFrame(json_data), spatial_info
            elif isinstance(json_data, dict):
                # If it's a dictionary, try to flatten it
                if 'features' in json_data:
                    # GeoJSON format - this is spatial!
                    spatial_info['is_spatial'] = True
                    if 'crs' in json_data:
                        spatial_info['crs'] = json_data['crs']
                    
                    features = json_data['features']
                    flattened = []
                    geometry_types = set()
                    
                    for feature in features:
                        row = {}
                        if 'properties' in feature:
                            row.update(feature['properties'])
                        if 'geometry' in feature:
                            geom_type = feature['geometry'].get('type', 'Unknown')
                            row['geometry_type'] = geom_type
                            geometry_types.add(geom_type)
                        flattened.append(row)
                    
                    spatial_info['geometry_types'] = list(geometry_types)
                    spatial_info['geometry_column'] = 'geometry_type'
                    
                    return pd.DataFrame(flattened), spatial_info
                else:
                    # Try to convert dict to DataFrame
                    try:
                        return pd.DataFrame([json_data]), spatial_info
                    except:
                        # If that fails, try to normalize
                        return pd.json_normalize(json_data), spatial_info
        elif file_path.endswith('.csv'):
            return pd.read_csv(StringIO(response.text)), spatial_info
        elif file_path.endswith('.geojson'):
            gdf = gpd.read_file(StringIO(response.text))
            spatial_info['is_spatial'] = True
            spatial_info['crs'] = str(gdf.crs) if gdf.crs else "Unknown/Not set"
            spatial_info['geometry_column'] = 'geometry'
            spatial_info['geometry_types'] = list(gdf.geometry.geom_type.unique())
            return pd.DataFrame(gdf.drop(columns='geometry')), spatial_info
    
    # Original local file handling
    file_ext = Path(file_path).suffix.lower()
    
    if file_ext == '.csv':
        return pd.read_csv(file_path), spatial_info
    elif file_ext in ['.xlsx', '.xls']:
        return pd.read_excel(file_path), spatial_info
    elif file_ext in ['.shp', '.geojson', '.gpkg']:
        # These are spatial formats
        gdf = gpd.read_file(file_path)
        spatial_info['is_spatial'] = True
        spatial_info['crs'] = str(gdf.crs) if gdf.crs else "Unknown/Not set"
        spatial_info['geometry_column'] = 'geometry'
        spatial_info['geometry_types'] = list(gdf.geometry.geom_type.unique())
        return pd.DataFrame(gdf.drop(columns='geometry')), spatial_info
    elif file_ext == '.json':
        # Better JSON handling for local files too
        with open(file_path, 'r') as f:
            json_data = json.load(f)
        
        if isinstance(json_data, list):
            return pd.DataFrame(json_data), spatial_info
        elif isinstance(json_data, dict):
            if 'features' in json_data:
                # GeoJSON format - this is spatial!
                spatial_info['is_spatial'] = True
                if 'crs' in json_data:
                    spatial_info['crs'] = json_data['crs']
                
                features = json_data['features']
                flattened = []
                geometry_types = set()
                
                for feature in features:
                    row = {}
                    if 'properties' in feature:
                        row.update(feature['properties'])
                    if 'geometry' in feature:
                        geom_type = feature['geometry'].get('type', 'Unknown')
                        row['geometry_type'] = geom_type
                        geometry_types.add(geom_type)
                    flattened.append(row)
                
                spatial_info['geometry_types'] = list(geometry_types)
                spatial_info['geometry_column'] = 'geometry_type'
                
                return pd.DataFrame(flattened), spatial_info
            else:
                try:
                    return pd.DataFrame([json_data]), spatial_info
                except:
                    return pd.json_normalize(json_data), spatial_info
    elif file_ext == '.parquet':
        return pd.read_parquet(file_path), spatial_info
    elif file_ext == '.tsv':
        return pd.read_csv(file_path, sep='\t'), spatial_info
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")

def detect_coordinate_columns(df):
    """Detect potential coordinate columns in non-spatial files"""
    coord_candidates = []
    
    # Common coordinate column names
    lat_names = ['lat', 'latitude', 'y', 'y_coord', 'northing', 'lat_deg']
    lon_names = ['lon', 'lng', 'longitude', 'x', 'x_coord', 'easting', 'lon_deg']
    
    # Check for exact matches (case insensitive)
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in lat_names or col_lower in lon_names:
            coord_candidates.append(col)
    
    # Check for partial matches
    for col in df.columns:
        col_lower = col.lower()
        for lat_pattern in ['lat', 'y_', 'north']:
            if lat_pattern in col_lower and col not in coord_candidates:
                coord_candidates.append(col)
                break
        for lon_pattern in ['lon', 'lng', 'x_', 'east']:
            if lon_pattern in col_lower and col not in coord_candidates:
                coord_candidates.append(col)
                break
    
    return coord_candidates

def safe_unique_count(series):
    """Safely count unique values, handling unhashable types"""
    try:
        return series.nunique()
    except TypeError:
        # For unhashable types like dicts/lists, convert to string first
        return series.astype(str).nunique()

def safe_value_counts(series):
    """Safely get value counts, handling unhashable types"""
    try:
        return series.value_counts(dropna=False)
    except TypeError:
        # For unhashable types like dicts/lists, convert to string first
        return series.astype(str).value_counts(dropna=False)

def analyze_data(df, spatial_info):
    """Analyze dataframe and categorize fields"""
    
    analysis = {
        'total_columns': len(df.columns),
        'total_rows': len(df),
        'unique_fields': [],
        'categorical_fields': [],
        'high_null_fields': [],
        'nested_fields': [],
        'spatial_info': spatial_info,
        'coordinate_candidates': []
    }
    
    # If not explicitly spatial, check for coordinate columns
    if not spatial_info['is_spatial']:
        analysis['coordinate_candidates'] = detect_coordinate_columns(df)
    
    for col in df.columns:
        null_count = df[col].isnull().sum()
        null_percentage = (null_count / len(df)) * 100
        unique_count = safe_unique_count(df[col])
        unique_ratio = unique_count / len(df) if len(df) > 0 else 0
        
        col_info = {
            'name': col,
            'dtype': str(df[col].dtype),
            'null_count': null_count,
            'null_percentage': null_percentage,
            'unique_count': unique_count
        }
        
        # Check if column contains nested data (dicts, lists)
        is_nested = False
        if df[col].dtype == 'object':
            # Check first few non-null values to see if they're nested
            non_null_values = df[col].dropna()
            if len(non_null_values) > 0:
                sample_values = non_null_values.head(3)
                for val in sample_values:
                    if isinstance(val, (dict, list)):
                        is_nested = True
                        break
        
        # High null fields (>20%)
        if null_percentage > 20:
            analysis['high_null_fields'].append(col_info.copy())
        
        # Categorize fields
        if is_nested:
            # Nested fields (dicts, lists, etc.)
            col_info['nested_type'] = 'dict/list'
            # Show first few values as examples
            non_null_values = df[col].dropna()
            if len(non_null_values) > 0:
                examples = []
                for val in non_null_values.head(3):
                    if isinstance(val, dict):
                        examples.append(f"Dict with keys: {list(val.keys())}")
                    elif isinstance(val, list):
                        examples.append(f"List with {len(val)} items")
                    else:
                        examples.append(str(val)[:50] + "..." if len(str(val)) > 50 else str(val))
                col_info['examples'] = examples
            analysis['nested_fields'].append(col_info)
        elif df[col].dtype in ['int64', 'float64'] or unique_ratio > 0.5:
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
            value_counts = safe_value_counts(df[col])
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
    print(f"Nested fields: {len(analysis['nested_fields'])}")
    print(f"High null fields (>20%): {len(analysis['high_null_fields'])}")
    
    # Spatial Information
    spatial_info = analysis['spatial_info']
    if spatial_info['is_spatial']:
        print(f"\n{'='*60}")
        print("SPATIAL INFORMATION")
        print(f"{'='*60}")
        print(f"Spatial data: YES")
        print(f"CRS: {spatial_info['crs']}")
        print(f"Geometry column: {spatial_info['geometry_column']}")
        print(f"Geometry types: {', '.join(spatial_info['geometry_types'])}")
        
        # CRS warnings
        if spatial_info['crs'] == "Unknown/Not set":
            print("⚠️  WARNING: No CRS defined! This may cause issues with spatial operations.")
        elif "4326" in str(spatial_info['crs']):
            print("ℹ️  Geographic CRS detected (WGS84) - good for global data")
        else:
            print("ℹ️  Projected CRS detected - good for local/regional analysis")
    else:
        print(f"\nSpatial data: NO")
        
        # Check for coordinate candidates
        if analysis['coordinate_candidates']:
            print(f"\n{'='*60}")
            print("POTENTIAL COORDINATE COLUMNS")
            print(f"{'='*60}")
            print("The following columns might contain coordinate data:")
            for col in analysis['coordinate_candidates']:
                print(f"  - {col}")
            print("💡 Consider converting to spatial format if these are coordinates")
    
    # Nested Fields
    if analysis['nested_fields']:
        print(f"\n{'='*60}")
        print("NESTED FIELDS (DICTS/LISTS)")
        print(f"{'='*60}")
        
        for field in analysis['nested_fields']:
            print(f"\n{field['name']} ({field['dtype']})")
            print(f"  Unique values: {field['unique_count']}")
            print(f"  Null count: {field['null_count']} ({field['null_percentage']:.1f}%)")
            if 'examples' in field:
                print("  Examples:")
                for example in field['examples']:
                    print(f"    {example}")
    
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
    df, spatial_info = load_data(file_path)
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Analyze data
    analysis = analyze_data(df, spatial_info)
    
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