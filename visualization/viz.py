import argparse
import sys
from pathlib import Path
import warnings

import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
import folium

warnings.filterwarnings("ignore")

# Supported basemaps
BASEMAP_PROVIDERS = {
    "osm": ctx.providers.OpenStreetMap.Mapnik,
    "carto": ctx.providers.CartoDB.Positron,
    "dark": ctx.providers.CartoDB.DarkMatter,
    "satellite": ctx.providers.Esri.WorldImagery,
}


def load_geospatial_data(file_path):
    path = Path(file_path)

    if path.is_dir():
        shp_files = list(path.glob("*.shp"))
        if not shp_files:
            raise ValueError(f"No .shp file found in directory: {path}")
        file_path = shp_files[0]
        print(f"📁 Detected shapefile: {file_path.name}")

    elif not path.exists():
        raise FileNotFoundError(f"File or directory does not exist: {file_path}")

    ext = path.suffix.lower()
    if ext in ['.shp', '.geojson', '.json', '.gpkg', '.kml']:
        return gpd.read_file(str(file_path))
    else:
        raise ValueError(f"Unsupported geospatial format: {ext}")


def add_custom_scalebar(ax, length_km=5, segments=2):
    # Scalebar in bottom-right (axes fraction)
    x_start = 0.7
    y_start = 0.05
    bar_total_km = length_km * segments

    # Convert axes fraction to data coordinates
    trans = ax.transAxes
    inv = ax.transData.inverted()
    x0, y0 = inv.transform(trans.transform((x_start, y_start)))
    x1, _ = inv.transform(trans.transform((x_start + 0.2, y_start)))
    bar_width = (x1 - x0) / segments
    bar_height = bar_width / 100  # very thin

    for i in range(segments):
        color = 'black' if i % 2 == 0 else 'white'
        ax.add_patch(plt.Rectangle((x0 + i * bar_width, y0), bar_width, bar_height,
                                   facecolor=color, edgecolor='black'))
        ax.text(x0 + i * bar_width, y0 + bar_height * 5,
                f"{i * length_km}", ha='center', va='bottom', fontsize=8)

    ax.text(x0 + segments * bar_width, y0 + bar_height * 5,
            f"{bar_total_km} km", ha='center', va='bottom', fontsize=8)


def plot_static_map(gdf, basemap='carto', title=None, output=None, color_by=None):
    gdf = gdf.to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(10, 10))

    if color_by and color_by in gdf.columns:
        gdf.plot(ax=ax, column=color_by, legend=True, alpha=0.6, edgecolor='black')
    else:
        gdf.plot(ax=ax, alpha=0.6, edgecolor='black')

    if basemap and basemap in BASEMAP_PROVIDERS:
        try:
            ctx.add_basemap(ax, source=BASEMAP_PROVIDERS[basemap])
        except Exception as e:
            print(f"⚠️ Basemap could not be added: {e}")

    add_custom_scalebar(ax, length_km=5, segments=2)

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')

    ax.set_axis_off()
    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=300)
        print(f"✅ Saved static map to {output}")
    else:
        plt.show()


def plot_interactive_map(gdf, title=None, output=None):
    gdf = gdf.to_crs(epsg=4326)

    bounds = gdf.total_bounds
    center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
    m = folium.Map(location=center, zoom_start=13, tiles='OpenStreetMap', control_scale=True)

    folium.GeoJson(gdf).add_to(m)

    if title:
        title_html = f'<h4 style="font-size:16px;font-weight:bold">{title}</h4>'
        m.get_root().html.add_child(folium.Element(title_html))

    if output:
        m.save(output)
        print(f"✅ Saved interactive map to {output}")
    else:
        m.show()


def main():
    parser = argparse.ArgumentParser(description="Geospatial Data Visualizer")
    parser.add_argument("file_path", help="Path to shapefile, geojson, etc.")
    parser.add_argument("--mode", choices=["static", "interactive"], default="static")
    parser.add_argument("--basemap", choices=BASEMAP_PROVIDERS.keys(), default="osm")
    parser.add_argument("--title", help="Map title")
    parser.add_argument("--output", help="Path to save the map (image or HTML)")
    parser.add_argument("--color_by", help="Field to color polygons by (static mode only)")

    args = parser.parse_args()
    gdf = load_geospatial_data(args.file_path)
    print(f"✅ Loaded geospatial data: {len(gdf)} features, CRS: {gdf.crs}")

    if args.mode == "static":
        plot_static_map(gdf, basemap=args.basemap, title=args.title, output=args.output, color_by=args.color_by)
    else:
        plot_interactive_map(gdf, title=args.title, output=args.output)


if __name__ == "__main__":
    main()
