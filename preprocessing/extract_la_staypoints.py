import json
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box

import json
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box
import os

def process_staypoints_file(i):
    input_path = f"/home/Shared/Staypoint/Split/part-00001/part-00001_split-{i}.geojson"
    output_path = f"/home/Shared/Staypoint/LosAngeles/df_la_part-00001_split-{i}.parquet"
    
    # Load staypoints
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded data from part-0000{i}.geojson successfully.\n")
    
    # Flatten GeoJSON to DataFrame
    df = pd.json_normalize(data['features'])
    # Extract lat/lon and create Point geometry
    df['geometry.latitude'] = df['geometry.coordinates'].apply(lambda x: x[1])
    df['geometry.longitude'] = df['geometry.coordinates'].apply(lambda x: x[0])
    
    print(f"Converted geojson to dataframe with {len(df)} rows.\n")
    
    df['geometry'] = df.apply(lambda row: Point(row['geometry.longitude'], row['geometry.latitude']), axis=1)
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326") 
        
    # Load LA County multipolygon only once
    boundary_path = "/home/Shared/la_county_boundary.geojson"
    multipolygon_gdf = gpd.read_file(boundary_path)
    la_multipolygon = multipolygon_gdf.unary_union

    # LA bounding box
    la_bbox = {
        'min_lat': 32.75004,
        'max_lat': 34.823302,
        'min_lon': -118.951721,
        'max_lon': -117.646374
    }
    la_bbox_poly = box(la_bbox['min_lon'], la_bbox['min_lat'], la_bbox['max_lon'], la_bbox['max_lat'])
    
    # Fast bbox filter
    gdf_bbox = gdf[gdf.intersects(la_bbox_poly)]
    
    # Accurate geometry filter
    gdf_la = gdf_bbox[gdf_bbox.within(la_multipolygon)]
    print(f"Number of points in LA: {len(gdf_la)}")
    
    # Save result
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    gdf_la.to_parquet(output_path, index=False)
    print(f"Saved df_la_part-0000{i}.parquet successfully.\n")

if __name__ == "__main__":
    # Process each part file from 0 to 9
    for chunk_index in range(10):
        process_staypoints_file(chunk_index)
    print("All files processed successfully.")
