import json
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely import wkt
from shapely.errors import WKTReadingError
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree

def wkt_load_helper(wkt_str):
    try:
        if pd.isna(wkt_str):
            return None
        return wkt.loads(wkt_str)
    except (WKTReadingError, TypeError, AttributeError):
        return None


# Safely create geometry: use polygon if available, else fallback to Point
def load_wkt_geometry(row):
    geom = wkt_load_helper(row['polygon_wkt'])
    if geom is None and pd.notna(row.get('longitude')) and pd.notna(row.get('latitude')):
        return Point(row['longitude'], row['latitude'])
    return geom

def load_parquet_file(file_path: str) -> pd.DataFrame:
    """Load a parquet file and return it as a pandas DataFrame."""
    try:
        df = pd.read_parquet(file_path)
        print(f"Loaded data from {file_path} successfully.")
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error

def load_csv_file(file_path: str) -> pd.DataFrame:
    """Load a CSV file and return it as a pandas DataFrame."""
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded data from {file_path} successfully.")
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error

def POI_attribution(df_la: pd.DataFrame, df_pois: pd.DataFrame, d_thrs=50, lat_col='lat', lon_col='long') -> pd.DataFrame:
    df_la = df_la.copy()
    df_la['original_index'] = df_la.index

    # Parse POI geometries (polygons or fallback to lat/lon Points)
    df_pois = df_pois.copy()
    df_pois['geometry'] = df_pois.apply(load_wkt_geometry, axis=1)
    gdf_pois = gpd.GeoDataFrame(df_pois, geometry='geometry', crs="EPSG:4326")

    # Create staypoint geometries
    gdf_la = gpd.GeoDataFrame(
        df_la,
        geometry=gpd.points_from_xy(df_la[lon_col], df_la[lat_col]),
        crs="EPSG:4326"
    )
    
    # Step 1: Polygon intersection
    joined = gpd.sjoin(gdf_la, gdf_pois, how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep='first')] # Assumption: keep first match if multiple polygons intersect    

    # Prefix POI columns with 'safegraph.'
    poi_columns = set(gdf_pois.columns) - {"geometry"}
    for col in poi_columns:
        if col in joined.columns:
            joined = joined.rename(columns={col: f"safegraph.{col}"})

    # Optional renaming cleanup
    joined = joined.rename(columns={
        "safegraph.safegraph_place_id": "safegraph.place_id",
        "safegraph.safegraph_brand_ids": "safegraph.brand_ids",
        "safegraph.parent_safegraph_place_id": "safegraph.parent_place_id"
    })

    # Separate matched and unmatched
    matched = joined[~joined['safegraph.place_id'].isna()].copy()
    unmatched = joined[joined['safegraph.place_id'].isna()].copy()
    
    matched['safegraph.dist_to_poi'] = 0.0  # Already matched via intersection
    unmatched['safegraph.dist_to_poi'] = np.nan  # Initialize unmatched distances to NaN

    print(f"Number of matched points: {len(matched)}")
    print(f"Number of unmatched points: {len(unmatched)}")
    
    # Step 2: Nearest POI for unmatched staypoints
    if not unmatched.empty:
        # Convert lat/lon to radians for haversine distance
        poi_coords_rad = np.radians(gdf_pois[['latitude', 'longitude']].values)

        tree = BallTree(poi_coords_rad, metric='haversine')

        # Unmatched staypoints in radians
        unmatched_coords = np.radians(np.vstack([unmatched.geometry.y.values, unmatched.geometry.x.values]).T)

        dist, ind = tree.query(unmatched_coords, k=1)
        dist_meters = dist[:, 0] * 6371000  # Convert radians to meters

        unmatched = unmatched.reset_index(drop=True)
        unmatched['safegraph.dist_to_poi'] = dist_meters

        # Find nearest POIs
        nearest_pois = gdf_pois.iloc[ind[:, 0]].reset_index(drop=True)

        # Keep only those within distance threshold
        matched_by_dist_idx = unmatched['safegraph.dist_to_poi'] <= d_thrs
        matched_by_dist = unmatched[matched_by_dist_idx].copy()
        unmatched = unmatched[~matched_by_dist_idx].copy()
        nearest_pois = nearest_pois[matched_by_dist_idx].reset_index(drop=True)
        
        unmatched = unmatched.reset_index(drop=True)
        safegraph_cols = [col for col in matched_by_dist.columns if col.startswith("safegraph.")]
        matched_by_dist = matched_by_dist.drop(columns=safegraph_cols, errors='ignore')
        matched_by_dist = matched_by_dist.reset_index(drop=True)
        
        if not matched_by_dist.empty:
            # Transfer all POI attributes
            for col in gdf_pois.columns:
                if col not in ['geometry', 'centroid']:
                    matched_by_dist[f"safegraph.{col}"] = nearest_pois[col].values
            matched_by_dist = matched_by_dist.rename(columns={
                "safegraph.safegraph_place_id": "safegraph.place_id",
                "safegraph.safegraph_brand_ids": "safegraph.brand_ids",
                "safegraph.parent_safegraph_place_id": "safegraph.parent_place_id"
            })
            matched_by_dist['safegraph.dist_to_poi'] = dist_meters[matched_by_dist_idx]
            unmatched['safegraph.dist_to_poi'] = np.nan

        print(f"Number of matched by distance: {len(matched_by_dist)}")
        
    print(f"Final unmatched points after distance check: {len(unmatched)} / {len(df_la)}")
    
    final_df = pd.concat([matched, matched_by_dist, unmatched], ignore_index=True)
    # resort by original index to maintain the order of staypoints
    final_df = final_df.sort_values(by='original_index').reset_index(drop=True)
    final_df = final_df.drop(columns=['original_index'], errors='ignore')  # Clean up original index column
    
    return final_df

if __name__ == "__main__":
    # Load datasets
    d_thrs = 100  # distance threshold in meters
    city = "Houston"  # "Houston" or "LosAngeles"
    dir_path = "/mnt/disk/data/trajfm_veraset_splits/veraset/"
    # Load staypoints and POIs based on city
    if city == "Houston":
        df = load_parquet_file(dir_path + "Staypoint/Houston/veraset_staypoints-march_5_10_15_20_25.parquet")
        print(f"Loaded Houston staypoints: {len(df)} rows")
        pois = load_csv_file("/mnt/disk/data/POI_data/Safegraph/safegraph_pois_Houston.csv")
        to_save_name = "veraset_visits-march_5_10_15_20_25"
        lat_col = 'lat'
        lon_col = 'long'
    elif city == "LosAngeles":
        df = load_parquet_file(dir_path + "Staypoint/LosAngeles/df_la_part-00001_split-9.parquet")
        # keep the first 10 rows for testing
        print(f"Loaded LA staypoints: {len(df)} rows")
        pois = load_csv_file("/mnt/disk/data/POI_data/Safegraph/safegraph_pois_LA_County.csv")
        to_save_name = "df_la_part-00001_split-9"
        lat_col = 'geometry.latitude'
        lon_col = 'geometry.longitude'

    # Perform POI attribution
    df = POI_attribution(df, pois, d_thrs, lat_col=lat_col, lon_col=lon_col) # threshold in meters
    
    # Save the result to a new parquet file
    output_file = dir_path + f"Visits/{city}/{to_save_name}_{d_thrs}.parquet"
    df.to_parquet(output_file, index=False)
    
    print(f"POI attribution completed. Output saved to {output_file}")
    print("POI attribution completed successfully.")