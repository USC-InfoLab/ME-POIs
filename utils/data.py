import torch
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from typing import Tuple
import json
from utils.constants import SG_CATEGORIES_MAPPING
from math import cos, radians, sqrt
from utils.modeling import transfer_usage_distributions, transfer_usage_by_text_similarity

def pretrain_dataset_split(
    data: pd.DataFrame,
    val_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset by User ids.
    """
    users = data['user_id'].unique()
    n_val = int(len(users) * val_ratio)

    rng = pd.Series(users).sample(frac=1.0, random_state=seed)
    val_users = set(rng.iloc[:n_val])
    
    train_df = data[~data['user_id'].isin(val_users)].reset_index(drop=True)
    val_df = data[data['user_id'].isin(val_users)].reset_index(drop=True)
    
    return train_df, val_df

def eval_dataset_split(
    dataset: pd.DataFrame,
    task: str,
    keep_coords = False,
    ratio: float = 0.2,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split POI ids and their associated labels for evaluation tasks.
    """
    # Keep only POIs for which we have associated metadata
    if task == 'open_hours':
        dataset = dataset[dataset['open_hours'].apply(lambda x: not np.all(x == 0))].reset_index(drop=True)
    elif task == 'busyness':
        if 'busyness' not in dataset.columns:
            print("Constructing busyness vector from busyness_0 to busyness_23 columns")
            busyness_cols = [f'busyness_avg_{i:02d}' for i in range(24)]
            busyness_array = dataset[busyness_cols].to_numpy()
            dataset['busyness'] = list(busyness_array)
        dataset = dataset[dataset['busyness'].apply(lambda x: not np.all(np.array(x) == 0.0))].reset_index(drop=True)

    # Deduplicate POIs by taking first label per place_id
    if keep_coords:
        poi_labels_df = dataset.groupby('place_id').first().reset_index()
        poi_labels_df = poi_labels_df[['place_id', 'safegraph.location_name', 'place_lat', 'place_lon', task]]
    else:
        poi_labels_df = dataset.groupby('place_id')[task].first().reset_index()
    print(f"Total unique POIs for task '{task}': {len(poi_labels_df)}")
    
    # Split using sklearn
    train_df, val_df = train_test_split(
        poi_labels_df, test_size=ratio, random_state=seed, shuffle=True
    )

    return train_df, val_df

def load_dataset(
    path: str,
    file_name: str,
    loc_encoder_type: str,
    area_bbox: Tuple[float, float, float, float] = None,
    area_timezone: str = None
) -> pd.DataFrame:
    """
    Load a dataset from a file.
    """
    df = _load_veraset_dataset(path, file_name)
    # Preprocess Veraset dataset
    df_prep = _prep_veraset_dataset(df, loc_encoder_type, area_bbox, area_timezone)
        
    return df_prep

def _load_veraset_dataset(path: str, file_name: str) -> pd.DataFrame:
    """
    Load Veraset dataset from parquet files.
    """
    df = pd.read_parquet(os.path.join(path, file_name))
    print(f"Loaded Veraset dataset with {len(df)} rows.")
    print(f"Timespan of Veraset dataset: {df['arrival_time'].min()} - {df['departure_time'].max()}")
        
    return df

def normalize_coordinates_zscore(df: pd.DataFrame, lat_col: str, lon_col: str) -> pd.DataFrame:
    """
    Normalize lat/lon coordinates using z-score.
    """
    df['lat'] = (df[lat_col] - df[lat_col].mean()) / df[lat_col].std()
    df['lon'] = (df[lon_col] - df[lon_col].mean()) / df[lon_col].std()
    
    return df

def normalize_coordinates_in_bbox(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    to_range: Tuple = (0, 1),
    area_bbox: Tuple[float, float, float, float] = None
) -> pd.DataFrame:
    """
    Computes the area bbox from the dataset and normalizes the coordinates within that bbox.
    The coordinates are normalized to the range [0, 1] by default, or to [-1, 1] if specified.
    """
    if area_bbox is None:
        min_lat, max_lat = df[lat_col].min(), df[lat_col].max()
        min_lon, max_lon = df[lon_col].min(), df[lon_col].max()
        print(f"Normalizing coordinates within bbox: lat [{min_lat}, {max_lat}], lon [{min_lon}, {max_lon}]")
    else:
        min_lat, min_lon, max_lat, max_lon = area_bbox

    df['lat'] = (df[lat_col] - min_lat) / (max_lat - min_lat)
    df['lon'] = (df[lon_col] - min_lon) / (max_lon - min_lon)

    if to_range == (-1, 1):
        df['lat'] = df['lat'] * 2 - 1
        df['lon'] = df['lon'] * 2 - 1

    return df

def normalize_time_minmax(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """
    Normalize time column to a range of [0, 1] using min-max scaling.
    """
    # Convert to datetime with timezone awareness (UTC), then to Pacific Time
    df[time_col] = pd.to_datetime(df[time_col], utc=True).dt.tz_convert('America/Los_Angeles')

    min_time = df[time_col].min()
    max_time = df[time_col].max()

    # Normalize the time column to [0, 1]
    df[time_col] = (df[time_col].dt.tz_localize(None) - min_time.tz_localize(None)) / (max_time - min_time)

    return df

def normalize_visit_time(
    df: pd.DataFrame,
    arrival_col: str = "arrival_time",
    departure_col: str = "departure_time",
    user_col: str = "user_id",
    timezone: str = "America/Los_Angeles",
) -> dict:
    """
    Normalize arrival and departure times of visits in the dataset. For each time feature we
    extract the hour of day, day of week, and days since the start of the dataset.
    """
    # Sort by user and arrival time first
    df = df.sort_values([user_col, arrival_col]).reset_index(drop=True)
    
    # Convert timestamps to local timezone
    arr_ts = pd.to_datetime(df[arrival_col], utc=True).dt.tz_convert(timezone)
    dep_ts = pd.to_datetime(df[departure_col], utc=True).dt.tz_convert(timezone)

    ## Normalize arrival time
    # Compute the normalized value of the hour within the day. 
    # 0.0 is midnight, 0.5 is noon, 1.0 is the next midnight.
    df[f"{arrival_col}_hour_of_day"] = (
        arr_ts.dt.hour + arr_ts.dt.minute / 60 + arr_ts.dt.second / 3600
    ) / 24.0 # scale to day

    # Compute the normalized value of the day within the week.
    # 0.0 is Monday midnight, 0.5 is Wednesday noon, 1.0 is next Monday midnight.
    df[f"{arrival_col}_day_of_week"] = (
        arr_ts.dt.dayofweek + df[f"{arrival_col}_hour_of_day"]
    ) / 7.0 # scale to week

    # Compute the number of days since the start of the dataset.
    # The start time is the minimum arrival time in the dataset.
    # The value is normalized within the year since the start.
    start_time = arr_ts.min()
    df[f"{arrival_col}_days_since_start"] = (
        (arr_ts - start_time).dt.total_seconds() / (3600 * 24)
    ) / 365.0  # scale to year

    ## Normalize departure time
    df[f"{departure_col}_hour_of_day"] = (
        dep_ts.dt.hour + dep_ts.dt.minute / 60 + dep_ts.dt.second / 3600
    ) / 24.0 # scale to day

    df[f"{departure_col}_day_of_week"] = (
        dep_ts.dt.dayofweek + df[f"{departure_col}_hour_of_day"]
    ) / 7.0 # scale to week

    df[f"{departure_col}_days_since_start"] = (
        (dep_ts - start_time).dt.total_seconds() / (3600 * 24)
    ) / 365.0  # scale to year

    # Compute duration of stay and travel time in hours
    df['duration'] = (dep_ts - arr_ts).dt.total_seconds() / 3600
    prev_dep_ts = dep_ts.shift(1)
    df['travel_time'] = (arr_ts - prev_dep_ts).dt.total_seconds() / 3600

    # First visit for each user equals travel_time = 0
    df.loc[df.groupby(user_col).nth(0).index, 'travel_time'] = 0

    # Clip outliers
    max_duration = df['duration'].quantile(0.99)
    df.loc[df['duration'] > max_duration, 'duration'] = max_duration

    max_travel_time = df['travel_time'].quantile(0.99)
    df.loc[df['travel_time'] > max_travel_time, 'travel_time'] = max_travel_time

    return df

def convert_open_hours_json_to_vec(open_hours_str: str) -> np.ndarray:
    """
    Convert open hours JSON string to a binary vector representing the open hours of a place.
    """
    vector = np.zeros(168, dtype=np.uint8) # 168 hours in a week (7 days * 24 hours)

    if not isinstance(open_hours_str, str) or open_hours_str.strip() == '':
        return vector

    try:
        schedule = json.loads(open_hours_str)
    except json.JSONDecodeError:
        return vector

    day_to_index = {"Mon": 0, "Tue": 1, "Wed": 2,
                    "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}

    for day, intervals in schedule.items():
        if day not in day_to_index:
            continue
        day_idx = day_to_index[day]

        for interval in intervals:
            if len(interval) != 2:
                continue
            try:
                start_hour = int(interval[0].split(':')[0])
                end_hour = int(interval[1].split(':')[0])
            except ValueError:
                continue

            base_idx = day_idx * 24

            if end_hour <= start_hour:
                vector[base_idx + start_hour:base_idx + 24] = 1
                next_day_idx = ((day_idx + 1) % 7) * 24
                vector[next_day_idx:next_day_idx + end_hour] = 1
            else:
                vector[base_idx + start_hour:base_idx + end_hour] = 1

    return vector

def convert_is_closed(closed_date: str) -> int:
    """
    Convert closed date string to a binary value indicating if 
    the place is permanately closed.
    """
    if closed_date is None or pd.isna(closed_date):
        return 0
    else:
        return 1

def _prep_veraset_dataset(
    df: pd.DataFrame,
    loc_encoder_type: str,
    area_bbox: Tuple[float, float, float, float] = None,
    area_timezone: str = None
) -> pd.DataFrame:
    """
    Preprocess Veraset dataset for POI visit modeling. This function normalizes
    coordinates, visit times, and prepares categories and labels.
    Args: 
        df: Raw Veraset DataFrame.
        loc_encoder_type: Type of location encoder used in the model.
    Returns:
        Preprocessed DataFrame.
    """
    
    # GeoCLIP location encoder requires normalized lat/lon in [-1, 1]
    # Rest of location encoders are usually normalzed to [0, 1]
    if loc_encoder_type == 'geoclip':
        print("Using GeoCLIP location encoder, normalizing coordinates to [-1, 1]")
        to_range = (-1, 1)
    else:
        print(f"Using {loc_encoder_type} location encoder, normalizing coordinates to [0, 1]")
        to_range = (0, 1)
    
    # Normalize coordinates within the datasets bbox
    df = normalize_coordinates_in_bbox(df, 
                                    lat_col='place_lat', 
                                    lon_col='place_lon', 
                                    to_range=to_range,
                                    area_bbox=area_bbox)
    
    # Normalize arrival/departure time of visits
    df = normalize_visit_time(df, 
                            arrival_col='arrival_time', 
                            departure_col='departure_time', 
                            user_col='user_id',
                            timezone=area_timezone)
    
    # Map POI types to 10 high level categories
    df = map_to_high_level_category(df)
    
    # Prepare dataset labels
    df['open_hours'] = df['safegraph.open_hours'].apply(convert_open_hours_json_to_vec)
    df['is_closed'] = df['safegraph.closed_on'].apply(convert_is_closed)
    
    return df

    
def map_to_high_level_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map NAICS codes to 10 high-level categories.
    """
    category_id_map = {
        "Arts & Entertainment": 1,
        "College & University": 2,
        "Food": 3,
        "Professional & Other Places": 4,
        "Nightlife Spot": 5,
        "Outdoors & Recreation": 6,
        "Shop & Service": 7,
        "Travel & Transport": 8,
        "Residence": 9,
        "Other": 0
    }
    
    # Get top-level NAICS code and assign a custom category
    df['top_category'] = df['safegraph.top_category'].map(SG_CATEGORIES_MAPPING).fillna("Other")
    # Map top-level categories to category IDs
    df['category_id'] = df['top_category'].map(category_id_map)
    
    # Check for null values in category_id
    assert df['category_id'].isnull().sum() == 0, "Null values found in category_id"
    
    return df

def normalize_sigmas(sigmas_km: list, area_bbox: Tuple[float, float, float, float]) -> list:
    """
    Normalize Gaussian sigmas based on the diagonal of the area bbox.
    Args:
        sigmas_km: List of Gaussian sigmas in kilometers.
        area_bbox: Tuple of (min_lat, min_lon, max_lat, max_lon)."""
    
    min_lat, min_lon, max_lat, max_lon = area_bbox
    
    center_lat = (min_lat + max_lat) / 2
    delta_lat_km = (max_lat - min_lat) * 111  # approx km per degree lat
    delta_lon_km = (max_lon - min_lon) * 111 * cos(radians(center_lat))

    diag_km = sqrt(delta_lat_km**2 + delta_lon_km**2)  # bbox diagonal in km

    # Normalize each sigma
    sigmas_norm = [s / diag_km for s in sigmas_km]
    print(f"Normalized sigmas: {sigmas_norm} based on bbox diagonal of {diag_km:.2f} km")
    
    return sigmas_norm

def load_anchor_pois(dir_path: str, file_path: str, dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Load precomputed daily and weekly distributions from anchors.
    """
    anchors = pd.read_parquet(os.path.join(dir_path, file_path))
    # Keep only the safegraph.place_id and relevant columns for merging
    anchors = anchors[['safegraph_place_id', 'daily', 'weekly', 'num_visits']]
    # Build a poi lookup table from the whole dataset
    poi_lookup = (
        dataset[["safegraph.place_id", "place_id", "lat", "lon", "top_category", "category_id"]]
            .drop_duplicates("place_id")
            .rename(columns={"safegraph.place_id": "safegraph_place_id"})
    )
    
    anchors_enriched = anchors.merge(poi_lookup, on="safegraph_place_id", how="left")
    
    return anchors_enriched

def get_sparse_pois(dataset: pd.DataFrame, anchor_pois: pd.DataFrame) -> pd.DataFrame:
    """
    Returns POIs with sparse visits.    
    """
    anchor_ids = set(anchor_pois['place_id'])
    all_pois = dataset[["safegraph.place_id", "place_id", "lat", "lon", "top_category", "category_id"]].drop_duplicates("place_id")
    sparse_pois = all_pois[~all_pois['place_id'].isin(anchor_ids)].reset_index(drop=True)
    
    return sparse_pois

def compute_anchor_precomputed_weights(
    dir_path: str,
    anchors_df: str,
    dataset: pd.DataFrame,
    sigmas: list,
    city: str,
    area_bbox: list
) -> dict:
    """
    Precompute weights for each sigma between sparse POIs and anchor POIs.
    Args:
        dir_path: Directory path where the anchor POIs file is located.
        anchors_df: DataFrame of anchor POIs with precomputed daily and weekly usage distributions.
        dataset: The full dataset DataFrame.
        sigmas: List of Gaussian sigmas in kilometers.
        city: City name (used for caching).
        area_bbox: Tuple of (min_lat, min_lon, max_lat, max_lon).
    Returns:
        A dictionary mapping each sigma to its corresponding DataFrame of transferred distributions.
    """
    # Extract sparse POIs from the dataset
    sparse_df = get_sparse_pois(dataset, anchors_df)
    
    # Normalize sigmas based on the area bbox
    sigmas_norm = normalize_sigmas(sigmas, area_bbox)
    
    # Precompute weights for each sigma between sparse and anchor POIs
    result_dict = {}
    for idx in range(len(sigmas)):
        print(f"Processing sigma = {sigmas[idx]}...")
        result_df = transfer_usage_distributions(
            sparse_pois_df=sparse_df,
            anchors_df=anchors_df,
            sigma=sigmas_norm[idx]
        )

        sigma_str = str(sigmas[idx]).replace('.', '')
        output_path = f"{dir_path}/cache/{city}/transferred_distributions_sigma_{sigma_str}.parquet"
        result_df.to_parquet(output_path, index=False)
        result_dict[sigmas[idx]] = result_df
    
    return result_dict

def compute_anchor_precomputed_text_sim_weights(
    dir_path: str,
    anchors_df: str,
    dataset: pd.DataFrame,
    city: str,
    text_embeds: dict,
    top_k: int = 10
) -> dict:
    """
    Precompute weights between sparse POIs and anchor POIs based on text similarity.
    Args:
        dir_path: Directory path where the anchor POIs file is located.
        anchors_df: DataFrame of anchor POIs with precomputed daily and weekly usage distributions.
        dataset: The full dataset DataFrame.
        text_embeds: Dictionary mapping place_id to text embedding vectors.
        city: City name (used for caching).
        top_k: Number of top similar anchors to consider for transfer.
    Returns:
        None. Saves the transferred distributions to a parquet file.
    """
    # Extract sparse POIs from the dataset
    sparse_df = get_sparse_pois(dataset, anchors_df)
    
    print(f"Processing text similarity based transfer with top_k = {top_k}...")
    result_df = transfer_usage_by_text_similarity(
        sparse_pois_df=sparse_df,
        anchors_df=anchors_df,
        text_embeds=text_embeds,
        top_k=top_k
    )

    output_path = f"{dir_path}/cache/{city}/transferred_distributions_text_sim_topk_{top_k}.parquet"
    result_df.to_parquet(output_path, index=False)
    print(f"Saved text similarity based transferred distributions to {output_path}")

    return result_df

def load_anchor_precomputed_weights(dir_path: str, sigmas: list, city: str) -> dict:
    """
    Load precomputed weights for each sigma from cached parquet files.
    Args:
        dir_path: Directory path where the cached files are located.
        sigmas: List of Gaussian sigmas in kilometers.
    Returns:
        A dictionary mapping each sigma to its corresponding DataFrame of transferred distributions.
    """
    result_dict = {}
    for sigma in sigmas:
        sigma_str = str(sigma).replace('.', '')
        file_path = f"{dir_path}/cache/{city}/transferred_distributions_sigma_{sigma_str}.parquet"
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Cached file not found: {file_path}")
        
        df = pd.read_parquet(file_path)
        result_dict[sigma] = df
        print(f"Loaded precomputed weights for sigma = {sigma} from {file_path}")
    
    return result_dict

def load_anchor_precomputed_text_sim_weights(dir_path: str, top_k: int = 10, city: str = 'LosAngeles') -> pd.DataFrame:
    """
    Load precomputed text similarity based weights from cached parquet file.
    Args:
        dir_path: Directory path where the cached file is located.
        top_k: Number of top similar anchors considered for transfer.
    Returns:
        DataFrame of transferred distributions based on text similarity.
    """
    file_path = f"{dir_path}/cache/{city}/transferred_distributions_text_sim_topk_{top_k}.parquet"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cached file not found: {file_path}")
    
    df = pd.read_parquet(file_path)
    print(f"Loaded precomputed text similarity based weights from {file_path}")
    
    return df

def get_usage_dicts(
    dir_path: str,
    file_path: str,
    dataset: pd.DataFrame,
    sigmas: list,
    text_embeds: dict,
    city: str,
    area_bbox: list,
    recompute: bool = False,
    distr_col: str = "weekly"
) -> Tuple[dict, dict, dict]:
    """
    Get usage distribution dictionaries for given sigmas.
    In total we get 3 dictionaries:
        1. anchor_map: Mapping anchor POI ids to their precomputed weekly or daily distribution
        2. sparse_multiscale_map: Mapping each sparse POI id to a weekly or daily transferred distribution
        3. anchor_text_sim_map: Mapping anchor POI ids to their text similarity based distribution
            The distributions are transferred according to sigma values (multiscale)
    Args:
        dir_path: Directory path where the anchor POIs file is located.
        file_path: File path of the anchor POIs file.
        dataset: The full dataset DataFrame.
        sigmas: List of Gaussian sigmas in kilometers.
        text_embeds: Dictionary mapping place_id to text embedding vectors.
        city: City name (used for caching).
        area_bbox: Tuple of (min_lat, min_lon, max_lat, max_lon).
        recompute: Whether to recompute the transferred distributions or load from cache.
        distr_col: Column name for the distribution to use ('weekly' or 'daily').
    Returns:
        A tuple of two dictionaries: (anchor_map, sparse_multiscale_map).
    """
    # Load anchors precomputed daily and weekly distributions from file
    anchors_usage_df = load_anchor_pois(dir_path, file_path, dataset)
    
    # Compute or load precomputed anchor-based usage weights for sparse POIs
    if recompute:
        multiscale_usage_dict = compute_anchor_precomputed_weights(
            dir_path, anchors_usage_df, dataset, sigmas, city, area_bbox)
        if text_embeds is not None:
            text_usage_df = compute_anchor_precomputed_text_sim_weights(
                dir_path, anchors_usage_df, dataset, city, text_embeds
            )
    else:
        multiscale_usage_dict = load_anchor_precomputed_weights(
            dir_path, sigmas, city
        )
        if text_embeds is not None:
            text_usage_df = load_anchor_precomputed_text_sim_weights(
                dir_path, top_k=10, city=city
            )

    # Construct a dictionary mapping from place_id to distributions
    anchor_map = {
        int(place_id): (torch.tensor(vec, dtype=torch.float32).clamp_min(0)
               / (torch.tensor(vec, dtype=torch.float32).clamp_min(0).sum() + 1e-12))
        for place_id, vec in zip(anchors_usage_df["place_id"], anchors_usage_df[distr_col])
    }
    
    # Construct a nested dictionary for multiscale sparse POI distributions
    sparse_multiscale_map = {}
    for sigma in sigmas:
        df = multiscale_usage_dict[sigma]
        sigma_map = {
            int(place_id): (torch.tensor(vec, dtype=torch.float32).clamp_min(0)
                   / (torch.tensor(vec, dtype=torch.float32).clamp_min(0).sum() + 1e-12))
            for place_id, vec in zip(df["place_id"], df["pred_" + distr_col])
        }
        sparse_multiscale_map[sigma] = sigma_map
    
    if text_embeds is None:
        sparse_text_sim_map = None
    else:
        sparse_text_sim_map = {
            int(place_id): (torch.tensor(vec, dtype=torch.float32).clamp_min(0)
                / (torch.tensor(vec, dtype=torch.float32).clamp_min(0).sum() + 1e-12))
            for place_id, vec in zip(text_usage_df["place_id"], text_usage_df["pred_" + distr_col])
        }

    return anchor_map, sparse_multiscale_map, sparse_text_sim_map

