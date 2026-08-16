import pandas as pd
import math
import time
from functools import lru_cache

from pyparsing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

EARTH_RADIUS_KM = 6371.0088

def to_radians(lat: float, lon: float):
    """
    Convert latitude and longitude to radians.
    """
    return np.radians(np.column_stack([lat, lon]))

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float):
    """
    Compute the Haversine distance between two points on the Earth in kilometers.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))

def initial_bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float):
    """
    Calculate the initial bearing (in degrees) from point 1 to point 2.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(dlon)
    bearing = (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0
    return bearing

def bearing_to_compass(bearing_deg: float, n_dirs: int = 16) -> str:
    """
    Map a bearing (in degrees) to a compass direction label.
    """
    labels_16 = [
        "North", "North-Northeast", "Northeast", "East-Northeast",
        "East", "East-Southeast", "Southeast", "South-Southeast",
        "South", "South-Southwest", "Southwest", "West-Southwest",
        "West", "West-Northwest", "Northwest", "North-Northwest"
]
    idx = int((bearing_deg / 360.0) * n_dirs) % n_dirs
    return labels_16[idx]

class NeighborIndex:
    def __init__(
            self,
            df: pd.DataFrame,
            lat_col: str = "place_lat",
            lon_col: str = "place_lon"
        ):
        self.df = df.reset_index(drop=True)
        self.lat_col = lat_col
        self.lon_col = lon_col
        # Convert coordinates to radians
        self.coords_rad = to_radians(self.df[lat_col].values, self.df[lon_col].values)
        # Build BallTree for fast nearest neighbor search
        self.tree = BallTree(self.coords_rad, metric="haversine")

    def get_neighbors(self, poi_idx: int, k: int = 10, max_radius_km: float = 100) -> list[dict]:
        """
        Return up to k nearest neighbors within max_radius_km for POI idx (excluding itself).
        """
        rad = max_radius_km / EARTH_RADIUS_KM  # convert km to radians
        ind = self.tree.query_radius(self.coords_rad[poi_idx:poi_idx+1], r=rad, return_distance=True)
        idxs = ind[0][0]
        dists_rad = ind[1][0]
        # Exclude self
        mask = idxs != poi_idx
        idxs = idxs[mask]
        dists_rad = dists_rad[mask]
        if len(idxs) == 0:
            return []

        # Sort by distance, take top-k
        order = np.argsort(dists_rad)
        idxs = idxs[order][:k]
        dists_km = dists_rad[order][:k] * EARTH_RADIUS_KM

        # Prepare neighbor rows
        src = self.df.iloc[poi_idx]
        lat1, lon1 = float(src[self.lat_col]), float(src[self.lon_col])
        lat2 = self.df.iloc[idxs][self.lat_col].to_numpy(dtype=float)
        lon2 = self.df.iloc[idxs][self.lon_col].to_numpy(dtype=float)
        bearings = initial_bearing_deg(lat1, lon1, lat2, lon2)
        dirs = [bearing_to_compass(b) for b in bearings]

        neighbors = []
        for rank, (j, dk, dr) in enumerate(zip(idxs, dists_km, dirs), start=1):
            row = self.df.iloc[j]
            neighbors.append({
                "rank": rank,
                "poi_id": row.get("poi_id", j),
                "name": row.get("location_name") or "",
                "category": row.get("top_category") or "",
                "lat": float(row[self.lat_col]),
                "lon": float(row[self.lon_col]),
                "distance_km": float(dk),
                "direction": dr,
            })
        return neighbors

def make_nominatim(user_agent="poi_text_embedder"):
    """
    Create a Nominatim geocoder to reverse geocode coordinates
    to human-readable addresses.
    """
    geolocator = Nominatim(user_agent=user_agent, timeout=10)
    # Usage policy: at most ~1 req/sec
    reverse = RateLimiter(geolocator.reverse, min_delay_seconds=1)
    return reverse

@lru_cache(maxsize=100_000)
def reverse_geocode(lat: float, lon: float, reverse_fn = None):
    """
    Reverse geocode a pair of latitude and longitude coordinates.
    """
    if reverse_fn is None:
        return {}
    try:
        loc = reverse_fn((lat, lon), language="en", zoom=16, addressdetails=True)
        addr = getattr(loc, "raw", {}).get("address", {}) if loc else {}
        # Normalize a few fields commonly present
        return {
            "place": addr.get("neighbourhood") or addr.get("suburb") or addr.get("hamlet"),
            "city": addr.get("city") or addr.get("town") or addr.get("village") or addr.get("county"),
            "state": addr.get("state"),
            "country": addr.get("country"),
        }
    except Exception:
        return {}

def compose_address_text(addr_dict: dict) -> str:
    """
    Compose a human-readable address text from the address components.
    """
    parts = []
    if addr_dict.get("place"): parts.append(addr_dict["place"])
    if addr_dict.get("city"): parts.append(addr_dict["city"])
    if addr_dict.get("state"): parts.append(addr_dict["state"])
    if addr_dict.get("country"): parts.append(addr_dict["country"])
    return ", ".join([str(p) for p in parts if p])

def compose_neighbors_text(neighbors: list[dict], dist: int = 100) -> str:
    """
    Compose a human-readable text from the neighboring POIs.
    Format: '0.6 km South-West: Theater District'
    """
    if not neighbors:
        return f"Nearby Places: None within {dist} km."
    lines = []
    for n in neighbors:
        name = n["name"] if n["name"] else f"POI_{n['poi_id']}"
        lines.append(f"{n['distance_km']:.1f} km {n['direction']}: {name}")
    return "Nearby Places: " + "; ".join(lines) + "."

def compose_poi_text(
        row: dict,
        addr_text: str,
        neighbors_text: str,
        lat_col="place_lat",
        lon_col="place_lon",
        name_col="location_name",
        cat_col="top_category"
    ) -> str:
    """
    Final text string to feed to pretrained text encoders.
    """
    name = row.get(name_col) or "Unknown place"
    category = row.get("top_category") or ""
    cat = f"({category})" if category else ""
    lat, lon = float(row[lat_col]), float(row[lon_col])
    coord_text = f"Coordinates: {lat:.6f}, {lon:.6f}."
    address_text = f"Address: {addr_text}." if addr_text else "Address: Unknown."
    return f"{name} {cat}. {coord_text} {address_text} {neighbors_text}"

def build_text_inputs_for_pois(
    df: pd.DataFrame,
    lat_col="place_lat",
    lon_col="place_lon",
    use_reverse_geocode=True,
    user_agent="poi_text_embedder",
    k_neighbors=10,
    max_radius_km=100,
    city="LosAngeles"
):
    """
    Build text inputs for POIs. Each POI will have a corresponding text input
    that includes its coordinates, address and neighboring POIs. This follows
    the approach presented at https://arxiv.org/pdf/2310.06213.
    Args:
        df (pd.DataFrame): Input DataFrame with POI data.
        lat_col (str): Name of the column containing latitude information.
        lon_col (str): Name of the column containing longitude information.
        use_reverse_geocode (bool): Whether to use reverse geocoding to get address information.
        user_agent (str): User agent to use for reverse geocoding.
        k_neighbors (int): Number of neighboring POIs to consider.
        max_radius_km (int): Maximum radius (in km) to search for neighboring POIs.
    Returns:
        pd.DataFrame: DataFrame with a new column 'text_input' containing the text inputs for each POI.
    """
    df = df.reset_index(drop=True).copy()
    # Reverse geocoder
    if use_reverse_geocode:
        reverse_fn = make_nominatim(user_agent)
    else:
        reverse_fn = None

    # Neighbor index
    nidx = NeighborIndex(df, lat_col=lat_col, lon_col=lon_col)

    text_inputs = []
    for i, row in df.iterrows():
        # Get address of a POI with reverse geocoding
        if use_reverse_geocode:
            addr = reverse_geocode(float(row[lat_col]), float(row[lon_col]), reverse_fn=reverse_fn) if reverse_fn else {}
        else:
            # Create address dict from safegraph attributes
            if city == "LosAngeles":
                addr = {
                    "place": row.get("street_address"),
                    "city": row.get("city"),
                    "state": row.get("region"),
                    "country": row.get("postal_code")
                }
            elif city == "Houston":
                addr = {
                    "place": row.get("safegraph.street_address"),
                    "city": row.get("safegraph.city"),
                    "state": row.get("safegraph.region"),
                    "country": row.get("safegraph.postal_code")
                }
            else:
                addr = {}
        addr_text = compose_address_text(addr)

        # Get the nearest neighbors
        neighbors = nidx.get_neighbors(i, k=k_neighbors, max_radius_km=max_radius_km)
        neighbors_text = compose_neighbors_text(neighbors)

        # Compose the final text input for the POI
        name_col = "location_name" if "location_name" in df.columns else "safegraph.location_name"
        cat_col = "top_category" if "top_category" in df.columns else "safegraph.top_category"
        poi_descr_text = compose_poi_text(row, 
            addr_text, 
            neighbors_text, 
            lat_col=lat_col, 
            lon_col=lon_col, 
            name_col=name_col, 
            cat_col=cat_col
        )
        text_inputs.append(poi_descr_text)

    df["text_input"] = text_inputs
    return df

if __name__ == "__main__":
    sfg_data_path = "/mnt/disk/data/POI_data/Safegraph/safegraph_pois_Houston_cleaned.csv"
    # read the data
    df_sfg_pois = pd.read_csv(sfg_data_path)
    city = "Houston"
    if city == "LosAngeles":
        lat_col = "latitude"
        lon_col = "longitude"
    elif city == "Houston":
        lat_col = "latitude"
        lon_col = "longitude"
    else:
        raise ValueError(f"City {city} not supported.")
    # Print the columns of the DataFrame
    df = build_text_inputs_for_pois(df = df_sfg_pois,
                                    use_reverse_geocode=False,
                                    user_agent="poi_text_embedder",
                                    lat_col="latitude",
                                    lon_col="longitude",
                                    k_neighbors=10,
                                    max_radius_km=100)
    # keep only safegraph id and text
    df = df[["safegraph_place_id", "text_input"]]
    df.to_csv("/mnt/disk/data/POI_data/Safegraph/pois_Houston_text_description_cleaned.csv", index=False)