# Object-based-FM

* Using python v3.13.2
* To install dependencies, run `pip install -r requirementst.txt`.

# Dev Notes

### Data assumptions

1. Staypoints that are not attributed to a POI still exist in the pre-training dataset.
    - These staypoints still have location + time information but the category= 0, and the poi_ids = 0 (SAME to `PAD` value).

2. Attributed staypoints (= Visits) have the following attributes. 
    - Location
    - Time of arrival
    - Time of departure
    - Category => following Safegraph's top category attribute
    - Naics codes => from safefraph
    - Naics 2 digit codes => the first 2 digits indicate higher level category.

3. Some POIs might still not have category information. In this case the category is again `UNKNOWN`.

4. All `UNKNOWN` and `PADDED` VALUES are set to 0. Please see `utils/constants.py` file.


### Pretraining

1. Pre training utilizes the whole dataset. We can potentially keep some users out to evaluate the models MLM pre-training. 


LA bbox: [32.80798, -118.944405, 34.820696, -117.652404]
LA num pois: 39557
window size: 32

Houston bbox: 29.557009000000004, -95.558418, 29.949892, -95.158592
Houston num pois: 32160 / 28419