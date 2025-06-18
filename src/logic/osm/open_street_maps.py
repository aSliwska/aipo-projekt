from osm.countries import all_countries
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from geopy import distance
import pandas as pd
from functools import partial
from osm.caching import Cache
import numpy as np

def predict_place(countries_by_road_side, countries_by_road_signs, countries_by_car_license_plates, 
                  regions_by_car_license_plates, countries_by_language, places_with_distances):
    """Calls OpenStreetMaps to find the most fitting place based on given params.
    Keyword arguments:
    countries_by_road_side -- list of country names that match the side of the road the vehicle in the video drove on
    countries_by_road_signs -- list of country names that match the road signs that appear in the video
    countries_by_car_license_plates -- list of country names that match the vehicle license plates that appear in the video
    regions_by_car_license_plates -- list of region names that match the vehicle license plates that appear in the video
    countries_by_language -- list of country names that match the language of text found in the video
    places_with_distances -- dictionary of characteristic words (places) found in text in the video that could point to 
        a location, with distance (in kilometers) as values (0. if no distance found)
    Returns:
        ([latitude, longitude], radius) - where radius in meters represents the precision of the prediction
        or (None, None) when nothing was found.

    Determines the country by finding the most fitting one using hardcoded weights for each country input list. 
    Queries Nominatim API for each combination (cartesian product) of "country, region, place", 
    but only uses another country if nothing is found for the first one. After receiving lists of coordinates
    for each query, finds the final median (lat, lon) and the final radius as 
    max( distance to farthest found point + distance from that point in places_with_distances ).

    """

    countries_possibilities = __get_country_probabilities(countries_by_road_side, countries_by_road_signs, countries_by_car_license_plates, countries_by_language)

    country_idx = -1 if len(countries_possibilities) == 0 else 0

    df = __query_osm_for_locations(country_idx, countries_possibilities, regions_by_car_license_plates, places_with_distances)

    if df.empty:
        return None, None # nothing found

    df['points'] = df['location'].apply(lambda locations: [(loc.latitude, loc.longitude) for loc in locations])

    return __get_median_coords_and_max_radius(df)


def __query_osm_for_locations(country_idx, countries_possibilities, regions_by_car_license_plates, places_with_distances):
    df = pd.DataFrame()

    geolocator = Nominatim(user_agent="geoguesser_app_agh_wfiis_aipo")
    geocode = partial(RateLimiter(geolocator.geocode, min_delay_seconds=1), language='pl', exactly_one=False)
    cache = Cache()

    # search until we find at least one existing location (from best fitting country to worst)
    while df.empty and country_idx < len(countries_possibilities):
        df['query'], df['distance'] = __get_queries(country_idx, countries_possibilities, regions_by_car_license_plates, places_with_distances)

        # send query - country with the best score, each of the appearing regions, each of the keywords (cartesian product)
        df['location'] = df['query'].apply(lambda query: __get_geocoded_result(geocode, cache, query))
        df = df.dropna()
        country_idx += 1
    
    return df


def __get_median_coords_and_max_radius(df):
    points = [] # (lat, lon)
    distances = [] # distance in meters
    for index, row in df.iterrows():
        for point in row['points']:
            points.append(point)
            distances.append(row['distance'] * 1000.)

    median_coords = np.median(points, axis=0).tolist()
    max_radius = 0

    for i in range(len(points)):
        r = distances[i] + distance.distance(points[i], median_coords).m
        if r > max_radius:
            max_radius = r

    return median_coords, max_radius


def __get_geocoded_result(geocode, cache, query):
    location = cache.query_cached(query)

    if location is False:
        print('fetching not cached query:', query)
        location = geocode(query)
        cache.save_to_cache(query, location)
    
    return location


def __get_queries(country_idx, countries, regions, places_with_distances):
    country = [] if country_idx == -1 else [countries[country_idx][0]]

    # if countries and regions empty
    if country_idx == -1 and len(regions) == 0:
        return pd.DataFrame({'query': [], 'distance': []}) # place name alone is not enough
    
    df_countries = pd.DataFrame({'country': country})
    df_regions = pd.DataFrame({'region': regions})
    df_places = pd.DataFrame({'place': list(places_with_distances.keys()), 'distance' : list(places_with_distances.values())})

    df = df_countries if df_regions.empty else df_regions if df_countries.empty else df_countries.merge(df_regions, how='cross')
    df = df if df_places.empty else df_places if df.empty else df.merge(df_places, how='cross')

    columns = []
    if not df_countries.empty:
        columns.append('country')
    if not df_regions.empty:
        columns.append('region')
    if not df_places.empty:
        columns.append('place')

    df['query'] = df[columns].agg(', '.join, axis=1)
    return df['query'], df['distance']


def __get_country_probabilities(countries_by_road_side, countries_by_road_signs, countries_by_car_license_plates, countries_by_language):
    # add points to countries based on input
    road_side_weight = 0.5
    road_signs_weight = 0.6
    license_plates_weight = 0.4
    language_weight = 0.6

    countries_possibilities = {country : 0. for country in all_countries}

    for country in countries_by_road_side:
        countries_possibilities[country] += road_side_weight
    
    for country in countries_by_road_signs:
        countries_possibilities[country] += road_signs_weight
    
    for country in countries_by_car_license_plates:
        countries_possibilities[country] += license_plates_weight
    
    for country in countries_by_language:
        countries_possibilities[country] += language_weight
    
    # filter out countries with a score of 0
    countries_possibilities = dict(filter(lambda item: item[1] != 0., countries_possibilities.items()))

    # convert to list of (key, value) touples and sort by value descending
    countries_possibilities = sorted(countries_possibilities.items(), key=lambda kv: kv[1], reverse=True)

    return countries_possibilities

if __name__ == "__main__":
    countries_by_road_side = ["Poland", "Portugal", "Hungary", "Iceland", "Georgia"]
    countries_by_road_signs = ["Poland", "Portugal", "Hungary", "Germany"] 
    countries_by_language = ["Poland", "Portugal", "Germany"] 
    countries_by_car_license_plates = ["Hungary", "Poland", "Ukraine"]

    regions_by_car_license_plates = ["Kielce", "Kraków"]
    places_with_distances = {"Żabka": 0, "Galeria Krakowska": 0.5}

    print(predict_place(countries_by_road_side, countries_by_road_signs, countries_by_car_license_plates, 
                  regions_by_car_license_plates, countries_by_language, places_with_distances))
    

