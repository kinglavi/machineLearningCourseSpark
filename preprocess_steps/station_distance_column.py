from geopy import distance
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType


def calculate_distance(
        start_station_lat, start_station_lang,
        end_station_latitude, end_station_longitude):
    start_station_coords = (start_station_lat, start_station_lang)
    end_station_coords = (end_station_latitude, end_station_longitude)

    return distance.distance(start_station_coords, end_station_coords).km


def station_distance_column(observation_df):
    calc_distance = udf(calculate_distance, FloatType())

    observation_df = observation_df.withColumn(
        'station_distance',
        calc_distance(
            observation_df['start_station_latitude'], observation_df['start_station_longitude'],
            observation_df['end_station_latitude'], observation_df['end_station_longitude'])
    )

    return observation_df
