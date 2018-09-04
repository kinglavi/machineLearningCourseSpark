from __future__ import print_function

import six


def show_stats(df):
    pandas_df = df.describe().toPandas().transpose()
    return pandas_df


def show_correlation(observation_df, numeric_columns):
    for column in numeric_columns:
        if not (isinstance(observation_df.select(column).take(1)[0][0], six.string_types)):
            print("Correlation to duration_sec for ", column, observation_df.stat.corr('duration_sec', column))


correlation = {
    "end_station_latitude": 0.00692779921467,
    "end_station_longitude": -0.00142525613598,
    "end_time_day_in_week": 0.00304284585533,
    "end_time_hour": 0.00372852220773,
    "end_time_is_evening": -0.00184025285163,
    "end_time_is_morning": -0.014175229857,
    "end_time_is_noon": 0.0128139789473,
    "end_time_month": -0.0260895688832,
    "end_time_year": 0.0353374904848,
    "start_station_latitude": 0.00693670400941,
    "start_station_longitude": -0.000336424073019,
    "start_time_day_in_week": 0.00211089358795,
    "start_time_hour": 0.0115397385249,
    "start_time_is_evening": -0.000728426737693,
    "start_time_is_morning": -0.0189349802247,
    "start_time_is_night": 0.0053031047373,
    "start_time_is_noon": 0.0173768584594,
    "start_time_month": -0.0251690476598,
    "start_time_year": 0,
    "age": -0.00496274235803,
    "is_child": 0,
    "is_young": -0.000119870854425,
    "is_middle": -0.00112421218675,
    "is_old": 0.00339662187612,
    "member_gender_number": 0.0407852570194,
    "station_distance": 0.0941418952255,
    "user_type_number": 0.145240314545}
