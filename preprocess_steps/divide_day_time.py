from pyspark import Row


def add_hour_column(rdd, date_column):
    row_dict = rdd.asDict()
    hour = rdd[date_column].hour
    row_dict[date_column + '_is_morning'] = 1 if 6 <= hour < 12 else 0
    row_dict[date_column + '_is_noon'] = 1 if 12 <= hour < 18 else 0
    row_dict[date_column + '_is_evening'] = 1 if 18 <= hour < 22 else 0
    row_dict[date_column + '_is_night'] = 1 if 22 <= hour < 24 or 0 <= hour < 6 else 0

    return Row(**row_dict)


def divide_day_time_step(observation_df, date_column):
    observation_df = observation_df.rdd.map(lambda rdd: add_hour_column(rdd, date_column)).toDF()

    return observation_df
