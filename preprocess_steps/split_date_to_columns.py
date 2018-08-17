from pyspark import Row


def split_date(rdd, date_column):
    row_dict = rdd.asDict()

    row_dict[date_column + '_hour'] = rdd[date_column].hour
    row_dict[date_column + '_year'] = rdd[date_column].year
    row_dict[date_column + '_month'] = rdd[date_column].month
    row_dict[date_column + '_day_in_week'] = rdd[date_column].weekday()
    row_dict[date_column + '_day_in_week'] = rdd[date_column].day

    return Row(**row_dict)


def split_date_to_columns(observation_df, column_name):
    return observation_df.rdd.map(lambda rdd: split_date(rdd, column_name)).toDF()
