def replace_null_with_average(observation_df, column_name):

    average = observation_df.agg({column_name: "avg"}).collect()[0].asDict()['avg({})'.format(column_name)]

    observation_df = observation_df.fillna({column_name: average})

    return observation_df
