from pyspark.sql.functions import when


def user_type_column_to_numeric(observation_df):
    return observation_df.withColumn(
        'user_type',
        when(observation_df.user_type == 'Customer', 1).
        when(observation_df.user_type == 'Subscriber', 0))
