from pyspark.sql.functions import when


def add_age_range_column(observation_df):
    """
    is_child: 0 - 17
    is_young: 18 - 39
    is_middle: 40 - 60
    is_old: 60+
    :param observation_df:
    :return:
    """

    observation_df = observation_df.withColumn(
        "is_child",
        when(observation_df.age < 18, 1).otherwise(0))

    observation_df = observation_df.withColumn(
        "is_young",
        when((18 <= observation_df.age) & (observation_df.age < 40), 1).otherwise(0))

    observation_df = observation_df.withColumn(
        "is_middle",
        when((40 <= observation_df.age) & (observation_df.age < 60), 1).otherwise(0))

    observation_df = observation_df.withColumn(
        "is_old",
        when(observation_df.age >= 60, 1).otherwise(0))

    return observation_df
