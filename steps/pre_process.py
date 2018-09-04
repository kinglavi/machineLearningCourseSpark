from pyspark.ml.feature import StringIndexer

from preprocess_steps.add_age_column import add_age_column
from preprocess_steps.age_range_column import add_age_range_column
from preprocess_steps.divide_day_time import divide_day_time_step
from preprocess_steps.replace_nulls import replace_null_with_average
from preprocess_steps.split_date_to_columns import split_date_to_columns
from preprocess_steps.station_distance_column import station_distance_column


def pre_process(observation_df):
    observation_df = split_date_to_columns(observation_df, 'start_time')

    observation_df = split_date_to_columns(observation_df, 'end_time')

    observation_df = divide_day_time_step(observation_df, 'start_time')

    observation_df = divide_day_time_step(observation_df, 'end_time')

    observation_df = replace_null_with_average(observation_df, 'member_birth_year')

    observation_df = add_age_column(observation_df)

    observation_df = add_age_range_column(observation_df)

    # Remove rows with gender null # TODO: maybe do not remove those rows.
    observation_df = observation_df.na.drop(subset=['member_gender'])

    # observation_df = gender_column_to_0_1_2(observation_df)
    indexer = StringIndexer(inputCol="member_gender", outputCol="member_gender_number")
    observation_df = indexer.fit(observation_df).transform(observation_df)

    observation_df = station_distance_column(observation_df)

    observation_df = observation_df.filter(observation_df.age < 110)

    indexer = StringIndexer(inputCol="user_type", outputCol="user_type_number")
    observation_df = indexer.fit(observation_df).transform(observation_df)

    return observation_df