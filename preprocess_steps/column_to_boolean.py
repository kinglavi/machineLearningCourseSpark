from pyspark.sql.functions import when


def gender_column_to_0_1_2(observation_df):
    observation_df = observation_df.withColumn(
        'member_gender',
        when(observation_df.member_gender == "Male", 0).
        when(observation_df.member_gender == "Female", 1).
        when(observation_df.member_gender == "Other", 2)
    )

    return observation_df
