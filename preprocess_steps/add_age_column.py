

def add_age_column(observation_df):
    return observation_df.withColumn(
        "age", observation_df.start_time_year - observation_df.member_birth_year)
