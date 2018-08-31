from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.regression import LinearRegression
from pyspark.sql.functions import monotonically_increasing_id

from general_utils.spark_connector import init_spark
from general_utils.statistic_utils import show_stats, show_correlation
from models.linear_regression_model import build_linear_regression_model, create_feature_column
from preprocess_steps.add_age_column import add_age_column
from preprocess_steps.age_range_column import add_age_range_column
from preprocess_steps.column_to_boolean import gender_column_to_0_1_2
from preprocess_steps.divide_day_time import divide_day_time_step
from preprocess_steps.replace_nulls import replace_null_with_average
from preprocess_steps.split_date_to_columns import split_date_to_columns
from preprocess_steps.station_distance_column import station_distance_column
from preprocess_steps.user_type_column_to_numeric import user_type_column_to_numeric


def read_data(spark, csv_file_path):
    return spark.read.load(
        csv_file_path,
        format='com.databricks.spark.csv', header='true', inferSchema='true')


def pre_process(observation_df):
    observation_df = split_date_to_columns(observation_df, 'start_time')

    observation_df = split_date_to_columns(observation_df, 'end_time')

    observation_df = divide_day_time_step(observation_df, 'start_time')

    observation_df = divide_day_time_step(observation_df, 'end_time')

    observation_df = replace_null_with_average(observation_df, 'member_birth_year')

    observation_df = add_age_column(observation_df)

    observation_df = add_age_range_column(observation_df)

    # Remove rows with gender null # TODO: maybe do not remove those rows.
    # observation_df = observation_df.na.drop(subset=['member_gender'])

    # observation_df = gender_column_to_0_1_2(observation_df)

    observation_df = station_distance_column(observation_df)

    # TODO: remove column with age bigger than 120

    indexer = StringIndexer(inputCol="user_type", outputCol="user_type_number")
    observation_df = indexer.fit(observation_df).transform(observation_df)

    return observation_df


INPUT_FILE_DATA_PATH = "./2017-fordgobike-tripdata.csv"
OUTPUT_FILE_DATA_PATH = "./201801-fordgobike-tripdata.csv"


def main():
    spark, sc = init_spark()

    observation_df = read_data(spark, INPUT_FILE_DATA_PATH)
    observation_df = pre_process(observation_df)

    observation_df.persist()

    # status_df = show_stats(observation_df)

    feature_columns = [column[0] for column in observation_df.dtypes if column[1] in
                       ['bigint', 'double', 'int', 'float']]

    # Remove not useful columns
    remove_features = [
            'duration_sec', 'member_birth_year', 'member_gender',
            'bike_id', 'start_station_id'
        ]
    feature_columns = [feature for feature in feature_columns if feature not in remove_features]

    # Just to watch with features has high correlation
    # show_correlation(observation_df, feature_columns)

    lr_model = build_linear_regression_model(observation_df, feature_columns=feature_columns)

    new_bike_data_unchanged = read_data(spark, OUTPUT_FILE_DATA_PATH)
    # New data has already duration_sec column so we should drop it..
    new_bike_data = new_bike_data_unchanged.drop("duration_sec")
    new_bike_data = pre_process(new_bike_data)
    new_bike_data.persist()

    new_bike_data_vector = create_feature_column(
        new_bike_data, feature_columns, ['features'])

    predictions = lr_model.transform(new_bike_data_vector)
    predictions.select("prediction", "features").show()

    predictions = predictions.withColumn('id', monotonically_increasing_id())
    new_bike_data_unchanged = new_bike_data_unchanged.withColumn('id', monotonically_increasing_id())
    new_bike_data_predicted = new_bike_data_unchanged.join(predictions, 'id', "outer")

    new_bike_data_predicted.write.csv("./predicted_2018_bika_data.csv")

    pass


if __name__ == '__main__':
    main()
