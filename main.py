from __future__ import print_function

from general_utils.spark_utils import init_spark, marge_two_df_by_order
from general_utils.statistic_utils import show_correlation
from models.gradient_boosted_tree_regression_model import build_gradient_boosted_tree_regression
from models.utils import create_feature_column
from steps.pre_process import pre_process
from steps.reader import read_data, read_new_data_and_pre_process

INPUT_FILE_DATA_PATH = "./2017-fordgobike-tripdata.csv"
NEW_DATA_FILE_PATH = "./201801-fordgobike-tripdata.csv"
PREDICT_RESULT_FILE_PATH = "./201801-fordgobike-tripdata-predictions"


def main():
    spark, sc = init_spark()

    observation_df = read_data(spark, INPUT_FILE_DATA_PATH)

    observation_df = pre_process(observation_df)

    observation_df.persist()

    # Remove not useful columns
    remove_features = [
        'duration_sec', 'member_birth_year', 'member_gender',
        'bike_id', 'start_station_id', 'end_station_id',
        'start_time_year', 'is_child', 'end_time_year',
        'end_time_is_evening', 'start_time_is_evening'
    ]
    feature_columns = [column[0] for column in observation_df.dtypes if column[1] in
                       ['bigint', 'double', 'int', 'float'] and column[0] not in remove_features]

    # status_df = show_stats(observation_df)
    # Just to watch with features has high correlation
    # show_correlation(observation_df, feature_columns)

    model = build_gradient_boosted_tree_regression(
        observation_df, feature_columns=feature_columns)

    new_bike_data, new_bike_data_unchanged = read_new_data_and_pre_process(
        spark, NEW_DATA_FILE_PATH)

    new_bike_data_vector = create_feature_column(
        new_bike_data, feature_columns, ['features'])

    predictions = model.transform(new_bike_data_vector)
    predictions = predictions.drop("features")

    new_bike_data_predicted = marge_two_df_by_order(predictions, new_bike_data_unchanged)

    new_bike_data_predicted.toPandas().to_csv(PREDICT_RESULT_FILE_PATH)
    new_bike_data_predicted.write.csv(PREDICT_RESULT_FILE_PATH, header=True)

    print("Successfully wrote predicted data to file.")


if __name__ == '__main__':
    main()
