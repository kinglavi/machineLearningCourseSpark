from steps.pre_process import pre_process


def read_data(spark, csv_file_path):
    return spark.read.load(
        csv_file_path,
        format='com.databricks.spark.csv', header='true', inferSchema='true')


def read_new_data_and_pre_process(spark, file_path):
    new_bike_data_unchanged = read_data(spark, file_path)
    # New data has already duration_sec column so we should drop it..
    new_bike_data = new_bike_data_unchanged.drop("duration_sec")
    new_bike_data = pre_process(new_bike_data)
    new_bike_data.persist()

    return new_bike_data, new_bike_data_unchanged
