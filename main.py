import six
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

from general_utils.spark_connector import init_spark
from preprocess_steps.add_age_column import add_age_column
from preprocess_steps.age_range_column import add_age_range_column
from preprocess_steps.column_to_boolean import gender_column_to_0_1_2
from preprocess_steps.divide_day_time import divide_day_time_step
from preprocess_steps.replace_nulls import replace_null_with_average
import pandas

from preprocess_steps.split_date_to_columns import split_date_to_columns
from preprocess_steps.station_distance_column import station_distance_column
from preprocess_steps.user_type_column_to_numeric import user_type_column_to_numeric


def main():
    spark, sc = init_spark()

    observation_df = spark.read.load(
        "./2017-fordgobike-tripdata.csv",
        format='com.databricks.spark.csv', header='true', inferSchema='true')

    observation_df = split_date_to_columns(observation_df, 'start_time')

    observation_df = split_date_to_columns(observation_df, 'end_time')

    observation_df = divide_day_time_step(observation_df, 'start_time')

    observation_df = divide_day_time_step(observation_df, 'end_time')

    observation_df = replace_null_with_average(observation_df, 'member_birth_year')

    observation_df = add_age_column(observation_df)

    observation_df = add_age_range_column(observation_df)

    # Remove rows with gender null # TODO: maybe do not remove those rows.
    observation_df = observation_df.na.drop(subset=['member_gender'])

    observation_df = gender_column_to_0_1_2(observation_df)

    observation_df = station_distance_column(observation_df)

    observation_df = user_type_column_to_numeric(observation_df)

    # TODO: remove column with age bigger than 120

    # df = show_stats(observation_df)

    numeric_columns = [column[0] for column in observation_df.dtypes if column[1] in
                       ['bigint', 'double', 'int', 'float']]
    observation_df.persist()
    # for column in numeric_columns:
    #     if not (isinstance(observation_df.select(column).take(1)[0][0], six.string_types)):
    #         print("Correlation to duration_sec for ", column, observation_df.stat.corr('duration_sec', column))

    vector_assembler = VectorAssembler(
        inputCols=numeric_columns,
        outputCol='features')
    vector_observation_df = vector_assembler.transform(observation_df)
    vector_observation_df = vector_observation_df.select(['features', 'duration_sec'])
    vector_observation_df.show(3)

    splits = vector_observation_df.randomSplit([0.7, 0.3])
    train_df = splits[0]
    test_df = splits[1]

    lr = LinearRegression(
        featuresCol='features', labelCol='duration_sec',
        maxIter=10, regParam=0.3, elasticNetParam=0.8)

    lr_model = lr.fit(train_df)
    print("Coefficients: " + str(lr_model.coefficients))
    print("Intercept: " + str(lr_model.intercept))

    training_summary = lr_model.summary
    print("RMSE: %f" % training_summary.rootMeanSquaredError)
    print("r2: %f" % training_summary.r2)

    train_df.describe().show()

    lr_predictions = lr_model.transform(test_df)
    lr_predictions.select("prediction", "duration_sec", "features").show(5)

    lr_evaluator = RegressionEvaluator(
        predictionCol="prediction",
        labelCol="duration_sec", metricName="r2")
    print("R Squared (R2) on test data = %g" % lr_evaluator.evaluate(lr_predictions))

    test_result = lr_model.evaluate(test_df)
    print("Root Mean Squared Error (RMSE) on test data = %g" % test_result.rootMeanSquaredError)

    print("numIterations: %d" % training_summary.totalIterations)
    print("objectiveHistory: %s" % str(training_summary.objectiveHistory))
    training_summary.residuals.show()

    predictions = lr_model.transform(test_df)
    predictions.select("prediction", "duration_sec", "features").show()


def show_stats(df):
    pandas_df = df.describe().toPandas().transpose()
    return pandas_df


if __name__ == '__main__':
    main()
