from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor


def create_feature_column(df, feature_columns, output_columns):
    # Create new column with all of the features
    vector_assembler = VectorAssembler(
        inputCols=feature_columns,
        outputCol='features')
    vector_df = vector_assembler.transform(df)
    vector_df = vector_df.select(output_columns)

    return vector_df


def build_linear_regression_model(observation_df, feature_columns):
    # Create new column with all of the features
    vector_observation_df = create_feature_column(
        observation_df, feature_columns, ['features', 'duration_sec'])

    train_df, test_df = vector_observation_df.randomSplit([0.7, 0.3])

    lr = RandomForestRegressor(
        featuresCol='features', labelCol='duration_sec')
    rfr_model = lr.fit(train_df)

    test_predictions = rfr_model.transform(test_df)
    test_predictions.select("prediction", "duration_sec", "features").show(5)

    evaluator = MulticlassClassificationEvaluator(
        predictionCol='prediction', labelCol="duration_sec",
        metricName="accuracy")
    print("RMSEon test data = %g" % evaluator.evaluate(test_predictions))

    # test_result = rfr_model.evaluate(test_df)

    return rfr_model
