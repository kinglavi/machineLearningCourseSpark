from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.regression import RandomForestRegressor

from models.utils import create_feature_column


def build_random_forest_regressor_model(observation_df, feature_columns):
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
    print("RMSE on test data = %g" % evaluator.evaluate(test_predictions))

    # test_result = rfr_model.evaluate(test_df)

    return rfr_model