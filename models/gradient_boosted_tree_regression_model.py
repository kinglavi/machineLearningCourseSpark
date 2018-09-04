from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import GBTRegressor

from models.utils import create_feature_column


def build_gradient_boosted_tree_regression(observation_df, feature_columns):
    # Create new column with all of the features
    vector_observation_df = create_feature_column(
        observation_df, feature_columns, ['features', 'duration_sec'])

    train_df, test_df = vector_observation_df.randomSplit([0.7, 0.3])
    model = GBTRegressor(featuresCol="features", labelCol="duration_sec", maxIter=15)

    model = model.fit(train_df)

    test_predictions = model.transform(test_df)

    test_predictions.select("prediction", "duration_sec", "features").show(5)

    evaluator = RegressionEvaluator(
        predictionCol='prediction', labelCol="duration_sec",
        metricName="rmse")
    print("RMSE on test data = %g" % evaluator.evaluate(test_predictions))

    evaluator = RegressionEvaluator(
        predictionCol='prediction', labelCol="duration_sec",
        metricName="r2")

    print("R2 on test data = %g" % evaluator.evaluate(test_predictions))

    return model


