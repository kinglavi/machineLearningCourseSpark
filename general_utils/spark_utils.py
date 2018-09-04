from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id


def init_spark():
    spark = SparkSession.builder.appName("HelloWorld").getOrCreate()
    sc = spark.sparkContext
    return spark, sc


def marge_two_df_by_order(df, other_df):
    df = df.withColumn('id', monotonically_increasing_id())
    other_df = other_df.withColumn('id', monotonically_increasing_id())
    return other_df.join(df, 'id', "outer")