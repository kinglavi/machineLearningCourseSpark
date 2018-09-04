from pyspark.ml.feature import VectorAssembler


def create_feature_column(df, feature_columns, output_columns):
    # Create new column with all of the features
    vector_assembler = VectorAssembler(
        inputCols=feature_columns,
        outputCol='features')
    vector_df = vector_assembler.transform(df)
    vector_df = vector_df.select(output_columns)

    return vector_df