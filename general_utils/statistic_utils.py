from __future__ import print_function

import six


def show_stats(df):
    pandas_df = df.describe().toPandas().transpose()
    return pandas_df


def show_correlation(observation_df, numeric_columns):
    for column in numeric_columns:
        if not (isinstance(observation_df.select(column).take(1)[0][0], six.string_types)):
            print("Correlation to duration_sec for ", column, observation_df.stat.corr('duration_sec', column))
