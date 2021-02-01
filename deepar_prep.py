# spark script to format data for DeepAR
import findspark
findspark.init('/Users/matt.wheeler/spark')
import os
os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages com.amazonaws:aws-java-sdk-pom:1.10.34,org.apache.hadoop:hadoop-aws:2.7.2 pyspark-shell'
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
spark = SparkSession.builder.appName('ops').getOrCreate()

sc = spark.sparkContext

import pyspark.sql.functions as f
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import lit, isnan
from pyspark.sql import Window
from pyspark.ml.feature import StringIndexer
#import boto3

def retrieve_aws_creds():

    try:
        access_key = os.getenv('ACCESSKEY')
        secret_key = os.getenv('SECRETKEY')
        print("$$$$$$$$$$$$$$$$$$$")
        print("AWS creds loaded :)")
        
    except:
        print("AWS creds unable to load...")
        exit()

    return access_key, secret_key


def read_create_mindate(file, series_field, granularity, target_field):

    df = spark.read.csv(file, header=True)

    # Create min_date for each series
    df_mindate = df.groupby(series_field).agg(f.min(granularity).alias('mindate')).persist()

    # Join mindates back to df
    df_joined = df.join(df_mindate,[series_field], how='left')

    # Create target from field to forecast
    df_joined = df_joined.withColumn('target', df_joined[target_field])

    print("Preview of initial data:")
    df_joined.show(5)

    return df_joined




def train_split(df, max_train_date, series_field, granularity):

    train_data = df.filter(df[granularity] <= max_train_date)
    test_data = df
    validation_set = df.filter(df[granularity] > max_train_date)

    print("Training Data Metrics:")
    train_data.agg(f.countDistinct(series_field), f.countDistinct(granularity),\
        f.min(granularity), f.max(granularity), f.avg('target')).show()

    print("Full Test Data Metrics:")
    test_data.agg(f.countDistinct(series_field), f.countDistinct(granularity),\
        f.min(granularity), f.max(granularity), f.avg('target')).show()

    print("Validation Data Metrics:")
    validation_set.agg(f.countDistinct(series_field), f.countDistinct(granularity),\
        f.min(granularity), f.max(granularity), f.avg('target')).show()

    return train_data, test_data


def string_index(train_data, test_data, input, output):
    # encoding of categorical fields
    indexer = StringIndexer(inputCol=input, outputCol=output, handleInvalid="keep")
    train_data = indexer.fit(train_data).transform(train_data)
    test_data = indexer.fit(test_data).transform(test_data)

    # Categorical fields need to be INT
    train_data = train_data.withColumn(output, train_data[output].cast('int'))
    test_data = test_data.withColumn(output, test_data[output].cast('int'))

    return train_data, test_data


def create_array_formats(train_data, test_data, series_field, granularity):

    # The batch transform data needs to have a dynamic feature array the size of Training+Validation
    # SLIM TRAINING DATA
    keep_cols=[series_field, granularity , 'target']
    train_data_slim = train_data.select([column for column in train_data.columns if column in keep_cols])

    # Join slim training period data back to Full test length dynamic feature data
    train_data_batch = test_data.select(series_field, "mindate","state_index","city_index", granularity, "cnt_holidays").join(train_data_slim,[series_field,granularity], how='left')

    print("**train_data_batch** preview (Target should be NULL in validation period)")
    train_data_batch.filter(train_data_batch[granularity] == "2020-02").show(10)

    w = Window.partitionBy(series_field).orderBy(granularity)

    sorted_list_train = train_data.withColumn('target', f.collect_list('target').over(w))\
    .withColumn('cnt_holidays', f.collect_list('cnt_holidays').over(w))\
    .groupBy(series_field)\
    .agg(f.max('target').alias('target'), 
        f.min('mindate').alias('start'), 
        f.min('state_index').alias('cat1'),
        f.min('city_index').alias('cat2'),
        f.max('cnt_holidays').alias('dynamic_feat1')
        )

    sorted_list_test = test_data.withColumn('target', f.collect_list('target').over(w))\
    .withColumn('cnt_holidays', f.collect_list('cnt_holidays').over(w))\
    .groupBy(series_field)\
    .agg(f.max('target').alias('target'), 
        f.min('mindate').alias('start'), 
        f.min('state_index').alias('cat1'),
        f.min('city_index').alias('cat2'),
        f.max('cnt_holidays').alias('dynamic_feat1')
        )

    sorted_list_train_batch = train_data_batch.withColumn('target', f.collect_list('target').over(w))\
    .withColumn('cnt_holidays', f.collect_list('cnt_holidays').over(w))\
    .groupBy(series_field)\
    .agg(f.max('target').alias('target'), 
        f.min('mindate').alias('start'), 
        f.min('state_index').alias('cat1'),
        f.min('city_index').alias('cat2'),
        f.max('cnt_holidays').alias('dynamic_feat1')
        )

    train_final = sorted_list_train.select(series_field,"start","target", f.array(["cat1","cat2"]).alias("cat"),
    f.array(["dynamic_feat1"]).alias("dynamic_feat"))

    test_final  = sorted_list_test.select(series_field,"start","target", f.array(["cat1","cat2"]).alias("cat"),
    f.array(["dynamic_feat1"]).alias("dynamic_feat"))

    train_batch  = sorted_list_train_batch.select(series_field,"start","target", f.array(["cat1","cat2"]).alias("cat"),
    f.array(["dynamic_feat1"]).alias("dynamic_feat"))

    print("Preview of DeepAR input data w/ Dynamic Feature(s):")
    train_final.show(5)

    return train_final, test_final, train_batch


def write_final_to_json(train, test, batch, bucket, pathkey, access_key, secret_key, sc):
    sc._jsc.hadoopConfiguration().set("fs.s3a.access.key", access_key)
    sc._jsc.hadoopConfiguration().set("fs.s3a.secret.key", secret_key)

    try:
        test.coalesce(1).write.mode('overwrite').json(f's3a://{bucket}/{pathkey}/training_data/test')
        train.coalesce(1).write.mode('overwrite').json(f's3a://{bucket}/{pathkey}/training_data/train')
        batch.coalesce(1).write.mode('overwrite').json(f's3a://{bucket}/{pathkey}/training_data/batch')
        print("Write to S3 in json format succeeded.")
        
    except:
        print("Write to json failed.")

def main():

    access_key, secret_key = retrieve_aws_creds()

    df_joined = read_create_mindate(file='2bdrm_by_zip_yearmonth.csv', series_field='zip', granularity='year_month', target_field='median_value')
    train_data, test_data = train_split(df=df_joined, max_train_date="2017-12-31", series_field='zip', granularity='year_month')
    train_data, test_data = string_index(train_data=train_data, test_data=test_data, input='State', output='state_index')
    train_data, test_data = string_index(train_data=train_data, test_data=test_data, input='City', output='city_index')
    train_final, test_final, train_batch = create_array_formats(train_data=train_data, test_data=test_data, series_field='zip', granularity='year_month')
    write_final_to_json(train=train_final, test=test_final, batch=train_batch,bucket='zillow-home-ts', pathkey='2bdrm_zip', access_key=access_key, secret_key=secret_key, sc=sc)

if __name__ == "__main__":
    main()