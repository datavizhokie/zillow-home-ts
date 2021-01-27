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

    print("Training Data Metrics:")
    train_data.agg(f.countDistinct(series_field), f.countDistinct(granularity),\
        f.min(granularity), f.max(granularity), f.avg('target')).show()

    print("Full Test Data Metrics:")
    test_data.agg(f.countDistinct(series_field), f.countDistinct(granularity),\
        f.min(granularity), f.max(granularity), f.avg('target')).show()

    return train_data, test_data


def create_array_formats(train_data, test_data, series_field, granularity):

    w = Window.partitionBy(series_field).orderBy(granularity)

    sorted_list_train = train_data.withColumn('target', f.collect_list('target').over(w)
                                        )\
    .groupBy(series_field)\
    .agg(f.max('target').alias('target'), 
        f.min('mindate').alias('start'), 
    #      f.max('month_list').alias('dynamic_feat1'),
    #      f.max('year_list').alias('dynamic_feat2')
        )

    sorted_list_test = test_data.withColumn('target', f.collect_list('target').over(w)
                                        )\
    .groupBy(series_field)\
    .agg(f.max('target').alias('target'), 
        f.min('mindate').alias('start'), 
    #      f.max('month_list').alias('dynamic_feat1'),
    #      f.max('year_list').alias('dynamic_feat2')
        )

    print("Preview of DF with Target arrays by Series:")
    sorted_list_train.show(5)

    #TODO: add logic for Cat's and DynFeat's
    # train_final = sorted_list_train.select("OFFENSE_CATEGORY_ID","start","target").persist()
                                       #f.array(["cat2"]).alias("cat"),
                                                    #f.array(["dynamic_feat1","dynamic_feat2"]).alias("dynamic_feat")).persist()
    # test_final = ...

    return sorted_list_train, sorted_list_test


def write_final_to_json(train, test, bucket, pathkey, access_key, secret_key, sc):
    sc._jsc.hadoopConfiguration().set("fs.s3a.access.key", access_key)
    sc._jsc.hadoopConfiguration().set("fs.s3a.secret.key", secret_key)

    try:
        test.coalesce(1).write.mode('overwrite').json(f's3a://{bucket}/{pathkey}/training_data/test')
        train.coalesce(1).write.mode('overwrite').json(f's3a://{bucket}/{pathkey}/training_data/train')
        print("Write to S3 in json format succeeded.")
        
    except:
        print("Write to json failed.")

def main():

    access_key, secret_key = retrieve_aws_creds()

    df_joined = read_create_mindate(file='2bdrm_by_zip_and_yearmonth_median_values.csv', series_field='zip', granularity='year_month', target_field='median_value')
    train_data, test_data = train_split(df=df_joined, max_train_date="2017-12-31", series_field='zip', granularity='year_month')
    sorted_list_train, sorted_list_test = create_array_formats(train_data=train_data, test_data=test_data, series_field='zip', granularity='year_month')
    write_final_to_json(train=sorted_list_train, test=sorted_list_test, bucket='zillow-home-ts', pathkey='2bdrm_zip', access_key=access_key, secret_key=secret_key, sc=sc)

if __name__ == "__main__":
    main()