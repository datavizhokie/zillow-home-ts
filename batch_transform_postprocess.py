# spark script for running batch transform and post-processing

import findspark
findspark.init('/Users/matt.wheeler/spark')

import os
os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages com.amazonaws:aws-java-sdk-pom:1.10.34,org.apache.hadoop:hadoop-aws:2.7.2 pyspark-shell'

from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
spark = SparkSession.builder.appName('ops').getOrCreate()

sc=spark.sparkContext

from pyspark.sql.types import *
import pyspark.sql.functions as f
from pyspark.sql.functions import *
from dateutil.parser import parse

import boto3
import sagemaker.amazon.common as smac
from sagemaker import get_execution_role
import re
import sagemaker
import numpy as np
import json
from time import gmtime, strftime

bucket  = 'zillow-home-ts'
pathkey ='2bdrm_zip'

region = boto3.Session(region_name='us-west-2', profile_name='mateosanchez') 
smclient = boto3.Session(region_name='us-west-2', profile_name='mateosanchez').client('sagemaker')

sess = sagemaker.Session(boto_session=boto3.Session(region_name='us-west-2', profile_name='mateosanchez'))
role ='arn:aws:iam::353643785282:role/service-role/AmazonSageMaker-ExecutionRole-20210115T123894'

job_name = 'derp'

model_name=job_name + '-model'
print(model_name)

model_data = r's3://{0}/{1}/{2}'.format('zillow-home-ts/2bdrm_zip/training_output', job_name, 'output/model.tar.gz')

containers  = {
    'us-east-1': '522234722520.dkr.ecr.us-east-1.amazonaws.com/forecasting-deepar:latest',
    'us-east-2': '566113047672.dkr.ecr.us-east-2.amazonaws.com/forecasting-deepar:latest',
    'us-west-2': '156387875391.dkr.ecr.us-west-2.amazonaws.com/forecasting-deepar:latest',
    'ap-northeast-1': '633353088612.dkr.ecr.ap-northeast-1.amazonaws.com/forecasting-deepar:latest'
}

container = containers['us-west-2']

primary_container = {
    'Image': container,
    'ModelDataUrl': model_data
}

create_model_response = smclient.create_model(
    ModelName = model_name,
    ExecutionRoleArn = role,
    PrimaryContainer = primary_container)

print(create_model_response['ModelArn'])

transformer = sagemaker.transformer.Transformer(
    model_name=model_name,
    instance_count=1,
    instance_type='ml.m4.10xlarge',
    strategy='SingleRecord',
    env= {
      "DEEPAR_FORWARDED_FIELDS" : '["zip"]'
    },
    assemble_with='Line',
    output_path="s3://{0}/batch_transform/predictions/raw/{1}".format(bucket, model_name),
    accept='application/jsonlines',
    max_payload=1,
    base_transform_job_name=job_name,
    sagemaker_session=sess,
)

input_key = f'{pathkey}/training_data/test/'
input_location = 's3://{}/{}'.format(bucket, input_key)

transformer.transform(
    data=input_location,
    content_type='application/jsonlines',
    split_type='Line',
    #join_source='Input'   <- can bring in the target array, but it's the entire array, not just pred period
)

transformer.wait()


model_name = job_name + '-model'

#TODO: fix issue with the following
# om.amazonaws.services.s3.model.AmazonS3Exception: Status Code: 400, AWS Service: Amazon S3, 
# AWS Request ID: 771A3B01040740DA, AWS Error Code: InvalidArgument, 
# AWS Error Message: Requests specifying Server Side Encryption with AWS KMS managed keys 
# require AWS Signature Version 4.

access_key = os.getenv('ACCESSKEY')
secret_key = os.getenv('SECRETKEY')
sc._jsc.hadoopConfiguration().set("fs.s3a.access.key", access_key)
sc._jsc.hadoopConfiguration().set("fs.s3a.secret.key", secret_key)
prediction_df=spark.read.json(f"s3a://{bucket}/batch_transform/predictions/raw/"+model_name+"/*json.out")
prediction_df=spark.read.json("batch_transform/*json.out")

prediction_df.count()

# Create date range
data = [("2018-01-01","2020-12-01")]
df_range = spark.createDataFrame(data, ["minDate", "maxDate"])

df_range= df_range.withColumn("monthsDiff", f.months_between("maxDate", "minDate"))\
    .withColumn("repeat", f.expr("split(repeat(',', monthsDiff), ',')"))\
    .select("*", f.posexplode("repeat").alias("month", "val"))\
    .withColumn("month", f.expr("add_months(minDate, month)"))\
    .select('month')

df_range = df_range.withColumn('month_string', col('month').cast('string'))

df_range.show(5)

# create array from data range DF
reference = df_range.orderBy('month_string').agg(f.collect_list(df_range.month_string)).collect()

date_array = np.array(reference)
date_array_sort = np.sort(date_array[0][0])

# Create Month column in prediction results dataframe
prediction_df=prediction_df.withColumn('month_of', f.array([f.lit(x) for x in date_array_sort]))

# create column for 50 quantile array
prediction_df_with_q50=(prediction_df
 .withColumn('Q5', prediction_df.SageMakerOutput.quantiles['0.5'])
 .drop('quantiles')
)

prediction_df_with_q50.show(5)

combine = f.udf(lambda x, y: list(zip(x, y)),
              ArrayType(StructType([StructField("month_of", StringType()),
                                    StructField("Q5", IntegerType())]))) #DoubleType

df_pred_combined = df_q50.withColumn("new", combine("month_of", "Q5"))\
       .withColumn("new", f.explode("new"))\
       .select("zip", f.col("new.month_of").alias("month_of"), f.col("new.Q5").alias("Q5"))

df_pred_combined = df_pred_combined.withColumn("month_of", date_format(col('month_of'),"yyyy-MM-dd").cast("date"))
df_pred_combined.show(5)


# Read in data with actual target to compare to predictions
test_df = spark.read.csv("2bdrm_by_zip_and_yearmonth_median_values.csv", header=True)
test_df = test_df.withColumn("month_of", f.trunc("year_month", "month"))
test_df.show(5)


# Join predictions to actual target
cond = [test_df.month_of == df_pred_combined.month_of, test_df.zip == df_pred_combined.zip]

df1 = test_df.alias('df1')
df2 = df_pred_combined.alias('df2')

results = df1.join(df2, cond, 'left').select('df1.*', df2.Q5.alias("predicted"))

results.where(col('predicted').isNotNull()).show(10)


results.coalesce(1).write.mode('overwrite').csv("preds/", header=True)