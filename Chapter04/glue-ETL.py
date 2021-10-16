import sys 

from awsglue.utils import getResolvedOptions 
from awsglue.transforms import Join 
from pyspark.context import SparkContext 
from awsglue.context import GlueContext 
from awsglue.job import Job 
import pandas as pd 
from datetime import datetime 
import uuid 
from pyspark.ml.feature import StringIndexer 

glueContext = GlueContext(SparkContext.getOrCreate()) 

logger = glueContext.get_logger() 

current_date = datetime.now() 

default_date_partition = f"{current_date.year}-{current_date.month}-{current_date.day}"   

default_version_id = str(uuid.uuid4()) 

default_bucket = "data-lake-demo-serving-dyping<your default bucket name>" 

default_prefix = "ml-customer-churn" 

target_bucket = "" 

prefix = "" 

day_partition ="" 

version_id = "" 

try: 

    args = getResolvedOptions(sys.argv,['JOB_NAME','target_bucket','prefix','day_partition','version_id']) 

    target_bucket = args['target_bucket'] 

    prefix = args['prefix'] 

    day_partition = args['day_partition'] 

    version_id = args['version_id'] 

except: 

    logger.error("error occured with getting arguments") 

if target_bucket == "":  

    target_bucket = default_bucket 

if prefix == "": 

    prefix = default_prefix 

if day_partition == "": 

    day_partition = default_date_partition 

if version_id == "": 

    version_id = default_version_id 

 # catalog: database and table names 

db_name = "customer_db" 

tbl_customer = "customer_data" 

tbl_churn_list = "churn_list" 

# Create dynamic frames from the source tables  

customer = glueContext.create_dynamic_frame.from_catalog(database=db_name, table_name=tbl_customer) 

churn = glueContext.create_dynamic_frame.from_catalog(database=db_name, table_name=tbl_churn_list) 

# Join the frames to create customer churn dataframe 

customer_churn = Join.apply(customer, churn, 'customerid', 'customerid') 

customer_churn.printSchema() 

# ---- Write out the combined file ---- 

current_date = datetime.now() 

str_current_date = f"{current_date.year}-{current_date.month}-{current_date.day}"   

random_version_id = str(uuid.uuid4()) 

output_dir = f"s3://{target_bucket}/{prefix}/{day_partition}/{version_id}" 

s_customer_churn = customer_churn.toDF() 

gender_indexer = StringIndexer(inputCol="gender", outputCol="genderindex") 

s_customer_churn = gender_indexer.fit(s_customer_churn).transform(s_customer_churn) 

geo_indexer = StringIndexer(inputCol="geography", outputCol="geographyindex") 

s_customer_churn = geo_indexer.fit(s_customer_churn).transform(s_customer_churn) 

s_customer_churn = s_customer_churn.select('geographyindex', 'estimatedsalary','hascrcard','numofproducts', 'balance', 'age', 'genderindex', 'isactivemember', 'creditscore', 'tenure', 'exited') 

s_customer_churn = s_customer_churn.coalesce(1) 

s_customer_churn.write.option("header","true").format("csv").mode('Overwrite').save(output_dir) 

logger.info("output_dir:" + output_dir) 

 