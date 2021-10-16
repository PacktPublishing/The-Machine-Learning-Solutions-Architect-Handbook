import json

import boto3

def lambda_handler(event, context):

s3 = boto3.resource('s3')

for record in event['Records']:

srcBucket = record['s3']['bucket']['name']

srckey = record['s3']['object']['key']

desBucket = "MLSA-DataLake-<your initials>"

desFolder = srckey[0:srckey.find('.')]

desKey = "bank_customer_db/" + desFolder + "/" + srckey

source= { 'Bucket' : srcBucket,'Key':srckey}

dest ={ 'Bucket' : desBucket,'Key':desKey}

s3.meta.client.copy(source, desBucket, desKey)

return {

'statusCode': 200,

'body': json.dumps('files ingested')

}