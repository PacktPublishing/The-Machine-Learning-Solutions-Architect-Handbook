cat << EOF > aws_secret_admin.yaml 

apiVersion: v1 

kind: Secret 

metadata: 

  name: aws-secret 

  namespace: admin 

type: Opaque 

data: 

  AWS_ACCESS_KEY_ID: <<your AWS access key>>  

  AWS_SECRET_ACCESS_KEY: <<your AWS secret key>>  

EOF 