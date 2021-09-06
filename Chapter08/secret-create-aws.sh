cat << EOF > aws_secret.yaml 

apiVersion: v1 

kind: Secret 

metadata: 

  name: aws-secret 

type: Opaque 

data: 

  AWS_ACCESS_KEY_ID: <<your aws access key>> 

  AWS_SECRET_ACCESS_KEY: <<you aws secret access key>> 

EOF 