curl --silent --location "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp 
 
chmod +x /tmp/eksctl 

sudo mv /tmp/eksctl ./bin/eksctl 

export PATH=$PATH:/home/cloudshell-user/bin

curl -o aws-iam-authenticator https://amazon-eks.s3.us-west-2.amazonaws.com/1.19.6/2021-01-05/bin/linux/amd64/aws-iam-authenticator

chmod +x ./aws-iam-authenticator

sudo mv aws-iam-authenticator ./bin/aws-iam-authenticator 

eksctl create cluster

eksctl get nodegroup --cluster=<cluster name> 

curl -LO "https://storage.googleapis.com/kubernetes-release/release/$(curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt)/bin/linux/amd64/kubectl" 

chmod +x ./kubectl 

sudo mv ./kubectl ./bin/kubectl 