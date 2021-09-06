cat << EOF > ubuntu.yaml 

apiVersion: v1 

kind: Pod 

metadata: 

  name: ubuntu 

  labels: 

    app: ubuntu 

spec: 

  containers: 

  - name: ubuntu 

    image: ubuntu:latest 

    command: ["/bin/sleep", "3650d"] 

    imagePullPolicy: IfNotPresent 

  restartPolicy: Always 

EOF 