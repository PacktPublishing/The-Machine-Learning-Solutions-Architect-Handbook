cat << EOF > sa-cluster-binding.yaml 

apiVersion: rbac.authorization.k8s.io/v1 

kind: ClusterRoleBinding 

metadata: 

  name: default-editor-binding 

  namespace: admin 

subjects: 

- kind: ServiceAccount 

  name: default-editor 

  namespace: admin 

roleRef: 

  kind: ClusterRole  

  name: cluster-admin  

  apiGroup: rbac.authorization.k8s.io 

EOF 