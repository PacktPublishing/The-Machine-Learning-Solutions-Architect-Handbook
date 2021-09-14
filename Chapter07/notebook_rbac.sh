cat << EOF > notebook_rbac.yaml 

apiVersion: rbac.istio.io/v1alpha1 

kind: ServiceRoleBinding 

metadata: 

  name: bind-ml-pipeline-nb-admin 

  namespace: kubeflow 

spec: 

  roleRef: 

    kind: ServiceRole 

    name: ml-pipeline-services 

  subjects: 

  - properties: 

      source.principal: cluster.local/ns/admin/sa/default-editor 

EOF 