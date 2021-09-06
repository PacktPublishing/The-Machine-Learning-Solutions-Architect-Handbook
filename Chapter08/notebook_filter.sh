cat << EOF > notebook_filter.yaml 

apiVersion: networking.istio.io/v1alpha3 

kind: EnvoyFilter 

metadata: 

  name: add-header 

  namespace: admin 

spec: 

  configPatches: 

  - applyTo: VIRTUAL_HOST 

    match: 

      context: SIDECAR_OUTBOUND 

      routeConfiguration: 

        vhost: 

          name: ml-pipeline.kubeflow.svc.cluster.local:8888 

          route: 

            name: default 

    patch: 

      operation: MERGE 

      value: 

        request_headers_to_add: 

        - append: true 

          header: 

            key: kubeflow-userid 

            value: admin@kubeflow.org 

  workloadSelector: 

    labels: 

      notebook-name: david 

EOF 

 