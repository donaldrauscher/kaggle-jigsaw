export WORKER_DEPLOY=$(kubectl get deploy --selector=component=worker --output=name)
export WORKER_NODE_POOL=google_container_node_pool.dask-cluster-np-workers
export PROJECT_ID=$(gcloud config get-value project -q)

kubectl scale --replicas=$2 ${WORKER_DEPLOY}
terraform apply -var project=${PROJECT_ID} -var worker_node_count=$1 -target=${WORKER_NODE_POOL} -auto-approve
