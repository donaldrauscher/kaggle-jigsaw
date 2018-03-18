export PROJECT_ID=$(gcloud config get-value project -q)
terraform apply -var project=${PROJECT_ID} -var node_count=$1 -auto-approve

export WORKER_DEPLOY=$(kubectl get deploy --selector=component=worker --output=name)
kubectl scale --replicas=$2 ${WORKER_DEPLOY}
