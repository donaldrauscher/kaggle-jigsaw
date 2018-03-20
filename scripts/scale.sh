export WORKER_DEPLOY=$(kubectl get deploy --selector=component=worker --output=name)
export PROJECT_ID=$(gcloud config get-value project -q)

if [ $1 = "up" ]; then
  terraform apply -var project=${PROJECT_ID} -var node_count=$2 -auto-approve
  kubectl scale --replicas=$3 ${WORKER_DEPLOY}
else
  kubectl scale --replicas=$3 ${WORKER_DEPLOY}
  terraform apply -var project=${PROJECT_ID} -var node_count=$2 -auto-approve
fi
