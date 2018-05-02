export PROJECT_ID=$(gcloud config get-value project -q)

terraform apply -var project=${PROJECT_ID} -auto-approve

gcloud container clusters get-credentials dask-cluster
gcloud config set container/cluster dask-cluster

helm init --wait # note: make sure running helm v2.8.2
helm install . --name dask -f values.yaml --wait
