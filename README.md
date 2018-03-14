
Using [Helm chart](https://github.com/kubernetes/charts/tree/master/stable/dask) to install Dask on a Kubernetes cluster.  Use `kubectl cp` to copy files to the Jupyter instance.  Can manually upload/download notebooks from Jupyter UI.

I specifically used Dask to parallelize hyperparameter tuning.  The [`dask-searchcv` package](http://dask-searchcv.readthedocs.io/en/latest/) provides implementations of sklearn’s GridSearchCV and RandomizedSearchCV classes.

``` bash
export PROJECT_ID=$(gcloud config get-value project -q)

terraform apply -var project=${PROJECT_ID}

gcloud container clusters get-credentials dask-cluster
gcloud config set container/cluster dask-cluster

helm init
helm install -f values.yaml stable/dask

export JUPYTER_POD=$(kubectl get pods --selector=component=jupyter --output=name | cut -d/ -f2)
kubectl exec -it $JUPYTER_POD -- mkdir data
kubectl cp data $JUPYTER_POD:data
kubectl cp model.ipynb $JUPYTER_POD:model.ipynb
kubectl cp model_param.yaml $JUPYTER_POD:model_param.yaml
```
