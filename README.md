
I modified this [Helm chart](https://github.com/kubernetes/charts/tree/master/stable/dask) which installs Dask on a Kubernetes cluster, adding support for [`nodeSelectors`](https://kubernetes.io/docs/concepts/configuration/assign-pod-node/).  This allows you to put Jupyter and the Dask scheduler in a separate node pool from the Dask workers; you can scale Dask workers up / down without inadvertently killing your Jupyter instance.

Use `kubectl cp` to copy files to/from the Jupyter instance.

I specifically used Dask to parallelize hyperparameter tuning.  The [`dask-searchcv` package](http://dask-searchcv.readthedocs.io/en/latest/) provides implementations of sklearn’s GridSearchCV and RandomizedSearchCV classes.

Initialize cluster, scale up/down, and destroy:
``` bash
time source scripts/initialize.sh
time source scripts/scale.sh <num_nodes> <num_workers>
time source scripts/destroy.sh
```

Copy models/params to/from Jupyter:
``` bash
export JUPYTER_POD=$(kubectl get pods --selector=component=jupyter -o jsonpath='{.items[0].metadata.name}')
kubectl cp model.ipynb $JUPYTER_POD:model.ipynb
kubectl cp $JUPYTER_POD:model.ipynb model.ipynb
kubectl cp $JUPYTER_POD:model_param.yaml model_param.yaml
```
