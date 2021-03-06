variable "project" {}

variable "region" {
  default = "us-central1"
}

variable "zone" {
  default = "us-central1-f"
}

variable "worker_node_count" {
  default = "2"
}

provider "google" {
  version = "~> 1.5"
  project = "${var.project}"
  region = "${var.region}"
}

resource "google_container_node_pool" "dask-cluster-np-scheduler" {
  name = "dask-cluster-np-scheduler"
  zone = "${var.zone}"
  cluster = "${google_container_cluster.dask-cluster.name}"
  node_count = "1"

  node_config {
    preemptible  = false
    machine_type = "n1-standard-4"
    oauth_scopes = ["https://www.googleapis.com/auth/devstorage.read_write"]
  }
}

resource "google_container_node_pool" "dask-cluster-np-workers" {
  name = "dask-cluster-np-workers"
  zone = "${var.zone}"
  cluster = "${google_container_cluster.dask-cluster.name}"
  node_count = "${var.worker_node_count}"

  node_config {
    preemptible  = false
    machine_type = "n1-standard-4"
    oauth_scopes = ["https://www.googleapis.com/auth/devstorage.read_write"]
  }
}

resource "google_container_cluster" "dask-cluster" {
  name = "dask-cluster"
  zone = "${var.zone}"

  lifecycle {
    ignore_changes = ["node_pool"]
  }

  node_pool {
    name = "default-pool"
  }
}
