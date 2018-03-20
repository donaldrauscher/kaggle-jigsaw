variable "project" {}

variable "region" {
  default = "us-central1"
}

variable "zone" {
  default = "us-central1-f"
}

variable "node_count" {
  default = "1"
}

provider "google" {
  version = "~> 1.5"
  project = "${var.project}"
  region = "${var.region}"
}

resource "google_container_node_pool" "dask-cluster-np" {
  name = "dask-cluster-np"
  zone = "${var.zone}"
  cluster = "${google_container_cluster.dask-cluster.name}"
  node_count = "${var.node_count}"

  node_config {
    preemptible  = false
    machine_type = "n1-standard-4"
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
