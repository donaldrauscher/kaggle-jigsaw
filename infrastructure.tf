variable "project" {}

variable "region" {
  default = "us-central1"
}

variable "zone" {
  default = "us-central1-f"
}

provider "google" {
  version = "~> 1.5"
  project = "${var.project}"
  region = "${var.region}"
}

resource "google_container_cluster" "dask-cluster" {
  name = "dask-cluster"
  zone = "${var.zone}"
  initial_node_count = "6"
  node_config {
    machine_type = "n1-highmem-4"
  }
}
