# Cloud-Native ML Model Deployment with Kubernetes

## Objective
Deploy a TensorFlow machine learning model in a cloud-native environment using Docker, Flask, and Kubernetes.

## Project Overview
This project demonstrates the complete workflow for serving a machine learning model in a production-style environment. A TensorFlow MNIST classifier was trained, containerized using Docker, exposed through a Flask API, and deployed to Kubernetes using Minikube.

## Technologies Used
- Python 3
- TensorFlow
- Flask
- Docker
- Kubernetes
- Minikube
- kubectl
- Linux
- Virtual Environments

## Key Skills Demonstrated
- ML model training and saving
- REST API model serving
- Docker containerization
- Kubernetes deployments and services
- Minikube cluster management
- Kubernetes troubleshooting
- Cloud-native ML deployment workflow

## Kubernetes Components
- Deployment
- Service
- NodePort networking
- Container orchestration

## Troubleshooting & Fixes

### Fixed Minikube Image Visibility Issue
Kubernetes failed with:

ErrImageNeverPull

Reason:
The Docker image was built in the host Docker daemon instead of Minikube's Docker environment.

Fix:
Used:

eval $(minikube docker-env)

Then rebuilt the image directly inside Minikube.

### Updated TensorFlow Container Strategy
Replaced outdated TensorFlow Docker base image approach with a modern Python 3.12 slim container for improved compatibility and package control.

### Updated TensorFlow Model Format
Used modern `.keras` model format instead of legacy `.h5`.

## Files
- `train_model.py` — model training script
- `mnist_model.keras` — trained TensorFlow model
- `app.py` — Flask prediction API
- `Dockerfile` — container definition
- `deployment.yaml` — Kubernetes deployment
- `service.yaml` — Kubernetes service exposure
