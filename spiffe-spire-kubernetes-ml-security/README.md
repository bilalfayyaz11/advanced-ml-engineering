# Zero-Trust Workload Identity with SPIFFE, SPIRE, and Kubernetes

## Objectives
This lab implements Zero-Trust service identity for a simulated machine learning workload using SPIFFE and SPIRE. It demonstrates how workload identities can be issued as X.509 SVID certificates and used alongside Kubernetes NetworkPolicy to restrict service-to-service communication.

## Tools Used
- Ubuntu 24.04
- Docker
- Minikube
- Kubernetes
- kubectl
- Calico CNI
- SPIFFE
- SPIRE Server
- SPIRE Agent
- OpenSSL
- Nginx
- curlimages/curl

## Key Skills Demonstrated
- Zero-Trust workload identity design
- SPIFFE ID registration and validation
- SPIRE Server and Agent configuration
- X.509 SVID certificate issuance
- Certificate SAN inspection with OpenSSL
- Kubernetes workload deployment
- Kubernetes Service exposure
- NetworkPolicy enforcement using Calico
- Service-to-service access control testing

## Architecture
The lab simulates a protected machine learning service inside Kubernetes. SPIRE issues a SPIFFE identity to the workload, while Kubernetes NetworkPolicy restricts access so only an approved client pod can reach the service.

## Validation Results
- SPIFFE ID issued successfully: `spiffe://example.org/ml-service`
- X.509 certificate contained the expected SPIFFE URI in Subject Alternative Name.
- Allowed client successfully accessed the ML service.
- Blocked client was denied access and timed out.

## Troubleshooting Log
- Replaced outdated SPIRE `v0.12.0` with a modern SPIRE release.
- Fixed missing SPIRE binary path issue after archive extraction.
- Fixed Docker permission issue by adding the user to the docker group.
- Fixed SPIRE Agent configuration syntax error caused by token placement.
- Fixed workload registration selector from `unix:user:ubuntu` to `unix:uid:1000`.
- Fixed incorrect SPIRE parent ID by using the real agent SPIFFE ID.
- Fixed Bash line-continuation error in `spire-server entry create`.
- Fixed NetworkPolicy not enforcing by recreating Minikube with Calico CNI.

