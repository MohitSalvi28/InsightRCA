#!/bin/bash
set -e  # Exit immediately if a command fails

echo "Deploying llm-log-analyzer..."
kubectl apply -f llm-log-analyzer.yaml

echo "Deploying RBAC for filebeat..."
kubectl apply -f rbac-filebeat.yaml

echo "Deploying filebeat..."
kubectl apply -f filebeat.yaml

echo "Deploying logstash..."
kubectl apply -f logstash.yaml

echo "Deploying error-generator..."
kubectl apply -f error-generator.yaml

echo "All components deployed successfully!"
