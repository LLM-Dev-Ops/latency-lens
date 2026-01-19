#!/bin/bash
# LLM-Latency-Lens IAM Setup Script
# --------------------------------------------------
# Creates service account and assigns minimum required permissions.
# Follows principle of least privilege.
#
# Usage: ./setup-iam.sh [PROJECT_ID]

set -euo pipefail

PROJECT_ID="${1:-agentics-dev}"
SERVICE_ACCOUNT_NAME="llm-latency-lens-sa"
SERVICE_ACCOUNT_EMAIL="${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

echo "=============================================="
echo "LLM-Latency-Lens IAM Setup"
echo "=============================================="
echo "Project: ${PROJECT_ID}"
echo "Service Account: ${SERVICE_ACCOUNT_EMAIL}"
echo "=============================================="

# 1. Create service account
echo ""
echo "Step 1: Creating service account..."
gcloud iam service-accounts create ${SERVICE_ACCOUNT_NAME} \
    --project="${PROJECT_ID}" \
    --display-name="LLM-Latency-Lens Service Account" \
    --description="Service account for LLM-Latency-Lens diagnostic layer" \
    2>/dev/null || echo "Service account already exists"

# 2. Assign IAM roles (least privilege)
echo ""
echo "Step 2: Assigning IAM roles..."

# Cloud Run Invoker - allows the service to invoke other Cloud Run services (ruvector-service)
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role="roles/run.invoker" \
    --condition=None \
    --quiet

# Secret Manager Secret Accessor - read secrets
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role="roles/secretmanager.secretAccessor" \
    --condition=None \
    --quiet

# Cloud Trace Agent - send traces to Cloud Trace
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role="roles/cloudtrace.agent" \
    --condition=None \
    --quiet

# Monitoring Metric Writer - write custom metrics
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role="roles/monitoring.metricWriter" \
    --condition=None \
    --quiet

# Logging Writer - write logs
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role="roles/logging.logWriter" \
    --condition=None \
    --quiet

echo ""
echo "Step 3: Creating secrets placeholders..."

# Create secrets (if they don't exist)
for SECRET_NAME in "llm-latency-lens-ruvector-url" "llm-latency-lens-ruvector-key" "llm-latency-lens-telemetry-endpoint"; do
    gcloud secrets describe ${SECRET_NAME} --project="${PROJECT_ID}" 2>/dev/null || \
    gcloud secrets create ${SECRET_NAME} \
        --project="${PROJECT_ID}" \
        --replication-policy="automatic" \
        --labels="app=llm-latency-lens,env=prod"
    echo "Secret ${SECRET_NAME}: OK"
done

echo ""
echo "=============================================="
echo "IAM Setup Complete"
echo "=============================================="
echo ""
echo "Service Account: ${SERVICE_ACCOUNT_EMAIL}"
echo ""
echo "Assigned Roles:"
echo "  - roles/run.invoker"
echo "  - roles/secretmanager.secretAccessor"
echo "  - roles/cloudtrace.agent"
echo "  - roles/monitoring.metricWriter"
echo "  - roles/logging.logWriter"
echo ""
echo "Next Steps:"
echo "  1. Add secret values:"
echo "     echo -n 'YOUR_RUVECTOR_URL' | gcloud secrets versions add llm-latency-lens-ruvector-url --data-file=-"
echo "     echo -n 'YOUR_RUVECTOR_KEY' | gcloud secrets versions add llm-latency-lens-ruvector-key --data-file=-"
echo "     echo -n 'YOUR_TELEMETRY_ENDPOINT' | gcloud secrets versions add llm-latency-lens-telemetry-endpoint --data-file=-"
echo ""
echo "  2. Deploy the service:"
echo "     gcloud builds submit --config=deploy/cloudbuild.yaml"
echo ""

# 4. Verify setup
echo "Step 4: Verifying setup..."
gcloud iam service-accounts describe ${SERVICE_ACCOUNT_EMAIL} --project="${PROJECT_ID}" --format="yaml(email,displayName)"

echo ""
echo "Done!"
