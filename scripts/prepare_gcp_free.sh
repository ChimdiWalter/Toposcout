#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 YOUR_GCP_PROJECT_ID [REGION]"
  exit 1
fi

PROJECT_ID="$1"
REGION="${2:-us-central1}"
BUCKET="${PROJECT_ID}-toposcout-inputs"
TOPIC="toposcout-image-events"

echo "Configuring project: ${PROJECT_ID}"
gcloud config set project "${PROJECT_ID}"
gcloud config set run/region "${REGION}"

echo "Enabling only the APIs used by the MVP..."
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  storage.googleapis.com \
  firestore.googleapis.com \
  pubsub.googleapis.com \
  secretmanager.googleapis.com

echo "Creating Standard Storage bucket in ${REGION} if needed..."
if ! gcloud storage buckets describe "gs://${BUCKET}" >/dev/null 2>&1; then
  gcloud storage buckets create "gs://${BUCKET}" \
    --location="${REGION}" \
    --default-storage-class=STANDARD \
    --uniform-bucket-level-access
else
  echo "Bucket already exists: gs://${BUCKET}"
fi

echo "Creating Pub/Sub topic if needed..."
if ! gcloud pubsub topics describe "${TOPIC}" >/dev/null 2>&1; then
  gcloud pubsub topics create "${TOPIC}"
else
  echo "Topic already exists: ${TOPIC}"
fi

echo
cat <<MSG
Base free-tier infrastructure is ready.

Bucket: gs://${BUCKET}
Topic:  ${TOPIC}
Region: ${REGION}

NEXT (manual, only once):
1. Create the DEFAULT Firestore Native database in ${REGION} if your project does not already have one:
   gcloud firestore databases create --location=${REGION}

2. Add your Gemini Developer API key to Secret Manager after creating the secret:
   printf '%s' 'YOUR_KEY' | gcloud secrets create GEMINI_API_KEY --data-file=-
   (If the secret already exists, add a version instead.)

Do not put the API key in Git.
MSG
