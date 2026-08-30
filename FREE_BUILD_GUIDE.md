# TopoScout: Free-First Build Guide

This build deliberately uses free local execution first, then the free Gemini Developer API tier and Google Cloud products with free quotas.

## Cost strategy

- Gemini reasoning: Gemini Developer API free tier (`gemini-3.5-flash`) for demo/non-sensitive inputs.
- Image computation: deterministic Python code; raw images do not need to be sent to Gemini.
- Cloud Run: request-based, minimum instances 0, maximum instances 1 for the hackathon demo.
- Cloud Storage: Standard storage in `us-central1`, keep demo files below 5 GB.
- Firestore: use exactly the `(default)` database; do not create a named database.
- Pub/Sub: one small topic and push subscription.
- Secret Manager: one active API-key secret.
- Cloud Build / Artifact Registry: keep only the few images needed for the final demo.
- Logging: default Cloud Run logs only; avoid high-volume debug logs.

## Step 1 — local deterministic smoke test (no account, no model cost)

```bash
python run_local.py demo_inputs/synthetic_science.png --output-dir demo_outputs
```

Expected: a mask plus a JSON evidence report.

## Step 2 — install the official agent tooling

Prerequisites: Python 3.11+, `uv`, and Node.js for Agents CLI skills.

```bash
uvx google-agents-cli setup
uv sync
uv run pytest
```

## Step 3 — free Gemini API key

Create a key in Google AI Studio and save it locally only:

```bash
cp .env.example .env
```

Set:

```dotenv
GEMINI_API_KEY=YOUR_KEY
TOPOSCOUT_MODEL=gemini-3.5-flash
```

Use only synthetic/public/authorized demo data on the free API tier. The free Gemini Developer API tier may use submitted content to improve Google products. TopoScout should send structured scientific metrics to Gemini rather than raw confidential imagery.

## Step 4 — run the ADK agent locally

```bash
agents-cli playground
```

Then instruct TopoScout to process an absolute image path and output directory.

## Step 5 — create a Google Cloud project

Use an eligible new-customer free trial if available. Choose `us-central1` for the MVP so Cloud Storage can qualify for its Always Free regional quota.

Authenticate:

```bash
gcloud auth login
gcloud auth application-default login
```

Set a project:

```bash
gcloud config set project YOUR_PROJECT_ID
```

## Step 6 — create only the free-tier infrastructure we need

```bash
./scripts/prepare_gcp_free.sh YOUR_PROJECT_ID us-central1
```

Then create the one free Firestore database if needed:

```bash
gcloud firestore databases create --location=us-central1
```

If `(default)` already exists, do not run the create command again.

## Step 7 — Cloud Run cost guardrails

When the app is ready to deploy, use these limits:

- region: `us-central1`
- request-based billing
- minimum instances: 0
- maximum instances: 1
- memory: start at 512 MiB
- CPU: start at 1
- concurrency: 4 or higher for light requests
- no GPU

The final deploy command will be generated after the web/API service is added.

## Step 8 — add state and event routing

The MVP cloud loop will be:

`Cloud Storage upload -> Pub/Sub -> Cloud Run -> Firestore state -> analysis artifacts in Cloud Storage`

Firestore stores only small JSON state records; Cloud Storage stores masks/reports.

## Step 9 — budget protection

In Google Cloud Billing, create budget alerts at low thresholds (for example $1, $5, and $10) and leave Cloud Run min instances at zero. A budget alert does not automatically cap spend, so also keep max instances at 1 and delete unneeded container revisions/artifacts after recording the demo.

## Step 10 — final hackathon deployment

Only after local ADK + local science pipeline + cloud state work:

```bash
agents-cli scaffold enhance -d cloud_run
agents-cli deploy --project YOUR_PROJECT_ID --region us-central1
```

Then verify:

```bash
agents-cli deploy --list
agents-cli deploy --status
```

Record Cloud Run, logs, and the `.run` URL for the demo video, then scale down/delete unneeded resources after capture.
