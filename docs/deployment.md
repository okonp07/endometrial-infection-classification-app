# Deployment Guide

This document describes the zero-cost deployment path for the endometrial classifier.

## 1. Train and export the model

Train the model in your notebook, then export the final inference artifact:

```bash
python scripts/export_model_artifacts.py \
  --model /absolute/path/to/final_model.keras \
  --output-model models/endometrial_classifier.keras \
  --labels infected uninfected
```

## 2. Create the GitHub repository

Create a new public GitHub repository and push this project to it.

Recommended branch:

- `main`

## 3. Create the Hugging Face Docker Space

Create a new public Space on Hugging Face and choose:

- SDK: `Docker`
- Visibility: `Public`

The `README.md` already includes the required front matter:

- `sdk: docker`
- `app_port: 7860`

## 4. Add GitHub secrets

In the GitHub repository settings, add:

- `HF_TOKEN`
- `HF_SPACE_REPO`

Example value for `HF_SPACE_REPO`:

```text
your-username/endometrial-infection-classifier
```

## 5. Push to main

When you push to `main`:

- `ci.yml` runs tests
- `sync-hf-space.yml` mirrors the repo to the Hugging Face Space

## 6. Verify the live Space

After the sync workflow succeeds, verify:

- `/health` returns a valid response
- the Gradio UI loads
- image predictions work

## Notes

- If the model file is large, keep a close eye on repository size and build times.
- Free CPU Spaces are appropriate for demos and lightweight production traffic, not heavy concurrent inference.
- If you later need a shared artifact store, you can add a free DVC remote or move the model to a public Hugging Face model repository.
