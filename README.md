---
title: Endometrial Infection Classifier
sdk: docker
app_port: 7860
license: mit
---

# Endometrial Infection Classifier

This repository is a zero-cost production scaffold for serving an endometrial image classifier with:

- `FastAPI` for the prediction API
- `Gradio` for the browser UI
- `Docker` for packaging
- `Hugging Face Docker Spaces` for free public hosting
- `GitHub Actions` for free CI/CD
- `DVC` for local artifact versioning without requiring a paid backend

## Zero-cost architecture

This setup stays on the free path by using:

- a public GitHub repository
- a free Hugging Face `CPU Basic` Docker Space
- GitHub Actions on the free tier
- DVC for local artifact management, with no paid remote required by default

What is deliberately not included:

- no paid cloud VM
- no paid database
- no paid model registry
- no paid inference endpoint

## Project structure

```text
endometrial-infection-zero-cost-app/
├── .github/workflows/
│   ├── ci.yml
│   └── sync-hf-space.yml
├── .dvc/
│   ├── .gitignore
│   └── config
├── artifacts/
│   └── class_names.json
├── docs/
│   ├── deployment.md
│   └── zero-cost-stack.md
├── models/
│   └── .gitkeep
├── scripts/
│   ├── export_model_artifacts.py
│   └── train_from_archives.py
├── src/endometrial_app/
│   ├── api.py
│   ├── config.py
│   ├── model.py
│   ├── schemas.py
│   ├── service.py
│   └── ui.py
├── tests/
│   ├── test_api.py
│   └── test_service.py
├── app.py
├── Dockerfile
├── Makefile
├── dvc.yaml
├── requirements-ci.txt
└── requirements.txt
```

## How the app works

1. Your notebook trains the classifier.
2. You export the chosen inference model into the `models/` directory.
3. The app loads the model at startup.
4. FastAPI exposes the prediction API.
5. Gradio provides the web interface on top of the same prediction service.
6. GitHub Actions tests the code and can sync the repo to a Hugging Face Docker Space.

## Model contract

The production app expects:

- a TensorFlow/Keras model at `models/endometrial_classifier.keras`, or a custom path via `MODEL_PATH`
- class names at `artifacts/class_names.json`
- an exported inference model that already contains any required preprocessing logic, or a model that expects resized RGB images shaped as `224 x 224`

The default class order is:

```json
["infected", "uninfected"]
```

If your model outputs a single sigmoid probability, the app treats that probability as the score for `uninfected`, and computes `infected` as `1 - p`.

## Export your notebook model

After training in the notebook, save the final model and register the class names:

```bash
python scripts/export_model_artifacts.py \
  --model /absolute/path/to/final_model.keras \
  --output-model models/endometrial_classifier.keras \
  --labels infected uninfected
```

This script copies the trained model into the app layout and writes `artifacts/class_names.json`.

## Or train directly from the original zip files

If you have not yet exported a model from the notebook, this repo can train one directly from your two archives:

```bash
python scripts/train_from_archives.py \
  --infected-zip "/Users/researchanddevelopment2/Desktop/Endometrial Infection Image Classification Using TensorFlow/infected.zip" \
  --uninfected-zip "/Users/researchanddevelopment2/Desktop/Endometrial Infection Image Classification Using TensorFlow/notinfected.zip"
```

That command will:

- extract the images
- remove exact duplicates
- create train, validation, and test splits
- train a MobileNetV2-based classifier
- save the final model to `models/endometrial_classifier.keras`
- write labels to `artifacts/class_names.json`
- write a training summary to `artifacts/training_summary.json`

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate

python -m pip install -r requirements.txt
make run
```

The app will be available at:

- UI: `http://127.0.0.1:7860/`
- Health: `http://127.0.0.1:7860/health`
- API docs: `http://127.0.0.1:7860/docs`

## API usage

Prediction endpoint:

```bash
curl -X POST "http://127.0.0.1:7860/api/predict" \
  -F "file=@sample_image.jpg"
```

Response shape:

```json
{
  "predicted_label": "infected",
  "predicted_index": 0,
  "confidence": 0.9412,
  "probabilities": {
    "infected": 0.9412,
    "uninfected": 0.0588
  }
}
```

## Docker

Build and run locally:

```bash
docker build -t endometrial-infection-app .
docker run -p 7860:7860 endometrial-infection-app
```

## DVC usage

This repo includes a DVC-friendly layout but does not force a paid remote.

Recommended free workflow:

1. Initialize DVC locally after cloning:

```bash
dvc init
```

2. Track the selected production model:

```bash
dvc add models/endometrial_classifier.keras
git add models/endometrial_classifier.keras.dvc .gitignore
git commit -m "Track production model with DVC"
```

3. If you later want a shared free remote, you can point DVC to a free backend you control.

## GitHub Actions

Included workflows:

- `ci.yml`: installs lightweight CI dependencies and runs tests
- `sync-hf-space.yml`: pushes the repo to a Hugging Face Docker Space when `main` updates

To enable Space sync, set these GitHub secrets:

- `HF_TOKEN`
- `HF_SPACE_REPO` with a value like `your-username/your-space-name`

## Deployment

The fastest zero-cost deployment path is:

1. Create a public GitHub repository.
2. Create a public Hugging Face Docker Space.
3. Add your Hugging Face token and Space id to GitHub secrets.
4. Push to `main`.
5. Let the GitHub Action sync the app to the Space.

Detailed instructions are in [docs/deployment.md](docs/deployment.md).

## Notes on zero-cost limits

- Hugging Face free Spaces sleep after inactivity.
- Free Spaces use CPU-only hardware.
- DVC is free, but any shared storage backend you add later may or may not be free.
- Keep the final inference model compact enough for CPU serving.

## License

This scaffold is released under the MIT License. See [LICENSE](LICENSE).
