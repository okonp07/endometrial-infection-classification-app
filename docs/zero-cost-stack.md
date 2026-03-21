# Zero-Cost Stack Notes

This project is designed around a practical zero-cost setup:

- `FastAPI`: free open-source API framework
- `Gradio`: free open-source ML UI framework
- `Docker`: free for personal, educational, open-source, and small business use
- `Hugging Face Docker Spaces`: free CPU hosting for public Spaces
- `GitHub Actions`: free CI/CD within free-tier limits
- `DVC`: free open-source artifact and data versioning

## Why this stack fits this project

- Image classification apps need a simple upload-and-predict UI.
- FastAPI gives a reusable API for future integrations.
- Gradio gives a fast ML-native front end.
- Docker makes the app portable.
- Hugging Face Spaces gives public hosting without paying for a server.
- GitHub Actions covers automated testing and syncing.
- DVC helps you version model artifacts without committing every model binary directly to Git.

## What to avoid if you want to stay free

- Paid cloud VMs
- Paid inference endpoints
- Paid experiment tracking platforms
- GPU upgrades on hosting unless you receive a grant
- Large private repositories with heavy CI usage
