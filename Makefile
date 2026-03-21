PYTHON ?= python3
PORT ?= 7860

.PHONY: run api test compile

run:
	PYTHONPATH=src $(PYTHON) -m uvicorn app:app --host 0.0.0.0 --port $(PORT)

api:
	PYTHONPATH=src $(PYTHON) -m uvicorn app:api_app --host 0.0.0.0 --port $(PORT)

test:
	PYTHONPATH=src $(PYTHON) -m pytest

compile:
	PYTHONPATH=src $(PYTHON) -m compileall app.py src tests scripts
