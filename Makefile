install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

format:
	black .

lint:
	ruff check .

test:
	pytest -q

run-api:
	uvicorn procurement_risk_detection_ai.app.api.main:app --app-dir src --reload --port 8000

run-ui:
	streamlit run app/ui/streamlit_app.py

precommit-install:
	pre-commit install
