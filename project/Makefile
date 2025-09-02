PY=python
ENV?=.venv
EXP?=churn-exp

init:
	python -m venv .venv && . .venv/bin/activate && pip install -U pip -r requirements.txt

train:
	MLFLOW_EXPERIMENT_NAME=$(EXP) $(PY) src/train.py --config configs/config.yaml

evaluate:
	MLFLOW_EXPERIMENT_NAME=$(EXP) $(PY) src/evaluate.py --config configs/config.yaml

test:
	pytest -q

lint:
	ruff check .
