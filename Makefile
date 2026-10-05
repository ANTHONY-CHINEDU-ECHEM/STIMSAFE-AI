.PHONY: install data train report all test frontend api dev docker clean

install:
	pip install -e ".[dev]"

data:
	python -m stimsafe.cli simulate

train:
	python -m stimsafe.cli train

report:
	python -m stimsafe.cli report

all:
	python -m stimsafe.cli all

test:
	pytest -q

frontend:
	cd frontend && npm install && npm run build

api:
	uvicorn stimsafe.api.main:app --host 0.0.0.0 --port 8000

dev:
	cd frontend && npm run dev

docker:
	docker compose up --build

clean:
	rm -rf data/raw frontend/dist frontend/node_modules .pytest_cache
