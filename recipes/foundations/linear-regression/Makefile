# make all runs both variants end to end.
.PHONY: all train train-manual train-nn eval predict test lint clean

DEVICE ?= gpu

all: train

train: train-manual train-nn

train-manual:
	python train_manual.py --device $(DEVICE)

train-nn:
	python train_nn.py --device $(DEVICE)

eval:
	python evaluate.py --variant manual
	python evaluate.py --variant nn

predict:
	python predict.py --variant nn

test:
	pytest tests/ -v

lint:
	ruff check . && ruff format --check .

clean:
	rm -rf checkpoints __pycache__ tests/__pycache__ .pytest_cache
