# make all runs both variants end to end.
.PHONY: all train train-binary train-softmax eval predict test lint clean

DEVICE ?= gpu

all: train

train: train-binary train-softmax

train-binary:
	python train_binary.py --device $(DEVICE)

train-softmax:
	python train_softmax.py --device $(DEVICE)

eval:
	python evaluate.py --variant binary
	python evaluate.py --variant softmax

predict:
	python predict.py

test:
	pytest tests/ -v

lint:
	ruff check . && ruff format --check .

clean:
	rm -rf checkpoints __pycache__ tests/__pycache__ .pytest_cache
