# make all reproduces every number in results.md from a clean checkout.
.PHONY: all train train-manual train-nn bench eval results predict test lint clean

DEVICE ?= gpu

all: train results

train: train-manual train-nn

train-manual:
	python train_manual.py --device $(DEVICE)

train-nn:
	python train_nn.py --device $(DEVICE)

# Both devices, for the CPU-vs-GPU rows of results.md.
bench:
	python train_manual.py --device cpu
	python train_nn.py --device cpu
	python train_manual.py --device gpu
	python train_nn.py --device gpu

eval:
	python evaluate.py --variant manual
	python evaluate.py --variant nn

results: train
	python make_results.py

predict:
	python predict.py --variant nn

test:
	pytest tests/ -v

lint:
	ruff check . && ruff format --check .

clean:
	rm -rf checkpoints results __pycache__ tests/__pycache__ .pytest_cache
