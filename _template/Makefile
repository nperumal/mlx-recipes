# make all reproduces every number in results.md from a clean checkout.
# A recipe that compares two implementations splits train into train-<variant>
# targets -- see recipes/foundations/linear-regression/Makefile.
.PHONY: all data train eval results predict test lint clean

DEVICE ?= gpu

all: train results

data:
	python prepare_data.py

train:
	python train.py --device $(DEVICE)

eval:
	python evaluate.py

results:
	python make_results.py

predict:
	python predict.py

test:
	pytest tests/ -v

lint:
	ruff check . && ruff format --check .

clean:
	rm -rf data checkpoints results __pycache__ tests/__pycache__ .pytest_cache
