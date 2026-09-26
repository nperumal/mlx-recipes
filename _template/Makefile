# make all runs the recipe end to end.
# A recipe that compares two implementations splits train into train-<variant>
# targets -- see recipes/foundations/linear-regression/Makefile.
.PHONY: all data train eval predict test lint clean

DEVICE ?= gpu

all: data train

data:
	python prepare_data.py

train:
	python train.py --device $(DEVICE)

eval:
	python evaluate.py

predict:
	python predict.py

test:
	pytest tests/ -v

lint:
	ruff check . && ruff format --check .

clean:
	rm -rf data checkpoints __pycache__ tests/__pycache__ .pytest_cache
