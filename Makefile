# Makefile cho đồ án Nhận diện tin giả (PhoBERT & SVM)

.PHONY: install test data features train evaluate baselines ablation all quick serve

install:
	python -m pip install -r requirements.txt

test:
	python -m pytest tests -q

data:
	python -m src.preprocessing.split_data data/raw/vn_news_merged.csv

features:
	python -m src.features.extract_phobert

train:
	python -m src.models.train_svm

evaluate:
	python -m src.evaluation.evaluate

baselines:
	python -m src.evaluation.compare_models

ablation:
	python -m experiments.ablation_preprocessing

all:
	python -m experiments.run_all

quick:
	python -m experiments.run_all --quick

serve:
	python -m app.app
