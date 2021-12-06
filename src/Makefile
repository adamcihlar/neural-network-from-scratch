all: compile run_nn evaluate

compile:
	g++  -std=c++11 ./src/nn/nn.cpp -o ./src/nn/nn -fpermissive -O3 -fopenmp

run_nn:
	./src/nn/nn

evaluate:
	python3 python_evaluator/evaluate.py actualTestPredictions data/fashion_mnist_test_labels.csv
	python3 python_evaluator/evaluate.py trainPredictions data/fashion_mnist_train_labels.csv
