// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: https://pvs-studio.com

// Dataset
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <utility>

// Matrix
#include <random>
#include <chrono>
#include <numeric>

#include <algorithm>
#include <ctime>

// Hack
#include <omp.h>

int NUM_THREADS = 16;
int CLASSES = 10;


/**
* Class for loading and saving data (predictions)
* and returning desired subset of inputs and/or labels.
* 
* Also contains method to randomly shuffle itself
* and method to return a new Dataset with a subset of its data.
*/
class Dataset
{
public:
	/**
	* Initialize Dataset directly with data
	* Provide both - inputs and labels 
	*/
	Dataset(
		std::vector<std::vector<double>> X_inp = { {} },
		std::vector<double> y_inp = {}
	) : X(X_inp),
		y(y_inp),
		X_cols(X_inp[0].size()),
		y_rows(y_inp.size())
	{
		if (X_inp[0].size() == 0) X_rows = 0; else X_rows = X_inp.size();
	}

	/**
	* Load inputs and save the dims
	* Method expects data in "mnist" format - 784 columns
	* Data minmax normalized by default
	*/
	void load_mnist_data(std::string fpath, bool normalize = true) {
		std::ifstream myfile(fpath);

		if (myfile.is_open()) {

			std::vector<std::vector<double>> res_mat;
			std::string line;

			double element;
			char delimiter;

			int rowcount = 0;
			int normalizer = 1;

			if (normalize) normalizer = 255;

			std::vector<double> int_vec(784);
			while (std::getline(myfile, line))
			{
				std::stringstream sline(line);

				int last_element_index = int_vec.size() - 1;


#pragma omp parallel for num_threads(NUM_THREADS)
				for (size_t i = 0; i < last_element_index; i++)
				{
					sline >> element >> delimiter;
					int_vec[i] = element / normalizer;
				}
				sline >> element;
				int_vec[last_element_index] = element / normalizer;

				res_mat.push_back(int_vec);
				rowcount++;
				if (rowcount % 1000 == 0) std::cout << rowcount << std::endl;
			}

			if (rowcount > 0) {
				X_rows = rowcount;
				X_cols = res_mat.at(rowcount - 1).size();
				X = res_mat;
				std::cout << "Dataset successfully loaded. Shape: (" << X_rows << " x " << X_cols << ")." << std::endl;
			}
			else {
				std::cerr << "No valid data found in the file.\n";
			}
		}
		else {
			std::cerr << "ERROR: Cannot open the file.\n";
		}
	}

	/**
	* Load labels and save the dims
	*/
	void load_labels(std::string fpath) {
		std::ifstream myfile(fpath);

		if (myfile.is_open()) {
			std::vector<double> res_vec;
			std::string line;
			double element;
			int rowcount = 0;

			while (myfile >> element) {
				res_vec.push_back(element);
				rowcount++;
				if (rowcount % 1000 == 0) std::cout << rowcount << std::endl;
			}

			if (rowcount > 0) {
				if (X_rows != 0 && X_rows != rowcount) {
					std::cerr << "Number of labels doesn't match number of samples. No data loaded to y.\n";
					return;
				}
				y_rows = rowcount;
				y = res_vec;
				std::cout << "Labels successfully loaded. Shape: (" << y_rows << " x 1)." << std::endl;
			}
			else {
				std::cerr << "No valid data found in the file.\n";
			}
		}
		else {
			std::cerr << "ERROR: Cannot open the file.\n";
		}

	}

	/**
	* Save predicted labels
	*/
	void save_labels(std::string fpath) {
		std::ofstream myfile(fpath);
		for (size_t i = 0; i < y_rows; i++) {
			myfile << y[i] << "\n";
		}
		myfile.close();
	}

	int get_X_rows() {
		return X_rows;
	}
	int get_X_cols() {
		return X_cols;
	}

	/**
	* Returns subset of X vectors between 'from' and 'to' index
	*/
	std::vector<std::vector<double>> get_subset_X(int from = 0, int to = -1) {
		if (to == -1 || to >= X_rows) to = X_rows;
		if (from < 0) from = 0;
		if (from >= to) return { {} };
		std::vector<std::vector<double>>::const_iterator first = X.begin() + from;
		std::vector<std::vector<double>>::const_iterator last = X.begin() + to;
		std::vector<std::vector<double>> subset(first, last); // ALLOC - dims known on init of Dataloader

		return subset;
	}

	/**
	* Returns subset of y vectors between 'from' and 'to' index
	*/
	std::vector<double> get_subset_y(int from = 0, int to = -1) {
		if (to == -1 || to >= y_rows) to = y_rows;
		if (from < 0) from = 0;
		if (from >= to) return {};
		std::vector<double>::const_iterator first = y.begin() + from;
		std::vector<double>::const_iterator last = y.begin() + to;
		std::vector<double> subset(first, last); // ALLOC - dims known on init of Dataloader

		return subset;
	}

	/**
	* Randomly changes order of input vectors and labels inplace while preserving the correct matching.
	*/
	void shuffle() {
		std::vector<int> indexes; // ALLOC - dims known on init of Dataset
		indexes.reserve(X.size());
		for (int i = 0; i < X.size(); ++i)
			indexes.push_back(i);
		std::random_shuffle(indexes.begin(), indexes.end());

		std::vector<std::vector<double>> X_shuffled(X_rows, std::vector<double>(X_cols)); // ALLOC
		std::vector<double> y_shuffled(X_rows); // ALLOC
		for (int i = 0; i < X_rows; i++) {
			X_shuffled[i] = X[indexes[i]];
			y_shuffled[i] = y[indexes[i]];
		}
		X = X_shuffled;
		y = y_shuffled;
	}

	/**
	* Randomly separates a desired share of data (typically validation dataset) 
	* while reducing the data in itself.
	*/
	Dataset separate_validation_dataset(double validation_share, bool random = true) {
		if ((validation_share <= 0) || (validation_share >= 1)) {
			std::cerr << "ERROR: Share of data for validation must be between 0 and 1.\n";
			return Dataset();
		}
		if (y_rows == 0) {
			std::cerr << "ERROR: Labels must be loaded to create validation dataset.\n";
			return Dataset();
		}
		int first_n_rows = round((1 - validation_share) * X_rows);
		int second_n_rows = X_rows - first_n_rows;
		if (random)	shuffle();

		Dataset validation_dataset(get_subset_X(first_n_rows, X_rows), get_subset_y(first_n_rows, X_rows));
		X = get_subset_X(0, first_n_rows);
		y = get_subset_y(0, first_n_rows);
		if (X[0].size() == 0) X_rows = 0; else X_rows = X.size();
		X_cols = X[0].size();
		y_rows = y.size();

		return validation_dataset;
	}
	
	/**
	* Replaces true labels (if present) with predicted ones
	*/
	void set_y(std::vector<double> y_pred) {
		if (y_pred.size() != X_rows) std::cerr << "ERROR: Length of y does not correspond to length of X." << std::endl;
		y = y_pred;
		y_rows = y_pred.size();
	}

private:
	std::vector<std::vector<double>> X;
	std::vector<double> y;
	int X_cols;
	int X_rows;
	int y_rows;
};


/**
* Parent class for generating random numbers
* 
* Only requires the child classes to be able to sample from their distribution.
*/
class RandomGenerator
{
	virtual double get_sample() = 0;
};

/**
* Provides sampling from normal distribution
*/
class NormalRandomGenerator : public RandomGenerator
{
public:
	NormalRandomGenerator(double mean = 0, double std = 1) : mean(mean), std(std) {
		std::random_device rd;

		distribution = std::normal_distribution<double>(mean, std);
		gen = std::mt19937(rd());
	}

	double get_sample() {
		return distribution(gen);
	}
private:
	std::normal_distribution<double> distribution;
	std::mt19937 gen;
	double mean, std;
};

/**
* Provides sampling from Bernoulli distribution
*/
class BernoulliGenerator : public RandomGenerator
{
public:
	BernoulliGenerator(double p = 0.5) : p(p) {
		std::random_device rd;
		gen = std::mt19937(rd());
		distribution = std::bernoulli_distribution(p);
	}

	double get_sample() {
		return distribution(gen);
	}

private:
	std::bernoulli_distribution distribution;
	std::mt19937 gen;
	double p;
};


/**
* Stores matrix type data and basic metadat (nrows, ncols)
* Implements all needed matrix operations.
*/
class Matrix
{
public:
	Matrix() : values(0, std::vector<double>(0)), shape({0,0}) {}

	/**
	* Initialize with given shape and default value 
	*/
	Matrix(unsigned int nrow, unsigned int ncol, double default_value = 0) : 
		values(nrow, std::vector<double>(ncol, default_value)), 
		cachedValues(nrow, std::vector<double>(ncol, default_value)), 
		shape({ nrow, ncol }) {}

	/**
	* Initialize with given data 
	*/
	Matrix(std::vector<std::vector<double>> data) : values(data), cachedValues(data), shape({data.size(), data[0].size()}) {}

	/**
	* Initialize with given shape 
	*/
	Matrix(unsigned int nrow, unsigned int ncol, double mean, double std) : 
		values(nrow, std::vector<double>(ncol)),
		cachedValues(nrow, std::vector<double>(ncol)), 
		shape({ nrow, ncol }) {
		NormalRandomGenerator randgen(mean, std);
		for (size_t i = 0; i < shape[0]; i++) {
			for (size_t j = 0; j < shape[1]; j++) {
				values[i][j] = randgen.get_sample();
			}
		}
	}

	void print_matrix() {
		if (!values.empty() && !values.at(0).empty()) {
			for (int i = 0; i < shape[0]; i++) {
				std::cout << "[ ";
				for (int j = 0; j < shape[1]; j++) {
					std::cout << values[i][j] << " ";
				}
				std::cout << "]" << std::endl;
			}
		}
		else {
			std::cout << "Empty matrix.\n";
		}
	}

	void print_shape() {
		std::cout << "(";
		for (size_t i = 0; i < shape.size(); i++) {
			std::cout << " " << shape.at(i);
		}
		std::cout << " )" << std::endl;
	}

	std::vector<unsigned int> get_shape() {
		return shape;
	}
	std::vector<std::vector<double>> get_values() {
		return values;
	}

	/** 
	* Reset values and update the shape if changing the whole matrix
	*/
	void set_values(std::vector<std::vector<double>> in_values) {
		values = in_values;
		shape = { in_values.size(), in_values[0].size() };
	}
	void set_value(int nrow, int ncol, double value) {
		values[nrow][ncol] = value;
	}
	void set_row(int nrow, std::vector<double> in_values) {
		values[nrow] = in_values;
	}

	/** 
	* Matrix multiplication
	*/
	Matrix dot(Matrix* second) {
		int ncols1 = get_shape()[1];
		int nrows2 = second->get_shape()[0];
		if (ncols1 == nrows2) {
			int nrows1 = get_shape()[0];
			int ncols2 = second->get_shape()[1];
			Matrix result(nrows1, ncols2);

#pragma omp parallel for num_threads(NUM_THREADS)
			for (int i = 0; i < nrows1; i++) {
				for (int k = 0; k < nrows2; k++) {
					for (int j = 0; j < ncols2; j++) {
						result.values[i][j] += values[i][k] * second->values[k][j];
					}
				}
			}
			return result;
		}
		else {
			std::cerr << "Nonconformable dimensions: mat1 is (x " << ncols1 << " ) but mat2 is (" << nrows2 << " y).\n";
			return Matrix();
		}
	}

	/**
	* Sum two matrices with same shape
	*/
	Matrix sum(Matrix* second) {
		if (get_shape() == second->get_shape()) {
			int nrow = get_shape()[0];
			int ncol = get_shape()[1];
#pragma omp parallel for num_threads(NUM_THREADS)
			for (int i = 0; i < nrow; i++) {
				for (int j = 0; j < ncol; j++) {
					cachedValues[i][j] = values[i][j] + second->values[i][j];
				}
			}
			return Matrix(cachedValues);
		}
		else {
			std::cerr << "Nonconformable dimensions, both dimensions must match.\n";
			return Matrix();
		}
	}
	Matrix sum(Matrix second) {
		if (get_shape() == second.get_shape()) {
			int nrow = get_shape()[0];
			int ncol = get_shape()[1];
#pragma omp parallel for num_threads(NUM_THREADS)
			for (int i = 0; i < nrow; i++) {
				for (int j = 0; j < ncol; j++) {
					cachedValues[i][j] = values[i][j] + second.values[i][j];
				}
			}
			return Matrix(cachedValues);
		}
		else {
			std::cerr << "Nonconformable dimensions, both dimensions must match.\n";
			return Matrix();
		}
	}

	/**
	* Elementwise multiplication by another matrix (with same shape)
	* or by vector with corresponding lenght (equal to ncols).
	*/
	Matrix multiply(Matrix* multiplier) {
		if (multiplier->get_shape()[0] == 1 && shape[1] == multiplier->get_shape()[1]) {
#pragma omp parallel for num_threads(NUM_THREADS)
			for (size_t i = 0; i < shape[0]; i++) {
				for (size_t j = 0; j < shape[1]; j++) {
					cachedValues[i][j] = values[i][j] * multiplier->get_values()[0][j];
				}
			}
		}
		else if (multiplier->get_shape() == shape) {
#pragma omp parallel for num_threads(NUM_THREADS)
			for (size_t i = 0; i < shape[0]; i++) {
				for (size_t j = 0; j < shape[1]; j++) {
					cachedValues[i][j] = values[i][j] * multiplier->get_values()[i][j];
				}
			}
		}
		else {
			std::cerr << "Nonconformable dimensions, multiplier must be either vector of shape (1, matrix.ncol) or matrix with matching dimensions.\n";
		}
		return Matrix(cachedValues);
	}

	Matrix multiply(Matrix multiplier) {
		if (multiplier.get_shape()[0] == 1 && shape[1] == multiplier.get_shape()[1]) {
#pragma omp parallel for num_threads(NUM_THREADS)
			for (size_t i = 0; i < shape[0]; i++) {
				for (size_t j = 0; j < shape[1]; j++) {
					cachedValues[i][j] = values[i][j] * multiplier.get_values()[0][j];
				}
			}
		}
		else if (multiplier.get_shape() == shape) {
#pragma omp parallel for num_threads(NUM_THREADS)
			for (size_t i = 0; i < shape[0]; i++) {
				for (size_t j = 0; j < shape[1]; j++) {
					cachedValues[i][j] = values[i][j] * multiplier.get_values()[i][j];
				}
			}
		}
		else {
			std::cerr << "Nonconformable dimensions, multiplier must be either vector of shape (1, matrix.ncol) or matrix with matching dimensions.\n";
		}
		return Matrix(cachedValues);
	}

	/**
	* Elementwise multiplication by single value.
	*/
	Matrix scalar_mul(double multiplier) {
#pragma omp parallel for num_threads(NUM_THREADS)
		for (size_t i = 0; i < shape[0]; i++) {
			for (size_t j = 0; j < shape[1]; j++) {
				cachedValues[i][j] = values[i][j] * multiplier;
			}
		}
		return Matrix(cachedValues);
	}

	/**
	* Returns new transposed matrix.
	*/
	Matrix get_transposed() {
		std::vector<std::vector<double>> transposed(shape[1], std::vector<double>(shape[0]));
#pragma omp parallel for num_threads(NUM_THREADS)
		for (size_t i = 0; i < transposed.size(); i++) {
			for (size_t j = 0; j < transposed[i].size(); j++) {
				transposed[i][j] = values[j][i];
			}
		}
		Matrix result(transposed);
		return result;
	}

	/**
	* Sums columns and returns it as new Matrix.
	*/
	Matrix col_sums() {
		std::vector<double> sums(shape[1]);
#pragma omp parallel for num_threads(NUM_THREADS)
		for (size_t i = 0; i < shape[0]; i++)
			for (size_t j = 0; j < shape[1]; j++)
				sums[j] += values[i][j];
		return Matrix({ sums });
	}

private:
	std::vector<unsigned int> shape;
	std::vector<std::vector<double>> values;
	std::vector<std::vector<double>> cachedValues;

};


/**
* Structure for passing input features in batches together with labels to the NeuralNetwork.
*/
struct Batch {
	Matrix* X, * Y;
};


/**
* Class responsible for taking subset of data from the underlying dataset
* and feeding them to the NeuralNetwork.
* Rembers what rows it has already given.
*/
class DataLoader
{
public:
	/**
	* Initialize DataLoader with underlying Dataset and required batch size.
	*/
	DataLoader(Dataset* dataset, int batch_size = 32) :
		batchSize(batch_size),
		sourceDataset(dataset),
		rowsTotal(dataset->get_X_rows()),
		rowsGiven(0),
		exhausted((sourceDataset->get_X_rows() <= rowsGiven)),
		X_sample(Matrix(batch_size, dataset->get_X_cols())),
		y_sample(Matrix(batch_size, 1))
	{}

	/**
	* Returns one Batch and updates its state.
	*/
	Batch get_sample() {
		if (exhausted) return {};
		rowsGiven += batchSize;
		exhausted = (sourceDataset->get_X_rows() <= rowsGiven);

		X_sample.set_values(sourceDataset->get_subset_X(rowsGiven - batchSize, rowsGiven));
		y_sample.set_values((one_hot_encode(sourceDataset->get_subset_y(rowsGiven - batchSize, rowsGiven))));

		sample = { &X_sample, &y_sample };
		return sample;
	}

	/**
	* Returns one sample and updates its state.
	*/
	Batch get_one_sample() {
		if (exhausted) return {};
		rowsGiven += 1;
		exhausted = (sourceDataset->get_X_rows() <= rowsGiven);

		Matrix X_mat(sourceDataset->get_subset_X(rowsGiven - 1, rowsGiven)); // ALLOC - dims known on init of DataLoader
		Matrix Y_mat(one_hot_encode(sourceDataset->get_subset_y(rowsGiven - 1, rowsGiven))); // ALLOC - dims known on init of DataLoader

		Batch sample = { &X_mat, &Y_mat };
		return sample;
	}

	/**
	* Returns whole dataset in Batch format.
	*/
	Batch get_all_samples() {
		Matrix X_mat(sourceDataset->get_subset_X()); // ALLOC - dims known on init of DataLoader
		Matrix Y_mat(one_hot_encode(sourceDataset->get_subset_y())); // ALLOC - dims known on init of DataLoader

		Batch sample = { &X_mat, &Y_mat };
		return sample;
	}

	/**
	* Creates one hot encoded vectors from labels
	* to be able to feed them to NeuralNetwork.
	*/
	std::vector<std::vector<double>> one_hot_encode(std::vector<double> labels) {
		std::vector<std::vector<double>> one_hot_labels(labels.size(), std::vector<double>(CLASSES));
		for (size_t i = 0; i < labels.size(); i++) {
			one_hot_labels[i][labels[i]] = 1;
		}
		return one_hot_labels;
	}

	/**
	* Creates labels from one hot encoded predictions returned from NeuralNetwork
	* to be able to assign them back to Dataset.
	*/
	std::vector<double> one_hot_decode(std::vector<std::vector<double>> one_hot_labels) {
		std::vector<double> labels(one_hot_labels.size()); // ALLOC - dims depend on n_rows of dataset
		for (size_t i = 0; i < one_hot_labels.size(); i++) {
			std::vector<double>::iterator result = std::max_element(one_hot_labels[i].begin(), one_hot_labels[i].end());
			labels[i] = std::distance(one_hot_labels[i].begin(), result);
		}
		return labels;
	}

	/**
	* Assigns the predicted labels back to the underlying Dataset.
	*/
	void assign_predicted_labels(std::vector<double> y_pred) {
		sourceDataset->set_y(y_pred);
	}

	/**
	* Checks if all rows from Dataset were given.
	*/
	bool is_exhausted() {
		return exhausted;
	}

	/**
	* Make the instance forget how many rows it gave.
	*/
	void reset() {
		rowsGiven = 0;
		exhausted = (sourceDataset->get_X_rows() <= rowsGiven);
	}

	/**
	* Reshuffles the underlying dataset.
	*/
	void shuffle_dataset() {
		sourceDataset->shuffle();
	}

	int get_n_rows() {
		return rowsTotal;
	}
	int get_batch_size() {
		return batchSize;
	}

private:
	Dataset* sourceDataset;
	int batchSize;
	int rowsGiven;
	int rowsTotal;
	bool exhausted;
	Matrix X_sample;
	Matrix y_sample;
	Batch sample;
};


/**
* Class storing two Matrices - weights and bias
* Impements forward pass.
* Can reset its weights.
* Two regularization methods are defined on the Layer's level - dropout and weight decay.
*/
class Layer {
public:
	/**
	* Initialization of the Layer with He random initialization of weights
	* @params:
	* n of inputs to the layer
	* n of outputs from the layer
	* percent of dropout
	* weight_decay
	*/
	Layer(int n_inputs, int n_outputs, double dropout, double weight_decay) :
		w0(Matrix(1, n_outputs, 0, sqrt(2.0 / n_inputs))),
		weights(Matrix(n_inputs, n_outputs, 0, sqrt(2.0 / n_inputs))),
		weightsShape(weights.get_shape()),
		w0Shape(w0.get_shape()),
		dropout(dropout),
		weightDecay(weight_decay),
		dropoutMask(Matrix(1, n_outputs)) {}

	/**
	* Matrix multiplication of the input with weights and sum of the result with bias.
	* For training the dropout is switched on, for inference no dropout.
	*/
	Matrix pass(Matrix* inputs, bool dropout_switch_on) {
		int inputs_nrow = inputs->get_shape()[0];
		if (inputs_nrow != w0ext.get_shape()[0]) {
			w0ext = Matrix(inputs_nrow, w0Shape[1]);
		}

		for (int i = 0; i < inputs_nrow; i++)
			w0ext.set_row(i, w0.get_values()[0]);

		if (dropout != 0.0 && dropout_switch_on) {
			BernoulliGenerator b_rand(1 - dropout);
			for (size_t i = 0; i < weightsShape[1]; i++) {
				dropoutMask.set_value(0, i, b_rand.get_sample() / (1 - dropout));
			}
			return Matrix(inputs->dot(&weights).sum(&w0ext).multiply(&dropoutMask)); // ALLOC - dims known when calling nn.train / nn.predict
		}
		else {
			return Matrix(inputs->dot(&weights).sum(&w0ext)); // ALLOC - dims known when calling nn.train / nn.predict
		}
	}

	Matrix get_weights() { return weights; }

	/**
	* Reset its weights with updated ones and apply the weight decay if specified.
	*/
	void set_weights(Matrix new_weights) {
		if (weightDecay == 0.0) weights.set_values(new_weights.get_values());
		else weights.set_values(new_weights.scalar_mul(1 - weightDecay).get_values());
	}

	std::vector<unsigned int> get_weights_shape() { return weightsShape; }

	Matrix get_bias() { return w0; }

	/**
	* Reset its biases with updated ones and apply the weight decay if specified.
	*/
	void set_bias(Matrix new_weights) {
		if (weightDecay == 0.0) w0.set_values(new_weights.get_values());
		else w0.set_values(new_weights.scalar_mul(1 - weightDecay).get_values());
	}

	std::vector<unsigned int> get_bias_shape() { return w0Shape; }

	/**
	* Creates space for extended Matrix of biases
	* with numbers of rows determined by the batch size.
	*/
	void get_ready_for_pass(DataLoader* dataloader) {
		w0ext = Matrix(dataloader->get_batch_size(), w0Shape[1]);
	}

private:
	Matrix weights;
	Matrix w0;
	Matrix w0ext;
	std::vector<unsigned int> weightsShape;
	std::vector<unsigned int> w0Shape;
	double dropout;
	Matrix dropoutMask;
	double weightDecay;
};


/**
* Parent class for all activation functions.
* Requires implementation of evaluation and derivation of a layer
* and extends these methods to be able to evaluate/derive whole batch.
*/
class ActivationFunction {
public:
	/**
	* Applies a specific activiation function to whole batch.
	*/
	Matrix evaluate_batch(Matrix* batch_inner_potentials) {
		Matrix result(batch_inner_potentials->get_shape()[0], batch_inner_potentials->get_shape()[1]); // ALLOC - I know the dimensions when calling nn.train
		for (int i = 0; i < batch_inner_potentials->get_shape()[0]; i++) {
			result.set_row(i, evaluate_layer(batch_inner_potentials->get_values()[i]));
		}
		return result;
	}

	/**
	* Applies derivation of a specific activiation function to whole batch.
	*/
	Matrix derive_batch(Matrix* batch_neuron_outputs, Matrix* batch_y_true = &Matrix()) {
		Matrix result(batch_neuron_outputs->get_shape()[0], batch_neuron_outputs->get_shape()[1]); // ALLOC - I know the dimensions when calling nn.train
		if (batch_y_true->get_shape()[1] > 0) {
			for (size_t i = 0; i < batch_neuron_outputs->get_shape()[0]; i++) {
				result.set_row(i, derive_layer(batch_neuron_outputs->get_values()[i], batch_y_true->get_values()[i]));
			}
		}
		else {
			for (size_t i = 0; i < batch_neuron_outputs->get_shape()[0]; i++) {
				result.set_row(i, derive_layer(batch_neuron_outputs->get_values()[i]));
			}
		}
		return result;
	}

private:
	virtual std::vector<double> evaluate_layer(std::vector<double> inner_potentials) = 0;

	virtual std::vector<double> derive_layer(std::vector<double> neuron_outputs, std::vector<double> y_true = {}) = 0;

};

class ReLU :public ActivationFunction {
private:
	/**
	* Apply ReLU to the inner potentials of the layer.
	*/
	std::vector<double> evaluate_layer(std::vector<double> inner_potentials) {
		std::vector<double> result(inner_potentials.size(), 0); // ALLOC - I know the dimensions on init of NN
		for (size_t i = 0; i < inner_potentials.size(); i++) {
			if (inner_potentials[i] > 0) result[i] = inner_potentials[i];
		}
		return result;
	}

	/**
	* Apply derivative of ReLU to the layer.
	*/
	std::vector<double> derive_layer(std::vector<double> neuron_outputs, std::vector<double> y_true = {}) {
		std::vector<double> result(neuron_outputs.size(), 0); // ALLOC - I know the dimensions on init of NN
		for (size_t i = 0; i < neuron_outputs.size(); i++) {
			if (neuron_outputs[i] > 0) result[i] = 1;
		}
		return result;
	}

};

class Softmax :public ActivationFunction {
private:
	/**
	* Apply Softmax to the inner potentials of the layer.
	*/
	std::vector<double> evaluate_layer(std::vector<double> inner_potentials) {
		std::vector<double> result(inner_potentials.size(), 0); // ALLOC - I know the dimensions on init of NN
		double denominator = 0.0;
		double inner_max = inner_potentials[0];
		for (size_t i = 1; i < inner_potentials.size(); i++)
			if (inner_potentials[i] > inner_max) inner_max = inner_potentials[i];

		for (size_t i = 0; i < result.size(); i++)
			denominator += exp(inner_potentials[i] - inner_max);

		for (size_t i = 0; i < inner_potentials.size(); i++)
			result[i] = exp(inner_potentials[i] - inner_max) / denominator;

		return result;
	}

	/**
	* Apply derivative of Softmax with (!) Cross Entropy Loss function (!) to the layer.
	*/
	std::vector<double> derive_layer(std::vector<double> neuron_outputs, std::vector<double> y_true = {}) {
		std::vector<double> result(neuron_outputs.size()); // ALLOC - I know the dimensions on init of NN
		for (size_t i = 0; i < result.size(); i++) {
			result[i] = neuron_outputs[i] - y_true[i];
		}
		return result;
	}
};

/**
* Parent class for loss functions.
* Requires implementation loss calculation for singe sample
* and extends it to loss calculation for whole batch.
*/
class LossFunction {
public:
	virtual double calculate_loss(std::vector<double> y_true, std::vector<double> y_pred) = 0;

	/**
	* Applies the loss function to whole batch.
	*/
	double calculate_mean_batch_loss(Matrix* Y_true, Matrix* Y_pred) {
		batch_loss = 0.0;
		for (int i = 0; i < Y_true->get_shape()[0]; i++) {
			batch_loss += calculate_loss(Y_true->get_values()[i], Y_pred->get_values()[i]);
		}
		return batch_loss / Y_true->get_shape()[0];
	}

private:
	double batch_loss;
};

class CrossEntropyLoss :public LossFunction {
public:
	/**
	* Calculates loss on single sample.
	*/
	double calculate_loss(std::vector<double> y_true, std::vector<double> y_pred) {
		loss = 0.0;
		for (size_t i = 0; i < y_true.size(); i++) {
			loss += y_true[i] * log(y_pred[i]);
		}
		return -loss;
	}

private:
	double loss;

};


/**
* Parent class for all optimizers.
* Requires calculation of bias and weights update from given gradients.
*/
class Optimizer {
public:
	virtual std::vector<Matrix> calculate_bias_update(std::vector<Matrix> bias_grad) = 0;
	virtual std::vector<Matrix> calculate_weights_update(std::vector<Matrix> weights_grad) = 0;
	virtual void get_ready_for_optimization(std::vector<Layer*> nn_layers) = 0;
private:
	double learningRate;
};

class SGD :public Optimizer {
public:
	/**
	* Initialize SGD with learning rate, momentum and option of Nesterov momentum.
	*/
	SGD(double learning_rate, double momentum_alpha = 0.0, bool nesterov = false) : learningRate(learning_rate), momentumAlpha(momentum_alpha), nesterovMomentum(nesterov) {}

	/**
	* Calclutes the bias update from the given gradient.
	*/
	std::vector<Matrix> calculate_bias_update(std::vector<Matrix> bias_grad) {
		if (momentumAlpha != 0.0 && nesterovMomentum) {
			for (size_t i = 0; i < currentBiasUpdate.size(); i++) {
				previousBiasUpdate[i] = bias_grad[i].scalar_mul(-learningRate).sum(previousBiasUpdate[i].scalar_mul(momentumAlpha));
				currentBiasUpdate[i] = bias_grad[i].scalar_mul(-learningRate).sum(previousBiasUpdate[i].scalar_mul(momentumAlpha));
			}
			return currentBiasUpdate;
		}
		else if (momentumAlpha == 0.0) {
			for (size_t i = 0; i < currentBiasUpdate.size(); i++) {
				currentBiasUpdate[i] = bias_grad[i].scalar_mul(-learningRate);
			}
			return currentBiasUpdate;
		}
		for (size_t i = 0; i < currentBiasUpdate.size(); i++) {
			currentBiasUpdate[i] = bias_grad[i].scalar_mul(-learningRate).sum(previousBiasUpdate[i].scalar_mul(momentumAlpha));
		}
		previousBiasUpdate = currentBiasUpdate;
		return currentBiasUpdate;
	}

	/**
	* Calclutes the weights update from the given gradient.
	*/
	std::vector<Matrix> calculate_weights_update(std::vector<Matrix> weights_grad) {
		if (momentumAlpha != 0.0 && nesterovMomentum) {
			for (size_t i = 0; i < currentWeightsUpdate.size(); i++) {
				previousWeightsUpdate[i] = weights_grad[i].scalar_mul(-learningRate).sum(previousWeightsUpdate[i].scalar_mul(momentumAlpha));
				currentWeightsUpdate[i] = weights_grad[i].scalar_mul(-learningRate).sum(previousWeightsUpdate[i].scalar_mul(momentumAlpha));
			}
			return currentWeightsUpdate;
		}
		else if (momentumAlpha == 0.0) {
			for (size_t i = 0; i < currentWeightsUpdate.size(); i++) {
				currentWeightsUpdate[i] = weights_grad[i].scalar_mul(-learningRate);
			}
			return currentWeightsUpdate;
		}
		for (size_t i = 0; i < currentWeightsUpdate.size(); i++) {
			currentWeightsUpdate[i] = weights_grad[i].scalar_mul(-learningRate).sum(previousWeightsUpdate[i].scalar_mul(momentumAlpha));
		}
		previousWeightsUpdate = currentWeightsUpdate;
		return currentWeightsUpdate;
	}

	void get_ready_for_optimization(std::vector<Layer*> nn_layers) {
		set_weights_update_dimensions(nn_layers);
	}

	/**
	* Prepares the optimizer for training by setting the initial updates.
	*/
	void set_weights_update_dimensions(std::vector<Layer*> nn_layers) {
		for (size_t i = 0; i < nn_layers.size(); i++) {
			currentBiasUpdate.push_back(Matrix(nn_layers[i]->get_bias_shape()[0], nn_layers[i]->get_bias_shape()[1]));
			previousBiasUpdate.push_back(Matrix(nn_layers[i]->get_bias_shape()[0], nn_layers[i]->get_bias_shape()[1]));
			currentWeightsUpdate.push_back(Matrix(nn_layers[i]->get_weights_shape()[0], nn_layers[i]->get_weights_shape()[1]));
			previousWeightsUpdate.push_back(Matrix(nn_layers[i]->get_weights_shape()[0], nn_layers[i]->get_weights_shape()[1]));
		}
	}

private:
	double learningRate;
	double momentumAlpha;
	bool nesterovMomentum;
	std::vector<Matrix> currentBiasUpdate;
	std::vector<Matrix> currentWeightsUpdate;
	std::vector<Matrix> previousBiasUpdate;
	std::vector<Matrix> previousWeightsUpdate;
};


/**
* Parent class for metrics requiring method for calculation the metric for whole batch.
*/
class Metric {
public:
	virtual double calculate_metric_for_batch(Matrix* Y_true, Matrix* Y_pred) = 0;
};

class Accuracy :public Metric {
public:
	double calculate_metric_for_batch(Matrix* Y_true, Matrix* Y_pred) {
		count_true_in_batch = 0.0;
		for (size_t i = 0; i < Y_true->get_shape()[0]; i++) {
			count_true_in_batch += (Y_true->get_values()[i] == Y_pred->get_values()[i]);
		}
		return count_true_in_batch / Y_true->get_shape()[0];
	}

private:
	double calculate_metric(std::vector<double> y_true, std::vector<double> y_pred) {
		if (y_true == y_pred) return 1;
		return 0;
	}
	double count_true_in_batch;

};


class NeuralNetwork {
public:
	/**
	* Initialize NeuralNetwork if lengths of layers and activation functions are matching.
	*/
	NeuralNetwork(
		std::vector<Layer*> layers,
		std::vector<ActivationFunction*> activation_functions,
		Optimizer* optimizer,
		LossFunction* loss_function,
		Metric* metric
	) {
		if (layers.size() == activation_functions.size()) {
			this->layers = layers;
			activationFunctions = activation_functions;
			innerPotentials = std::vector<Matrix>(layers.size());
			neuronsOutputs = std::vector<Matrix>(layers.size() + 1);
			deltas = std::vector<Matrix>(layers.size());
			biasGradients = std::vector<Matrix>(layers.size());
			weightsGradients = std::vector<Matrix>(layers.size());
			countLayers = layers.size();
			this->optimizer = optimizer;
			lossFunction = loss_function;
			this->metric = metric;
			epochsDone = 0;
			oneHotPredictions = Matrix();

			this->optimizer->get_ready_for_optimization(this->layers);
		}
		else {
			std::cerr << "Number of layers must correspond to number of activation functions.\n";
		}
	}


	/**
	* Changes network's state (weights) by training for n epochs on train dataset
	* and validates the results after every epoch.
	* Displays basic info about the training.
	* Reshuffles train dataset at the beginning of every epoch if desired.
	*/
	void train(int epochs, DataLoader* train_dataset, DataLoader* validation_dataset, bool shuffle_train = true) {
		for (int i = 0; i < epochs; i++) {
			std::cout << "Epoch " << epochsDone + 1 << ":" << std::endl;
			train_epoch(train_dataset, shuffle_train);
			epochsDone++;
			display_train_metrics_from_last_epoch();
			validate_epoch(validation_dataset);
			display_validation_metrics_from_last_epoch();
			std::cout << std::endl;
		}
	}

	/**
	* Predict labels for given dataset.
	*
	* @param DataLoader pointing on dataset (typically test).
	* @return onehot encoded predictions.
	*/
	void predict(DataLoader* prediction_dataloader) {
		int n_predictions = prediction_dataloader->get_n_rows();
		int n_classes = layers[countLayers - 1]->get_weights().get_shape()[1];
		std::vector<std::vector<double>> one_hot_predictions(n_predictions, std::vector<double>(n_classes));

		// would be nice to pass the whole test dataset at once 
		// and get rid off this while
		prediction_dataloader->reset();

		for (size_t i = 0; i < n_predictions; i++) {
			Batch batch = prediction_dataloader->get_one_sample();
			forward_pass(batch, false);
			one_hot_predictions[i] = batch_output_probabilities_to_predictions()->get_values()[0];
		}

		std::vector<double> predictions = prediction_dataloader->one_hot_decode(one_hot_predictions);
		prediction_dataloader->assign_predicted_labels(predictions);

		//Batch batch = prediction_dataloader->get_all_samples();
		//forward_pass(batch);
		//one_hot_predictions = batch_output_probabilities_to_predictions().get_values();
		//std::vector<double> predictions = prediction_dataloader->one_hot_decode(one_hot_predictions);
		//prediction_dataloader->assign_predicted_labels(predictions);
	}



private:
	std::vector<Layer*> layers;
	std::vector<ActivationFunction*> activationFunctions;
	Optimizer* optimizer;
	LossFunction* lossFunction;
	Metric* metric;
	std::vector<Matrix> innerPotentials;
	std::vector<Matrix> neuronsOutputs;
	Matrix oneHotPredictions;
	std::vector<Matrix> deltas;
	std::vector<Matrix> biasGradients;
	std::vector<Matrix> weightsGradients;
	int countLayers;
	int epochsDone;
	std::vector<double> trainLossInEpoch;
	std::vector<double> validationLossInEpoch;
	std::vector<double> trainMetricInEpoch;
	std::vector<double> validationMetricInEpoch;

	/**
	* Iterates over all layers and activation functions to get predictions from input features.
	*/
	void forward_pass(Batch batch, bool dropout_switch_on) {
		neuronsOutputs[0] = (*batch.X);
		for (size_t i = 0; i < countLayers; i++)
		{
			innerPotentials[i].set_values(layers[i]->pass(&neuronsOutputs[i], dropout_switch_on).get_values());
			neuronsOutputs[i + 1].set_values(activationFunctions[i]->evaluate_batch(&innerPotentials[i]).get_values());
		}
	}

	/**
	* Iterates over all layers and activation functions backwards to get gradients.
	*/
	void backward_pass(Batch batch) {
		deltas[countLayers - 1].set_values(activationFunctions[countLayers - 1]->derive_batch(&neuronsOutputs[countLayers], batch.Y).get_values());

		weightsGradients[countLayers - 1] = neuronsOutputs[countLayers - 1].get_transposed().dot(&deltas[countLayers - 1]);
		biasGradients[countLayers - 1] = deltas[countLayers - 1].col_sums();

		for (int i = countLayers - 2; i >= 0; i--) {
			deltas[i] = activationFunctions[i]->derive_batch(&innerPotentials[i]).multiply(deltas[i + 1].dot(&layers[i + 1]->get_weights().get_transposed()));
			weightsGradients[i] = neuronsOutputs[i].get_transposed().dot(&deltas[i]);
			biasGradients[i] = deltas[i].col_sums();
		}
	}

	/**
	* Supplies the computed gradients to the optimizer and performs one optimization step
	*/
	void optimize() {
		update_layers(optimizer->calculate_bias_update(biasGradients), optimizer->calculate_weights_update(weightsGradients));
	}

	/**
	* Updates layers' weights and biases with updates returned from the optimizer.
	*/
	void update_layers(std::vector<Matrix> bias_update, std::vector<Matrix> weights_update) {
		for (size_t i = 0; i < countLayers; i++) {
			layers[i]->set_bias(layers[i]->get_bias().sum(&bias_update[i]));
			layers[i]->set_weights(layers[i]->get_weights().sum(&weights_update[i]));
		}
	}

	/**
	* Performs argmax on predicted probabilities to get one hot encoded predictions.
	*/
	Matrix* batch_output_probabilities_to_predictions() {
		oneHotPredictions.set_values(std::vector<std::vector<double>>(neuronsOutputs[countLayers].get_shape()[0], std::vector<double>(neuronsOutputs[countLayers].get_shape()[1])));
		double max = 0;
		int argmax = 0;
		for (size_t i = 0; i < oneHotPredictions.get_shape()[0]; i++) {
			max = 0;
			argmax = 0;
			for (size_t j = 0; j < oneHotPredictions.get_shape()[1]; j++) {
				if (neuronsOutputs[countLayers].get_values()[i][j] > max) {
					max = neuronsOutputs[countLayers].get_values()[i][j];
					argmax = j;
				}
			}
			oneHotPredictions.set_value(i, argmax, 1);
		}
		return &oneHotPredictions;
	}

	/**
	* Information about training from one training epoch.
	*/
	void display_train_metrics_from_last_epoch() {
		std::cout << "Train loss: " << trainLossInEpoch[epochsDone - 1] << std::endl;
		std::cout << "Train metric: " << trainMetricInEpoch[epochsDone - 1] << std::endl;
	}

	/**
	* Information about validation from one training epoch.
	*/
	void display_validation_metrics_from_last_epoch() {
		std::cout << "Validation loss: " << validationLossInEpoch[epochsDone - 1] << std::endl;
		std::cout << "Validation metric: " << validationMetricInEpoch[epochsDone - 1] << std::endl;
	}

	/**
	* Creates Matrix ready to receive oneHotPredictions with dims based on batch size.
	* TODO? reserve the Matrices also for innerPotentials, neuronOutputs, deltas,... ?
	*/
	void get_ready(DataLoader* dataloader) {
		oneHotPredictions = Matrix(dataloader->get_batch_size(), CLASSES);
	}

	/**
	* Train one epoch on given dataset.
	* Updates weights, stores basic info about training from the epoch.
	*
	* @param DataLoader pointing on training dataset.
	*/
	void train_epoch(DataLoader* train_dataloader, bool shuffle) {
		if (shuffle) train_dataloader->shuffle_dataset();
		train_dataloader->reset();

		get_ready(train_dataloader);

		for (size_t i = 0; i < countLayers; i++) {
			layers[i]->get_ready_for_pass(train_dataloader);
		}
		int _iter_in_epoch = 0;
		double _epoch_sum_of_average_batch_losses = 0;
		double _epoch_sum_of_average_batch_metric = 0;
		while (!train_dataloader->is_exhausted()) {
			Batch batch = train_dataloader->get_sample(); // ALLOC

			forward_pass(batch, true);
			backward_pass(batch);
			optimize();

			_epoch_sum_of_average_batch_losses += lossFunction->calculate_mean_batch_loss(batch.Y, &neuronsOutputs[countLayers]);
			_epoch_sum_of_average_batch_metric += metric->calculate_metric_for_batch(batch.Y, batch_output_probabilities_to_predictions());
			_iter_in_epoch++;
		}
		trainLossInEpoch.push_back(_epoch_sum_of_average_batch_losses / _iter_in_epoch);
		trainMetricInEpoch.push_back(_epoch_sum_of_average_batch_metric / _iter_in_epoch);
	}

	/**
	* Evaluate network on validation dataset.
	* Saves metrics about the validation.
	*
	* @param DataLoader pointing on validation dataset.
	*/
	void validate_epoch(DataLoader* validation_dataloader) {
		validation_dataloader->reset();

		get_ready(validation_dataloader);

		for (size_t i = 0; i < countLayers; i++) {
			layers[i]->get_ready_for_pass(validation_dataloader);
		}
		int _iter_in_epoch = 0;
		double _epoch_sum_of_average_batch_losses = 0;
		double _epoch_sum_of_average_batch_metric = 0;
		// would be nice to pass the whole validation dataset at once 
		// and get rid off this while
		// but the matrices would be probably too big
		while (!validation_dataloader->is_exhausted()) {
			Batch batch = validation_dataloader->get_sample(); // ALLOC ?

			forward_pass(batch, false);

			_epoch_sum_of_average_batch_losses += lossFunction->calculate_mean_batch_loss(batch.Y, &neuronsOutputs[countLayers]);
			_epoch_sum_of_average_batch_metric += metric->calculate_metric_for_batch(batch.Y, batch_output_probabilities_to_predictions());
			_iter_in_epoch++;
		}
		validationLossInEpoch.push_back(_epoch_sum_of_average_batch_losses / _iter_in_epoch);
		validationMetricInEpoch.push_back(_epoch_sum_of_average_batch_metric / _iter_in_epoch);
	}
};

int main() {

	auto start = std::chrono::high_resolution_clock::now();

	int batch_size = 64;
	double learning_rate = 0.0005;

	Dataset train;
	train.load_mnist_data("data/fashion_mnist_train_vectors.csv", true);
	train.load_labels("data/fashion_mnist_train_labels.csv");
	//train.load_mnist_data("data/fashion_mnist_train_vectors_00.csv", true);
	//train.load_labels("data/fashion_mnist_train_labels_00.csv");
	//train.load_mnist_data("../../data/fashion_mnist_train_vectors_00.csv", true);
	//train.load_labels("../../data/fashion_mnist_train_labels_00.csv");

	Dataset validation = train.separate_validation_dataset(0.2);

	DataLoader train_loader(&train, batch_size);
	DataLoader validation_loader(&validation, 200);

	Layer layer0(train.get_X_cols(), 256, 0.15, 0.0001);
	Layer layer1(256, 64, 0.0, 0.00001);
	Layer layer2(64, CLASSES, 0.0, 0.0);
	ReLU relu;
	Softmax softmax;
	SGD sgd(learning_rate, 0.95, true);
	CrossEntropyLoss loss_func;
	Accuracy acc;


	NeuralNetwork nn({ &layer0, &layer1, &layer2 }, { &relu, &relu, &softmax }, &sgd, &loss_func, &acc);

	nn.train(1, &train_loader, &validation_loader);

	//Dataset test;
	//test.load_mnist_data("data/fashion_mnist_test_vectors.csv", true);
	//DataLoader test_loader(&test, 200);
	//nn.predict(&test_loader);
	//test.save_labels("data/actualTestPredictions");

	//DataLoader infer_train(&train, 200);
	//nn.predict(&infer_train);
	//train.save_labels("data/trainPredictions");

	auto stop = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << duration.count() << std::endl;

}

