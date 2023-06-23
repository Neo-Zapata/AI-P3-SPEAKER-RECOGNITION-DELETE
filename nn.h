#include <eigen3/Eigen/Dense>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <cmath>
#include <map>
#include <numeric>
#include <filesystem>
namespace fs = std::filesystem;
using namespace std;
#define parameters_filename "parameters.bin"


struct matriz_pesos {
    Eigen::MatrixXd pesos;
  
    matriz_pesos(int n, int k) {
        // srand((unsigned)time(NULL));
        // pesos.resize(n, k);

        // for(int i = 0; i < n; i++) {
        //     for(int j = 0; j < k; k++) {
        //         pesos(i, j) = static_cast<double>(rand());
        //     }
        // }
        pesos.resize(n, k);
        pesos = Eigen::MatrixXd::Random(n, k);
    }
};



struct red_neuronal {

    vector<matriz_pesos> matrices_pesos;
    Eigen::MatrixXd X, true_labels;
    string activation_type, layers, neurons;
    map<string, Eigen::MatrixXd> parameters;
    double learning_rate;
    // int num_batches;

    red_neuronal(int input_nodes, vector<int> hidden_layers, int output_nodes, string type) { // constructor
        // assign random weights
        this->activation_type = type;
        this->learning_rate = 0.01;
        this->neurons = to_string(hidden_layers[0]);
        // cout << input_nodes << " - " << hidden_layers[0] << endl;

        int counter = 1;
        matrices_pesos.push_back(matriz_pesos(input_nodes, hidden_layers[0]));
        counter++;
        for (int i = 1; i < hidden_layers.size(); i++) {
            // cout << hidden_layers[i - 1] << " - " << hidden_layers[i ] << endl;
            matrices_pesos.push_back(matriz_pesos(hidden_layers[i - 1], hidden_layers[i]));
            counter++;
        }
        // cout << hidden_layers[hidden_layers.size() - 1] << " - " << output_nodes << endl;
        matrices_pesos.push_back(matriz_pesos(hidden_layers[hidden_layers.size() - 1], output_nodes));
        counter++;
        this->layers = to_string(counter);

        // parameters initialization
        for (int l = 0 ; l < matrices_pesos.size() ; l++) {
            parameters["W" + to_string(l+1)] = matrices_pesos[l].pesos;
            // parameters["B" + to_string(l+1)] = Eigen::MatrixXd::Zero(layer_dims[l], 1);
        } // W1 = matrix1 , W2 = matrix2
    }

    Eigen::MatrixXd forward(Eigen::MatrixXd A, map<string,Eigen::MatrixXd>& cache) {
        // Eigen::MatrixXd y_pred(rows, cols);
        // for(int i = 0 ; i < Xi.size() ; i++){
            // Eigen::VectorXd row = mini_batch.row(i);
        // Eigen::VectorXd layer = Current_layer;

        cache["A0"] = A; // input layer
        Eigen::MatrixXd new_A;
        for(int j = 0 ; j < matrices_pesos.size() ; j++){
            Eigen::MatrixXd W        = matrices_pesos[j].pesos;
            // cout << "mat_peso size: " << mat_peso.size() << endl;
            // cout << "mat_peso rows: " << mat_peso.rows() << endl;
            // cout << "mat_peso cols: " << mat_peso.cols() << endl;
            // cout << "layer size: " << Current_layer.size() << endl;
            // cout << "layer rows: " << Current_layer.rows() << endl;
            // cout << "layer cols: " << Current_layer.cols() << endl;
            Eigen::MatrixXd Z      = A * W;
            // cout << "Next_layer size: " << next_layer.size() << endl;
            // cout << "Next_layer rows: " << next_layer.rows() << endl;
            // cout << "Next_layer cols: " << next_layer.cols() << endl;
            if(j == matrices_pesos.size() - 1){
                // cout << "aplying soft" << endl;
                new_A = funcion_activacion(Z, "softmax");
                // new_A = funcion_activacion(Z, activation_type);
            } else {
                // cout << "aplying " << activation_type << endl;
                new_A = funcion_activacion(Z, activation_type);
            }
            // cout << "activated_layer size: " << activated_layer.size() << endl;
            // cout << "activated_layer rows: " << activated_layer.rows() << endl;
            // cout << "activated_layer cols: " << activated_layer.cols() << endl;
            A                   = new_A;

            cache["Z" + to_string(j+1)] = Z;
            cache["A" + to_string(j+1)] = A;
            // int count = 0;
            // for(int k = 0 ; k < mat_peso.cols() ; k++){
            //     Eigen::VectorXd row_peso = mat_peso.row(k);
            //     double dot_product = Xi.dot(row_peso);
            //     next_layer(count) = dot_product;
            //     count++;
            // }
            // Eigen::VectorXd activated_layer = funcion_activacion(next_layer, activation_type);
            // row = activated_layer;
        }
        // here we have the output vector
        // y_pred.row(i) = row;
        // }
        return A;
    }

    map<string, Eigen::MatrixXd> backpropagation(Eigen::MatrixXd y_pred, Eigen::MatrixXd y_real, map<string,Eigen::MatrixXd>& cache){
        map<string, Eigen::MatrixXd> gradients;

        // dC/dA
        Eigen::MatrixXd dA = (2*(y_pred - y_real)); // derivate of A with respect to Cost only if the cost function is MSE
        // cout << "dC/dA " << dA.rows() << " - " << dA.cols() << endl; // (1,24) - (1,2)
        // iterate in reverse order
        for(int i = matrices_pesos.size() ; i >= 1 ; i--){
            // cout << "i: " << i << endl;

            Eigen::MatrixXd A_prev = cache.at("A" + to_string(i-1)); // A[L-1] - A[L-2] (input)
            // cout << "A[L-i] = " << A_prev.rows() << " - " << A_prev.cols() << endl;

            Eigen::MatrixXd Z = cache.at("A" + to_string(i)); // Z[L] - Z[L-1]
            // cout << "Z[L] = " << Z.rows() << " - " << Z.cols() << endl;
            
            Eigen::MatrixXd dZ = dA.array() * (activation__derivative(Z, activation_type)).array(); // Da/dZ
            // cout << "dC/dA * dA/dZ = " << dZ.rows() << " - " << dZ.cols() << endl;

            Eigen::MatrixXd dW = (A_prev.transpose() * dZ)/y_pred.size(); // dZ/dW
            // cout << "dC/dW = dZ/dW * dC/dA * dA/dZ = " << dW.rows() << " - " << dW.cols() << endl;

            Eigen::MatrixXd new_dA = parameters.at("W" + to_string(i)) * dZ.transpose(); // dA = W[L] * dZ
            // cout << "new dA = " << new_dA.rows() << " - " << new_dA.cols() << endl;

            gradients["dW" + to_string(i)] = dW;

            dA = new_dA.transpose();
        }

        return gradients;
    }

    Eigen::MatrixXd activation__derivative(Eigen::MatrixXd Z, string type){
        if(type == "sigmoid"){
            // return x.array().exp() / (1 + x.array().exp()).pow(2);
            return Z.unaryExpr([](double x) { double sig = 1.0 / (1.0 + std::exp(-x)); return (sig * (1 - (1.0 / (1.0 + std::exp(-x))))); });
        } else if(type == "relu"){
            return (Z.array() > 0).cast<double>();
        } else if(type == "tanh"){
            return 1 - Z.array().tanh().pow(2);
        } else {
            cerr << "Error - not a valid type for activation function.\n";
            return Eigen::MatrixXd::Random(1,1);
        }
    }

    double calculate_error(Eigen::MatrixXd y_pred, Eigen::MatrixXd Yi){ // MSE
        Eigen::MatrixXd result = (y_pred - Yi).array().square();
        double loss = result.mean();
        return loss;
    }

    void update_parameters(map<string, Eigen::MatrixXd> gradients){
        for(int i = 0 ; i < matrices_pesos.size() ; i++){
            parameters["W" + to_string(i+1)] -= learning_rate * gradients["dW" + to_string(i+1)];
        }
    }

    // void save_parameters_to_file(const string filename, map<string, Eigen::MatrixXd> gradients){

    //     // Eigen::StdMap<std::string, Eigen::MatrixXd> eigenMatrixMap(gradients.begin(), gradients.end());
    //     // Eigen::save(eigenMatrixMap, filename);
    // }

    // std::map<std::string, Eigen::MatrixXd> load_parameters(const string filename) {
    //     Eigen::StdMap<std::string, Eigen::MatrixXd> eigenMatrixMap;
    //     Eigen::load(eigenMatrixMap, filename);
    //     return std::map<std::string, Eigen::MatrixXd>(eigenMatrixMap.begin(), eigenMatrixMap.end());
    // }

    void show_matrix(Eigen::MatrixXd mat){
        for (int i = 0 ; i < mat.rows() ; i++){
            for(int j = 0 ; j < mat.cols() ; j++){
                cout << round(mat(i,j)) << " ";
            }
        }
        cout << endl;
    }

    void train(Eigen::MatrixXd X, Eigen::MatrixXd true_labels){
        this->X = X;
        this->true_labels = true_labels;

        // cout << "X size: " << X.size() << endl;
        // cout << "X rows: " << X.rows() << endl;
        // cout << "X cols: " << X.cols() << endl;

        // cout << "true_labels size: " << true_labels.size() << endl;
        // cout << "true_labels rows: " << true_labels.rows() << endl;
        // cout << "true_labels cols: " << true_labels.cols() << endl;
        map<string, Eigen::MatrixXd> gradients;
        double last_error;
        for (int i = 0; i < X.rows(); ++i) {
            Eigen::VectorXd Xi = X.row(i); // (128, 1)
            Eigen::VectorXd Yi = true_labels.row(i); // (24, 1)

            Eigen::Matrix<double, 1, Eigen::Dynamic> Xi_matrix = Xi.transpose(); // (1, 128)
            Eigen::Matrix<double, 1, Eigen::Dynamic> Yi_matrix = Yi.transpose(); // (1, 24)
            
            // cout << Xi_matrix.rows() << endl;
            // cout << Xi_matrix.cols() << endl;
            // cout << Yi_matrix.rows() << endl;
            // cout << Yi_matrix.cols() << endl;

            // relevant informatioin in order to do backpropagation
            map<string, Eigen::MatrixXd> cache;

            // forward propagation
            // cout << "\nDoing forward:\n";
            Eigen::MatrixXd y_pred = forward(Xi_matrix, cache);
            // calculate the error
            double loss = calculate_error(y_pred, Yi_matrix);
            // cout << "\nError: " << loss;
            // backpropagation + update weights
            // cout << "\nBackpropagation\n" << endl;
            gradients = backpropagation(y_pred, Yi_matrix, cache);

            // cout << endl;
            // show_matrix(y_pred);
            // show_matrix(Yi_matrix);
            // cout << endl;

            // cout << "\nW update\n" << endl;
            update_parameters(gradients);
            last_error = loss;
        }
        // save_parameters_to_file(parameters_filename, gradients);
        storeMatricesToFile(gradients, parameters_filename);
        cout << "\nTraining Error: " << last_error;
        // vector<Eigen::MatrixXd> all_batches, true_labels_batch;
        // int batch_size = static_cast<int>(round(feature_vector.rows()/num_batches));
        // get_batches(batch_size, all_batches, true_labels_batch);

        // int counter = 0;
        // for(auto mini_batch: all_batches){
        //     Eigen::MatrixXd mini_true_labels_batch = true_labels_batch[counter];
        //     process_mini_batch(mini_batch, mini_true_labels_batch);
        //     counter++;
        // }        
    }

    // void get_batches(int batch_size, vector<Eigen::MatrixXd>& all_batches, vector<Eigen::MatrixXd>& true_labels_batch){
    //     for(int batch_ind = 0 ; batch_ind < num_batches ; batch_ind++){
    //         int counter = 0;
    //         Eigen::MatrixXd mini_batch(batch_size, feature_vector.cols());
    //         Eigen::MatrixXd mini_true_labels_batch(batch_size, true_labels.cols());
    //         for(int i = batch_ind * batch_size ; i < (batch_ind + 1) * batch_size ; i++){
    //             mini_batch.row(counter) = feature_vector.row(i);
    //             mini_true_labels_batch.row(counter) = true_labels.row(i);
    //             counter++;
    //         }
    //         all_batches.push_back(mini_batch);
    //         true_labels_batch.push_back(mini_true_labels_batch);
    //     }
    // }

    // void process_mini_batch(Eigen::MatrixXd mini_batch, Eigen::MatrixXd true_labels_mini_batch){
    //     // forward propagation
    //     Eigen::MatrixXd y_pred = forward(mini_batch, true_labels_mini_batch.rows(), true_labels_mini_batch.cols());
    //     // calculate the error
    //     double loss = calculate_error(y_pred, true_labels_mini_batch);
    //     // backpropagation + update weights
    //     backpropagation();
    // }

    // void evaluate(Eigen::MatrixXd x_test, Eigen::MatrixXd y_test){
    //     vector<Eigen::MatrixXd> all_batches, true_labels_batch;
    //     int batch_size = static_cast<int>(round(feature_vector.rows()/num_batches));
    //     get_batches(batch_size, all_batches, true_labels_batch);

    //     int counter = 0;
    //     for(auto mini_batch: all_batches){
    //         Eigen::MatrixXd mini_true_labels_batch = true_labels_batch[counter];
    //         process_mini_batch(mini_batch, mini_true_labels_batch);
    //         counter++;
    //     }
    //     // get the metrics
    // }

    Eigen::MatrixXd funcion_activacion(const Eigen::MatrixXd layer, string type) { // (1, k)
        if (type == "softmax"){
            Eigen::MatrixXd temp = layer.array().exp();
            double sum = temp.sum();
            return (temp/sum);
        } else if (type == "sigmoid"){
            return layer.unaryExpr([](double x) { return 1.0 / (1.0 + exp(-x)); });
        } else if (type == "relu"){
            return layer.unaryExpr([](double x) { return ((x > 0.0) ? x : 0.0); });
        } else if (type == "tanh"){
            return layer.unaryExpr([](double x) { return tanh(x); });
        } else {
            cerr << "Error - not a valid type for activation function.\n";
            return Eigen::MatrixXd::Random(1,1);
        }
    }

    void evaluate(Eigen::MatrixXd x_test, Eigen::MatrixXd y_test){
        map<string, Eigen::MatrixXd> parameters_loaded = retrieveMatricesFromFile(parameters_filename);
        this->parameters = parameters_loaded;
        this->X = x_test;
        this->true_labels = y_test;
        double last_error;
        vector<vector<double>> v_y_real;
        vector<vector<double>> v_y_pred;
        
        for (int i = 0; i < X.rows(); ++i) {
            Eigen::VectorXd Xi = X.row(i); 
            Eigen::VectorXd Yi = true_labels.row(i); 

            Eigen::Matrix<double, 1, Eigen::Dynamic> Xi_matrix = Xi.transpose(); // (1, 128)
            Eigen::Matrix<double, 1, Eigen::Dynamic> Yi_matrix = Yi.transpose(); // (1, 24)
            
            map<string, Eigen::MatrixXd> cache;

            // forward propagation
            // cout << "\nDoing forward:\n";
            Eigen::MatrixXd y_pred = forward(Xi_matrix, cache);
            double loss = calculate_error(y_pred, Yi_matrix);
            // cout << "\nError: " << loss;
            last_error = loss;

            vector<double> trueLabels = format_changer(Yi_matrix);
            vector<double> predLabels = format_changer(y_pred);

            v_y_pred.push_back(predLabels);
            v_y_real.push_back(trueLabels);
        }
        cout << "\nTesting Error: " << last_error;
        // save_to_file(v_y_real, v_y_pred);

        // for(int i = 0 ; i < v_y_pred[0].size() ; i++){
        //     cout << v_y_pred[0][i] << " ";
        // }
        // cout << endl;

        // for(int i = 0 ; i < v_y_real[0].size() ; i++){
        //     cout << v_y_real[0][i] << " ";
        // }
        // cout << endl;

        get_metrics(v_y_real, v_y_pred);

    }

    // Function to calculate the confusion matrix
    std::vector<std::vector<int>> calculateConfusionMatrix(const std::vector<int>& y_true, const std::vector<int>& y_pred) {
        std::vector<std::vector<int>> confusionMatrix(2, std::vector<int>(2, 0));

        for (int i = 0; i < y_true.size(); i++) {
            int trueLabel = y_true[i];
            int predictedLabel = y_pred[i];

            confusionMatrix[trueLabel][predictedLabel]++;
        }

        return confusionMatrix;
    }

    void get_metrics(vector<vector<double>> y_real, vector<vector<double>> y_pred){
        vector<double> prec;
        vector<double> rec;
        vector<double> f1scor;

        // vector<vector<int>> confusionMatrix = calculateConfusionMatrix(y_real, y_pred);
        // // Print the confusion matrix
        // for (const auto& row : confusionMatrix) {
        //     for (int value : row) {
        //         std::cout << value << " ";
        //     }
        //     std::cout << std::endl;
        // }

        for (size_t i = 0; i < y_pred.size(); ++i) {
            double precision = calculatePrecision(y_pred[i], y_real[i]);
            double recall = calculateRecall(y_pred[i], y_real[i]);
            double f1Score = calculateF1Score(precision, recall);
            prec.push_back(precision);
            rec.push_back(recall);
            f1scor.push_back(f1Score);
        }

        double sum = std::accumulate(prec.begin(), prec.end(), 0.0);
        double precision_mean = sum / prec.size();

        sum = std::accumulate(rec.begin(), rec.end(), 0.0);
        double recall_mean = sum / rec.size();

        sum = std::accumulate(f1scor.begin(), f1scor.end(), 0.0);
        double f1_mean = sum / f1scor.size();

        std::cout << "\nPrecision = " << precision_mean << ", Recall = " << recall_mean << ", F1 Score = " << f1_mean << endl;
    }

    double calculatePrecision(const std::vector<double>& predictions, const std::vector<double>& targets) {
        double truePositives = 0;
        double falsePositives = 0;

        for (size_t i = 0; i < predictions.size(); ++i) {
            if (predictions[i] == 1 && targets[i] == 1) {
                truePositives++;
            } else if (predictions[i] == 1 && targets[i] == 0) {
                falsePositives++;
            }
        }

        if (truePositives + falsePositives == 0) {
            return 0;  // handle the case when there are no positive predictions
        }

        return truePositives / (truePositives + falsePositives);
    }

    double calculateRecall(const std::vector<double>& predictions, const std::vector<double>& targets) {
        double truePositives = 0;
        double falseNegatives = 0;

        for (size_t i = 0; i < predictions.size(); ++i) {
            if (predictions[i] == 1 && targets[i] == 1) {
                truePositives++;
            } else if (predictions[i] == 0 && targets[i] == 1) {
                falseNegatives++;
            }
        }

        if (truePositives + falseNegatives == 0) {
            return 0;  // handle the case when there are no positive targets
        }

        return truePositives / (truePositives + falseNegatives);
    }

    double calculateF1Score(double precision, double recall) {
        if (precision + recall == 0) {
            return 0;  // handle the case when both precision and recall are 0
        }

        return 2 * (precision * recall) / (precision + recall);
    }

    vector<double> format_changer(const Eigen::MatrixXd& eigenMatrix) {
        vector<double> result;
        // result.reserve(eigenMatrix.cols());
        for(int j = 0 ; j < eigenMatrix.rows() ; j++){  // 0 - 1
            for (int i = 0; i < eigenMatrix.cols(); ++i){ // 0 - 25
                if(eigenMatrix(j,i) > 0.5){
                    result.push_back(1);
                    // cout << "1 ";
                } else{
                    result.push_back(0);
                    // cout << "0 ";
                }
                // result.push_back(round(eigenMatrix(0,i)));
            }
        }
        // cout <<  endl;
        return result;
    }

    void save_to_file(vector<vector<double>> y_real, vector<vector<double>> y_pred){
        
        // for(int i = 0 ; i < y_real.size() ; i++){
        //     for(int j = 0 ; j < y_real[i].size() ; j++){
        //         cout << "\n" << y_real[i][j] << " - " << y_pred[i][j] << endl;
        //     }
        // }

        // for(int i = 0 ; i < 24 ; i++){
        //     cout << y_real[0][i] << ", "; 
        // }
        // cout << endl;

        // for(int i = 0 ; i < 24 ; i++){
        //     cout << y_pred[0][i] << ", ";
        // }
        // cout << endl;



        string nene = activation_type + "_" + layers + "_" + neurons;
        string folder_name = "results/" + nene;
        string y_real_filename = folder_name + "/y_real_" + nene + ".bin";
        string y_pred_filename = folder_name + "/y_pred_" + nene + ".bin";

        if(!fs::exists(folder_name)){
            if(!fs::create_directories(folder_name)){
                cout << "Failed to created folders." << endl;
            }
        }

        ofstream file(y_real_filename, std::ios::binary);
        if (!file) {
            std::cerr << "Error opening file: " << y_real_filename << std::endl;
            return;
        }

        for (const auto& innerVec : y_real) {
            size_t size = innerVec.size();
            file.write(reinterpret_cast<const char*>(&size), sizeof(size));

            file.write(reinterpret_cast<const char*>(innerVec.data()), size * sizeof(double));
        }

        file.close();

        
        std::ofstream p_file(y_pred_filename, std::ios::binary);
        if (!p_file) {
            std::cerr << "Error opening file: " << y_pred_filename << std::endl;
            return;
        }

        for (const auto& innerVec : y_pred) {
            size_t size = innerVec.size();
            p_file.write(reinterpret_cast<const char*>(&size), sizeof(size));

            p_file.write(reinterpret_cast<const char*>(innerVec.data()), size * sizeof(double));
        }

        p_file.close();

    }

    void storeMatricesToFile(const std::map<std::string, Eigen::MatrixXd>& matrices, const std::string& filename) {
        std::ofstream file(filename, std::ios::binary);

        if (file.is_open()) {
            size_t numMatrices = matrices.size();
            file.write(reinterpret_cast<char*>(&numMatrices), sizeof(numMatrices));

            for (const auto& entry : matrices) {
                const std::string& matrixName = entry.first;
                const Eigen::MatrixXd& matrix = entry.second;

                // Write matrix name
                size_t matrixNameSize = matrixName.size();
                file.write(reinterpret_cast<char*>(&matrixNameSize), sizeof(matrixNameSize));
                file.write(matrixName.c_str(), matrixNameSize);

                // Write matrix dimensions
                Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Index rows = matrix.rows();
                Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Index cols = matrix.cols();
                file.write(reinterpret_cast<char*>(&rows), sizeof(rows));
                file.write(reinterpret_cast<char*>(&cols), sizeof(cols));

                // Write matrix data
                file.write(reinterpret_cast<const char*>(matrix.data()), matrix.size() * sizeof(double));
            }

            file.close();
            // std::cout << "Matrices stored in file: " << filename << std::endl;
        } else {
            std::cerr << "Unable to open file: " << filename << std::endl;
        }
    }

    std::map<std::string, Eigen::MatrixXd> retrieveMatricesFromFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        std::map<std::string, Eigen::MatrixXd> matrices;

        if (file.is_open()) {
            size_t numMatrices;
            file.read(reinterpret_cast<char*>(&numMatrices), sizeof(numMatrices));

            for (size_t i = 0; i < numMatrices; ++i) {
                // Read matrix name
                size_t matrixNameSize;
                file.read(reinterpret_cast<char*>(&matrixNameSize), sizeof(matrixNameSize));
                std::string matrixName(matrixNameSize, '\0');
                file.read(&matrixName[0], matrixNameSize);

                // Read matrix dimensions
                Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Index rows, cols;
                file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
                file.read(reinterpret_cast<char*>(&cols), sizeof(cols));

                // Read matrix data
                Eigen::MatrixXd matrix(rows, cols);
                file.read(reinterpret_cast<char*>(matrix.data()), matrix.size() * sizeof(double));

                matrices[matrixName] = matrix;
            }

            file.close();
            // std::cout << "Matrices retrieved from file: " << filename << std::endl;
        } else {
            std::cerr << "Unable to open file: " << filename << std::endl;
        }

        return matrices;
    }
};
