#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "data.h"
#include "feature_scaling.h"

typedef struct NN_results
{
    double ***w;
    double *w_output;
    double **b;
    double b_output;
    double ***output_sets;
    double *predictions;

} NN_results;

void NN_ReLU_Sigmoid(double *X, double *Y, int epochs, int m, int n, int hidden_layers, int *neurons);
NN_results forward_propagation(double *X, double *Y, int m, int n, int hidden_layers, int *neurons);
void back_propagation(double **w1, double *w2, double *b1, double b2, double *predictions, int m, int n, int layers, double *y, int neurons);

int main () {
    int hidden_layers = 1; // Apart from the input and output layers
    int neurons_per_layer = {8}; 
    z_score_normalized_X(&X[0][0], 100, 8);
    NN_ReLU_Sigmoid(&X[0][0], Y_classification, 100, 100, 8, hidden_layers, neurons_per_layer);
    return 0;
}

void NN_ReLU_Sigmoid(double *X, double *Y, int epochs, int m, int n, int hidden_layers, int *neurons) {
    
    for (int iter = 0; iter < epochs; iter ++) {
        NN_results model_results;
        model_results = forward_propagation(X, Y_classification, m, n, hidden_layers, neurons);
        // back_propagation(model_results.w1, model_results.w2, model_results.b1, model_results.b2, model_results.predictions, m, n, 1, Y_classification, n);
    }
}

NN_results forward_propagation(double *X, double *Y, int m, int n, int layers, int *neurons) {
    
    /* Declaring variables, allocating memory, etc */

    double *layer0 = (double *)malloc(n * sizeof(double));

    // Each layer has its set of weights pointing to an array of pointers
    // and a set of biases that are the same number as the number of neurons for that layer
    double ***weight_sets = (double ***)malloc(layers * sizeof(double));
    double **biases = (double **)malloc(layers * sizeof(double));
    for (int i = 0; i < layers; i++) {
        weight_sets[i] = (double **)malloc(neurons[i] * sizeof(double));
        biases[i] = (double *)calloc(neurons[i], sizeof(double));
        
        // Creating 2D array for neurons
        for (int j = 0; j < neurons[i]; j++) {
            // # of weights for each neuron = # of neurons in the previous layer
            if (i > 0)
                weight_sets[i][j] = (double *)calloc(neurons[i - 1], sizeof(double));
            else    // For the first hidden layer, # weights = # of features/neurons in input layer
                weight_sets[i][j] = (double *)calloc(neurons[i], sizeof(double));
        }
    }

    double *w_output_layer = (double *)calloc(neurons[layers - 1], sizeof(double));
    double b = 0.0;     // Bias for the output layer
    double *NN = (double *)calloc(m, sizeof(double));

    // For each training example, we will have a set of outputs
    double ***output_sets = (double ***)malloc(m * sizeof(double));
    for (int i = 0; i < m; i++) {
        output_sets[i] = (double **)calloc(layers, sizeof(double));
        for (int j = 0; j < layers; j++) 
            output_sets[i][j] = (double *)calloc(neurons[j], sizeof(double));
    }
    double error = 0.0;

    /* End of all the pre work like declaring variables, allocating memory, etc */

    /* Forward pass begins */
    // Looping through all the examlpes
    for (int i = 0; i < m; i++) {
        double result = 0.0;

        // Populating the input layer
        for (int j = 0; j < n; j++)
            layer0[j] = X[i * n + j];    

        for (int l = 0; l < layers; l++) {

            // Computing the output for each neuron using ReLU activation in Layer 1
            for (int j = 0; j < neurons[l]; j++) {
                double z_hidden = 0.0;  // z for hidden layer

                if (l > 0) {
                // Calculating the output of the neuron by looping through: all the outputs of the previous layer * weights
                    for (int k = 0; k < neurons[l - 1]; k++) {
                        z_hidden += (output_sets[i][l - 1][k] * weight_sets[l][j][k]);
                    }
                } else {    // For the first hidden layers, the weights will be multiplied by the # of features from input layer
                    for (int k = 0; k < n; k++) {
                        z_hidden += (layer0[k] * weight_sets[l][j][k]);
                    }
                }
                 
                z_hidden += biases[l][j];
                output_sets[i][l][j] = fmax(0.0, z_hidden);
            }
        }

        double z_output = 0.0;  // z for the output layer
        // Computing the final output (one neuron) using sigmoid acitvation
        for (int j = 0; j < neurons[layers - 1]; j++)
            z_output += (output_sets[i][layers - 1][j] * w_output_layer[j]);
        z_output += b;

        NN[i] = 1 / (1 + exp(-z_output));
        error += (NN[i] - Y[i]);
    }

    error /= m;

    NN_results final_results;
    final_results = get_NN_results(weight_sets, w_output_layer, biases, b, output_sets, NN);
    
    return final_results;
}

void back_propagation(double **w1, double *w2, double *b1, double b2, double *predictions, int m, int n, int layers, double *y, int neurons) {
    // Output layer calc; keeping it seperate
    double *dW1 = (double *)calloc(m, sizeof(double));
    
    for (int i = 0; i < m; i++) {
        double output = predictions[i] - y[i];
        output = output * predictions[i] * (1 - predictions[i]);
        // dW1[i] = 

        for (int l = layers; l > 0; l--) {
            for (int j = 0; j < neurons; j++) {
                ;
            }
        }
    }

}

NN_results get_NN_results(double ***w, double *w_output, double **b, double b_output, double ***output_sets, double *predictions) {
    NN_results results;

    results.w = w;
    results.w_output = w_output;
    results.b = b;
    results.b_output = b_output;
    results.output_sets = output_sets;
    results.predictions = predictions;

    return results;

}
