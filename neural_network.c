#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "data.h"
#include "feature_scaling.h"

typedef struct NN_params
{
    double ***w;
    double *w_output;
    double **b;
    double b_output;
    double ***output_sets;
    double *predictions;

} NN_params;

void NN_ReLU_Sigmoid(double *X, double *Y, int epochs, int m, int n, int hidden_layers, int *neurons);
NN_params forward_propagation(double *X, double *Y, int m, int n, int hidden_layers, int *neurons, double ***weight_sets, double *weight_output_layer, double **biases, double b);
void back_propagation(double ***w, double *w_output, double **b, double b_output, double ***output_sets, double *predictions, double *x, double *y, int m, int n, int layers, int *neurons, double alpha);
NN_params get_NN_results(double ***w, double *w_output, double **b, double b_output, double ***output_sets, double *predictions);
NN_params initialize_weights_biases(int layers, int *neurons);
void prediction_vs_target(double *x, double *y, double ***w, double *w_output, double **b, double b_output, int layers, int *neurons, int m, int n);

int main () {
    int hidden_layers = 1; // Excluding the input and output layers
    int *neurons_per_layer; 
    int neurons[] = {8};
    neurons_per_layer = neurons; 

    z_score_normalized_X(&X[0][0], 100, 8);
    NN_ReLU_Sigmoid(&X[0][0], Y_classification, 1, 100, 8, hidden_layers, neurons_per_layer);
    
    return 0;
}

void NN_ReLU_Sigmoid(double *X, double *Y, int epochs, int m, int n, int hidden_layers, int *neurons) {

    NN_params params = initialize_weights_biases(hidden_layers, neurons);
    NN_params model_predictions;

    for (int iter = 0; iter < epochs; iter ++) {
        model_predictions = forward_propagation(X, Y_classification, m, n, hidden_layers, neurons, params.w, params.w_output, params.b, params.b_output);
        back_propagation(model_predictions.w, model_predictions.w_output, model_predictions.b, model_predictions.b_output, model_predictions.output_sets, model_predictions.predictions, X, Y_classification, m, n, hidden_layers, neurons, 0.01);
    }

    prediction_vs_target(X, Y_classification, params.w, params.w_output, params.b, params.b_output, hidden_layers, neurons, 100, 8);

    for (int i = 0; i < hidden_layers; i++) {
        for (int j = 0; j < neurons[i]; j++) {
            free(params.w[i][j]);
        }
        free(params.w[i]);
        free(params.b[i]);
    }
    free(params.w);
    free(params.b);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < hidden_layers; j++) {
            free(params.output_sets[i][j]);
        }
        free(params.output_sets[i]);
    }
    free(params.output_sets);
    free(params.w_output);
    free(params.predictions);
}

NN_params forward_propagation(double *X, double *Y, int m, int n, int layers, int *neurons, double ***weight_sets, double *w_output_layer, double **biases, double b) {
    
    /* Declaring input layer, allocating memory, etc */
    double *layer0 = (double *)malloc(n * sizeof(double));
    // NN will store the final preditction of the network for each training example
    double *NN = (double *)calloc(m, sizeof(double));

    // For each training example, we'll have a set of outputs for each layer's each neuron
    double ***output_sets = (double ***)malloc(m * sizeof(double **));
    for (int i = 0; i < m; i++) {
        output_sets[i] = (double **)calloc(layers, sizeof(double *));
        for (int j = 0; j < layers; j++) 
            output_sets[i][j] = (double *)calloc(neurons[j], sizeof(double));
    }
    double error = 0.0;

    // Looping through all the examlpes
    for (int i = 0; i < m; i++) {

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

    NN_params final_results;
    final_results = get_NN_results(weight_sets, w_output_layer, biases, b, output_sets, &NN[0]);
    
    return final_results;
}

void back_propagation(double ***w, double *w_output, double **b, double b_output, double ***output_sets, double *predictions, double *x, double *y, int m, int n, int layers, int *neurons, double alpha) {
    
    double **delta = malloc(layers * sizeof(double *));
    for (int l = 0; l < layers; l++) {
        delta[l] = calloc(neurons[l], sizeof(double));
    }

    for (int i = 0; i < m; i++) {

        // Output layer delta
        double pred = predictions[i];
        // derivative of loss wrt z for output
        double output_error = pred - y[i];   
        double delta_out = output_error * pred * (1.0 - pred); // sigmoid derivative

        // Hidden layers deltas
        // Index for hidden layers: 0 -> layers-1
        for (int l = layers - 1; l > -1; l--) {
            for (int j = 0; j < neurons[l]; j++) {
                double sum = 0.0;

                if (l == layers - 1)  
                    // Last hidden layer connects to output
                    sum += delta_out * w_output[j];
                else {
                    // Connect to next hidden layer
                    for (int k = 0; k < neurons[l + 1]; k++)
                        sum += delta[l + 1][k] * w[l + 1][k][j];
                }


                // ReLU derivative for hidden layers
                double deriv = (output_sets[i][l][j] > 0.0) ? 1.0 : 0.0;
                delta[l][j] = sum * deriv;
            }
        }

        // Updating hidden layer weights/biases
        for (int l = 0; l < layers; l++) {
            int prev_neurons = (l == 0) ? n : neurons[l - 1];
            for (int j = 0; j < neurons[l]; j++) {
                for (int k = 0; k < prev_neurons; k++) {
                    double a_prev = (l == 0) ? x[i * n + k] : output_sets[i][l - 1][k];    
                    w[l][j][k] -= alpha * delta[l][j] * a_prev;
                }
                b[l][j] -= alpha * delta[l][j];
            }
        }

        // Updating output layer weights/bias
        for (int j = 0; j < neurons[layers - 1]; j++) {
            w_output[j] -= alpha * delta_out * output_sets[i][layers - 1][j];
        }
        b_output -= alpha * delta_out;
    }

    // Free delta arrays
    for (int l = 0; l < layers; l++) free(delta[l]);
    free(delta);
}


NN_params get_NN_results(double ***w, double *w_output, double **b, double b_output, double ***output_sets, double *predictions) {
    NN_params results;

    results.w = w;
    results.w_output = w_output;
    results.b = b;
    results.b_output = b_output;
    results.output_sets = output_sets;
    results.predictions = predictions;

    return results;

}

NN_params initialize_weights_biases(int layers, int *neurons) {

    NN_params w_b;

    double ***weight_sets = (double ***)malloc(layers * sizeof(double **));
    double **biases = (double **)malloc(layers * sizeof(double *));
    for (int i = 0; i < layers; i++) {
        weight_sets[i] = (double **)malloc(neurons[i] * sizeof(double *));
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

    w_b.w = weight_sets;
    w_b.w_output = w_output_layer;
    w_b.b = biases;
    w_b.b_output = b;

    return w_b;
}

void prediction_vs_target(double *x, double *y, double ***w, double *w_output, double **b, double b_output, int layers, int *neurons, int m, int n) {

    double *layer0 = (double *)malloc(n * sizeof(double));
    // NN will store the final preditction of the network for each training example
    double *NN = (double *)calloc(m, sizeof(double));

    // For each training example, we'll have a set of outputs for each layer's each neuron
    double ***output_sets = (double ***)malloc(m * sizeof(double **));
    for (int i = 0; i < m; i++) {
        output_sets[i] = (double **)calloc(layers, sizeof(double *));
        for (int j = 0; j < layers; j++) 
            output_sets[i][j] = (double *)calloc(neurons[j], sizeof(double));
    }
    // Looping through all the examlpes
    for (int i = 0; i < m; i++) {
        // Populating the input layer
        for (int j = 0; j < n; j++)
            layer0[j] = x[i * n + j];    

        for (int l = 0; l < layers; l++) {

            // Computing the output for each neuron using ReLU activation in Layer 1
            for (int j = 0; j < neurons[l]; j++) {
                double z_hidden = 0.0;  // z for hidden layer

                if (l > 0) {
                // Calculating the output of the neuron by looping through: all the outputs of the previous layer * weights
                    for (int k = 0; k < neurons[l - 1]; k++) {
                        z_hidden += (output_sets[i][l - 1][k] * w[l][j][k]);
                    }
                } else {    // For the first hidden layers, the weights will be multiplied by the # of features from input layer
                    for (int k = 0; k < n; k++) {
                        z_hidden += (layer0[k] * w[l][j][k]);
                    }
                }
                 
                z_hidden += b[l][j];
                output_sets[i][l][j] = fmax(0.0, z_hidden);
            }
        }

        double z_output = 0.0;  // z for the output layer
        // Computing the final output (one neuron) using sigmoid acitvation
        for (int j = 0; j < neurons[layers - 1]; j++)
            z_output += (output_sets[i][layers - 1][j] * w_output[j]);
        z_output += b_output;

        NN[i] = 1 / (1 + exp(-z_output));

        printf("%lf\t%lf\n", NN[i], y[i]);
    }

    free(layer0);
    free(NN);

    for (int i = 0; i < m; i++) {
        for (int l = 0; l < layers; l++) {
            free(output_sets[i][l]);
        }
        free(output_sets[i]);
    }
    free(output_sets);

}