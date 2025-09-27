#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "data.h"
#include "feature_scaling.h"

typedef struct NN_results
{
    double **w1;
    double *w2;
    double *b1;
    double b2;
    double *predictions;

} NN_results;

void NN_ReLU_Sigmoid(double *X, double *Y, int epochs, int m, int n);
NN_results forward_propagation(double *X, double *Y, int m, int n, int neuron_count, int layers);
void back_propagation(double **w1, double *w2, double *b1, double b2, double *predictions, int m, int n, int layers, double *y, int neurons);

int main () {
    z_score_normalized_X(&X[0][0], 100, 8);
    NN_ReLU_Sigmoid(&X[0][0], Y_classification, 100, 100, 8);
    return 0;
}

void NN_ReLU_Sigmoid(double *X, double *Y, int epochs, int m, int n) {
    
    for (int iter = 0; iter < epochs; iter ++) {
        NN_results model_results;
        model_results = forward_propagation(X, Y_classification, m, n, n, 1);
        back_propagation(model_results.w1, model_results.w2, model_results.b1, model_results.b2, model_results.predictions, m, n, 1, Y_classification, n);
    }
}

NN_results forward_propagation(double *X, double *Y, int m, int n, int neurons, int layers) {
    
    double *layer0 = (double *)malloc(n * sizeof(double));
    double *b1 = (double *)calloc(n, sizeof(double));    
    double *NN = (double *)calloc(m, sizeof(double));
    double **hidden_layers = (double **)calloc(m, sizeof(double));
    for (int i = 0; i < layers; i++) 
        hidden_layers[i] = (double *)calloc(n, sizeof(double));

    double **output_hidden = (double **)calloc(m, sizeof(double));
    for (int i = 0; i < m; i++) 
        output_hidden[i] = (double *)calloc(n, sizeof(double));

    double **W1 = (double **)malloc(n * sizeof(double *));
    double *W2 = (double *)calloc(n, sizeof(double));
    for (int i = 0; i < n; i++) 
        W1[i] = (double *)calloc(n, sizeof(double));

    double b2 = 0.0;    // Scalar bc output layer has only one neuron
    double error = 0.0;

    for (int i = 0; i < m; i++) {
        double *layer1 = (double *)malloc(n * sizeof(double));
        double result = 0.0;

        // Populating the input layer
        for (int j = 0; j < n; j++)
            layer0[j] = X[i * n + j];    

        for (int l = 0; l < layers; l++) {

            // Computing the output for each neuron using ReLU activation in Layer 1
            for (int j = 0; j < neurons; j++) {
                double z_hidden = 0.0;  // z for the hidden layer
                // Calculating the output of the neuron by looping through all the inputs
                for (int k = 0; k < neurons; k++)
                    z_hidden += (layer0[k] * W1[j][k]);
                z_hidden += b1[j];
                hidden_layers[l][j] = fmax(0.0, z_hidden);
            }

            output_hidden[i] = layer1;
        }

        double z_output = 0.0;  // z for the output layer

        // Computing the final output (one neuron) using sigmoid acitvation
        for (int j = 0; j < n; j++) 
            z_output += (layer1[j] * W2[j]);
        z_output += b2;

        free(layer1);
        NN[i] = 1 / (1 + exp(-z_output));
        error += (NN[i] - Y[i]);
    }

    error /= m;

    NN_results final_results;
    final_results = get_NN_results(W1, W2, b1, b2, NN);
    
    return final_results;
    // for (int i = 0; i < n; i++) {
    //     free(W1[i]);
    // }

    // free(W1);
    // free(b1);
    // free(W2);
    // free(layer0);

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

NN_results get_NN_results(double **w1, double *w2, double *b1, double b2, double *predictions) {
    NN_results results;

    results.w1 = w1;
    results.w2 = w2;
    results.b1 = b1;
    results.b2 = b2;
    results.predictions = predictions;

    return results;

}
