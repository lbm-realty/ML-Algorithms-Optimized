#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "data.h"
#include "feature_scaling.h"

void forward_propagation(double *X, double *Y, int m, int n);

int main () {
    z_score_normalized_X(&X[0][0], 100, 8);
    forward_propagation(&X[0][0], Y_classification, 100, 8);
    return 0;
}

void forward_propagation(double *X, double *Y, int m, int n) {
    
    double *layer0 = (double *)malloc(n * sizeof(double));
    double *NN_result = (double *)calloc(m, sizeof(double));
    double *b1 = (double *)calloc(n, sizeof(double));    

    double **W1 = (double **)malloc(n * sizeof(double *));
    double *W2 = (double *)calloc(n, sizeof(double));
    for (int i = 0; i < n; i++) 
        W1[i] = (double *)calloc(n, sizeof(double));

    double b2 = 0.0;    // Scalar bc output layer has only one neuron

    for (int i = 0; i < m; i++) {
        double *layer1 = (double *)malloc(n * sizeof(double));

        // Populating the input layer
        for (int j = 0; j < n; j++)
            layer0[j] = X[i * n + j];    

        // Computing the output for each neuron using ReLU activation in Layer 1
        for (int j = 0; j < n; j++) {
            double z_hidden = 0.0;  // z for the hidden layer
            for (int k = 0; k < n; k++)
                z_hidden += (layer0[k] * W1[j][k]);
            z_hidden += b1[j];
            layer1[j] = fmax(0.0, z_hidden);
        }

        double z_output = 0.0;  // z for the output layer
        // Computing the final output (one neuron)
        for (int j = 0; j < n; j++) 
            z_output += (layer1[j] * W2[j]);
        z_output += b2;

        free(layer1);
        NN_result[i] = 1 / (1 + exp(-z_output));

    }

    for (int i = 0; i < n; i++) {
        free(W1[i]);
    }
    
    free(W1);
    free(b1);
    free(W2);
    free(layer0);

}
