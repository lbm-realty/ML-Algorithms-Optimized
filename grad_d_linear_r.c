#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "data.h"
#include "feature_scaling.h"

/*
    Reference:
        - alpha -> Learning rate
        - epochs -> iterations
        - m -> number of training examples
        - n -> number of features
        - W -> array of weights
        - dW -> array of gradients
        - b -> bias
        - X -> training set
        - Y -> output set with answers
*/

double result[2];
double *find_weights(const double *x, const double *y, int epochs, int m, int n, double alpha);
void printArray(double *arr, int n);
void printMatrix(double *arr, int rows, int cols);

int main() {

    z_score_normalized_X(&X[0][0], 100, 8);
    z_score_normalized_Y(Y, 100);
    clock_t start = clock();
    double *result = find_weights(&X[0][0], Y, 100000, 100, 8, 0.01);
    clock_t end = clock();
    double run_time = (double)(end - start);
    
    printf("Weights: \n");
    printArray(result, 8);
    // printf("First 5 values of Y: \n");
    // printArray(Y, 4);
    // printf("First 5 rows of X: \n");
    // printMatrix(&X[0][0], 4, 8);
    printf("Run time for 10M iterations: %.2lf\n", run_time / 1000);
}

double *find_weights(const double *x, const double *y, int epochs, int m, int n, double alpha) {
    
    // The number of weights should be determined by the number of features
    double *W = (double *)calloc(n + 1, sizeof(double));
    double *dW = (double *)calloc(n, sizeof(double)), db = 0;
    double b = 0.0;

    for (int iter = 0; iter < epochs; iter++) {
        
        // Resetting the gradients for the ongoing iteration
        for (int i = 0; i < n; i++) 
            dW[i] = 0.0;

        // Going through all the training examples
        for (int i = 0; i < m; i++) {
            double f_prediction = 0.0;

            for (int j = 0; j < n; j++) 
                f_prediction += (W[j] * x[i * n + j]);
            double error = 0.0;
            f_prediction += b;
            error = f_prediction - y[i];

            for (int j = 0; j < n; j++) 
                dW[j] += error * x[i * n + j];
            db += error;
        }
        
        // Taking the average of the collected error
        for (int j = 0; j < n; j++)
            dW[j] /= m;
        db /= m;
        for (int j = 0; j < n; j++) 
            W[j] -= alpha * dW[j];
        b -= alpha * db;
    }

    W[n] = b;
    free(dW);
    return W;    
}

void printArray(double *arr, int n) {
    for (int i = 0; i < n + 1; i++)
        printf("%.2lf, ", arr[i]);
    printf("\n");
}

void printMatrix(double *arr, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++)
           printf("%.1lf, ", arr[i * cols + j]);
        printf("\n");
    }
    printf("\n");
}
