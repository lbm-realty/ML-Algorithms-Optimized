#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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
void print_arrray(double *arr, int n);
void print_matrix(double *arr, int rows, int cols);
void prediction_vs_target(double *x, double *y, double *w, double b, int m, int n);

int main() {

    z_score_normalized_X(&X[0][0], 100, 8);
    // scaled_Y(Y, 100);
    // clock_t start = clock();
    // double *result = find_weights(&X[0][0], Y_classification, 1000000, 100, 8, 0.01);
    // clock_t end = clock();
    // double run_time = (double)(end - start);

    // printf("Weights: \n");
    // print_arrray(result, 8);
    // printf("Run time for 10M iterations: %.2lf\n", run_time / 1000);
    // prediction_vs_target(&X[0][0], Y_classification, result, result[8], 100, 8);
    // print_matrix(&X[0][0], 100, 8);
    print_arrray(Y, 100);
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
            double exp_function = 0.0;

            for (int j = 0; j < n; j++) 
                exp_function += (W[j] * x[i * n + j]);

            exp_function += b;
            exp_function = 1 / (1 + exp(-exp_function));
            f_prediction = exp_function;
            double error = 0.0;
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

void prediction_vs_target(double *x, double *y, double *w, double b, int m, int n) {
    int count = 0;

    for (int i = 0; i < m; i++) {
        double f_prediction = 0.0;
        double exp_function = 0.0;

        for (int j = 0; j < n; j++) 
            exp_function += (w[j] * x[i * n + j]);

        exp_function += b;
        exp_function = 1 / (1 + exp(-exp_function));
        
        if (exp_function >= 0.5 && y[i] == 0)
            count += 1;
        if (exp_function < 0.5 && y[i] == 1)
            count += 1;

        // printf("%.1lf\t%.1lf\n", f_prediction, y[i]);
    }
    printf("%d\n", count);

}

void print_arrray(double *arr, int n) {
    for (int i = 0; i < n + 1; i++)
        printf("%.2lf, ", arr[i]);
    printf("\n");
}

void print_matrix(double *arr, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        printf("[");
        for (int j = 0; j < cols; j++)
           printf("%.1lf, ", arr[i * cols + j]);
        printf("],\n");
    }
    printf("\n");
}
