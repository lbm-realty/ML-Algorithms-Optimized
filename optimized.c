#include <stdio.h>
#include <stdlib.h>

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

int main() {
    const double X[][2] = {
        {1, 2},
        {2, 4},
        {3, 6},
        {4, 8}
    };
    const double Y[] = {7, 13, 19, 25};
    double *result = find_weights(&X[0][0], Y, 100000, 4, 2, 0.01);
    printArray(result, 2);
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
            f_prediction += b;
            double error = f_prediction - y[i];
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
        printf("%.2lf ", arr[i]);
    printf("\n");
}
