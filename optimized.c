#include <stdio.h>

double result[2];
double *find_weights(const double *x, const double *y, int epochs, int rows, int cols, double alpha);

int main() {
    const double X[][2] = {
        {1, 2},
        {2, 4},
        {3, 6},
        {4, 8}
    };
    const double Y[] = {7, 13, 19, 25};
    find_weights(X, Y, 100000000, 4, 2, 0.01);
}

double *find_weights(const double *x, const double *y, int epochs, int rows, int cols, double alpha) {
    double w = 0.0, b = 0.0;

    for (int iter = 0; iter < epochs; iter++) {
        double dw = 0, db = 0;

        for (int i = 0; i < rows; i++) {
            double f_pred = 0.0;
            for (int j = 0; j < cols; j++) 
                f_pred = w * x[i * cols + j] + b;
            double error = f_pred - y[i];
            for (int j = 0; j < 1; j++) {
                dw += error * x[i * cols + j];
                db += error;
            }
        }
        
        dw /= (rows / cols);
        db /= (rows / cols);
        w -= alpha * dw;
        b -= alpha * db;
        // printf("dw: %.2lf, db: %.2lf, w: %.2lf, b: %.2lf\n", dw, db, w, b);
    }

    result[0] = w;
    result[1] = b;
    printf("%lf %lf\n", result[0], result[1]);
    return result;
}


// printf("%lf %d %lf\n", *(&x[j]), x[j], &x[j]); 



// #include <stdio.h>
// #include <stdlib.h>
// #include <math.h>

// void gradient_descent(
//         double **X,      // m x n feature matrix
//         double *y,       // length m
//         double *theta,   // length n (weights)
//         int m,           // number of samples
//         int n,           // number of features
//         double alpha,    // learning rate
//         int iterations   // number of steps
// ) {
//     double *grad = malloc(n * sizeof(double));
//     if (!grad) { perror("malloc"); exit(1); }

//     for (int it = 0; it < iterations; it++) {
//         // zero the gradient
//         for (int j = 0; j < n; j++) grad[j] = 0.0;

//         // accumulate gradients
//         for (int i = 0; i < m; i++) {
//             double prediction = 0.0;
//             for (int j = 0; j < n; j++)
//                 prediction += X[i][j] * theta[j];

//             double error = prediction - y[i];
//             for (int j = 0; j < n; j++)
//                 grad[j] += error * X[i][j];
//         }

//         // update parameters
//         for (int j = 0; j < n; j++)
//             theta[j] -= (alpha / m) * grad[j];
//     }

//     free(grad);
// }

// int main(void) {
//     int m = 4;           // samples
//     int n = 2;           // features (including bias term if desired)

//     // allocate X (m x n)
//     double **X = malloc(m * sizeof(double*));
//     for (int i = 0; i < m; i++)
//         X[i] = malloc(n * sizeof(double));

//     // simple dataset: add a column of 1s for bias if needed
//     double rawX[4][2] = {
//         {1.0, 1.0},
//         {1.0, 2.0},
//         {1.0, 3.0},
//         {1.0, 4.0}
//     };
//     double y[4] = {3.0, 5.0, 7.0, 9.0}; // roughly y = 2x + 1

//     for (int i = 0; i < m; i++)
//         for (int j = 0; j < n; j++)
//             X[i][j] = rawX[i][j];

//     double theta[2] = {0.0, 0.0}; // initialize weights

//     gradient_descent(X, y, theta, m, n, 0.01, 10000);

//     printf("Learned parameters:\n");
//     for (int j = 0; j < n; j++)
//         printf("theta[%d] = %.6f\n", j, theta[j]);

//     for (int i = 0; i < m; i++) free(X[i]);
//     free(X);
//     return 0;
// }
