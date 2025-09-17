#include <stdio.h>

double *find_weights(const double *x, const double *y, int epochs, int rows, int cols, int size, int alpha);

int main() {
    const double X[][4] = { {12, 23, 32, 45}, {54, 45 ,34, 54} };
    const double Y[] = {3, 4, 7, 5};
    find_weights(X, Y, 10, 2, 4, 8, 0.1);
}

double *find_weights(const double *x, const double *y, int epochs, int rows, int cols, int size, int alpha) {
    double w = 0.0, b = 0.0;
    double result[] = {w, b};
    double dw = 0, db = 0;

    for (int iter = 0; iter < 1; iter++) {
        double error_sum_dw = 0.0, error_sum_db = 0.0;

        for (int j = 0; j < size; j++) {
            double f_pred = w * (*(&x[j])) + b;
            double error_dw = (y[j % 4] - f_pred) * (*(&x[j]));
            double error_db = (y[j % 4] - f_pred);
            error_sum_dw += error_dw;
            error_sum_db += error_db;
        }
        
        dw += error_sum_dw / (size / 4);
        db += error_sum_db / (size / 4);
        w = w - alpha * dw;
        b = b - alpha * db;
    }

    result[0] = w, result[1] = b;
    printf("%lf %lf\n", w, b);
    return result;
}


// printf("%lf %d %lf\n", *(&x[j]), x[j], &x[j]); 
