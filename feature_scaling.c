#include "feature_scaling.h"
#include <math.h>

void scaled_X(double *X, int rows, int cols) {
    for (int j = 0; j < cols; j++) {
        double mean = 0.0;
        
        for (int i = 0; i < rows; i++)
            mean += X[i * cols + j];
        mean /= rows;
        double std_dev = 0.0;
        
        for (int i = 0; i < rows; i++) {
            double square = pow((X[i * cols + j] - mean), 2.0);
            std_dev += square;
        }
        std_dev /= rows;
        std_dev = sqrt(std_dev);

        for (int i = 0; i < rows; i++) 
            X[i* cols + j] = (X[i * cols + j] - mean) / std_dev;
    }
}

void scaled_Y(double *Y, int rows) {
    for (int i = 0; i < rows; i++) {
        double mean = 0.0;
        mean += Y[i];
        mean /= rows;
        double std_dev = 0.0;
        
        for (int i = 0; i < rows; i++) {
            double square = pow((Y[i] - mean), 2.0);
            std_dev += square;
        }
        std_dev /= rows;
        std_dev = sqrt(std_dev);

        for (int i = 0; i < rows; i++) 
            Y[i] = (Y[i] - mean) / std_dev;
    }
}
