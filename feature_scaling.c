#include "feature_scaling.h"
#include <math.h>

void z_score_normalized_X(double *X, int rows, int cols) {
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

void z_score_normalized_Y(double *Y, int rows) {
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

void min_max_scaled_X(double *X, int rows, int cols) {
    // Figuring the max and the min
    double min = X[0], max = X[0];

    for (int i = 0; i < (rows * cols); i++) {
        if (X[i] < min)
            min = X[i];
        if (X[i] > max)
            max = X[i];
    }

    // Normalizing
    for (int i = 0; i < (rows * cols); i++) {
        double numerator = X[i] - min;
        double denominator = max - min;
        X[i] = numerator / denominator;
    }
    
}

void min_max_scaled_Y(double *Y, int rows) {
    // Figuring the max and the min
    double min = Y[0], max = Y[0];

    for (int i = 0; i < rows; i++) {
        if (Y[i] < min)
            min = Y[i];
        if (Y[i] > max)
            max = Y[i];
    }

    // Normalizing
    for (int i = 0; i < rows; i++) {
        double numerator = Y[i] - min;
        double denominator = max - min;
        Y[i] = numerator / denominator;
    }
}