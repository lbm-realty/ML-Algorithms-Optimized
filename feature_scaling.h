#ifndef FEATURE_SCALING_H
#define FEATURE_SCALING_H


void z_score_normalized_X(double *X, int rows, int cols);
void z_score_normalized_Y(double *Y, int rows);
void min_max_scaled_X(double *X, int rows, int cols);
void min_max_scaled_Y(double *Y, int rows);

#endif