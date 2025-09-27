import numpy as np
import random
import time

class Gradient_Descent:
    
    def grad_descent(x, y, epochs, m, n, alpha):
        w = x.shape[0]
        b = 0.0
        dw, db = 0.0, 0.0
        for iter in range(epochs):
            for i in range(m):
                f_pred = np.dot(w, x[i]) + b
                error = f_pred - y[i]
                dw += error * x[i]
                db += error
            dw /=  m
            db /=  m
            w -= alpha * dw
            b -= alpha * db

            if iter % 10000 == 0:
                print(f"For {iter}th iteration\n\t the values of w and b are: {w, b[0]}")
        
        return w, b

    def test(training_set, weights, bias):
        prediction = [0 for i in range(len(training_set))]

        for i in range(len(training_set)):
            prediction = np.dot(weights, training_set) + bias
        
        return prediction    

start_time = time.time()
weights = Gradient_Descent.grad_descent(x, y, 100000, 100, 8, 0.01)
print(weights)
end_time = time.time()
print(f"Run time for 100K iterations: {end_time - start_time}")
