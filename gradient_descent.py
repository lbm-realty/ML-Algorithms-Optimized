import numpy as np

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
                print(f"For {iter}th iteration\n\t the values of w and b are: {w[0], w[1], b[0]}")
        
        return w, b

    def test(training_set, weights, bias):
        prediction = [0 for i in range(len(training_set))]

        for i in range(len(training_set)):
            prediction[i] = np.dot(weights, training_set[i]) + bias
        
        return prediction    


x = np.array([
    [1, 2],
    [2, 4],
    [3, 6], 
    [4, 8]
])
y = np.array([7, 13, 19, 25])
# weights = Gradient_Descent.grad_descent(x, y, 100000, 4, 2, 0.01)
# print(weights)
results = Gradient_Descent.test(x, weights=[1.2, 2.4], bias=0.5)
print(results)