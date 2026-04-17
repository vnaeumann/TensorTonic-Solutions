import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    b=0
    x = np.array(X)
    w = np.ones(x.shape[1])
    N = x.shape[0]
    
    for epoch in range(steps):
        ypred = _sigmoid(x@w + b)
        eps = 1e-9
        ypred = np.clip(ypred, eps, 1 - eps)
        loss = -np.mean((y*np.log(ypred)) + (1-y)*np.log(1-ypred))
        dw = x.T @ (ypred-y) / N
        db = np.mean(ypred-y)
        w = w - lr*dw
        b = b - lr*db
        
        
    
    
    return w,b