def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.


    """
    x = x0
    w=0
    bb = 0
    f = a*(x**2) + (b*x) + c
    df = 2*a*x + b
    
    for step in range(steps):
        if abs(df) < 1e-8:
            break
            
        x = x - lr*df
        df = 2*a*x + b



    return x
    
        
