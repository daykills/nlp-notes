 Multiclass Classification

 - Multiclass Perceptron

predict ŷ = argmax_k w_k · x  
if ŷ != y:
    w_y      += x
    w_ŷ      -= x
# else: no update
