 Multiclass Classification

 - Multiclass Perceptron

predict ŷ = argmax_k w_k · x  
if ŷ != y:
    w_y      += x
    w_ŷ      -= x

```
initialize w[1..K] = 0
for epoch = 1..T:
    shuffle(training_set)
    for (x, y) in training_set:
        y_hat = argmax_k dot(w[k], x)       
        if y_hat != y:
            w[y]     = w[y]     + x
            w[y_hat] = w[y_hat] - x
return w
```
