# Learning-Hypergraphs-From-Signals-With-Dual-Smoothness-Prior
This is the repo for our ICASSP 2023 paper: [Learning Hypergraphs From Signals With Dual Smoothness Prior](https://arxiv.org/pdf/2211.01717).

## Recommend Environment:
python 3.7.10

pytorch 1.5.1

## Running Experiments:
```
python main.py
```

There are four hyperparameters that may need to be fine-tuned for different datasets: alpha, beta, step_size, and threshold.

In our experiments, we conducted a grid search for alpha and beta, ranging from 1e-3 to 1e+3, for step_size, ranging 
from 1e-4 to 1, and for threshold, ranging from 1e-4 to 5e-1.
