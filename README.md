# AICA


## Performance

The performance for a basic ANN used for MNIST image recognition is as follows (tested on an Intel i5 4670k):

|          | NO XSIMD | XSIMD   | OPENBLAS |
|----------|----------|---------|----------|
| Training | 40456ms  | 37873ms | 11923ms  |
| Testing  | 949ms    | 941ms   | 632ms    |
| Total    | 41404    | 38814   | 12555ms  |

