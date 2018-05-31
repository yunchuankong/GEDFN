# GEDFN

GEDFN: graph-embedded deep feedforward networks - Tensorflow implementation

The method is introduced in https://academic.oup.com/bioinformatics/advance-article-abstract/doi/10.1093/bioinformatics/bty429/5021680?redirectedFrom=fulltext.

## Prerequisites

The following packages are required for executing the main code file:

* NumPy http://www.numpy.org/
* Scikit-learn http://scikit-learn.org/stable/install.html
* Tensorflow https://www.tensorflow.org/install/

## Usage

### Data formats

* data matrix (example_expression.csv): a csv file with n rows and p+1 columns. n is the number of samples and p is the number of features (continuous variables, such as gene expression values). The additional column at last is the 0/1 binary outcome variable vector. n=100 and p=500 for the example dataset.
* feature graph (example_adjacency.txt): a txt file with p rows and p colunms, which is the corresponding adjacency matrix of the feature graph.

NOTE: no headers are allowed in both files.

### Run GEDFN

In the terminal, change the directory to the folder under which main.py is located, then type the command

```
 python main.py "example_expression.csv" "example_adjacency.txt" "var_impo.csv"
```

where var_impo.csv is the output file for variable importance and will be created by the program automatically. The program will run while printing logs

```
Epoch: 1 cost = 0.619800305 Training accuracy: 0.5  Training auc: 0.658
Epoch: 2 cost = 0.620009381 Training accuracy: 0.5  Training auc: 0.728
Epoch: 3 cost = 0.610391283 Training accuracy: 0.5  Training auc: 0.782
......
Epoch: 71 cost = 0.142398462 Training accuracy: 0.988  Training auc: 0.999
Epoch: 72 cost = 0.126102197 Training accuracy: 0.988  Training auc: 0.999
Epoch: 73 cost = 0.116139328 Training accuracy: 0.988  Training auc: 1.0
Epoch: 74 cost = 0.121380727 Training accuracy: 0.988  Training auc: 1.0
Epoch: 75 cost = 0.127119239 Training accuracy: 1.0  Training auc: 1.0
Epoch: 76 cost = 0.097086006 Training accuracy: 1.0  Training auc: 1.0
Early stopping.
*****===== Testing accuracy:  0.85  Testing auc:  0.94 =====*****
```

and the var_impo.csv file is seen in this repo.

### Hyperparameters and training options

Seen in the main.py.
