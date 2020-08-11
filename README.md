# Kernel Embedding-Based Clustering (KEBC)

Python implementation of the following paper

[Model-free Inference of Diffusion Networks using RKHS Embeddings.](https://link.springer.com/article/10.1007/s10618-018-00611-1)  
Shoubo Hu, Bogdan Cautis, Zhitang Chen, Laiwan Chan, Yanhui Geng, Xiuqiang He.  
*Data Mining and Knowledge Discovery* 33, no. 2 (2019): 499-525.

A brief description of KEBC algorithm is available at [https://amber0309.github.io/2017/06/07/kebc/](https://amber0309.github.io/2017/06/07/kebc/).

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites
- NumPy
- SciPy
- scikit-learn

We test the code using **Anaconda 4.3.0 64-bit for python 2.7** on windows. Any later version should still work perfectly.

## Running the tests

After installing allrequired packages, you can run *test_KEBC.py* to see whether ENCI could work normally.

The test does the following:
1. it generate 50 groups of data coming from 2 generative mechanisms and put all those groups in a list.
(Each group is a L by 2 *numpy array* where L is the number of points in the current group.)
2. KEBC is applied on the generated data set to form clusters.

## Apply on your data

### Usage

Import KEBC using

```
from KEBC_bochner import KEBC_cond_bochner, membership
```

Apply KEBC on your data
```
label = KEBC_cond_bochner(XY, label_true, k, score_flg)
```

### Description

Input of function **KEBC_cond_bochner()**

| Argument  | Description  |
|---|---|
|XY | input data, list of numpy arrays. rows of each array are i.i.d. samples, column of each array represent variables|
|label_true |  the ground truth of cluster label of each group|
|k |  the length of explicit kernel mapping|
|score_flg | output score file or not. 1 - yes, 0 - no|

Output of function **KEBC_cond_bochner()**

| Argument  | Description  |
|---|---|
|label   |    list of cluster label for each group|

**NOTE**: If the fisrt column is $X$ and the second is $Y$, the conditional distribution KEBC estimates for each group is $p(X|Y)$.

### Other function choices for clustering

#### KEBC.py
(Implementation of KEBC using kernel trick)

| Function  | Description  |
|---|---|
|KEBC_cond() | KEBC using conditional distributions|
|KEBC_marg() | KEBC using marginal distributions|

#### KEBC_bochner.py
(Implementation of KEBC using explicit kernel mapping (Bochner's theorem))

| Function  | Description  |
|---|---|
|KEBC_cond_bochner() | KEBC using conditional distributions|
|KEBC_marg_bochner() | KEBC using marginal distributions|

## Authors

* **Shoubo Hu** - shoubo DOT sub AT gmail DOT com
* **Zhitang Chen** - chenzhitang2 AT huawei DOT com

See also the list of [contributors](https://github.com/amber0309/KEBC/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
