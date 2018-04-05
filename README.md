# Kernel Embedding-Based Clustering (KEBC)

Python code of distribution clustering algorithm using kernel embedding (KEBC).

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

We test the code using **Anaconda 4.3.0 64-bit for python 2.7** on windows. Any later version should still work perfectly.

## Running the tests

After installing allrequired packages, you can run *test_KEBC.py* to see whether ENCI could work normally.

The test does the following:
1. it generate 30 groups of data coming from 2 generative mechanisms and put all those groups in a list.
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

### Notes

Input of function **KEBC_cond_bochner**

| Argument  | Description  |
|---|---|
|XY | input data, list of numpy arrays. rows of each array are i.i.d. samples, column of each array represent variables|
|label_true |  the ground truth of cluster label of each group|
|k |  the length of explicit kernel mapping|
|score_flg | output score file or not. 1 - yes, 0 - no|

Output:Output of function **KEBC_cond_bochner**

| Argument  | Description  |
|---|---|
|label   |    list of cluster label for each group|

If the fisrt column is $X$ and the second is $Y$, the conditional distribution KEBC estimates for each group is $p(X|Y)$.

### Other clustering choices

#### KEBC.py
(Implementation of KEBC using kernel trick)

| Function  | Description  |
|---|---|
|KEBC_cond | KEBC using conditional distributions|
|KEBC_marg | KEBC using marginal distributions|

#### KEBC_bochner.py
(Implementation of KEBC using explicit kernel mapping (Bochner's theorem))

| Function  | Description  |
|---|---|
|KEBC_cond_bochner | KEBC using conditional distributions|
|KEBC_marg_bochner | KEBC using marginal distributions|

## Authors

* **Shoubo Hu** - shoubo.sub@gmail.com
* **Zhitang Chen** - chenzhitang2@huawei.com

See also the list of [contributors](https://github.com/amber0309/KEBC/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
