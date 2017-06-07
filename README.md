# KEBC (Kernel Embedding-based Clustering)

Python Code for the KEBC algorithm
(Anaconda 4.3.0 64-bit for python 2.7 on Windows).

Content:

1. Code
2. Usage
3. Contact


-------------------------------------------------------------------------------
----------------------------------- 1. Code -----------------------------------
-------------------------------------------------------------------------------

**KEBC.py**  - Implementation of KEBC using kernel trick

| Function  | Description  |
|---|---|
|KEBC_cond | KEBC using conditional distributions|
|KEBC_cond_ref |  KEBC using conditional distributions with reference distribution|
|KEBC_marg | KEBC using marginal distributions|

**KEBC_bochner.py**  - Implementation of KEBC using explicit kernel mapping (Bochner's theorem)

| Function  | Description  |
|---|---|
|KEBC_cond_bochner | KEBC using conditional distributions|
|KEBC_marg_bochner | KEBC using marginal distributions|

**test_KEBC.py**  - a toy synthetic test

| Function  | Description  |
|---|---|
|gen_stat | compute the precision and recall of clustering results|
|Generate_XY  |  generate synthetic data|
|exp_synth | conduct synthetic experiment|

-------------------------------------------------------------------------------
---------------------------------- 2. Usage -----------------------------------
-------------------------------------------------------------------------------

### KEBC (KEBC.py)

~~~~
label = KEBC_cond(XY, label_true, score_flg)
label = KEBC_cond_ref(XY, label_true, score_flg)
label = KEBC_marg(XY, label_true, score_flg)
~~~~

Inputs:

| Argument  | Description  |
|---|---|
|XY | input data, list of numpy arrays. rows of each array are i.i.d. samples, column of each array represent variables|
|label_true |  the ground truth of cluster label of each group|
|score_flg | output score file or not. 1 - yes, 0 - no|

Output:

| Argument  | Description  |
|---|---|
|label   |    list of cluster label for each group|

### KEBC_bochner (KEBC_bochner.py)

~~~~
label = KEBC_cond_bochner(XY, label_true, k, score_flg)
label = KEBC_marg_bochner(XY, label_true, k, score_flg)
~~~~

Inputs:

| Argument  | Description  |
|---|---|
|XY | input data, list of numpy arrays. rows of each array are i.i.d. samples, column of each array represent variables|
|label_true |  the ground truth of cluster label of each group|
|k |  the length of explicit kernel mapping|
|score_flg | output score file or not. 1 - yes, 0 - no|

Output:

| Argument  | Description  |
|---|---|
|label   |    list of cluster label for each group|

### Synthetic experiment (test_KEBC.py)

~~~~
exp_synth(n_clu, n_grp)
~~~~

Inputs:

| Argument  | Description  |
|---|---|
|n_clu |    number of clusters|
|n_grp |    number of groups|


-------------------------------------------------------------------------------
--------------------------------- 3. Contact ----------------------------------
-------------------------------------------------------------------------------

E-Mail: 
shoubo.hu@gmail.com
chenzhitang2@huawei.com
