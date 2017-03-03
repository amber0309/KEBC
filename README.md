# KEBC
Kernel embedding-based clustering

Python Code for the KEBC algorithm

Content:

1. Code
2. Usage
3. Contact


-------------------------------------------------------------------------------
----------------------------------- 1. Code -----------------------------------
-------------------------------------------------------------------------------

The code is written in Python (Anaconda 4.3.0 64-bit for Windows).

   * <KEBC class="py"></KEBC>  - python implementation of KEBC
     * KEBC_cond           & KEBC using conditional distributions
     * KEBC_cond_ref       & KEBC using conditional distributions with reference distribution
     * KEBC_marg           & KEBC using marginal distributions

   * <test_KEBC class="py"></test_KEBC>  - a toy synthetic test
     * gen_stat            compute the precision and recall of clustering results
     * Generate_XY         generate synthetic data
     * exp_synth           conduct synthetic experiment

-------------------------------------------------------------------------------
---------------------------------- 2. Usage -----------------------------------
-------------------------------------------------------------------------------

For KEBC

label = KEBC\_cond(XY, label_true, score_flg)

Inputs:
  * XY          input data, list of numpy arrays. rows of each array are i.i.d.
              samples, column of each array represent variables
  * label_true  the ground truth of cluster label of each group
  * score_flg   output score file or not. 1 - yes, 0 - no

Output:
  * label       list of cluster label for each group

For synthetic experiment

exp_synth(n_clu, n_grp)

Inputs:
  * n_clu       number of clusters
  * n_grp       number of groups

-------------------------------------------------------------------------------
--------------------------------- 3. Contact ----------------------------------
-------------------------------------------------------------------------------

E-Mail: 
shoubo.hu@gmail.com
chenzhitang2@huawei.com
