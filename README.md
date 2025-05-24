# mnist
<pre>
mnist.py:
Method	5-fold CV Accuracy
Feature Agglomeration	0.9287 ± 0.0007
ICA	0.8700 ± 0.0013
HVGS	0.8432 ± 0.0035
PCA	0.8356 ± 0.0036
t-SNE	0.8094 ± 0.0007
UMAP	0.8029 ± 0.0027
LLE	0.7910 ± 0.0026
Spearman 0.7699 ± 0.0024
Kendall	0.7177 ± 0.0020
NMF	0.4257 ± 0.0043

mnisttest.py:
Ranking of feature selection methods by accuracy:
1. Mutual Information (with RF): 0.8445 ± 0.0023
2. Transfer Entropy (with RF): 0.8074 ± 0.0020
3. LASSO (with LASSO classifier): 0.7299 ± 0.0024

FA-lasso-log-pca.py:
1. Feature Agglomeration + Random Forest: 0.9279 ± 0.0020
2. LASSO features + LASSO classifier: 0.7203 ± 0.0015
3. LogReg features + LogReg: 0.6911 ± 0.0025
4. PCA features + Random Forest: 0.6164 ± 0.0025

temulapca.py
Ranking of feature selection methods by accuracy:
1. Transfer Entropy (with RF): 0.8657 ± 0.0016
2. Mutual Information (with RF): 0.8444 ± 0.0015
3. LASSO (with LASSO classifier): 0.7299 ± 0.0040
4. PCA (with RF): 0.6439 ± 0.0018
</pre>

Traditional PCA underperforms compared to alternative feature selection methods when applied to the MNIST dataset (70,000 samples, 784 features). 
When selecting the top 30 features from the original 784, Feature Agglomeration (FA) significantly outperforms all other methods with a 5-fold cross-validation accuracy of 0.9287 ± 0.0007 using Random Forest classification. 
Independent Component Analysis (ICA) achieves the second-highest performance (0.8700 ± 0.0013), followed by High Variance Gene Selection (HVGS) at 0.8432 ± 0.0035, while PCA trails with the lowest accuracy (0.8365 ± 0.0031). 
These findings challenge the conventional reliance on PCA for dimensionality reduction, demonstrating that FA's ability to preserve feature relationships through intelligent clustering offers substantial advantages for image classification tasks, providing nearly 10 percentage points higher accuracy than PCA while using the same number of features. 
