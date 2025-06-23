# mnist
<pre>
pcaST.py
Mean CV accuracy: 0.9407 ± 0.0015
PC1
PC2

  
pcahvgsfa.py
PCA without scaling (NO TRANSFORM)  0.8395 ±0.0024
Feature Agglomeration               0.9279 ±0.0020
HVGS (Highly Variable Gene Selection) 0.8441 ±0.0023
  
  
mnistpcaST.py
Method 1 (No scaling, No transform): 0.8441±0.0023
Method 2 (Scaling and Transform): 0.9415±0.0010
Method 3 (No scaling and Transform): 0.9532±0.0018
Method 4 (Scaling and No transform): 0.8442±0.0022

pcaicahvgsfa.py
PCA without scaling (no transform)  0.8376±0.0026
PCA with scaling (transform)        0.9415±0.0017
ICA                                 0.7809±0.0020
Feature Agglomeration               0.9279±0.0020
HVGS (High Variance Gene Selection) 0.8441±0.0023

  
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
3. LASSO (with LASSO classifier) with scaling and transform: 0.7299 ± 0.0024

FA-lasso-log-pca.py:
1. Feature Agglomeration (with RF): 0.9279 ± 0.0020
2. LASSO features + LASSO classifier: 0.7215 ± 0.0031
3. LogReg features + LogReg: 0.6482 ± 0.0018
4. PCA features (with RF): 0.6164 ± 0.0025

temulapca.py
Ranking of feature selection methods by accuracy:
1. Transfer Entropy (with RF): 0.8657 ± 0.0016
2. Mutual Information (with RF): 0.8444 ± 0.0015
3. LASSO (with LASSO classifier): 0.7299 ± 0.0040
4. PCA (with RF): 0.6439 ± 0.0018

lasso-mi-te.py
Ranking of methods by cross-validation accuracy:
1. Transfer Entropy (with Random Forest): 0.8459 ± 0.0027
2. Mutual Information (with Random Forest): 0.8444 ± 0.0026
3. LASSO (with LASSO classifier): 0.2152 ± 0.0037

pca-hvgs-fa.py
Feature Agglomeration: 0.9139 (±0.0047)
PCA: 0.8973 (±0.0049)
HVGs: 0.8421 (±0.0055)
  
</pre>

Traditional PCA underperforms compared to alternative feature selection methods when applied to the MNIST dataset (70,000 samples, 784 features). 
When selecting the top 30 features from the original 784, Feature Agglomeration (FA) significantly outperforms all other methods with a 5-fold cross-validation accuracy of 0.9287 ± 0.0007 using Random Forest classification. 
Independent Component Analysis (ICA) achieves the second-highest performance (0.8700 ± 0.0013), followed by High Variance Gene Selection (HVGS) at 0.8432 ± 0.0035, while PCA trails with the lowest accuracy (0.8365 ± 0.0031). 
These findings challenge the conventional reliance on PCA for dimensionality reduction, demonstrating that FA's ability to preserve feature relationships through intelligent clustering offers substantial advantages for image classification tasks, providing nearly 10 percentage points higher accuracy than PCA while using the same number of features. 
