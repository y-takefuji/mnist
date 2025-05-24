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
Spearman	0.7699 ± 0.0024
Kendall	0.7177 ± 0.0020
NMF	0.4257 ± 0.0043
</pre>



mnisttest.py:

FA-lasso-log-pca.py:

Traditional PCA underperforms compared to alternative feature selection methods when applied to the MNIST dataset (70,000 samples, 784 features). 
When selecting the top 30 features from the original 784, Feature Agglomeration (FA) significantly outperforms all other methods with a 5-fold cross-validation accuracy of 0.9287 ± 0.0007 using Random Forest classification. 
Independent Component Analysis (ICA) achieves the second-highest performance (0.8700 ± 0.0013), followed by High Variance Gene Selection (HVGS) at 0.8432 ± 0.0035, while PCA trails with the lowest accuracy (0.8365 ± 0.0031). 
These findings challenge the conventional reliance on PCA for dimensionality reduction, demonstrating that FA's ability to preserve feature relationships through intelligent clustering offers substantial advantages for image classification tasks, providing nearly 10 percentage points higher accuracy than PCA while using the same number of features. 
