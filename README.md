# FFt_ex1.py
This example code is a reproduction of the code in the reference next.  [Reference](https://www.youtube.com/watch?v=s2K1JfNR7Sc)  

## Data Shape!  
<img src="https://user-images.githubusercontent.com/71545160/118467187-8f011400-b73e-11eb-82ed-8ab009f1cac3.png" width="600" height="400">

## Frequency Analysis of the Noise - by FFT  
The second graph in the plot below shows the FFT analysis of the noise signal added to the original signal.
The horizontal axis represents the frequency, and the vertical axis represents the magnitude of each frequency.
<img src="https://user-images.githubusercontent.com/71545160/118467129-7e509e00-b73e-11eb-8ac3-78769bea97f5.png" width="600" height="400">

# FFt_ex2.py

```python
def FFT(sample_rate, duration, signal):
    ...
    return xf, yf, phase_ang
```
<img src="https://user-images.githubusercontent.com/71545160/171537184-f8a1506f-8f83-4860-9279-3da5d9ec2600.png" width="600" height="250">

:worried: Why are the results different with ex1 ?
    + In [ex1] code, it obtains **power spectrum** scale value. 
        ```python
        fhat = np.fft.fft(f,n)                     #compute the FFT
        PSD = fhat * np.conj(fhat)/n               #power spectrum 
        freq = (1/(dt*n)) * np.arange(n)           #x-axis of frequencies 
        L = np.arange(1,np.floor(n/2),dtype='int') #only plot 1st half
        ```

# FFt_ex3.py
```python
def FFT(sample_rate, duration, signal):
    ...
    return xf, amplitude_Hz, phase_ang
```
<img src="https://user-images.githubusercontent.com/71545160/171537283-3b8457b5-3ed1-4199-8b29-f673ce47ee0a.png" width="600" height="250">

<br>
<br>
<br>

---

<br>
<br>
<br>

# Sampler-and-Generator.py  
You can do [1], [2] using this file  
[1] Sampling  
Sampling from Github_set.csv  
[2] Generator  
You can generate extra columns like below on the new generated .csv file.  
<img src="https://user-images.githubusercontent.com/71545160/134902379-20d80c31-f13a-4ba8-8d05-af0eb679af55.png" width="600" height="400">  

Columns to be stacked could be set at the part shown below.  
```python
StackedNum = 4               # TobeStacked's copy Num
Samp_Num = 12 * StackedNum   # Sampling from [Github_gener_ex.csv] file.

TobeStacked_all = []

# 12
TobeStacked = np.array([[0, 0, 3, 0, 0, 3], [0, 0, 3, 0, 0, -3], [0, 0, -3, 0, 0, 3], [0, 0, -3, 0, 0, -3],
                        [0, 3, 0, 0, 3, 0], [0, 3, 0, 0, -3, 0], [0, -3, 0, 0, 3, 0], [0, -3, 0, 0, -3, 0],
                        [3, 0, 0, 3, 0, 0], [3, 0, 0, -3, 0, 0], [-3, 0, 0, 3, 0, 0], [-3, 0, 0, -3, 0, 0]])
TobeStacked = TobeStacked.reshape(-1, 6)
TobeStacked = pd.DataFrame(TobeStacked)
```

<br>
<br>
<br>

---

<br>
<br>
<br>

# SVM/SVM_classifier.py
svm_classifier using a scikit learn

If your data group is more than 3, </br> 
you should use below code for ensuring visualization.

```python
z = clf.predict(xy)
```

If your data group is smaller than 3, </br> 
you can also use below code for visualization with rpresentating of margins.

```python
z = clf.decision_function(xy)
```

<br>
<br>
<br>

---

<br>
<br>
<br>

# PCA_Reconstructing > pca_various_plt_test.py

![image](https://user-images.githubusercontent.com/71545160/167357642-268f0f09-f011-479c-b530-e6287b242219.png)

**Below description's [Reference](https://stats.stackexchange.com/questions/229092/how-to-reverse-pca-and-reconstruct-original-variables-from-several-principal-com)**

## How to reverse PCA and reconstruct original variables from several principal components?

PCA computes eigenvectors of the covariance matrix ("principal axes") and sorts them by their eigenvalues (amount of explained variance). The centered data can then be projected onto these principal axes to yield principal components ("scores"). For the purposes of dimensionality reduction, one can keep only a subset of principal components and discard the rest. (See here for a layman's introduction to PCA.)

Let Xraw be the n×p data matrix with n rows (data points) and p columns (variables, or features). After subtracting the mean vector μ from each row, we get the centered data matrix X. Let V be the p×k matrix of some k eigenvectors that we want to use; these would most often be the k eigenvectors with the largest eigenvalues. Then the n×k matrix of PCA projections ("scores") will be simply given by Z=XV.

This is illustrated on the figure below: the first subplot shows some centered data (the same data that I use in my animations in the linked thread) and its projections on the first principal axis. The second subplot shows only the values of this projection; the dimensionality has been reduced from two to one:

![image](https://user-images.githubusercontent.com/71545160/167358002-28cb8479-6a31-4660-b8a1-dcd06ab8bb3e.png)

In order to be able to reconstruct the original two variables from this one principal component, we can map it back to p dimensions with V⊤. Indeed, the values of each PC should be placed on the same vector as was used for projection; compare subplots 1 and 3. The result is then given by X^=ZV⊤=XVV⊤. I am displaying it on the third subplot above. To get the final reconstruction X^raw, we need to add the mean vector μ to that:

PCA reconstruction=PC scores⋅Eigenvectors⊤+Mean
Note that one can go directly from the first subplot to the third one by multiplying X with the VV⊤ matrix; it is called a projection matrix. If all p eigenvectors are used, then VV⊤ is the identity matrix (no dimensionality reduction is performed, hence "reconstruction" is perfect). If only a subset of eigenvectors is used, it is not identity.

This works for an arbitrary point z in the PC space; it can be mapped to the original space via x^=zV⊤.

Discarding (removing) leading PCs

### Discarding (removing) leading PCs

Sometimes one wants to discard (to remove) one or few of the leading PCs and to keep the rest, instead of keeping the leading PCs and discarding the rest (as above). In this case all the formulas stay exactly the same, but V should consist of all principal axes except for the ones one wants to discard. In other words, V should always include all PCs that one wants to keep.

Caveat about PCA on correlation

### Caveat about PCA on correlation

When PCA is done on correlation matrix (and not on covariance matrix), the raw data Xraw is not only centered by subtracting μ but also scaled by dividing each column by its standard deviation σi. In this case, to reconstruct the original data, one needs to back-scale the columns of X^ with σi and only then to add back the mean vector μ.

### Image processing example

This topic often comes up in the context of image processing. Consider Lenna -- one of the standard images in image processing literature (follow the links to find where it comes from). Below on the left, I display the grayscale variant of this 512×512 image.

![image](https://user-images.githubusercontent.com/71545160/167358236-a85e5443-047d-4440-823a-e8e117cdc363.png)

We can treat this grayscale image as a 512×512 data matrix Xraw. I perform PCA on it and compute X^raw using the first 50 principal components. The result is displayed on the right.

