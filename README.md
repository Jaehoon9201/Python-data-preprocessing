# FFt.py
This example code is a reproduction of the code in the reference next.  [Reference](https://www.youtube.com/watch?v=s2K1JfNR7Sc)  

## Data Shape!  
<img src="https://user-images.githubusercontent.com/71545160/118467187-8f011400-b73e-11eb-82ed-8ab009f1cac3.png" width="600" height="400">

## Frequency Analysis of the Noise - by FFT  
The second graph in the plot below shows the FFT analysis of the noise signal added to the original signal.
The horizontal axis represents the frequency, and the vertical axis represents the magnitude of each frequency.
<img src="https://user-images.githubusercontent.com/71545160/118467129-7e509e00-b73e-11eb-8ac3-78769bea97f5.png" width="600" height="400">

---

# Sanmpler-and-Generator.py  
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

---

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

---

# PCA_restoring > pca_various_plt_test.py

![image](https://user-images.githubusercontent.com/71545160/167357642-268f0f09-f011-479c-b530-e6287b242219.png)
