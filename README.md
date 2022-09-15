# Characteristics of FFT
+ Include this [FFT_ex1.py], all examples do FFT over the entire interval. If you change **FFT windows(unit: # of samples or seconds)** , a frequency resolution  decreases by the proportion **Fsamp/FFTwindows**. Eventhough, range of frequency analysis is not changed.

+ [Reference](https://support.ircam.fr/docs/AudioSculpt/3.0/co/Window%20Size.html) 
    + Lowest Detectable Frequency 
        + F0  = 5*(SR/Window Size)
        + ex ) F0 = 5*(44100/1024) ≃ 215 Hz.
        
    + WS = 5*Fsamp/F0
        + F0 = 440 Hz --> WS = 501
        + F0 = 100 Hz --> WS = 2205
        
    + TR = Window Size/Fsamp
        + If window size is big --> time domain is shrinked. 
        
    + FR = Fsamp/Window Size
        + The more bins, the more slices of frequency range we get.

+ **Summary** ([Reference](http://www.add.ece.ufl.edu/4511/references/ImprovingFFTResoltuion.pdf))
    + A greater frequency resolution results in a smaller time resolution. 


# FFt_ex1.py
+ This example code is a reproduction of the code in the reference next.  [Reference](https://www.youtube.com/watch?v=s2K1JfNR7Sc)  

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

+ :worried: Why are the results different with ex1 ?
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

+ :worried: Is this ex3 result same with matlab **positiveFFT** ? Matlab codes are represented below.
    + **Ans** : not perfectly same with the result of ex3's. 
    + **Reason** : Below matlab code is doing 'normalize' as 'X=fft(x)/N*2; % normalize the data'.  
    
                   But python [ex3] code get the magnitude values after 'abs' function('amplitude_Hz = 2*abs(Y)').    
                   
        ```matlab
        % test.m
        clear;clc;
        dt = 0.001; 
        t= 0:dt:1;
        noise = 2.5 .*rand(length(t),1 );
        x= sin(2*pi*50*t) + sin(2*pi*120*t);

        positiveFFT(x',length(x'));

        Magnitude=(abs(ans))';
        Order=0:1:length(Magnitude)-1;
        bar(Order, Magnitude);

        grid on;
        xlabel('Harmonic Order','FontSize',15);
        ylabel('Mag','FontSize',15);
        set(gca,'FontSize',15);
        grid on;
        ```
        
        ```matlab
        % poisitiveFFT.m
        function [X,freq]=positiveFFT(x,Fs)
           N=length(x); %get the number of points
           k=0:N-1;     %create a vector from 0 to N-1
           T=N/Fs;      %get the frequency interval
           freq=k/T;    %create the frequency range
           X=fft(x)/N*2; % normalize the data
         if X(1)~=X(N)
             X(1)=X(1)/2;
         end
           %only want the first half of the FFT, since it is redundant
           cutOff = ceil(N/2); 

           %take only the first half of the spectrum
           X = X(1:cutOff);
           freq = freq(1:cutOff);
        ```
        <img src="https://user-images.githubusercontent.com/71545160/171540248-a3936fcb-4817-4269-9cae-e83d026e732f.png" width="600" height="250">


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

<br>
<br>
<br>

---

<br>
<br>
<br>

# Data save to a text file

+ It makes text file with target data.
+ If you set a below code, you can distinguish the data with lines.

    ```python
        for i in range(len(data1)):
            if i % 10 == 0 and i > 0:
                file_1.write('\n')
                file_1.write(',')
                file_2.write('\n')
                file_2.write(',')
            elif i % 10 != 0 and i > 0:
                file_1.write(',')
                file_2.write(',')
    ```
    
    + All data are saved in string format before saving them.

<img src="https://user-images.githubusercontent.com/71545160/172093975-c392683c-172e-4e7e-9e81-ef69e850b0c0.png" width="400" height="200">
<img src="https://user-images.githubusercontent.com/71545160/172093981-6e573062-2456-4745-8c43-02f6bf4141d9.png" width="400" height="200">


# CompSens (Compressive Sensing)

<img src="https://user-images.githubusercontent.com/71545160/178152292-1062afce-4689-43fc-91a1-da878214c341.png" width="600" height="600">

## real time domain plot 

![image](https://user-images.githubusercontent.com/71545160/178152327-63cacc5b-d73a-4a3f-b1c8-8e739ad1b93f.png)

## DCT domain plot

![image](https://user-images.githubusercontent.com/71545160/178152333-a6939215-8e17-4c19-a35f-e21d03530784.png)


# Spectrogram_plot

If you want to place results in low frequency on bottom, 
active a below code.
```python
spec = np.flip(spec, axis=0)
```

## 100Hz
<img src="https://user-images.githubusercontent.com/71545160/180639722-1fdacb22-c7b0-41d1-a34e-dd1a1609426d.png" width="300" height="300">

## 200Hz
<img src="https://user-images.githubusercontent.com/71545160/180639737-117d2f27-b214-486f-a4b6-186379dcac37.png" width="300" height="300">

## 2000Hz
<img src="https://user-images.githubusercontent.com/71545160/180639738-ce13c8c3-e4a1-4e22-9792-aadac8e3c0eb.png" width="300" height="300">

## 10000Hz
<img src="https://user-images.githubusercontent.com/71545160/180639736-23b6de04-fcb8-48de-9b4f-366044b5862d.png" width="300" height="300">


# wavelet_trans_basic_ex( = Continuous Wavelet Transform(CWT) )

<img src="https://user-images.githubusercontent.com/71545160/180641708-68080060-87b4-40f2-90aa-c730de648585.png" width="600" height="300">

<br>
<br>
<br>

---

<br>
<br>
<br>

# Path Example

<img src="https://user-images.githubusercontent.com/71545160/190367315-8b35e23d-e1fe-44af-afc8-bb57e38bfeaf.png" width="450" height="600">


