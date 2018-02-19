# CEM
The implementation for Constrastive Explanations Method (CEM)

To find the pertinent positives (PP), 

```python3
python3 main.py -i 2953 --mode PP --kappa 20 --gamma 100
```
This would find the PP of image 2953 in the MNIST dataset.

![Results_PP_orig](/Results/PP_ID2953_Gamma_100.0/Orig_original5.png)
![Results_PP_delta](/Results/PP_ID2953_Gamma_100.0/Delta_id2953_kappa10.0_Orig5_Adv3_Delta5.png)

From left to right: the original image and the pertinent positive component. This PP in Image 2953 is sufficient to be classified as 5.

To find the pertinent negatives (PN),

```python3
python3 main.py -i 340 --mode PN --kappa 20 --gamma 100
```
This would find the PN of image 340 in the MNIST dataset.

![Results_PN_orig](/Results/PN_ID340_Gamma_100.0/Orig_original3.png)
![Results_PN_delta](/Results/PN_ID340_Gamma_100.0/Delta_id340_kappa10.0_Orig3_Adv5_Delta8.png)
![Results_PN_adv](/Results/PN_ID340_Gamma_100.0/Adv_id340_kappa10.0_Orig3_Adv5_Delta8.png)

From left to right: the original image, the pertinent negative component and the image composed of the original image and PN. If we add PN to Image 340, it would be classified as 5.

The arugment `kappa` and `gamma` are tuning parameters for the optimization setup. For more details, please refer to the paper.
