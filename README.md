# TableGAN-MCA

TableGAN-MCA Pytorch code

Paper： TableGAN-MCA: Evaluating Membership Collisions of GAN-Synthesized Tabular Data Releasing

ACM Reference Format:

Aoting  Hu,  Renjie  Xie,  Zhigang  Lu,  Aiqun  Hu,  and  Minhui  Xue.  2021.TableGAN-MCA: Evaluating Membership Collisions of GAN-SynthesizedTabular Data Releasing. InProceedings of the 2021 ACM SIGSAC Conference on Computer and Communications Security (CCS ’21), November 15–19,2021, Virtual Event, Republic of Korea. ACM, New York, NY, USA. 17 pages. 

## Introduction

Input: a copy of synthetic datasets

Output: a small dataset that most of it are real.


The basic implementation is shown in notebook TableGAN-MCA.ipynb.

Tabular Datasets: Adult

We support the follwoing net:
-   'WGAN with gradient penalty',
-   'WGAN with weight decay
-   'Differentially private WGAN with weight decay'
-   'CTGANSynthesizer', (Chen et al. 2019. “Modeling Tabular data using Conditional GAN”)
-   'TVAESynthesizer', (Chen et al. 2019. “Modeling Tabular data using Conditional GAN”)

All nets are work with tabular data.



<<<<<<< HEAD
Input: a copy of synthetic datasets

Output: a small dataset that most of it are real.




