Adversarial Attack to Capsule Networks
===================================

This repo. is to research adversarial attack performance for CapsNets.

In those days, deep learning has shown an attractive results on several applications (speech / image recognization). However, from 2014, there are research papers that deep learning can be easiy fool by very low noise. In NIPS 2017 workshop, Ian Goodfellow opened adversarial non-targeted/targeted attack and defense competition for this problem, and In ICLR2018, lots of researches for adversarial examples are submitted. In this situation, we read G. Hinton's paper, Dynamic routing between capsules, and tried to validate the robustness of CapsNets on the paper to well-known adversarial attack.

**The CapsNets part on this repo. is based on InnerPeace-Wu's ones, https://github.com/InnerPeace-Wu/CapsNet-tensorflow.**

Settings
----------

This implementation is on tensorflow 1.2.1, and the detailed setting is followed InnerPeace-Wu's one.

Attackers
----------------

### Fast Gradient Sign Method (FGSM)

Fast Gradient Sign Method is to generate adversarial images with gradient information from loss function and true label.

https://arxiv.org/pdf/1607.02533.pdf

`
python attack_gsm.py --max_iter=1
`

This module is based on gongzhitaao's git and sample attack codes from Google Brain.

https://github.com/gongzhitaao/tensorflow-adversarial

### Basic Iteration (basic iter.)

Basic Iteration is to generate adversarial images with iteratively running FGSM.

https://arxiv.org/pdf/1607.02533.pdf

`
python attack_gsm.py --max_iter=<iter_num>
`

This module is based on gongzhitaao's git and sample attack codes from Google Brain.

https://github.com/gongzhitaao/tensorflow-adversarial

### Least-likely Class Method (step l.l)

Least-likely Class Method is to generate adversarial images by not using true label but "least-likely" label.

https://arxiv.org/pdf/1607.02533.pdf

`
python attack_llcm.py --max_iter=1
`

This module is based on gongzhitaao's git and sample attack codes from Google Brain.

https://github.com/gongzhitaao/tensorflow-adversarial

### Iterative Least-likely Class Method (iter. l.l)

Iterative Least-likely Class Method is to generate adversarial images with iteratively running step l.l.

https://arxiv.org/pdf/1607.02533.pdf

`
python attack_llcm.py --max_iter=<iter_num>
`

This module is based on gongzhitaao's git and sample attack codes from Google Brain.

https://github.com/gongzhitaao/tensorflow-adversarial

### Adversarial Attack based on adversarial generative learning(AGL)

This module is to generate adversairal noise can make the model mis-predict data added with that.

In here it, three loss terms are as bellowed.

1. reverse cross-entropy for true label (rce) => getting insight from FGSM
2. cross-entropy for the lease label (ce_ll) => getting insight from step l.l
3. cross-entropy for top-1 selected label exept true label (ce_s) => new things

`
python attack_gan.py
`

The Agents
----------------

### Naiive conv model (Baseline)

This module is not consisted with adversarial prior to defense.

The structure of this is as bellowed.

3 convolutional layers with Relu activation function, the kernel and feature size of which are 3 and [64,128,256], respectively.
After 3 conv. layers, 2 linear layers with Relu and softmax activation function are used to get prediction of them.

The accuracy for MNIST original test data is about 99%.

### Adversarial Training with adversarial examples from FGSM and Basic iter.

This module is to regularize baseline model with FGSM and Basic iter.

On each epoch, adversarial examples are repeatly generated and learned with the agent.

### Adversarial Training with adversarial examples from step l.l and iter. l.l

This module is to regularize baseline model with step l.l.

On each epoch, adversarial examples are repeatly generated and learned with the agent.

Experiments Results (MNIST)
----------------------------

- CapsNets is learned with 1.5K iteration, and training/test acc. are 0.995/0.991, respectively.

- Epsilon is calculated with infinity norm

- Max epoch is 50

- Max iter. for basic iter. and iter. l.l is 5.

- gif file is adversarial examples with max_epsilon from 5 to 50

### FGSM

![alt tag](https://github.com/jaesik817/adv_attack_capsnet/blob/master/figures/aaa.PNG)

FGSM|0 |5 |10 |15 |20 |25 |30 |35 |40 |45 |50 
----|--|--|---|---|---|---|---|---|---|---|--
naiive model|0.990 |0.984 |0.966 |0.941 |0.920 |0.883 |0.826 |0.759 |0.683 |0.606 |0.529 
adv. Training with FGSM|0.992 |0.988 |0.984 |0.984 |0.979 |0.975 |0.958 |0.965 |0.960 |0.957 |0.955 
adv. Training with basic iter.|0.992 |0.989 |0.986 |0.977 |0.975 |0.974 |0.971 |0.969 |0.962 |0.964 |0.958 
adv. Training with step l.l|0.992 |0.987 |0.978 |0.973 |0.966 |0.958 |0.955 |0.949 |0.928 |0.942 |0.934 
adv. Training with iter l.l|0.992 |0.986 |0.980 |0.967 |0.958 |0.968 |0.961 |0.959 |0.953 |0.946 |0.942 
CapsNet|0.991 |0.975 |0.942 |0.901 |0.854 |0.812 |0.785 |0.765 |0.749 |0.736 |0.721 

![alt tag](https://github.com/jaesik817/adv_attack_capsnet/blob/master/figures/FGSM.gif)

### Basic iter.

![alt tag](https://github.com/jaesik817/adv_attack_capsnet/blob/master/figures/basic_iter.PNG)

Basic iter.|0 |5 |10 |15 |20 |25 |30 |35 |40 |45 |50 
-----------|--|--|---|---|---|---|---|---|---|---|--
naiive model|0.990 |0.986 |0.974 |0.952 |0.915 |0.866 |0.807 |0.736 |0.681 |0.647 |0.623 
adv. Training with FGSM|0.992 |0.989 |0.987 |0.987 |0.982 |0.966 |0.966 |0.960 |0.959 |0.950 |0.944 
adv. Training with basic iter.|0.992 |0.989 |0.989 |0.982 |0.979 |0.977 |0.977 |0.974 |0.971 |0.970 |0.965 
adv. Training with step l.l|0.992 |0.989 |0.982 |0.979 |0.973 |0.966 |0.959 |0.946 |0.924 |0.934 |0.915 
adv. Training with iter l.l|0.992 |0.988 |0.984 |0.974 |0.968 |0.972 |0.965 |0.962 |0.955 |0.947 |0.942 
CapsNet|0.991 |0.975 |0.936 |0.877 |0.800 |0.721 |0.647 |0.579 |0.517 |0.465 |0.423 

![alt tag](https://github.com/jaesik817/adv_attack_capsnet/blob/master/figures/basic_iter.gif)

### step l.l

![alt tag](https://github.com/jaesik817/adv_attack_capsnet/blob/master/figures/step_ll.PNG)

step l.l|0 |5 |10 |15 |20 |25 |30 |35 |40 |45 |50 
--------|--|--|---|---|---|---|---|---|---|---|--
naiive model|0.990 |0.988 |0.982 |0.971 |0.951 |0.923 |0.883 |0.817 |0.740 |0.637 |0.511 
adv. Training with FGSM|0.992 |0.991 |0.990 |0.991 |0.988 |0.975 |0.980 |0.983 |0.980 |0.980 |0.978 
adv. Training with basic iter.|0.992 |0.989 |0.990 |0.984 |0.983 |0.984 |0.985 |0.981 |0.981 |0.981 |0.978 
adv. Training with step l.l|0.992 |0.989 |0.987 |0.987 |0.985 |0.982 |0.979 |0.973 |0.967 |0.972 |0.970 
adv. Training with iter l.l|0.992 |0.989 |0.988 |0.985 |0.983 |0.977 |0.977 |0.973 |0.971 |0.969 |0.966 
CapsNet|0.991 |0.991 |0.979 |0.946 |0.880 |0.797 |0.702 |0.619 |0.546 |0.484 |0.431 

![alt tag](https://github.com/jaesik817/adv_attack_capsnet/blob/master/figures/step_ll.gif)

### iter l.l

![alt tag](https://github.com/jaesik817/adv_attack_capsnet/blob/master/figures/iter_ll.PNG)

iter l.l|0 |5 |10 |15 |20 |25 |30 |35 |40 |45 |50 
--------|--|--|---|---|---|---|---|---|---|---|--
naiive model|0.990 |0.989 |0.986 |0.979 |0.964 |0.942 |0.897 |0.837 |0.754 |0.650 |0.550 
adv. Training with FGSM|0.992 |0.991 |0.991 |0.992 |0.990 |0.980 |0.985 |0.986 |0.985 |0.981 |0.978 
adv. Training with basic iter.|0.992 |0.990 |0.991 |0.987 |0.987 |0.987 |0.988 |0.986 |0.984 |0.985 |0.983 
adv. Training with step l.l|0.992 |0.990 |0.990 |0.989 |0.988 |0.987 |0.985 |0.977 |0.973 |0.974 |0.969 
adv. Training with iter l.l|0.992 |0.990 |0.989 |0.987 |0.987 |0.982 |0.982 |0.980 |0.977 |0.976 |0.975 
CapsNet|0.991 |0.991 |0.985 |0.964 |0.933 |0.895 |0.854 |0.819 |0.787 |0.755 |0.733 

![alt tag](https://github.com/jaesik817/adv_attack_capsnet/blob/master/figures/iter_ll.gif)

Discussion
-------------

CapsNet also fall in the trap of every type adversarial examples. This structure can be one of the hint to solve adversarial problem, however experiment results show CapsNets is not free to adversarial attack.
