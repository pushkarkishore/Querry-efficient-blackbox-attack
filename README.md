# SeqGAN

# Improvement

Now, we can use python 3 and customized data to create new sequences from provided sequence.


## Requirements: 
* **Tensorflow-gpu 2.1.0**
* Python 3.6.13
* CUDA 10.1.243 (For GPU)

## Introduction

We provide example codes to repeat the synthetic data experiments with oracle evaluation mechanisms.
To run the experiment with default parameters:
```
$ python sequence_gan.py
```
You can change the all the parameters in `sequence_gan.py`.

The experiment has two stages. In the first stage, use the positive data provided by the oracle model and Maximum Likelihood Estimation to perform supervise learning. In the second stage, use adversarial training to improve the generator.

After running the experiments, you could get the negative log-likelihodd performance saved in `save/experiment-log.txt` like:
```
pre-training...
epoch:	0	nll:	11.416485
epoch:	5	nll:	11.371629
epoch:	10	nll:	11.412918
epoch:	15	nll:	11.407625
epoch:	20	nll:	11.409526
epoch:	25	nll:	11.418101
epoch:	30	nll:	11.435586
epoch:	35	nll:	11.472502
epoch:	40	nll:	11.456929
epoch:	45	nll:	11.457744
epoch:	50	nll:	11.444353
epoch:	55	nll:	11.458662
epoch:	60	nll:	11.468146
epoch:	65	nll:	11.482325
epoch:	70	nll:	11.495163
epoch:	75	nll:	11.464915
adversarial training...
epoch:	0	nll:	11.502315
epoch:	5	nll:	11.575931
epoch:	10	nll:	11.575464
epoch:	15	nll:	11.61273
epoch:	20	nll:	11.569257
epoch:	25	nll:	11.5669985
epoch:	30	nll:	11.676801
epoch:	35	nll:	11.852644
epoch:	40	nll:	11.847681
epoch:	45	nll:	11.8924055
epoch:	50	nll:	12.061672
epoch:	55	nll:	12.578293
epoch:	60	nll:	12.431416
epoch:	65	nll:	11.905843
epoch:	70	nll:	11.6158905
epoch:	75	nll:	11.663459
epoch:	80	nll:	11.766123
epoch:	85	nll:	12.023353
epoch:	90	nll:	12.189504
epoch:	95	nll:	12.548395
epoch:	100	nll:	14.238763
epoch:	105	nll:	15.610122
epoch:	110	nll:	15.82634
epoch:	115	nll:	15.624449
epoch:	120	nll:	14.803987
epoch:	125	nll:	14.005474
epoch:	130	nll:	13.433034
epoch:	135	nll:	13.1244
epoch:	140	nll:	12.548196
epoch:	145	nll:	12.33089
epoch:	150	nll:	11.556798
epoch:	155	nll:	10.403472
epoch:	160	nll:	9.735615
epoch:	165	nll:	9.510079
epoch:	170	nll:	9.456528
epoch:	175	nll:	9.452542
epoch:	180	nll:	9.467561
epoch:	185	nll:	9.463689
epoch:	190	nll:	9.445531
epoch:	195	nll:	9.430949
epoch:	199	nll:	9.425596
```

Note: this code is based on the https://github.com/LantaoYu/SeqGAN and https://github.com/ofirnachum/sequence_gan. Thanks to LantaoYu and ofirnachum.
