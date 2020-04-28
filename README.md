# sigvae-torch
A Pytorch implementation of http://papers.nips.cc/paper/9255-semi-implicit-graph-variational-auto-encoders

The work is re-implemented with python3/3.6.3 and torch 1.4.0

# Updates 
### (w.r.t. the original [implementation](https://github.com/sigvae/SIGraphVAE) )
We had a minor adjustment on the encoder structure, namely, instead of using individual network branches to produce mu and sigma, we let them share the first hidden layer. This update on the encoder cuts down redundant network weights and improves the model performance. The options of encoder structure is coded up in the argument "encsto", the encoder stochasticity. Set it to 'full' to inject randomness into both mu and sigma, and produces different sigma for all (K+J) outputs. Set it to 'semi' so that sigma is produced deterministically from node features.  

# Usage
For example, run sigvae-torch on cora dataset with bernoulli-poisson decoder, and semi-stochastic encoder with the following command
```
>>>python train.py --dataset-str cora --gdc bp --encsto semi
```
The arguments are default to ones that yield optimal results.

# Acknowledgement
This work is developed from https://github.com/zfjsail/gae-pytorch, thank you [@zfjsail](https://github.com/zfjsail) for sharing!
Also appreciate the technical support from [@Chaojie](https://chaojiewang94.github.io/).
