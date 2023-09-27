# Entropy regularized part-of-speech tagger

Part-of-speech tagging, also called grammatical tagging is the process of marking up a word in a text (corpus) as corresponding to a particular part of speech, based on both its definition and its context.
In this work, realized during the NLP course at ETHz, a neural POS tagger with over 90% performace accuracy is developed.
The main characteristics of the project are:
* BERT encodings utilization for taking  context of each word into account.
* Dynamic programming to solve a series of otherwise intractable problems.
* Computation in log-space for numerical stability.
* Entropy regularization to improve performances.

## Modelling and architecture: 

We can model POS tagging task as a path search on a lattice. 

![simple example](POS_modelling.svg)

In particular, we can associate to each edge a score and assign paths (i.e. taggings) with the sum of their edge-scores.

Edge-scores are also additively modelled. Two components are taken into account:
* emissions: representing the likelihood for the 'tag at the end of the edge' to be associated with the corresponding word.
* transitions: modelling tag-tag transitions likelihoods with markovian assumption.

Emissions are computed using a bidirectional LSTM module mounted upon a BERT transformer, while transitions are modelled directly as parameters. 
Both emission network's parameters and transitions are learnt with Adam optimizer.

The loss is formed by negative log-likelihood regularized along with a scaled penalty term given by entropy of the probability distribution returned by the model.

Backward algorithm on different semirings was used to efficiently compute probability normalizer, entropy and the maximum path on the lattice.

## Results
Accuracy on test set for three different $\beta$ values. 

|  $\beta$  | Accuracy  |
|-----------|-----------|
| 0.0       | 0.919     |
| 0.1       | 0.921     |
| **1.0**   | **0.925** |

With $\beta$ being the scaling factor of the entropy term (0 corresponding to regularization absence).

## Dataset
English language version of Universal Dependencies dataset [Nivre et al., 2017].


## USAGE
For ready-to-go usage, you can simply run the [main](main.py) and enter a sentence. Before doing so, you can install the dependencies by running:
```
pip install -r requirements.txt
``` 
