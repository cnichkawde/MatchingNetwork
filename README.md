# Matching Networks implementation in Keras
Implementation of [Matching Networks for One Shot Learning](https://arxiv.org/abs/1606.04080) in Keras

In order to train a 5-way 1-shot model run:
```
python matchingnetwork.py
```
Train a model with Full Context Embedding (FCE) defined as Siamese like pairwise interactions with max pooling:
```
python matchingnetworkwithrelationalembedding.py
```

## References
[1] **Matching Networks for One Shot Learning**, Oriol Vinyals, Charles Blundell, Timothy Lillicrap, Koray Kavukcuoglu and Daan Wierstra, https://arxiv.org/abs/1606.04080, 2016 <br/>
[2] **Siamese network for one shot-image recognition**, G Koch, R Zemel, and R Salakhutdinov, ICML Deep Learning workshop, 2015 <br/>
[3] **A simple neural network module for relational reasoning**, Adam Santoro, David Raposo, David G.T. Barrett, Mateusz Malinowski, Razvan Pascanu, Peter Battaglia and Timothy Lillicrap, https://arxiv.org/abs/1706.01427, 2017
