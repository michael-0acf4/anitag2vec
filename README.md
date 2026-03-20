# anitag2vec

anitag2vec is a vector embedding primarily focused on Danbooru, Pixiv, MAL, etc type of tags.

# Why?

If you have your own local gallery or index of things you like, which, to be fair you most likely probably don't BUT having a recommendation system is quite laborious without a fuzzy component to it.

I mean, sure you can do tag based statistics but you will have to manually group similar tags and somehow also account for spelling variation. With a vector embedding, problem solved! Just pin something you like then get recommended co**similar** stuff.

There are many off-the-shelf vector embeddings, but they are primarily designed for general-purpose tasks such as sentence embeddings. While you can still adapt them for other use cases, many models are sensitive to token order and the exact phrasing of inputs.

# Architecture

## Layout
Fundamentally, creating embeddings for a list of tags is about mapping an unordered set of items into a $N$ dimensional vector.

The task is quite similar to what [Deep Sets](https://arxiv.org/abs/1703.06114) paper do:

$$
M: T \rightarrow \mathbb{R} ^ N
$$

with function/model $M$ to be permutation invariant

$$
M(T) = \phi_2 (\sum_{t \in T} {\phi_1 (t)}) 
$$

Which sounds reasonable but the issue is that most tags will have various spellings such as 1girl ~ 女の子 ~ girl, meaning you'd still have to group similar items somehow (I suppose you can use a dedicated word embedding for that task but that gives us another dependency).

**anitag2vec** assumes a simple transformer encoder architecture. We can drop positional encoding as order doesn't really matter since we care only about how each tag correlates/attends to each other within the set.

For the token embeddings, I use a [BPE encoder](https://en.wikipedia.org/wiki/Byte-pair_encoding).


```
                    batch of tags 
                         |
                   [BPE encoder]
                         |
                [batch of token ids]
                         |
                       (B, I)
                         |
                   [ Embedding LUT ]
                      (B, I, I)
                         |
   [  Transformer(d_model=D, nheads=.., nlayers=..)  ]
                      (B, D, D)
                    /           \
                   /             \
          [ Mean Pool ]         [ Max Pool ]
        context extract       highlight extract
            (B, D)                 (B, D)
                   \             /
                     [   (+)   ]
                       (B, 2D)
                          |
                      [ Linear ]
                          |
                        (B, O)
```

## Training and loss function

The truth is in the distribution, it is held in the dataset.
As for the loss, I use contrastive learning discussed in 
[Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748) paper and explored in
[InfoNCE](https://arxiv.org/abs/2407.00143) paper.

The model implements 2 ideas:

1. Permutation invariance:

We approximate permutation invariance by augmenting each set with random permutations of its elements.

2. Context relevance as objective:

The idea is that within a batch $B$, the model outputs $O \leftarrow M(B)$, then we determine how each sample's matching output distribution ressembles the others within that batch i.e. compute the self-similarity of the output $S :=  norm \{O\} . norm \{O\}^T$ giving a square matrix of dimension $size(B) \times size(B) $.

For example $S[i][j]$ encodes exactly how much the i-th sample ressembles the j-th sample within the batch.

Our target is $\text{diag } S$, and it will all be just a bunch of 1s since at some position $k$, $O[k][k]$ is exactly how much k-th sample ressembles to itself.

The trick is to **augment** the dataset by having two versions with each having some of its elements hidden, so to reformulate we have $O_{1} \leftarrow M(\text{aug }B)$ and $O_{2} \leftarrow M(\text{aug }B)$, resulting in new the similarity matrix $S :=  norm \{O_{1}\} . norm \{O_{2}\}^T$. 

And in principle for any sample $k$, we want $S[k][k]$ to reach $1$ which is similar at heart to having it "predict" the hidden tokens. To be precise, the objective is to make diagonal similarities large relative to off-diagonal ones. This is optimized using a cross-entropy loss over the similarity matrix $S$.
