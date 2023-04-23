## ---- include = FALSE---------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)
library(attention)

## -----------------------------------------------------------------------------
# encoder representations of four different words
word_1 = matrix(c(1,0,0), nrow=1)
word_2 = matrix(c(0,1,0), nrow=1)
word_3 = matrix(c(1,1,0), nrow=1)
word_4 = matrix(c(0,0,1), nrow=1)

## -----------------------------------------------------------------------------
# stacking the word embeddings into a single array
words = rbind(word_1,
              word_2,
              word_3,
              word_4)

## -----------------------------------------------------------------------------
print(words)

## -----------------------------------------------------------------------------
# initializing the weight matrices (with random values)
set.seed(0)
W_Q = matrix(floor(runif(9, min=0, max=3)),nrow=3,ncol=3)
W_K = matrix(floor(runif(9, min=0, max=3)),nrow=3,ncol=3)
W_V = matrix(floor(runif(9, min=0, max=3)),nrow=3,ncol=3)

## -----------------------------------------------------------------------------
# generating the queries, keys and values
Q = words %*% W_Q
K = words %*% W_K
V = words %*% W_V

## -----------------------------------------------------------------------------
# scoring the query vectors against all key vectors
scores = Q %*% t(K)
print(scores)

## -----------------------------------------------------------------------------
weights = ComputeWeights(scores)

## -----------------------------------------------------------------------------
print(weights)

## -----------------------------------------------------------------------------
# computing the attention by a weighted sum of the value vectors
attention = weights %*% V

## -----------------------------------------------------------------------------
print(attention)

