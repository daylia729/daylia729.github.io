---
title: MoE
description: MoE
date: 2025-09-29
tags:
  - cs336
---

### Mixture if experts
* Replace big feedforward with (many) small feedforward networks and a selector layer.
### Why are MoEs gettig popular?
* Same FLOP,more param does better
* Faster to train MoEs
* parallelizable to many devices
### What varies?
#### Routing function
* How we are going to routhe or essentially match tokens to experts
* Not all experts will process every token
##### Token Choice(almost all the MoEs)
* each token is going to have a srt of preference for different experts,and will choose the top K for each token
##### Expert Choice
* each expert is going to have a rank preference over tokens,and will choose the top K tokens for each expert
##### Global Assignment
* make sure that the mapping between experts and tokens is somehow balanced
#### Coomon routing variants in detail
* Top-k
* Hashing
#### Expert sizes
#### Training objectives
