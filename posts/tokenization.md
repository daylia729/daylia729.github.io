---
title: tokenization
description: tokenization
date: 2025-09-14
tags:
  - 大模型
---

* Tokenization is the process of taking raw text which is generally represented as Unicode strings and turning it into a set of integers.
* A tokenizer is a class that implements the encode and decode methods.
* The vocabulary size is number of possible tokens(integers).
https://tiktokenizer.vercel.app/?model=gpt-4o

#### Character-based tokenization
* This method allocate each one slot in vocabulary for every character.
* Some characters appear way more frequently than others and some code poits are actually really large.
* This is not a effective way to use your buget.

#### Byte-based tokenization
* Unicode strings can be represented as a sequence of bytes,which can be represented by integers between 0 and 255.(one byte is fixed to consist of 8 binary bits)
* The most common Unicode encoding is UTF-8,which is a variable-length character encoding for Unicode. It is widely used for transmitting and storing text on the internet.
* compression ratio == 1,it is terrible,which means the sequence will be too long.

#### Word-based tokenization
* The number of words is huge(like for Unicode characters).
* Many words are rare and the model won't learn much about them.
* New words we haven't seen during training get a special UNK token,which is ugly and can mess up perplexity(PPL).

#### Byte Pair Encoding(BPE)
* Basic idea:train the tokenizer on raw text to automatocally detemine the vocabulary.
* Common sequences of characters are represented by a single token,rare sequences are represented by many tokens.
https://huggingface.co/learn/llm-course/zh-CN/chapter6/5
https://github.com/rsennrich/subword-nmt

