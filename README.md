# Gloss Attention for Gloss-free Sign Language Translation

This is the official implementation of the GASLT paper.

## Introduction

Most sign language translation (SLT) methods to date require the use of gloss annotations to provide additional supervision information, however, the acquisition of gloss is not easy.
To solve this problem, we first perform an analysis of existing models to confirm how gloss annotations make SLT easier.
We find that it can provide two aspects of information for the model, 1) it can help the model implicitly learn the location of semantic boundaries in continuous sign language videos, 2) it can help the model understand the sign language video globally.
We then propose gloss attention, which enables the model to keep its attention within video segments that have the same semantics locally, just as gloss helps existing models do.
Furthermore, we transfer the knowledge of sentence-to-sentence similarity from the natural language model to our gloss attention SLT network (GASLT) to help it understand sign language videos at the sentence level.
Experimental results on multiple large-scale sign language datasets show that our proposed GASLT model significantly outperforms existing methods.

 The current expected release time of the full version codes and data is at the CVPR-2023 conference (before June 20, 2023). Please star us and stay tuned!