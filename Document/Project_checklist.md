# Checklist for project

## Frame the problem and look at the big picture

### Objective

- Create one-voice polyphony melody having good quality from the model.
- The created melody need to have these properties:
  - Controllable Vietnamese and/or Western classical style.
  - Controllabel four basic mood in music.
  - Contain the given melody or at least infering from the given melody.
- Experiment with insight idea using for music:
  - Experiment CVAE (The idea from age regression paper).
  - Test new way of music presentation:
    - Incorporate some more information.
    - Wide matrix with lots of 0 value.

### Frame model

- Offline
- Unsupervised

### Measuring performance

For each of mentioned objectives, we find a metric for measuring each of them (if possible). The metric is quantitative and qualitative (if possible). Most of the metric are got from the related paper.

- Accuracy (For given melody objective and quality of music objective).
- Loss function (For demonstrating converging property of the model).
- Hamming loss (For create melody objective).
- High-level information of style, emotion (For style and mood objective, the ability of capturing high-level information in the latent space).

We need to compare at least two different kind of models under these measuring. These models are:

- CVAE model (the most prominent model)
- VAE model + label injeted to the decoder (the discriminator of decoder are removed)
- VAE model without label injected (if possible, using to compare related paper)

## Get the data

- Didive training set, validation set, test set.

Information of source, type of the data are described in the paper.

## Explore the data

Information of source, type of the data are described in the paper.

## Prepare the data for better implementation for project

The preprocess and normalized of the data are described in the paper.

## Explore different models and choose the best ones

Testing only this model

## Fine-tune the selected model

### Fine-tune parameters

- Loss function: binary cross entropy \
- Number of neuron per layers, number of layers.
- Mini-batch (unstateful RNN) \ Gradient batch (stateful RNN).

### Checking with test set

## Present the solution, model, complete the paper

### Create some results, with these consumptions

- One Vietnamese melody and Western label.