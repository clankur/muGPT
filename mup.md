# Scaling Hyperparameters using Maximal Update Parameterization (muP)

Exploring the best approaches for applying mu-transfer.

## Goal

Determine what hyperparameters are more transferable between the base model and target models when applying muP. Specifically, investigating these hyperparameters:

- Width
- Depth
- Total training steps
- Number of warm up steps
- Gradient Clipping

As a part of this investigation, we are also attempting to measure the impact of the attention multiplier and output multiplier used by muP and the importance of applying 0 initialization to the LM head and query's weights.

## Investigation Procedure

- Perform hyperparameter sweep on base model and target model as a benchmark
- See how transfer works along changing the following hyperparameters:
  - Width
    - For attention, we can consider widening both by increasing the number of heads and increasing the d_head.
  - Depth
  - Total training steps
  - Number of warm up steps
  - Gradient Clipping
  - A combination of: width + depth + total training steps + number of warm up steps
- Check performance after training the target model when you scale and apply muP when scaling each of the hyperparameters

## Measuring performance

We'd want to measure loss after training and compare it relative to our benchmark models by observing the steps to achieve the same loss.

## Findings

### Do models with same dimensionality have transferable hyperparameters?

Below, both of these models share the same dimensionality - only differing in number of heads and head_dim though coming out to be the same size of model (13 million parameters) - they have different optimal learning rates for the same number of steps.

### Transferability of a muP models

A 10m model did not scale well past 37m, but 37m to 270m transferred well and outperformed SP with a LR found through an LR sweep and is 2k steps ahead of the SP baseline.
These results also hold with a 1B model, where muP was 4500 steps ahead. You can view the results of my experiments of the 270m experiments [here](https://embed.clear.ml/projects/*/compare-experiments;ids=1151de73c92c49baaa612fd2a1567ed8,80acd1b6b7fc4fb7ad3800b4ecaa3be2/scalars/graph?metricVariants=loss&metricName=&params=loss) and the 1b experiments [here](https://embed.clear.ml/projects/*/compare-experiments;ids=8da892f490744918b675c4b071860d48,eebfcd7638784437ac5faf0836a3cb5b/scalars/graph?metricVariants=loss&metricName=&params=loss).

### Zero initialization for the queries and output layer

After looking over our coordinate checks, we noticed that the muP models' queries were exploding, in line with a finding in the muP paper.

## Pending questions for future investigations

### Clarifying Ambiguous Hyperparameters from the muP Paper

In the muP paper, it mentions that attention and output multipliers have to be tuned for the base model, but I still have yet to review the importance of tuning and setting these parameters. My preliminary analysis without doing an investigation into the matter shows that muP transfer does not require tuning them as setting them to 1 still enables them to transfer.

In this same vein of question, does the clip value hyperparameter matter when applying gradient clipping? The paper says it stays constant relative to width implying that the clip value only needs to be tuned for the base model and its family of models would transfer this parameter.

To determine the importance of these, I plan on running [ClearML's Optuna](https://clear.ml/docs/latest/docs/integrations/optuna/) to sweep the parameters for the 37m model and compare how muP transfer performs when tuning each of these vs. leaving them as is vs. using SP.

### Limits to muP transfer

Another line of question that remains unclear to me at this moment is the limits of how small a model can be while still getting the benefits of applying muP transfer. This stems from my finding that shows that a 10m model wasn't able to scale well, indicating there are limits to how much a model can scale and apply muP transfer. I plan on seeing how well muP transfer applies to a 2B model to see if scaling a 37m model 60x with muP can match SP's performance.

## Experiments run so far

- [270m muP vs SP without gradient clipping](https://embed.clear.ml/projects/*/compare-experiments;ids=1151de73c92c49baaa612fd2a1567ed8,80acd1b6b7fc4fb7ad3800b4ecaa3be2/scalars/graph?metricVariants=loss&metricName=&params=loss)
- [1b muP vs SP without gradient clipping](https://embed.clear.ml/projects/*/compare-experiments;ids=8ba8cdbca4094bab8a458e9416fc97be,8da892f490744918b675c4b071860d48/scalars/graph?metricVariants=loss&metricName=&params=loss)
