# Minimizing HBM using SharedKV

## Goal

Past papers have found success in applying grouped query attention (GQA) to lower high bandwidth memory (HBM) usage with little degradation in model performance. In this similar line of reasoning, I am proposing calculating the values by applying a projection matrix and the keys are simply a result of applying RoPE to these values. Using this shared KV would cut KVCache in half by only needing to store a single cache for both the keys and values, though the keys would need to have rope re-applied at inference time.

## Findings

So far early results show the approach is promising and doesn't result in a significant model degradation. Comparing a 1b run with SharedKV to the 1b multi-head attention (MHA) baseline run it appears SharedKV lags 7k steps behind MHA, but note this is without performing an learning rate sweep for the model using SharedKV. It's unclear if it would require a different learning rate, but seeing that they have different layout for attention it is something to rule out as a confounding variable. To see the full experiment comparison, you can view it [here on ClearML.](https://embed.clear.ml/projects/*/compare-experiments;ids=1151de73c92c49baaa612fd2a1567ed8,80acd1b6b7fc4fb7ad3800b4ecaa3be2/scalars/graph?metricVariants=loss&metricName=&params=loss%3E)  

## Future experiments

As a part of my next batch of experiments, I'd like to determine how it compares to a model using GQA and if they can be used in conjunction, running the following experiments:

- GQA with 4 queries per K, V ~ 1/4 HBM to the baseline MHA
- GQA with 2 queries per K, V ~ 1/2 HBM
- GQA with 2 queries per shared K, V ~ 1/4 HBM
- Shared KV ~ 1/2 HBM

## Links to past experiments

Here a links to past experiments to see how SharedKV compares to MHA:

- [270m Shared KV vs MHA](https://embed.clear.ml/projects/*/compare-experiments;ids=1151de73c92c49baaa612fd2a1567ed8,80acd1b6b7fc4fb7ad3800b4ecaa3be2/scalars/graph?metricVariants=loss&metricName=&params=loss>)
- [1b Shared KV vs MHA](https://embed.clear.ml/projects/*/compare-experiments;ids=1151de73c92c49baaa612fd2a1567ed8,80acd1b6b7fc4fb7ad3800b4ecaa3be2/scalars/graph?metricVariants=loss&metricName=&params=loss%3E)  
