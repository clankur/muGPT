# Minimizing HBM using SharedKV

## Goal

Past papers have found success in applying grouped query attention (GQA) to lower high bandwidth memory (HBM) usage with little degradation in model performance. In this similar line of reasoning, I am proposing calculating the values by applying a projection matrix and the keys are simply a result of applying RoPE to these values. Using this shared KV would cut KVCache in half by only needing to store a single cache for both the keys and values, though the keys would need to have rope re-applied at inference time.

## Findings

So far early results show the approach is promising and doesn't result in a significant model degradation. Comparing a 270m run with SharedKV to the 270m baseline run it appears SharedKV lags slightly and note this is without performing an learning rate sweep for the model using SharedKV. It's unclear if it would require a different learning rate, but seeing that they have different layout for attention it is something to rule out as a confounding variable.

<iframe
  src="https://embed.clear.ml/widgets/?type=scalar&objectType=task&xaxis=iter&objects=1151de73c92c49baaa612fd2a1567ed8&objects=80acd1b6b7fc4fb7ad3800b4ecaa3be2&metrics=grad_norm&variants=grad_norm&light=true&company=34fa63d488454711ab18afa5e77895d0"
  style="width:100%; height:300px;"
></iframe>

Scaling this up to a 1B model using shared KV....

## Future experiments

As a part of my next batch of experiments, I'd like to determine how it compares to a model using GQA and if they can be used in conjunction, running the following experiments:

GQA with 4 queries per K, V ~ 1/4 HBM to the baseline
GQA with 2 queries per K, V ~ 1/2 HBM
GQA with 2 queries per shared K, V ~ 1/4 HBM
Shared KV ~ 1/2 HBM

You can view the results of past runs here on ClearML.
