# Reducing Memory Bandwidth Requirements using SharedKV

## Goal

Recent research has demonstrated the effectiveness of grouped query attention (GQA) in reducing memory bandwidth requirements, while maintaining model performance.  I propose an alternative method to achieve similar efficiency gains: calculating values by applying a projection matrix, while deriving keys by applying rotary positional encoding (RoPE) to these values. This shared key-value (KV) approach would halve the KV cache size, requiring storage for only a single cache that serves both keys and values. However, this optimization would necessitate reapplying RoPE to the keys during inference.

## Findings

[Comparing the models at 37m](https://embed.clear.ml/projects/*/compare-experiments;ids=0249f68bc4a04509a8415290abb78fe7,5f3a814b7d1b4e71bb9f2527348c0ea2,6c5d1d9ead334f42a2bddb76752047d3,fdab9d2101444844b79acfbf27fe0684/scalars/graph?metricVariants=loss&metricName=&params=loss), MHA performs achieves the same loss in 12% fewer training steps than GQA, 13% fewer training steps than SharedKV, 15% fewer steps than MQA, indicating all these models are competitive with one another.

<https://embed.clear.ml/projects/*/compare-experiments;ids=3c8dabf252c54c55bd100f4eff2a0c67,6b693d7ccea34457bfe169f977eb386c,6c5d1d9ead334f42a2bddb76752047d3,8cb9bbe524e94a5994e7772524c20e0a/scalars/graph?metricVariants=loss&metricName=&params=loss>

- GQA achieves the same loss as MHA with 9% additional training steps
- SharedKV achives the same loss as GQA with 1.2% additional training steps and the same loss as MHA with 11% additional training steps
- MQA achieves the same loss as GQA with 4.1% additinal steps and MHA with 13% additional training steps

So far early results show the approach is promising and doesn't result in a significant model degradation. Comparing a 1b run with SharedKV to the 1b multi-head attention (MHA) baseline run it appears SharedKV lags 7k steps behind MHA, but note this is without performing an learning rate sweep for the model using SharedKV. It's unclear if it would require a different learning rate, but seeing that they have different layout for attention it is something to rule out as a confounding variable. To see the full experiment comparison, you can view it [here on ClearML.]()  

## Future experiments

As a part of my next batch of experiments, I'd like to determine how it compares to a model using GQA and if they can be used in conjunction, running the following experiments:

- GQA with 4 queries per K, V ~ 1/4 HBM to the baseline MHA
- GQA with 2 queries per K, V ~ 1/2 HBM
- GQA with 2 queries per shared K, V ~ 1/4 HBM
- Shared KV ~ 1/2 HBM

## Links to past experiments

Here a links to past experiments to see how SharedKV compares to MHA:

- [37m Shared KV vs GQA vs MQA vs MHA](https://embed.clear.ml/projects/*/compare-experiments;ids=0249f68bc4a04509a8415290abb78fe7,5f3a814b7d1b4e71bb9f2527348c0ea2,6c5d1d9ead334f42a2bddb76752047d3,fdab9d2101444844b79acfbf27fe0684/scalars/graph?metricVariants=loss&metricName=&params=loss)
- [270m Shared KV vs MHA]()
- [1b Shared KV vs MHA]()
