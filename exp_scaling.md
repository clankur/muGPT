# Benchmarking Exponent Scaling

Recently, [Everett et al.](https://arxiv.org/pdf/2407.05872v2)  found that applying per layer learning rate prescription can enable hyperparameter transfer for other parameterization strategies besides muP and actually found that applying it to standard parameterization (SP) outperforms muP. I am attempting in reproducing this results and determining the limits of extensibility.

## Experiments

To verify the findings I performed parameter scaling from a 37m base up to 1b models and compared it to two baselines - a model trained using SP without parameter scaling and the other being a muP model.

[270m SP with tuned LR vs 270m muP transfer with 37m base vs 270m parameter transfer using SP with 37m base and fully aligned exponents](https://app.clear.ml/projects/c6c821d0a24e402eb4879dbe3ce93e2b/compare-experiments;ids=df7e20341b944c7685fcc054975aa21c,b85c64948d2747799e141fe99d41efa8,1151de73c92c49baaa612fd2a1567ed8/scalars/graph)

[1b SP with tuned LR vs 1b muP transfer with 37m base vs 1b parameter transfer using SP with 37m base and fully aligned exponents](https://app.clear.ml/projects/*/compare-experiments;ids=b9044d8fd148453ab592d8839615f78f,95b1306d3bf243a4a601d41f2fd40760,8ba8cdbca4094bab8a458e9416fc97be/scalars/graph)

**Note** to view interactive plots of their loss from ClearML you will *need* to create an account with ClearML.

## Pending questions

### Alternative parameterization approaches

Currently, I've only written an implementation of exponent scaling for SP and muP, though I have yet to benchmark the performance of applying per-layer learning rates to muP. Future work would include assessing its performance and also benchmarking NTK parameterizations's performance as Everett et al. found that the best model when applying per-layer “no alignment”.

### Alignment

One of the findings in the paper that might also be worth investigating is the influence of alignment as they mention that unaligned exponents enables modest performance of models, the exponents do not appear to capture the learning rate scaling behavior as well as the fully-aligned exponents since the power law exponents are not close to zero. They also caution that the influence of the exponent's alignment may vary per use case.

So far I have only performed parameter scaling with fully aligned exponents, but in future runs I would run experiments to either verify the performance gains observed in the paper or verify their note that the type alignments may matter per use case.

### Limits to parameter scaling

For both muP and exponent scaling, I observed their were some performance degradation depending on the selected base model, as a 13m base model **[need to quantify this]** underperformed the benchmark SP model with no scaling when performing parameter scaling to 270m BUT a 37m base model was able to outperform the same benchmark model.
