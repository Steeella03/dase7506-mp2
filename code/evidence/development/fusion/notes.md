# Fusion development handoff

No test references or test evaluation were used. All numbers below are
provisional development measurements; the root agent runs official scoring.

## Fixed search schedule
Initial: linear_listwise + mlp64_raw + text/full.
Second: linear_listwise + mlp128_hard + text/no_arithmetic.
Each round used the same predeclared coarse grid: text weights
[0, .25, .5, .75, 1], answer-group sum weights [0, .5, 1],
explicit vote weights [0, .1, .25]. Total: 90 configurations.

Initial selected 743/798 overall and 300/355 hard.
Second selected 749/798 overall and 306/355 hard. Both used
text=.75 and each embedding=.125, with no added pooling or vote weight.
The hard subset has at least two distinct valid parsed answers, computed
solely from input candidate text. Invalid candidate answers are masked.

The two embedding logits and the text logits are separately converted to
masked softmax probabilities before mixing. A repeated final answer can
optionally aggregate candidate evidence, but the coarse grid did not select
this option. Text models already contain consensus features, so this is
not evidence that all cross-candidate information is useless.

Best bundles contain copies of all required weights plus relative paths.
The inference code reads bank text and embeddings each time; it does not
use validation score arrays, IDs or sample-count assumptions.

Meaningful ablations include embedding_only, embedding_vote, text_only,
text_vote, without_text, text_full and text_no_consensus. Since the best
configuration has zero explicit votes and pooling, without_votes and
without_pooling intentionally equal the best and should not be presented
as distinct improvements. text_full restores arithmetic features;
text_no_consensus removes consensus and also has arithmetic enabled, so
it is not a perfectly isolated final-model consensus ablation.

Exact original component settings and frozen reconstruction commands are
in reproduction_map.json and reproduce_embedding_configs.json.
