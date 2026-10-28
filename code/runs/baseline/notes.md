# Training-free baselines: implementation handoff

Implementation: `methods/baselines.py`. Both `predict(bank, arrays, checkpoint)`
and `candidate_scores(bank, arrays, checkpoint)` are available. The default is
`majority`; a JSON checkpoint can select a mode using `{"mode": "majority"}`.
There are no learned weights, seeds or external dependencies beyond NumPy and
the supplied `common.parse_answer`. Configuration files are the only inference
assets. Candidate scores have shape `[N, 4]`; they are heuristic scores, not
calibrated probabilities. Embeddings, IDs and reference answers are not used
at inference.

## Majority and tie breaking

Each candidate's final numeric answer is parsed using the unchanged official
parser. Equivalent fractions and decimals therefore agree. A valid answer
receives the number of candidates that give that answer; an unparseable answer
receives zero votes. `argmax` returns the first maximal candidate in supplied
candidate order. Thus a 2+2 or 1+1+1+1 tie is resolved by candidate position.
This is a deliberate reproducible baseline, not a claim that position is
meaningful. It is independent of question IDs and bank length.

Other modes:
- `first`: selects candidate zero.
- `shortest` / `longest`: selects minimum / maximum text character length.
- `majority_shortest`: majority score minus `0.001 * length / max_length`;
  shorter text breaks vote ties but cannot override one additional vote.
- `shortest_valid`: prefers a parseable answer, then shorter text.

## Descriptive training observations

`analysis/profile_training.py` read train inputs and train references only and
wrote `training_profile.json` plus the six small mode configurations. The run
used the existing Python 3.12.14 environment, NumPy 2.5.3 and one thread per
BLAS/OpenMP library. No test references or test scoring were accessed.

Of 4,755 training questions, 711 / 732 / 1,070 / 2,242 have exactly 1 / 2 / 3 /
4 correct candidates. The all-correct groups therefore account for 47.15% of
training questions. Position-wise candidate correctness is approximately
75.04--75.79%, showing little simple position advantage. There are 871
unparseable candidates among 19,020 total (4.58%). Mean candidate text length
is 480.19 characters: correct candidates average 464.84 and incorrect ones
527.41. These marginal length statistics do not establish a causal relation.

| Heuristic | Descriptive training accuracy (%) |
|---|---:|
| first | 75.3733 |
| majority | 85.2576 |
| shortest | 70.3470 |
| longest | 77.7497 |
| majority_shortest | 84.0589 |
| shortest_valid | 74.5952 |

These are provisional training-only descriptions, not official validation or
acceptance results. Training includes easy unanimous groups whereas the
official test bank has at least two different valid parsed answers per group.
The training accuracy should not be presented as a test-performance estimate.
The difference between marginal correct-answer length and the `shortest`
selection result illustrates why group-level comparisons are necessary.

## Commands for the coordinator

From `code/`, official validation can use, for example:

```text
python predict.py --method methods/baselines.py --checkpoint runs/baseline/majority.json --split validation --output runs/baseline/majority-validation-predictions.json
python score_predictions.py --split validation --predictions runs/baseline/majority-validation-predictions.json --output runs/baseline/majority-validation-score.json
```

The root agent owns those official runs, resource measurement, acceptance and
any decision about final method selection. No outstanding implementation issue
is known; no final acceptance claim is made here.
