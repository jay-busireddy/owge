# OWGE Experiment 3 — Associative Dream Consolidation

## Scientific role

This is the third and final mechanism-adequacy experiment in the current OWGE sister-paper sequence.

1. Experiment 1 (existing): exploratory laptop pilot. Use only as baseline/exploration.
2. Experiment 2 (existing): 40-seed confirmatory simple-replay experiment. Its result showed that replay alone did not establish O4 > O1, while selective weighting O3 > exhaustive O2 was strongly supported.
3. Experiment 3 (this package): tests a more faithful O4 mechanism with associative episodic memory, validated dream recombination, durable consolidation, pathway diagnostics, and compositional transfer.

The earlier experiments must not be overwritten or reinterpreted. Experiment 3 is a new prespecified test, not an attempt to continue sampling Experiment 2 until a desired p-value appears.

## Primary hypothesis

Primary endpoint: held-out Phase-C compositional-transfer BALANCED ACCURACY.

H0-P: mean balanced accuracy(O4D) <= mean balanced accuracy(O1).
HA-P: mean balanced accuracy(O4D) > mean balanced accuracy(O1).

Decision rule: one-sided paired test at alpha = 0.05. p < 0.05 rejects H0-P. p >= 0.05 means fail to reject H0-P; it does not prove H0-P.

## Secondary hypotheses

S1: O4D > O4R on Phase-C balanced accuracy.
    Tests associative validated dream consolidation versus simple replay.

S2: O4D > O4M on Phase-C balanced accuracy.
    Tests whether dream adds value beyond associative external memory alone.

S3: O4M > O3 on Phase-C balanced accuracy.
    Tests associative memory versus weighted observation without external memory.

S4: O3 > O1 on Phase-B balanced accuracy.
    Tests weighted peripheral observation when delayed peripheral information is useful.

S5: O4D > O4R on I_durable_joint.
    Tests durable internalization after removing short-term support.

S6: O4D > O4M on associative retrieval precision.

S7: O4D > O4M on the specifically unseen Phase-C (central=0, peripheral=0) composition.

Secondary tests are Holm-corrected as a family.

## Observer definitions

O1  = central-only observer.
O3  = context-weighted observer with fixed peripheral reserve rho.
O4R = O3 + prioritized replay only (replay baseline).
O4M = O3 + associative episodic memory graph, no dream.
O4D = O3 + associative episodic memory graph + validated dream recombination + durable LoRA consolidation.

## Curriculum / environments

The same agent passes sequentially through three regimes so the final task can depend on previous knowledge.

Phase A — central-sufficient:
- label follows the central factor;
- peripheral factor is independent;
- sophisticated memory should not be required.

Phase B — delayed peripheral information:
- central cue is noisy (default reliability 0.65);
- transient peripheral cue is more informative (default reliability 0.86);
- peripheral cue appears early and must survive until the terminal action.

Phase C — compositional transfer:
- target is the equality/XNOR relation between central and peripheral factors;
- training exposes (0,1), (1,0), and (1,1), but not (0,0);
- held-out test contains all four combinations;
- the specifically unseen (0,0) positive case is separately measured;
- primary C test uses familiar visual styles to isolate compositional transfer from visual OOD shift.

C_style_shift is a separate secondary test using held-out visual styles. It is not part of the primary hypothesis.

## Why balanced accuracy is primary

Phase-C factor combinations can create unequal class frequencies. Balanced accuracy prevents a trivial majority-class policy from appearing successful.

## Dream mechanism

O4D does not simply replay an old episode.

1. Draw a central donor from earlier A/C memory.
2. Draw a peripheral donor from B/C memory, preferably from a different phase/style.
3. Recombine central and peripheral fragments into a candidate episode.
4. Validate the candidate label against the synthetic causal generator's known Phase-C rule.
5. During the dream gradient update, EXTERNAL MEMORY IS DISABLED so the candidate must flow through the durable LoRA pathway rather than being answered by lookup.
6. After the update, store the validated dream episode in associative memory with novelty-weighted consolidation strength.

This implements: prior memory -> association -> recombination -> validation -> parameter internalization -> memory consolidation.

## Fairness controls

Within each paired seed:
- same frozen base checkpoint;
- same initial LoRA A/B matrices for all observers;
- same real-data training batch indices;
- same common update schedule;
- same model size and LoRA rank;
- same number of optimizer steps per epoch;
- O4D dream updates REPLACE control updates rather than adding extra optimizer steps;
- O4R replay updates also live inside the same fixed update budget;
- primary confirmatory seeds are fresh and disjoint from Experiments 1 and 2.

## Base model and anti-central-bias correction

The base is pretrained on a balanced generic integration rule where either central or peripheral information can matter. It is not pretrained with the earlier central=0.95 / peripheral=0.50 bias.

The tiny CNN preserves coarse spatial layout with adaptive 2x2 pooling before projection. This prevents the early peripheral patch from being erased by global 1x1 pooling before recurrent processing.

## Internalization diagnostics

The experiment does not equate 'large LoRA norm' with successful internalization.

It reports:
- I_param: causal performance contribution of LoRA;
- I_external_memory: causal contribution of associative durable memory;
- I_short_term: contribution of recurrent within-episode memory;
- I_durable_joint: performance retained with durable stores while short-term state is removed, relative to a stripped durable baseline;
- LoRA Frobenius norm per adapted block;
- singular spectrum/effective rank;
- number of significant singular directions;
- fraction of materially changed weights;
- fractions of materially positive and negative parameter changes.

## Pathway / association diagnostics

The experiment records:
- active hidden units;
- active embedding dimensions;
- retrieved memory nodes;
- graph edges traversed;
- phase/style diversity of retrieved paths;
- fraction of retrieved dream records;
- retrieval label precision;
- query-to-memory active-dimension Jaccard overlap (pathway reuse);
- pathway union active units;
- memory growth and dream growth.

These are mechanistic diagnostics. They are not, by themselves, evidence of smartness; behavioral transfer remains the primary endpoint.

## Information theory

For every test regime the generator records:
- H(Y);
- I(C;Y);
- I(P;Y);
- I(P;Y|C);
- combination entropy;
- factor-combination coverage.

Information theory characterizes the environment; it is not inserted into OWGE merely to make the score favorable.

## OWGE scalar

The script reports the earlier/legacy OWGE proxy only as a diagnostic. Do NOT tune its coefficients based on Experiment-3 outcomes. Stage operationalization is being tested first. If a later revised scalar is proposed, it should be defined using training/pilot data and tested on a separate dataset.

## Confirmatory sample

The confirmatory preset uses 40 paired seeds, fresh and disjoint from prior experiments. Do not stop early based on p-values and do not add seeds after inspecting the result. If a new replication is desired, define it as a separate experiment with new seeds before running it.
