# OWGE sister-paper experiment design

## Research purpose

The experiment does **not** try to prove a universal definition of intelligence. It tests predictions inside the narrower OWGE boundary defined in the conceptual paper: whether an agent transforms partially observed evidence into useful, retrievable, transferable behavior efficiently under finite processing/memory resources.

The conceptual paper's observer families are treated as falsifiable policies:

- **O1 central-only:** preserves the obvious central cue; removes weak peripheral evidence.
- **O2 uniform exhaustive:** makes essentially all visual evidence available.
- **O3 weighted associative:** high central access plus bounded peripheral reserve.
- **O4 weighted + validated replay:** O3 plus prioritized replay that is retained only when validation performance does not degrade.
- **O5R retrieval-disconnected:** same weighted access/replay family as O4, but recurrent retrieval is deliberately broken by resetting hidden state at every step.

## Why synthetic images first

A real-image dataset is useful later for external validity, but it is a poor first falsification environment because we do not know the true causal relevance of every object, we cannot guarantee pretrained-model contamination, and conditional mutual information is difficult to control.

Synthetic scenes give exact control over:

1. the hidden state;
2. which cue is central;
3. which cue is peripheral and delayed;
4. how predictive the cue is;
5. how many distractors exist;
6. whether a relation reverses;
7. which visual families are held out for transfer.

## Scene families

All families implement the same latent causal structure with different visual surfaces.

- **crossing** — training style
- **warehouse** — training style
- **factory** — held-out transfer/test style
- **harbor** — held-out transfer/test style
- **base** — base-pretraining-only style

Each episode has six frames by default. The hidden state is binary: `hazard=0/1`.

- A large central shape is visible throughout the episode and is only moderately predictive in the POMDP.
- A smaller peripheral cue appears only at an early time step and can contain additional information about the hidden state.
- A billboard-like token is causally unrelated to the hazard; it is included so a later goal-switch experiment can test the post-publication attention-vs-retention refinement.
- Random objects are causally irrelevant distractors.

## POMDP role

The POMDP is the **experimental world**, not a component of the OWGE definition.

The true state is hidden. The agent gets only noisy images. A useful peripheral cue can appear early and disappear before the final action. An observer must retain enough history to use it later.

The final action is binary: choose safe/no-hazard vs hazard. Reward is +1 for a correct final action and -1 for an incorrect final action.

## MDP role

The MDP is a sanity/control condition. The same hidden state is made explicitly observable through a state beacon in every frame. If OWGE's peripheral-memory mechanisms are specifically useful under partial observability, observer differences should shrink when the state is fully visible.

A clean control hypothesis is:

`(O3 - O1 performance gap in POMDP) > (O3 - O1 performance gap in MDP)`.

## Information-theory role

Information theory describes the environment; it is not inserted into the OWGE smartness equation.

The generator measures:

- `H(hazard)`
- `I(central; hazard)`
- `I(peripheral; hazard)`
- `I(peripheral; hazard | central)`
- `I(billboard; hazard)` — should remain approximately zero

The peripheral-accuracy sweep changes the amount of useful delayed information and tests whether the O3-O1 performance gap emerges as conditional mutual information increases.

## Neural model / LoRA role

The default base is a deliberately tiny model:

- 3 small convolution layers
- 32-dimensional embedding
- 32-unit GRU memory
- 2-action policy head

The verified implementation contains about **12,074 base parameters**.

Every observer starts from an exact copy of the **same base checkpoint**. The base is frozen. Long-term adaptation occurs through two low-rank adapters:

1. a low-rank residual update in embedding space;
2. a low-rank update on the hidden-to-action policy matrix.

LoRA is therefore an experimental proxy for low-cost long-term internalization/adaptation. It is **not** used as the short-term episodic memory. The GRU hidden state is the short-term memory used to carry a weak cue across an episode.

## Reinforcement-learning role

Observer LoRA adapters are updated with REINFORCE from final task reward. All observers use the same optimizer, learning rate, rank, base model, episode sets, and action space.

O4/O5R additionally maintain a priority replay buffer. Replay updates are tentative: the adapter state is saved, replay is applied, and the replay update is reverted if fixed validation reward decreases.

## Train/validation/test split

Do **not** randomly split frames. Entire episodes remain in one split.

- Base pretraining: `base` visual style only.
- Adaptation/training: crossing + warehouse.
- Validation: disjoint crossing + warehouse episode seeds.
- Final test/transfer: factory + harbor, disjoint episode seeds.

This tests both novel episodes and visual transfer.

## Statistical hypotheses

Statistically, `O4 > O1` is an alternative/prediction, not the null.

Examples:

- H0: `mean(O3) <= mean(O1)`; HA: `mean(O3) > mean(O1)` under non-zero delayed peripheral information.
- H0: `mean(O4) <= mean(O3)`; HA: `mean(O4) > mean(O3)` under stable reusable structure.
- H0: O5R end-to-end OWGE is not lower than O4; HA: O5R is lower.
- Reversal experiment: HA can deliberately predict O4 < O3 if old replay slows adaptation after the cue relationship flips.

All main comparisons are paired across matched environment seeds.

## What counts as evidence

Do **not** reject a null because LoRA matrices differ. Different weights alone do not establish better learning.

Primary evidence:

- held-out accuracy/reward;
- transfer performance;
- memory-ablation performance;
- resource cost;
- OWGE components and OWGE+;
- paired statistics/effect sizes.

Secondary mechanistic evidence:

- LoRA update norm;
- pairwise LoRA update cosine similarity;
- learning curves;
- replay acceptance/gain.

## Main falsification criterion

The strongest test of the proposed smartness measure is not simply whether O4 wins. It is whether **OWGE predicts held-out adaptive performance better than simpler quantities** such as accuracy, reward, memory breadth, or LoRA magnitude.

If OWGE ranks models highly but that ranking consistently fails to predict held-out transfer/adaptation, the definition/operational score should be revised or rejected.

## Published v3 vs post-publication refinement

The published v3 says attention weights represent both processing and retention priority. The later billboard thought experiment exposes a real ambiguity: low current attention need not imply zero memory trace, and relevance can change with goal/time.

This does **not** invalidate the published v3 hypotheses used here. But the next conceptual version should separate:

- current processing attention `a_ti`;
- transient memory-write/trace strength `mu_ti`;
- goal- and time-conditioned relevance `r_ti(g, tau)`.

The current experiment deliberately keeps that goal-switch extension outside the primary hypotheses so the first empirical sister paper remains focused. After the primary study, the same generator can add a second final query about the earlier billboard to test transient weak traces directly.
