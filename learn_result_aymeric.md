## Prompt variation

No prompt variation seem to create real edge
Augmenting the number of generation for first round to 15 per LLM reduce the number of non valid seed at the beggining drastically.

On average from one generation to the next though we do retry we still lose 10% of the child which are then mere copy because LLM failed to create right children

## Optimisation issue

There is a high variance between gurobi seeds which is often higher than what a cut can bring -> maybe studying more than one seed in the GA can be a great idea

The fitness use only the gap and not the LP relaxation which could be a good proxy too

