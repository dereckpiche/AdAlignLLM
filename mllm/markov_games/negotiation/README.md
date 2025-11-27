## Negotiation Games: core mechanics and variants

This family of games feature two agents who, in each round, may briefly communicate and then simultaneously propose how to split a fixed resource (most commonly 10 coins). Rewards are the amount kept multiplied by an agent’s per-unit value. The starting speaker alternates deterministically across rounds.

Communication is optional and variant-dependent: some settings encourage rich messaging to share private information, while others remove messaging entirely to focus on allocation behavior.

Proportional splitting is used when the two proposals exceed the available total: allocations are scaled proportionally rather than discarded. This preserves a useful learning signal even when agents over-claim.

### Variants (in increasing difficulty) 

- No‑Press Split 
  - Multiple item types (e.g., hats, balls, books)
  - The item values for each agent are public. 
  - No communication; agents go straight to making split proposals. 
  - Motivation: mirrors no‑communication setups (e.g., Advantage Alignment) while keeping the split decision nontrivial.

- Trust-and-Split RPS (TAS-RPS)
  - Single item type (coins)
  - Each round, a rock–paper–scissors hand draw creates a strong asymmetry: the winner’s per-coin value is 10, the loser’s is 1.
  - Each agent initially sees only their own hand and must communicate to coordinate an optimal split.
  - Motivation: enforce large value disparity so one’s own value reveals little about the other’s (avoiding ceiling effects) and incentivize meaningful communication.






