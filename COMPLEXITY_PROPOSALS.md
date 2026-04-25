# Complexity Proposals: Making the Flight Rebooking Environment Non-Trivial

## The Core Problem With the Current Environment

Right now, a simple greedy algorithm beats the environment:

```
1. list_passengers()
2. list_alternative_flights()
3. For each passenger sorted by (has_hard_group, has_ssr, has_deadline, priority_tier):
     find first flight where cabin matches, SSR is supported, deadline met, seats > 0
     book_passenger() or book_group()
4. finalize_plan()
```

This is roughly ~40 lines of Python. No LLM needed. The environment can be solved programmatically because:

- **All information is available upfront.** Two calls (`list_passengers` + `list_alternative_flights`) give you everything you need. There is no hidden information, no uncertainty, no evolving state beyond seat decrements.
- **Constraints are binary and fully specified.** SSR either matches or doesn't. Deadline either met or not. There is no judgment, interpretation, or ambiguity.
- **Decisions are independent.** Booking PAX-001 on FL-201 has no downstream consequence beyond decrementing one seat counter. There are no cascading effects, no "booking this passenger here makes it impossible to satisfy a constraint 5 steps later" scenarios that the agent must reason about.
- **The optimal strategy is the obvious strategy.** The system prompt literally tells the agent: "book constrained passengers first, match cabin, use book_group for groups." Following these instructions mechanically is optimal.
- **The data is static and small.** 8/15/25 passengers, 3/4/5 flights. An LLM can hold the entire problem in context and pattern-match to a solution.

The environment tests an LLM's ability to **follow instructions and format JSON correctly**, not its ability to **reason, plan, adapt, or make trade-offs under uncertainty**.

Below are proposals organized from highest-impact to incremental, each designed to inject genuine reasoning challenges that resist programmatic shortcuts.

---

## 1. Dynamic World: Stochastic Mid-Episode Events

### The Problem It Solves
Currently, the world is frozen. The agent plans once and executes. There is no adaptation, no reaction, no recovery.

### Proposal

Introduce **mid-episode events** that fire at random steps (or after certain triggers) and force the agent to re-plan:

| Event | What Happens | Why It's Hard |
|-------|-------------|---------------|
| **Flight cancellation (secondary)** | An alternative flight the agent was using disappears. Passengers already booked on it become unbooked. | Agent must detect the damage, re-assess remaining capacity, re-book affected passengers. Tests recovery planning. |
| **Capacity change** | A flight's cabin availability suddenly drops (e.g., crew deadheading claims 3 economy seats) or increases (larger aircraft swapped in). | Agent can't assume its plan is still valid after every booking. Must re-check. |
| **New passenger injection** | A high-priority passenger is added mid-episode (e.g., a crew member who must deadhead, or a missed connection passenger). | Agent must integrate new information, possibly re-prioritize. |
| **SSR equipment failure** | A flight's SSR support changes (e.g., wheelchair lift breaks). Passengers already booked with that SSR must be moved. | Agent must handle involuntary rebooking, not just forward planning. |
| **Deadline shift** | A connecting flight is delayed, relaxing one passenger's deadline. Or it departs earlier, tightening it. | Agent must re-evaluate whether current bookings are still valid. |

### Why an LLM Needs This
A programmatic solver can be written for a static problem. Mid-episode disruptions break the "plan once, execute" pattern and require the agent to:
- Detect what changed (from observation diffs)
- Assess impact on existing bookings
- Decide what to unbook/rebook
- Prioritize recovery vs. continuing forward

This is **contextual reasoning under change**, not constraint satisfaction.

### Implementation Sketch
- Add an `events` list to `EpisodeState` that triggers at configured step numbers (or probabilistically).
- Events modify `flight_availability`, `passengers`, or `bookings` directly.
- The observation includes an `events` field listing what just changed.
- Add an `unbook_passenger(passenger_id)` tool so the agent can undo bookings in response to disruptions.

---

## 2. Incomplete and Asymmetric Information

### The Problem It Solves
Currently, `list_passengers()` + `list_alternative_flights()` = complete knowledge. There is no reason to ever call `get_passenger_details` except to fill a formality before `book_passenger`. The "information gathering" phase is trivial.

### Proposal

#### 2a. Hidden Constraints

Don't reveal all passenger constraints in `list_passengers()`. The summary should show **only** ID, priority tier, and group ID. Flags like `has_ssr` and `has_deadline` should be **removed from the summary** or made unreliable (e.g., "may have SSR" for 50% of passengers who actually don't).

The agent must call `get_passenger_details(pid)` to learn the actual constraints. But each call costs a step. With 25 passengers and a 90-step budget, spending 25 steps just on reconnaissance is expensive.

**The reasoning challenge**: The agent must decide *which* passengers to investigate in detail vs. which to book speculatively. It must balance information cost against risk of failed bookings.

#### 2b. Noisy / Uncertain Availability

Flight availability shown by `list_alternative_flights()` should be **approximate, not exact**. Show a range ("4-6 economy seats") or a staleness indicator ("as of 3 steps ago"). The only way to get exact current availability is `get_flight_details(flight_id)`, which costs a step.

**The reasoning challenge**: The agent must decide when to trust cached information vs. when to pay the step cost to refresh. If it books based on stale data, the booking might fail (wasting a step + penalty).

#### 2c. Information Costs (Step Budget Pressure)

Make the step budget **tighter relative to the problem size**. Currently, easy has 30 steps for 8 passengers (3.75 steps/passenger), medium has 60 for 15 (4 steps/passenger), hard has 90 for 25 (3.6 steps/passenger). This is generous -- the agent can afford to call every info tool for every entity.

Tighten to ~2.0-2.5 steps/passenger. Now the agent **cannot** inspect every passenger and every flight. It must be strategic: inspect the passengers most likely to have constraints, skip the ones that are probably straightforward.

### Why an LLM Needs This
A programmatic solver optimizes over known data. When data is incomplete, the agent must make **judgment calls about information value** -- "Is it worth spending a step to check if PAX-H019 has an SSR, or should I just try booking and handle failure?" This is a hallmark of LLM reasoning.

---

## 3. Trade-Offs With No Dominant Solution (Pareto Conflicts)

### The Problem It Solves
Currently, there is almost always a "correct answer" -- the optimal assignment achieves grader > 0.99 across all tiers. Every constraint can be satisfied simultaneously. There are no real trade-offs.

### Proposal

#### 3a. Capacity Scarcity That Forces Impossible Choices

Design data where **it is mathematically impossible to satisfy all constraints**. Examples:
- Only 1 flight supports UM, but it has 2 business seats, and 3 business-class passengers need UM. Someone gets downgraded.
- A hard group of 4 needs economy, but the only SSR-compatible flight has 3 economy seats. The group must be split (violating hard-group integrity) or someone must go to premium economy (cabin mismatch).
- Two passengers have conflicting deadlines on the same scarce flight. Only one can have the seat.

**The reasoning challenge**: The agent must decide *which* constraint to violate and *for whom*. This requires understanding the relative costs (priority-weighted penalties) and making a judgment call. There is no "right answer" -- only better and worse trade-offs.

#### 3b. Upgrade/Downgrade Economics

Add a **cost dimension**. Upgrading a passenger from economy to business isn't free -- it has a dollar cost to the airline. Downgrading has a compensation cost. The grader should include a `cost_efficiency_score` that penalizes wasteful upgrades.

Now the agent faces a genuine tension: "Do I upgrade PAX-003 to business (satisfying their booking but costing $800) or downgrade them to... wait, there's no economy left. Do I upgrade the tier-5 passenger cheaply or the tier-1 passenger expensively?"

#### 3c. Multi-Objective Tension in the Grader

Redesign the grader so that **maximizing one component necessarily hurts another** at the data level. For example:
- Putting everyone on the earliest flight maximizes deadline scores but violates group integrity (not enough seats for groups on that flight).
- Matching everyone's cabin perfectly leaves no room for the SSR passengers on compatible flights.
- Keeping groups together forces some members into downgraded cabins.

The current grader components are mostly orthogonal. Make them **conflict**.

### Why an LLM Needs This
Programmatic solvers can optimize a single objective. Multi-objective optimization under constraint conflicts requires **value judgments** -- weighing incommensurable goods. LLMs are better at this than scripts because they can reason about "what matters more in this specific situation."

---

## 4. Passenger Communication and Negotiation (Natural Language Reasoning)

### The Problem It Solves
Currently, the environment is purely mechanical. The agent never needs to understand, interpret, or generate natural language. It's just JSON in, JSON out.

### Proposal

#### 4a. Natural Language Passenger Profiles

Replace structured passenger data with **free-text notes** that the agent must parse:

```json
{
    "passenger_id": "PAX-H003",
    "name": "Amit Banerjee",
    "agent_notes": "VIP corporate account. Traveling with his wheelchair-bound mother (PAX-H004) who MUST be on the same flight. Has a board meeting in Delhi starting 3pm -- absolutely cannot arrive after 2:30pm. Will accept premium economy if business unavailable but will file a formal complaint. His admin called twice about this.",
    "priority_tier": 2,
    "original_cabin": "business"
}
```

The SSR flags, group relationships, deadlines, and preferences are **embedded in natural language**, not structured fields. The agent must extract them.

**Why this is hard for scripts**: A programmatic solver can't parse "wheelchair-bound mother (PAX-H004) who MUST be on the same flight" into `group_id: GRP-X, group_integrity: hard`. The LLM must do NLU.

#### 4b. Ambiguous / Contradictory Information

Some notes should be **ambiguous**: "Prefers morning flights but afternoon is okay if it's direct." Some should **contradict** the structured fields: the note says "must arrive by 2pm" but the structured deadline says "14:30". The agent must decide which to trust, or ask for clarification.

#### 4c. Passenger Satisfaction Tool

Add a `notify_passenger(passenger_id, message)` tool that simulates telling the passenger about their new flight. The passenger "responds" with satisfaction/dissatisfaction based on how well their preferences were met. This response affects a `customer_satisfaction_score` grader component.

The agent must craft appropriate messages (empathetic for downgrades, reassuring for tight deadlines) and handle pushback ("I will NOT fly economy. Rebook me in business or I want a full refund.").

### Why an LLM Needs This
This is the **killer feature** for distinguishing LLM agents from scripts. No programmatic solver can read free-text notes, resolve ambiguity, or craft empathetic notifications. This is exactly what LLMs excel at.

---

## 5. Multi-Step Dependency Chains (Planning Depth)

### The Problem It Solves
Currently, every booking decision is locally optimal and independent. There is no need to think more than one step ahead.

### Proposal

#### 5a. Rebooking (Undo + Redo)

Add an `unbook_passenger(passenger_id)` tool. This frees the seat but costs a step and incurs a small penalty (representing passenger disruption). Now the agent can make mistakes and recover -- but should it?

The interesting case: the agent booked PAX-001 on FL-201 early in the episode. Later, it discovers that FL-201 is the *only* SSR-compatible flight for PAX-010, and it's now full. The agent must decide: unbook PAX-001 (who is tier-1 and well-placed) to make room for PAX-010 (who has a hard SSR constraint), or leave PAX-010 unbooked (hurting coverage and SSR scores).

**The reasoning challenge**: The agent must reason about the global consequence of local decisions and sometimes undo earlier work. This requires planning depth > 1.

#### 5b. Conditional Bookings / Holds

Add a `hold_seat(flight_id, cabin, duration)` tool that temporarily reserves a seat for N steps. If not converted to a booking within that window, the hold expires and the seat returns to inventory. Holds cost a step but don't commit.

**The reasoning challenge**: The agent must manage a portfolio of holds -- reserving scarce seats while it investigates whether they're actually needed. Letting holds expire wastes steps; holding too aggressively starves other bookings of inventory. This is **resource management under time pressure**.

#### 5c. Booking Dependencies (Pre-Requisites)

Some bookings should require prerequisites:
- Booking an unaccompanied minor requires that a **guardian confirmation** tool be called first.
- Booking on a codeshare flight requires **partner airline approval** (a tool call that might fail 30% of the time).
- Booking a group requires that **all members' details have been fetched** first.

**The reasoning challenge**: The agent must plan its tool-call sequence, not just its booking assignments. The order matters.

---

## 6. Richer Constraint Types

### The Problem It Solves
The current constraints (SSR, groups, deadlines, cabin match) are few and simple. They can all be evaluated with a single comparison.

### Proposal

#### 6a. Connecting Itinerary Constraints

Some passengers aren't just going to DEL -- they have onward connections. The agent must ensure:
- Minimum connection time (MCT) between the rebooked flight's arrival and the onward flight's departure (e.g., 90 minutes for domestic, 150 minutes for international).
- If the connection is at a different terminal, add 30 minutes to MCT.
- If the onward flight is on a different airline, the agent must check interline baggage agreements.

This is no longer "does the flight arrive before the deadline?" -- it's "does the flight arrive early enough, accounting for terminal, airline, and connection type?"

#### 6b. Regulatory and Operational Constraints

- **Unaccompanied minors (UM)**: Can only fly on flights that have a trained cabin crew member assigned. Some flights have UM slots (max 2 UMs per flight). This is different from SSR support -- even if the flight "supports UM," it might already have 2 UMs booked.
- **Pet conflicts**: A `pet_cabin` passenger cannot be on the same flight as a passenger with a documented pet allergy (new passenger field: `pet_allergy: bool`). The agent must check for conflicts.
- **Wheelchair boarding**: WCHR passengers require extra boarding time. If a flight already has 3+ WCHR passengers, boarding time increases and the departure may be delayed (affecting downstream deadlines for *other* passengers on that flight).

**The reasoning challenge**: Constraints interact across passengers. Booking one passenger changes the constraint landscape for others. The agent can't evaluate each passenger in isolation.

#### 6c. Loyalty and Compensation Policies

- Passengers have loyalty status (`gold`, `silver`, `none`) that determines what compensation they're entitled to.
- If a gold member is downgraded, they must be offered lounge access + meal voucher.
- If any passenger waits more than 4 hours (departure time of new flight vs. original), they're entitled to a hotel voucher.
- Compensation has a budget ceiling per episode. The agent must allocate limited compensation dollars wisely.

**The reasoning challenge**: The agent must understand *policies* (if-then rules embedded in natural language or a policy document) and apply them correctly. This is interpretive, not computational.

---

## 7. Procedural Data Generation (Combinatorial Diversity)

### The Problem It Solves
The current data is 3 static JSON files. After a few episodes, the LLM has memorized the optimal assignment. There is no generalization.

### Proposal

#### 7a. Seed-Based Procedural Generation

Use the `seed` parameter in `reset()` to procedurally generate fresh passenger manifests and flight pools for every episode. Parameters:
- Number of passengers: 10-50 (continuous range, not 3 fixed tiers)
- Number of flights: 3-8
- SSR density: 0-40% of passengers
- Group density: 0-30% of passengers in groups
- Deadline density: 0-30% of passengers
- Capacity scarcity: surplus ratio 1.5x to 0.8x (over/under-provisioned)
- SSR distribution: how many flights support which SSRs

**Why this matters**: The LLM can't memorize solutions. Every episode is a new problem. It must generalize its strategy, not pattern-match to seen data.

#### 7b. Adversarial Data Generation

Generate data specifically designed to break common LLM failure modes:
- **Greedy traps**: The locally optimal assignment for passenger 1 blocks the globally optimal assignment for passengers 2-5.
- **Distractor flights**: Flights that look attractive (lots of seats, early departure) but don't support critical SSRs.
- **False scarcity**: One cabin appears full on all flights, but a careful combination of split-cabin bookings across flights can accommodate everyone.
- **Priority inversion traps**: A tier-1 passenger with no constraints vs. a tier-5 passenger with a critical SSR. The agent should prioritize the constrained tier-5 passenger first (constraint scarcity > priority tier for *booking order*).

---

## 8. Richer Reward Signal (Reward Decomposition)

### The Problem It Solves
The current reward signal is noisy and doesn't teach the agent *why* a decision was good or bad. A booking that gets +0.30 gives no signal about whether the choice was good for coverage vs. cabin match vs. group integrity.

### Proposal

#### 8a. Decomposed Reward Feedback

Instead of a single reward number, return a **reward breakdown** at every step:

```json
{
    "reward": 0.35,
    "breakdown": {
        "coverage_delta": 0.10,
        "cabin_match_delta": 0.15,
        "group_integrity_delta": 0.00,
        "deadline_delta": 0.05,
        "ssr_delta": 0.00,
        "efficiency_delta": 0.05,
        "opportunity_cost": -0.03
    },
    "explanation": "Booked tier-1 PAX in matching cabin on SSR-compatible flight. However, this used the last business seat on FL-201, which may constrain options for PAX-H004 (also needs business)."
}
```

**Why this matters**: The agent can learn *what aspect* of its decision was good/bad and adjust. "I keep losing on group_integrity_delta -- I should book groups earlier."

#### 8b. Opportunity Cost Signal

After each booking, compute what the agent *gave up* by making that choice. "By putting PAX-001 on FL-201, you consumed the last UM-compatible business seat. PAX-003 also needs UM + business and now has no valid flight."

This is a **counterfactual** signal that requires global awareness. It teaches the agent to think about downstream consequences.

#### 8c. Progressive Difficulty Reward Scaling

As the agent gets better (higher grader scores over episodes), scale the reward function to be harsher. Early episodes: any booking gets positive reward. Later episodes: only optimal or near-optimal bookings get positive reward. This creates a **curriculum** that prevents reward hacking.

---

## 9. Multi-Agent / Competitive Dynamics

### The Problem It Solves
The environment is single-player. The agent optimizes against a static world. There's no strategic interaction.

### Proposal

#### 9a. Shared Inventory (Competitive Rebooking)

Multiple cancelled flights, each with their own agent. All agents share the same pool of alternative flights. When Agent A books a seat, it's gone for Agent B too.

**The reasoning challenge**: The agent must anticipate that capacity is being consumed by others and act more urgently on scarce resources. It can't leisurely investigate every option.

#### 9b. Passenger Agents

Some passengers are "active" -- they can reject a booking and request alternatives. A tier-1 business passenger might reject an economy booking and demand business, even if the agent thinks economy is the best available option. The agent must negotiate or find a better option.

---

## 10. Long-Horizon Reasoning (Temporal Complexity)

### The Problem It Solves
Currently, all flights depart "today" and there's no temporal structure beyond HH:MM times. The planning horizon is flat.

### Proposal

#### 10a. Multi-Day Rebooking Window

Flights span 3 days. Early flights today have more capacity but require hotel + meals for passengers originally departing tomorrow. Late flights tomorrow are cheaper but leave passengers stranded longer. Some passengers have hard deadlines ("must be in Delhi by tomorrow 9am for a wedding").

**The reasoning challenge**: The agent must reason about time-cost trade-offs across days, not just "which flight has seats."

#### 10b. Rolling Availability

Flight availability changes over time (simulated). Early in the episode, FL-201 has 10 economy seats. By step 30, it might have 6 (others booked by a simulated "normal booking" background process). The agent must decide: book now at sub-optimal cabin match, or wait for better information but risk losing seats.

**The reasoning challenge**: This is the classic explore-exploit dilemma. Act early with imperfect information, or wait and risk losing inventory.

---

## 11. Policy and Regulation Interpretation

### The Problem It Solves
Currently, all rules are hardcoded in the environment. The agent doesn't need to *understand* policies -- the environment enforces them.

### Proposal

#### 11a. Policy Document Tool

Add a `read_policy(topic)` tool that returns a natural-language policy document. Topics include:
- "compensation" -> rules about what compensation each tier/status gets
- "unaccompanied_minors" -> regulations about UM handling
- "pet_travel" -> restrictions on pets
- "oversales" -> what to do when capacity is insufficient

The agent must **read and interpret** these policies to make correct decisions. The environment does NOT enforce policies -- it only enforces physical constraints (seat availability). If the agent violates a policy (e.g., doesn't offer lounge access to a downgraded gold member), the grader penalizes it.

**Why this matters**: Policy interpretation is a core LLM capability that no programmatic solver can match. The policies can be updated between episodes without changing code, testing the agent's ability to adapt to new rules.

#### 11b. Exception Handling

Some passengers have notes like "CEO of our biggest corporate account -- authorized for any exception up to $5000." The agent must recognize when standard policy doesn't apply and use judgment about when to make exceptions.

---

## 12. Observability and Debugging (Meta-Reasoning)

### The Problem It Solves
Currently, the agent just acts. It never needs to explain its reasoning, review its work, or catch its own mistakes.

### Proposal

#### 12a. Explain Tool

Add an `explain_booking(passenger_id)` tool that asks the agent to justify why it made a specific booking. The environment evaluates whether the explanation is consistent with the actual constraints and scoring. Good explanations get bonus reward; inconsistent explanations get penalized.

**Why this matters**: This tests whether the agent actually *understands* its decisions or is just pattern-matching. It forces metacognition.

#### 12b. Audit Mode

After the agent calls `finalize_plan()`, present it with 3-5 "audit questions":
- "PAX-H003 was booked on FL-203 in economy, but their original cabin was business. Why wasn't FL-201 business used instead?"
- "GRP-H01 was split across FL-201 and FL-202. This is a hard group. Justify this decision."

The agent must answer these questions correctly to receive full grader credit. If its answer reveals a mistake, it gets a chance to fix it (at a step cost).

---

## Summary: Proposed Complexity Tiers

### Tier A -- High Impact, Moderate Implementation Effort

| # | Proposal | What It Tests |
|---|----------|--------------|
| 1 | Stochastic mid-episode events | Adaptation, recovery planning |
| 3a | Impossible-to-satisfy-all constraint sets | Value judgment, trade-off reasoning |
| 7a | Procedural data generation | Generalization (no memorization) |
| 2a | Hidden constraints (info cost) | Strategic information gathering |

### Tier B -- High Impact, Higher Implementation Effort

| # | Proposal | What It Tests |
|---|----------|--------------|
| 4a | NL passenger profiles | Language understanding, information extraction |
| 5a | Unbook/rebook (planning depth) | Global reasoning, undoing decisions |
| 11a | Policy document interpretation | Reading comprehension, rule application |
| 6b | Cross-passenger constraint interactions | Systems thinking |

### Tier C -- Differentiation Features

| # | Proposal | What It Tests |
|---|----------|--------------|
| 4c | Passenger notification/negotiation | Empathy, communication |
| 8b | Opportunity cost signal | Counterfactual reasoning |
| 10b | Rolling availability (explore-exploit) | Decision-making under uncertainty |
| 12b | Audit mode post-finalization | Metacognition, self-correction |

### Minimum Viable Complexity Upgrade

If you implement **only 3 things**, make them:

1. **Procedural data generation** (7a) -- eliminates memorization, the single biggest weakness.
2. **Impossible constraint sets** (3a) -- forces trade-off reasoning instead of "follow the obvious path."
3. **Hidden constraints with step-budget pressure** (2a + 2c) -- forces strategic information gathering instead of "dump everything then solve."

These three changes together transform the environment from "can the LLM follow instructions and format JSON?" to "can the LLM plan under uncertainty, make trade-offs, and manage scarce resources?"
