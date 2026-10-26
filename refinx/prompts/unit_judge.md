You are auditing one run of a financial research agent against a reference annotation of the task. Judge only what is visible below: the agent's messages, its tool calls, the tool results it received, and its final response. Do not use your own knowledge of the facts; the annotation is the reference.

The annotation is in English; the agent may have worked in Chinese. Match by meaning. Company names in parentheses and stock codes identify entities across languages.

Some units say that information was unavailable, undisclosed or unverifiable as of the cutoff. Such a unit does not hold if a tool result in the trajectory, dated on or before the cutoff, shows that information. Then judge the fact "not_observed", the constraint "not_applicable", the dependency "not_covered", and the answer point "supported" when the response's conclusion rests on that tool result.

## Units to judge

### Context constraints — judge the final response
{constraints}
For each: "satisfied" if the response respects the constraint; "violated" if it contradicts or ignores it in a way that matters for its conclusions; "not_applicable" if the response never touches what the constraint governs.

### Answer points — judge the final response, then the trajectory
{answer_points}
For each: "missing" if the response does not give this conclusion or gives a conflicting one. Otherwise "supported" if the conclusion rests on tool results or calculations visible in the trajectory, or follows directly from what the question itself states (for example, asking for information the question leaves out); "unsupported" if the response asserts it without such backing.

### Facts — judge everything the agent wrote (messages and final response)
{facts}
For each: "correct" if the agent states this fact, or the specific value in it, consistently with the statement; "incorrect" if the agent states a conflicting entity, period, value or claim; "not_observed" if the agent never states it.

### Dependencies — judge how the agent used the prerequisite facts
{dependencies}
For each: "not_covered" if the agent never uses the prerequisites toward a conclusion; "valid" if it uses them and meets the requirement; "invalid" if it uses them but breaks the requirement (for example the wrong entity, period, metric scope, unit or currency, or an inference the inputs do not support).

### Numeric targets — read the final response
{numeric_targets}
For each: the number the response gives for this target, converted to the target's unit, or null if it gives none.

### Computations — read the agent's messages and final response
List at most 10 computations whose results the agent states, preferring those its conclusions use; skip any whose inputs are not visible in the trajectory. For each, write "expression": a plain arithmetic expression (numbers, + - * / ** and parentheses) built from the inputs the agent used and scaled so it yields the stated quantity in the stated unit, e.g. (4.794/2.898-1)*100 for a stated growth of 65.4%; "stated_result": the number exactly as the agent wrote it, without units or thousands separators, e.g. "65.4"; and a reason naming where the inputs came from. Do not check the arithmetic yourself.

### Extra claims — read the final response
List at most 5 substantive claims in the response that no answer point covers (skip restatements, caveats, disclaimers, remarks about the agent's own tools or data access, and generic advice). For each: "supported" if it rests on tool results or calculations visible in the trajectory or on the question itself, otherwise "unsupported".

## Output
Only one JSON object. Every listed unit appears exactly once, with its id. Each reason is one short sentence.
{"constraints": [{"unit_id": "C1", "verdict": "...", "reason": "..."}], "answer_points": [{"unit_id": "A1", "verdict": "...", "reason": "..."}], "facts": [{"unit_id": "F1", "verdict": "...", "reason": "..."}], "dependencies": [{"unit_id": "D1", "verdict": "...", "reason": "..."}], "numeric_values": [{"unit_id": "N1", "value": null, "reason": "..."}], "computations": [{"expression": "...", "stated_result": "...", "reason": "..."}], "extra_claims": [{"claim": "...", "verdict": "...", "reason": "..."}]}

## Task
Question (information cutoff {as_of} UTC):
{query}

## Agent trajectory (visible part)
{trajectory}

## Final response
{final_answer}
