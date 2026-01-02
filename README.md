# Jung Boundary Accountability
**Jung as a Boundary: Psychological Accountability When Meaning Is Committed**  
— *Ghost Drift: Disagreement Cannot Dissolve Responsibility*

> **Victory Condition (fixed):**  
> It must be impossible to say: “Different interpretations exist, therefore nobody is responsible.”  
> Under this protocol, disagreement can only do one of the following:
> 1) **Commit an alternative boundary** and inherit responsibility, or  
> 2) **Remain unbounded** and be logged as such—without dissolving the original responsibility surface.

---

## What this repository is
This repo is the public artifact for a **Jung-first humanities revival** that refuses the standard escape hatches:

- “It’s just interpretation.”
- “That’s only metaphor.”
- “Criteria changed after the fact.”

Instead of trying to “win” as a natural science paper, the goal here is stricter:

**Make meaning non-retroactive.**  
If meaning is invoked to justify a decision, it must be **committed to a boundary**—scope, assumptions, checks, and failure conditions—so responsibility cannot evaporate afterward.

Jung is used **as a labeling vocabulary for escape patterns** (projection, rationalization, shadow-avoidance, individuation-as-owning meaning), not as the causal proof.  
The irreversibility is enforced by **commit / ledger / rebuttal protocol**.

---

## The core idea (one paragraph)
In institutions and AI deployments, responsibility often evaporates because evaluation criteria can be rewritten after outcomes (“correct under the criteria at the time”). This project defines an accountability mechanism where meaning becomes binding only when it is committed to an explicit boundary. Once committed, any rebuttal must either (a) propose an alternative committed boundary and inherit responsibility, or (b) remain recorded as unbounded objection that cannot dissolve the original responsibility surface. Under this rule, “interpretation” stops functioning as a projection screen and becomes owned meaning.

---

## What is “Jung-first” here
Most “humanities x AI” projects begin with *Humanities* and use Jung as an example. This project does the opposite:

- **Jung is the entry point (the nucleus).**
- The paper and protocol treat Jungian mechanisms as **where agency hides** when responsibility is avoided.
- The system-level fix is not “better interpretation,” but **bounded commitment**.

This is not “therapy content.”  
This is **accountability engineering for meaning**.

---

## Protocol summary (the non-escape structure)

### 1) Boundary Packet (BP)
A **Boundary Packet** is the minimal unit that makes meaning binding.

It explicitly contains:
- **Objective** (what decision/output is for)
- **Scope** (what is included/excluded)
- **Constraints** (time, resources, audience, deployment context)
- **Assumptions** (what is presumed true)
- **Checks** (how falsification happens)
- **PASS/FAIL definition** (or responsibility triggers if PASS/FAIL is impossible)
- **Stakes** (decision_owner, time_horizon, harm_model)

A BP is **canonicalized** and hashed into a **BP_commit**.

### 2) Response Packet (RP)
A **Response Packet** binds an output to a BP_commit, including:
- the input text,
- the output text,
- the claimed assumptions vs claims separation,
- the check hooks / evaluation results (if applicable),
- the execution metadata (model/version/runtime).

It is canonicalized and hashed into an **RP_commit**.

### 3) Ledger (append-only)
A ledger is an append-only chain of rows that binds:
- BP_commit,
- RP_commit,
- row_commit linkage,
- head_commit.

This is the mechanism that makes “later reinterpretation” expensive and auditable.

### 4) Rebuttal Protocol (the key “no escape” move)
To disagree with a committed boundary/output, a critic must choose:

**A. Alternative Boundary (responsibility inheritance):**  
Provide BP_alt (their own boundary), produce BP_alt_commit, and accept that responsibility now attaches to that boundary.

**B. Unbounded Objection (logged but non-dissolving):**  
State the objection without a committed boundary; it is recorded as **unbounded** and cannot dissolve the original BP_commit’s responsibility surface.

This is the formal answer to:  
“Different interpretations exist, therefore nobody is responsible.”

---

## Minimal schemas (human-readable)

### Boundary Packet (BP) – minimal fields
```json
{
  "bp_version": "1.0",
  "objective": "...",
  "scope": { "in": ["..."], "out": ["..."] },
  "constraints": { "time": "...", "resources": "...", "audience": "..." },
  "assumptions": ["...", "..."],
  "checks": [
    { "id": "C1", "description": "...", "method": "...", "fail_condition": "..." }
  ],
  "pass_fail": { "pass_if": "...", "fail_if": "..." },
  "stakes": { "decision_owner": "...", "time_horizon": "...", "harm_model": "..." },
  "prev_commit": null,
  "witness": { "type": "optional", "ref": "optional" }
}
```

### Response Packet (RP) – minimal fields
```json
{
  "rp_version": "1.0",
  "bp_commit": "sha256:...",
  "input": "...",
  "output": "...",
  "claims": ["...", "..."],
  "supports": ["...", "..."],
  "assumptions_used": ["...", "..."],
  "checks_eval": [
    { "id": "C1", "result": "PASS|FAIL|NA", "notes": "..." }
  ],
  "runtime": { "model": "...", "date": "...", "env": "..." }
}
```

### Ledger row – minimal fields
```json
{
  "row_id": 12,
  "bp_commit": "sha256:...",
  "rp_commit": "sha256:...",
  "prev_row_commit": "sha256:...",
  "row_commit": "sha256:..."
}
```

---

## Reproducibility: what “evidence” means here
This project treats “evidence” as **non-retroactive artifacts**, not persuasion.

### Evaluation protocol (QSI/ATS/ΔATS)
- Use matched prompt pairs:
  - **LS** (low structure) vs **HS** (high structure)
  - Include **Negative Controls** (style/length without boundary binding)
  - Include **Failure Cases** (moving target, refusal of commitments, unauditable goals)
- Score responses with:
  - **QSI** (question structure)
  - **ATS** (accountability traceability)
- Report:
  - **ΔATS = ATS(HS) − ATS(LS)**
  - And ensure NC does not rise comparably

**Important:** ATS is not “quality.” It is whether the output is auditable without rewriting criteria.

---

## Non-retroactivity: what actually makes it hard
A ledger is only as hard as its anchoring.

This repo uses a two-layer stance:

1) **Local non-retroactivity (baseline):**
- canonicalization + commits (BP_commit/RP_commit)
- append-only row commits
- consistent head_commit storage

2) **Third-party non-retroactivity (when required):**
- anchor ledger head externally (witnessing) so a single operator cannot rewrite history undetected.

Common anchoring options (choose based on your threat model):
- signed commits/tags + release artifacts
- mirrored repository hosting
- time-stamped witness references (any independent mechanism you can cite)

This is not “perfect security.”  
It is the minimum structure needed so that “criteria changed” becomes audit-visible.

---

## Repo layout (recommended)
- `paper/` — main paper (HTML/PDF)
- `supplementary/` — S1/S2 case materials and figures
- `prompt_sets/` — LS/HS/NC/FC matched prompts
- `rubric/` — QSI/ATS scoring sheets
- `schema/` — BP/RP/Ledger schemas (json)
- `examples/` — sample BP, RP, ledger rows, and a head_commit example
- `figures/` — diagrams used in the paper
- `notes/` — design notes / limitations / threat model

---

## How to participate (the “no escape” contribution rules)
### If you want to rebut the paper’s claim
Choose one:

**A) Boundary rebuttal (inherits responsibility)**
- Submit a BP_alt (your boundary packet)
- Provide BP_alt_commit
- Explain how your checks would evaluate PASS/FAIL (or responsibility triggers)

**B) Unbounded objection (logged)**
- Submit an objection without a BP_alt_commit
- It will be recorded as “unbounded” and treated as non-dissolving critique

This is the project’s central move: criticism is welcomed, but it cannot erase responsibility without committing to its own boundary.

---

## Limitations (fixed, not hidden)
- This is not a claim that Jungian concepts are “scientifically proven” in the natural-science sense.
- Jung is used as a **labeling vocabulary** for responsibility-avoidance patterns.
- The hardness comes from **commit + ledger + rebuttal protocol**, not from interpretation.
- Third-party non-retroactivity depends on the chosen witness/anchoring mechanism.

These limitations are explicit by design: they prevent “interpretation drift” from being used as a retrofit escape.

---

## Citation (repository)
If you reference this work, cite the repository and the committed release/tag you used.

```bibtex
@misc{jung_boundary_accountability_2025,
  title        = {Jung Boundary Accountability: Non-Retroactive Accountability for Meaning},
  author       = {Manny and GhostDrift Mathematical Institute},
  year         = {2025},
  howpublished = {GitHub repository (commit/tag pinned)},
  note         = {URL: https://ghostdrifttheory.github.io/jung-boundary-accountability/}
}
```

---

## License
Choose a license that matches your intended reuse (e.g., CC BY for text, MIT/Apache-2.0 for code).  
If you want maximum adoption for protocols and tooling, permissive licenses are usually aligned with that goal.

