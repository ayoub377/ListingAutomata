---
name: verify-changes
description: Rigorous, evidence-based verification of code changes before claiming work is done. Shows the actual diff hunk by hunk, explains what changed and how it works by tracing the code, and proves claims with executed evidence — including running the new tests against the OLD code to show they would have caught the problem, and injecting deliberate bugs to show the tests notice. Use this skill whenever you have finished implementing, fixing, or refactoring code and are about to report success; whenever the user asks to verify, review, prove, double-check, or "show me the diff / what changed"; before a commit, PR, or merge; and whenever "tests pass" is the only evidence offered. Trigger even if the user doesn't say "verify" — any moment where you would otherwise write "done, all tests pass" is this skill.
---

# Verify Changes

The job here is not to summarise the change. It is to **establish, with evidence the user can check, whether the change does what was asked — and nothing else.**

"Tests pass" is weak evidence on its own, especially when the same agent wrote both the code and the tests. Tests written alongside the code tend to encode the author's assumptions, so they can pass while the code is wrong, pass against the old code too, or not touch the changed lines at all. This skill exists to close that gap.

## Stance: review what is on disk, not what you remember

You probably wrote this code. That is exactly why your memory of it is unreliable: you will see what you intended to write. So:

- Work **only from the diff and the files as they are now.** Re-read them. Do not describe code from memory.
- Assume the change is wrong until the evidence says otherwise. Your task is to look for reasons it fails, not reasons it works.
- A claim you cannot back with evidence gets labelled unverified. Saying "I didn't verify X" is a correct, useful outcome. Implying you verified something you didn't is the worst possible outcome of this skill.

## Evidence levels

Tag every claim in the report with the strongest evidence you actually have. Never upgrade a tag.

| Tag | Meaning | Required backing |
|---|---|---|
| `[RAN]` | You executed something in this session and saw the result | The exact command and a real excerpt of its output |
| `[READ]` | You traced it through the code | `path:line` references for each step of the trace |
| `[INFERRED]` | Follows from reasoning, not directly observed | The reasoning, stated so the user can check it |
| `[UNVERIFIED]` | Plausible but not checked | What would verify it |

Within `[RAN]`, evidence about tests has a strength ordering too:
1. **Tests pass on the new code** — necessary, weakest.
2. **Tests fail on the old code** (fail-before / pass-after) — shows the tests detect the change.
3. **Tests kill a mutant** of the new logic — shows the tests pin down *this* behaviour, not just "something changed".
4. **Direct observation** — running the program, a request, a REPL call, and seeing the behaviour.

Never paraphrase output you didn't see. If a command's output was truncated or you didn't run it, say so.

## Workflow

Skill scripts live in this skill's `scripts/` directory. Resolve that path first; the commands below write it as `$SKILL/scripts`. Every script works in a throwaway git worktree and **never modifies the user's working tree**. Artifacts go to `.verify/` at the repo root (it ignores itself).

### Step 1 — Pin down what was asked

Before looking at code, write the requirements as a numbered checklist, from the user's actual request (re-read it). Include implied requirements (e.g. "don't break the existing API"). This checklist is what the diff gets judged against, so it must come from the request, not from the diff.

### Step 2 — Collect the real diff

```bash
bash "$SKILL/scripts/collect_diff.sh"            # auto base: merge-base with main, else HEAD
bash "$SKILL/scripts/collect_diff.sh" --base <ref>  # when the right base is known
```

This captures committed + uncommitted + **untracked** files (plain `git diff` silently misses new files), classifies each as source / test / config / migration / generated / docs, writes `full.patch`, `source.patch`, `tests.patch`, and prints **red flags**: removed assertions, added skips/`.only`, trivial asserts, deleted tests, swallowed exceptions, suppressions (`noqa`, `@ts-ignore`, `#pragma warning disable`), debug prints, sleeps.

Check the base is right. If the user's work started from a specific commit, pass `--base`. A wrong base makes every later step meaningless.

If the repo isn't a git repo, say so and fall back to comparing against backups or the user's description, with all claims capped at `[READ]`.

### Step 3 — Scope check

Map diff to requirements in both directions:
- Each requirement → the hunk(s) that implement it. A requirement with no hunk is **not done**.
- Each hunk → the requirement it serves. A hunk that serves none is **unrequested** and must be listed: formatting churn, renamed things, "while I was here" refactors, config or dependency changes, deleted code.

Generated files (lockfiles, build output) are listed but not reviewed line by line — but a lockfile changing when no dependency was supposed to change is itself a finding.

### Step 4 — Review each change, hunk by hunk

Group related hunks into logical changes. For each one:

1. **Show the diff.** Paste the actual hunks in a ```` ```diff ```` block. Do not paraphrase code the user hasn't seen. For very large hunks, show the meaningful part and say what was elided.
2. **What changed** — one or two plain sentences on the behavioural difference, not the textual one ("orders of exactly 100 now ship free; previously they paid 15", not "changed `>` to `>=`").
3. **How it works** — trace an execution path through the new code with `path:line` at each step: entry point → changed lines → result. For a bug fix, trace the failing input through old and new code and show where they diverge.
4. **Blast radius** — who calls this? Search for call sites of any changed function, signature, return type, exception, default value, or schema. Look for callers that relied on the old behaviour. Report what you searched and what you found.
5. **Adversarial pass** — for the changed logic, walk through: empty / null / zero; boundaries (off-by-one on both sides); error paths and what happens on exception; types and units; concurrency or ordering if relevant; idempotency for anything that writes. For each case, state whether the code handles it and how you know.
6. **Evidence and confidence** — the tagged evidence for this change, then `high / medium / low` confidence with the reason.

### Step 5 — Prove the tests mean something

Passing tests are the start. Run these, most important first:

**a. Tests pass now.** Run the relevant tests and the broader suite. Quote the summary line. Note any skipped tests and why.

**b. Fail-before / pass-after.** For each changed or new test file, check that it fails against the old code:

```bash
bash "$SKILL/scripts/fail_before.sh" --cmd "pytest -q tests/test_x.py" --tests "tests/test_x.py"
```

Run it **per test file** — one import error in a multi-file run can abort the whole collection and hide real results. Read the verdict and then read `.verify/before.log` yourself:
- **STRONG**: failed on old code, passed on new. Confirm the failure is the assertion you expected, not an unrelated error.
- **PARTIAL**: failed only on import / compile / missing-symbol errors. That proves a new function exists, not that it behaves correctly. Follow with mutations.
- **WEAK**: passes on old code too. The tests don't demonstrate the change. Say so plainly; then strengthen the tests (and rerun) or report the gap.

Pure refactors are the exception: there, tests *should* pass before and after. The evidence is behavioural equivalence, so rely on mutations and on existing coverage.

**c. Mutation spot-checks.** Pick the 1–3 lines that carry the change's logic and break each on purpose:

```bash
bash "$SKILL/scripts/mutate_check.sh" --file src/pricing.py \
  --from 'total >= 100' --to 'total > 100' --cmd "pytest -q tests/test_pricing.py"
```

Good mutations still compile and flip behaviour: boundary (`>=`→`>`), boolean (`and`→`or`), remove the new guard, return early, drop the new retry/increment, swap arguments. A **SURVIVED** mutant is a concrete gap: name the behaviour no test pins down. A mutant killed only by a syntax/import error doesn't count.

**d. Test integrity review.** Read `tests.patch` in full. For every modified *pre-existing* test, show its diff and justify it. Changed expected values, removed or loosened assertions, added skips, broadened `except`/`catch`, and mocks of the very unit under test are red flags: the test may have been bent to fit the code. See `references/test-smells.md` for the full list.

**e. Direct observation where cheap.** If the change is reachable by running something (a CLI command, an HTTP request, a REPL call), do it and quote the output. This is the most convincing evidence for the user.

Language-specific commands (coverage of changed lines, running single tests, dependency pitfalls in worktrees) are in `references/ecosystems.md`. Read the section for the project's stack before Step 5.

### Step 6 — Report

Use this structure. Keep prose tight; the diffs and evidence carry the weight.

````markdown
# Verification: <one-line description of the change>

**Verdict:** VERIFIED | VERIFIED WITH GAPS | NOT VERIFIED | BROKEN
**Base:** <ref> → working tree   **Files:** N source, N test, N other

## Requirements
| # | Requirement | Implemented in | Evidence | Status |
|---|---|---|---|---|
| 1 | ... | `path:line` | [RAN] fail-before STRONG | ✅ |
| 2 | ... | — | — | ❌ not done |

## Changes
### 1. <behavioural title>
```diff
<actual hunk>
```
**What changed:** ...
**How it works:** 1) `a.py:12` ... 2) `b.py:40` ... → result
**Blast radius:** searched `<pattern>`; N call sites; <findings>
**Edge cases:** ...
**Evidence:**
- [RAN] `<command>` → `<output excerpt>`
- [READ] ...
**Confidence:** high — <why>

## Test evidence
| Test file | Pass now | Fail-before | Mutations (killed/total) |
|---|---|---|---|

## Unrequested changes
## Red flags and how each was resolved
## Not verified
<each item, and what would verify it>
## Risks / follow-ups
````

Verdict rules — apply them strictly:
- **BROKEN**: any relevant test fails now, or tracing shows the change does not do what was asked.
- **NOT VERIFIED**: a requirement is missing, or the core claims rest only on `[READ]`/`[INFERRED]`.
- **VERIFIED WITH GAPS**: requirements met and tests pass, but some fail-before is WEAK/PARTIAL, a mutant survived, a red flag is unresolved, or something is in "Not verified".
- **VERIFIED**: every requirement has `[RAN]` evidence, fail-before is STRONG for behaviour changes (or equivalence is shown for refactors), the key mutants are killed, and every red flag is explained.

Most real changes land on VERIFIED WITH GAPS. That is fine — a precise list of gaps is worth far more than an unearned VERIFIED.

## If you find problems

Report them first, as findings, with evidence. Fix only if the user asked you to fix things or clearly expects it; after any fix, rerun from Step 2 — a fix is a new change and needs its own evidence. Never weaken a test to make it pass.