# Postmortem — incident NN: <short name>

**Date:** YYYY-MM-DD · **Detected by:** <alert / dashboard / a human> ·
**Time to detect:** <minutes> · **Time to resolve:** <minutes>

---

## The five sentences

Write exactly five. The limit is the point: it forces you to separate *what
happened* from *how you found out*, which is the distinction the whole session
is about. If you cannot write these five, you have not finished diagnosing —
you have only stopped the bleeding.

1. **Context** — what was normal before this started.
2. **Signal** — what you actually observed, and *which instrument showed it*.
   Name the panel, the query or the SQL. "The model got worse" is not a signal.
3. **Action** — what you did, in order, including the things that turned out to
   be wrong. The dead ends are the most useful part for the next person.
4. **Outcome** — what changed, with a number.
5. **Lesson** — what you would change so the *next* one is caught faster. If the
   lesson is "be more careful", you have not found it yet.

---

## Evidence

Paste the actual numbers. A postmortem without evidence is a story.

```
<PromQL, SQL, or CLI output — before and after>
```

## Timeline

| Time | Event |
|---|---|
| | |

## What made this hard to find

The interesting question is not "what broke" but "why did it take N minutes".
Was the signal missing, aggregated away, or present and unread?

## Follow-up

- [ ] Alert or test that would have caught this sooner
- [ ] Runbook entry, if this class of failure is new
