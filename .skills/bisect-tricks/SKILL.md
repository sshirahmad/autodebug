# Bisect Tricks - AutoDebug

## Key Insight: Match the Repro to the Bug

When finding the culprit commit, the repro script defines WHAT the bug is.
If the repro calls `black.reformat_many()`, then the bug is specifically about
`reformat_many` not handling some error case. The bug "didn't exist" before
`reformat_many` was introduced, even if similar code existed elsewhere.

## Strategy

1. Read the repro script carefully - it tells you exactly what function/behavior is broken
2. Find when that function was introduced
3. The commit that introduced the function (without the fix) is usually the culprit
4. Verify with `git show <sha> -- <file>` that the buggy code was introduced there

## Example

Bug: `reformat_many` doesn't handle OSError from ProcessPoolExecutor
- `reformat_many` introduced in commit X
- Fix added in commit Y (after HEAD)
- Known-good = parent of X
- Culprit = X

## Verification Gate

Use `git show <sha> -- <file>` to confirm the buggy code was introduced.
The parent should NOT have the buggy code (or the function at all).
