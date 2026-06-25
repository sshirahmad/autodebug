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

## Pickaxe Self-Verification

When `git log -S "<distinctive_string>"` returns exactly one commit in the range,
that commit demonstrably added the buggy code. No further verification needed.

Example: searching for `maybe_decrement_after_for_loop_variable` returns exactly
one commit `c26daa4` — that's the culprit.

Always try the pickaxe search first with a distinctive function/method name from
the bug report before resorting to bisect.

## ios_banner whitespace stripping bug

The bug was introduced in the very first commit that created the ios_banner module (6e56a61535). The `.strip()` calls were present from the very beginning — they weren't added by a later commit. The fix came later in commit 52f3ce8a80 which changed `.strip()` to `.strip('\n')` and removed the `str(text).strip()` from `map_params_to_obj`.

Key lesson: When a bug is in the original commit that created a file, pickaxe `-S '.strip()'` with `--reverse` will find it as the first commit introducing that code.

## Debugging skip_defaults issue in FastAPI

When `response_model_skip_defaults=True` is set on a path operation:
1. `serialize_response` is called with `skip_defaults=True` and a `field` (response_field)
2. `field.validate(response, ...)` creates a new model instance via a CLONED field class (from `create_cloned_field`)
3. The cloned field creates a new subclass, and validation creates an instance where `__fields_set__` contains ALL fields
4. When `jsonable_encoder(value, skip_defaults=True)` then calls `value.dict(skip_defaults=True)`, nothing is skipped because Pydantic thinks all fields were explicitly set
5. This means `skip_defaults` has no effect when there's a response_model

The `else` branch (no field) also has a bug: `skip_defaults` is not forwarded to `jsonable_encoder`.

Root cause: `field.validate()` through the cloned field loses the original `__fields_set__` information, making `skip_defaults` ineffective.
