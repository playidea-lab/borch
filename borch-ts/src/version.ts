/**
 * Which build this is.
 *
 * ## Why a constant rather than reading package.json
 *
 * Bundlers disagree about importing it, and in a browser the file is not there at all.
 * So the number is written by hand in two places, and `tests/test_version.py` fails
 * when they disagree — a version that has drifted is a quiet thing.
 *
 * ## What it is for
 *
 * A manifest carries `runtime.ts`, a semver range saying which builds its weights run
 * on. The receiver can only compare that against something if it knows what it is, and
 * until now it did not — so the field was written and never read.
 */
export const VERSION = "0.3.0";
