"""Deprecated one-off SSIS-575 experiment.

The previous implementation hard-coded one title, reused an existing GPT draft,
and converted Japanese surface patterns into semantic fields.  It was not a
general Codex inference engine and must not be used as an autonomous release
path.
"""

raise SystemExit(
    "tools/codex_gen3.py is disabled because it is a title-specific legacy "
    "experiment, not an evidence-complete translation engine. Use "
    "tools/codex_autonomous_release.py prepare/finalize with reviewed decisions, "
    "or the official translation queue and strict application workflow. The "
    "previous script is archived under experiments/legacy-oneoff when the "
    "hardening migration is applied."
)
