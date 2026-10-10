# Session captures (copied by hand from terminal output, 2026-10-10)

These were seen in the working session's terminal and not saved by a script. They are verbatim copies, not machine-captured transcripts, so treat them as lower-grade evidence than the `.jsonl` and log files in this folder.

## INFO version field follows the environment (run on Ai-core, our 0.17.3 binary)

Mounted binary check inside the test container:
`af4f07bbe3759206013e` (first 20 hex of sha256 of `/usr/local/bin/flash_serve`, equal to `binaries/flash_serve` here).

INFO reply with the image env as shipped (`HALOGEN_IMAGE_VERSION=0.17.2`), saved in probe-0.17.3.jsonl:
`I 1 0 262144 7 1 1 0 110 64 4 262144 2 1 1 524288 1 0.17.2 2 6 gguf 1 0 0 0 0 - 2048`

INFO reply on a restart with `-e HALOGEN_IMAGE_VERSION=0.17.3` (not saved by a script):
`I 1 0 262144 7 1 1 0 110 64 4 262144 2 1 1 524288 1 0.17.3 2 6 gguf 1 0 0 0 0 - 2048`

## Live service cache before and after (halogen-swift, Ai-core)

Before the test run (survey command output):
`61G	/srv/halogen-cache` and one subtree `a665fe16b2767d6d`.

After the test run and the service restart:
`8.0K	/srv/halogen-cache/a665fe16b2767d6d`, and the restarted service logged
`prompt cache on DISK: removed another configuration's subtree d19a28430273f25c (0.00 GiB; HALOGEN_CACHE_PRUNE_OLD=1)`
`prompt cache on DISK at /cache/a665fe16b2767d6d ...: 0 records in 0 lineages, 0.00 GiB held`.

## Downtime

Service stopped, test container started, restarted afterwards. The route check printed
`hermes route health=200 after ~48s` and `engine=PONG`. The stop-to-restore span of about 15 minutes is an estimate from command timestamps, not a measurement.
