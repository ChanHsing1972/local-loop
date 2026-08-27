# Video script (target: 105-110 seconds)

## Before recording

- Use a neutral terminal prompt and relative or `/tmp` paths; hide the menu bar, account avatar, notifications, Git remote owner, and all API values.
- Copy `demo/price_project` to `/tmp/localloop-price-demo` and confirm that its two intended tests fail.
- Run `localloop doctor` before recording, then clear the terminal so no model list or account information is visible.
- Rehearse three times and keep the most reliable shorter take as a backup.

## Timeline and narration

**0-10 seconds:** “LocalLoop is a coding agent whose conversation loop, context management, local tools, termination, and error handling are implemented from scratch.” Show the README architecture diagram.

**10-72 seconds:** Show the two initial failing tests. Run the prepared task with `--auto-approve` in the disposable `/tmp` copy. Keep tool names, write diff, test failure or success, and final summary visible; speed up pauses rather than cutting away the causal sequence.

**72-100 seconds:** Show `agent.py` for the bounded loop, then `tools.py` for workspace resolution, SHA-256 stale-write protection, argv-only execution, timeout, and environment sanitization. Show the passing test line.

**100-110 seconds:** “The deterministic test suite covers normal multi-turn behavior, malformed calls, retries, context compaction, resume, and security boundaries. The controls reduce accidents but are not an OS sandbox.”

## Export check

Export H.264 MP4 at 1080p or 720p, verify duration is below 120 seconds and size below 200 MB, then watch the final export once with sound.

