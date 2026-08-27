# Design defense notes

## One-sentence architecture

LocalLoop is a bounded state machine: the model proposes typed function calls, the program validates and executes them locally, matching results are appended to history, and the model observes those results before deciding the next action.

## Questions to be ready for

### Why native tool calling instead of asking for JSON in text?

Native calls provide explicit call IDs, names, and argument boundaries. This removes brittle markdown parsing and lets multiple calls and their results remain correctly associated. A model without native function calling is rejected by `doctor` rather than silently using an unreliable fallback.

### Why is using the OpenAI client allowed here?

It is only a model-vendor transport client. It does not manage the conversation, choose tools, execute code, store files, compact context, retry the workflow, or decide termination; those responsibilities are implemented in this repository.

### Why argv instead of shell command strings?

An argv array is passed directly to the process launcher with `shell=False`, so shell metacharacters are data rather than executable syntax. This reduces injection risk and makes the exact command visible for approval. It intentionally sacrifices pipes and redirection.

### What does SHA-256 protect?

`read_file` hashes the entire current file. Updating an existing file requires that hash. If another process changes the file between read and write, LocalLoop rejects the write and asks the model to read again, preventing silent stale overwrites.

### How does context compaction remain API-valid?

Messages are grouped so an assistant tool-call message and all immediately following tool results are never separated. Old complete groups become ordinary assistant summaries; recent groups remain exact. If even recent interactions exceed the budget, the run stops clearly instead of sending malformed history.

### Why deterministic compaction rather than model summarization?

It costs no extra call, cannot hallucinate facts, and is easy to test. The complete transcript remains in JSONL, while only the request view is shortened. The tradeoff is that old semantic detail can be lost.

### When does the loop stop?

It stops on a final non-empty assistant message, maximum steps, total duration, three identical consecutive call batches, empty output, provider/configuration failure, context exhaustion, or user interruption. Each terminal event is recorded.

### What is and is not secured?

Resolved file paths cannot leave the selected workspace; sensitive paths are blocked; writes require approval and a current hash; subprocesses receive a narrow secret-free environment, no shell, a timeout, and output limits. This is defense against mistakes, not a kernel sandbox: an approved arbitrary executable can still have side effects available to the current user.

### How is resume kept consistent?

The append-only JSONL contains versioned metadata and every exact system, user, assistant, and tool message. Resume reconstructs the same message sequence. A partial last line is treated as corruption rather than guessed, making failure visible.

### What would be added with another week?

An OS sandbox/container backend, streaming output, patch-oriented editing for very large files, exact model-specific token counting, property-based path tests, and an evaluation set with task success and tool-efficiency metrics.

