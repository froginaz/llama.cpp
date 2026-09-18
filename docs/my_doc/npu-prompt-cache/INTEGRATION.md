# INTEGRATION.md - Applying npu-prompt-cache to the NPU ggml-backend

Audience: the coding agent implementing the dNPU ggml-backend (Phase-1
profile: single `GGML_OP_CUSTOM` node, firmware-internal KV, `tokens/pos ->
logits` API, host sampling, variable ubatch). Read `00-handoff-npu-backend.md`
first if you have not. This document tells you exactly where each module
plugs in, what the firmware/driver must guarantee, and in which order to
integrate and verify.

---

## 0. What you received

| File | Module | Runs where | Depends on |
|---|---|---|---|
| `src/backend_stream_tracker.hpp` | **BackendStreamTracker** | inside your ggml-backend, at graph_compute time | nothing (header-only) |
| `src/kv_prefix_cache.hpp/.cpp` | **KvLedger** | host runtime layer (standalone runner, or a `llama_memory_i` impl) | a device trim command + epoch echo |
| `tests/*` | executable spec | host, `make test` | none (FakeDev) |

The two modules are **independent**. They solve the same problem (prefix
reuse when the host cannot read device KV) at two different layers. You will
most likely integrate the tracker first and the ledger later - see the
decision table below.

| Your situation | Use |
|---|---|
| llama.cpp drives the backend; you cannot or do not want to touch llama.cpp host code yet | **Path A: tracker only** (zero new firmware commands) |
| You own a standalone runner (no llama.cpp server) | **Path B: KvLedger** as the runner's planner |
| You are implementing `llama_kv_cache_dnpu` (a `llama_memory_i`) | **Path B'**: absorb KvLedger into it; keep epoch fencing internal |
| llama-server with `cache_prompt` on top of your backend | tracker still useful as a safety net; do NOT duplicate the server's LCP with a second host planner |

---

## 1. Path A - BackendStreamTracker in graph_compute (do this first)

### 1.1 Why this path is special

It requires **no new firmware protocol**. The tracker never trims, never
clears, never reads device KV. Invalidation happens purely by overwrite.
If the three firmware invariants in 1.4 already hold (Phase-1 mostly does),
this path turns "agent framework resends the whole prompt every turn" from
O(prompt) firmware work into O(new suffix) - the dominant TTFT win - while
staying correctness-safe on every edge (divergence, rewind, gaps).

### 1.2 Where to hook

In your backend's graph_compute handler for the CUSTOM node, before
dispatching to firmware:

```cpp
// one tracker instance per ggml-backend context, persistent across calls
BackendStreamTracker tracker;

// per forward call: build the view from the node inputs
ForwardView v;
v.seq          = 0;                    // Phase-1: single sequence
v.pos_begin    = ((const int32_t *) inp_pos->data)[0];
v.n_tokens     = (int32_t) inp_tokens->ne[0];
v.tokens       = (const token_t *) inp_tokens->data;
v.wants_output = /* see 1.3 */;

ForwardDecision d = tracker.observe(v);
if (d.skip_compute) {
    return GGML_STATUS_SUCCESS;        // KV for this window is already on device
}
// dispatch [d.compute_from, d.compute_from + d.n_compute) to firmware
```

Data sources in llama.cpp: the token window arrives as the `inp_tokens`
input tensor (I32, `ne[0] == n_tokens`) and absolute positions as `inp_pos`.
For the single-node design both are srcs (or registered side inputs) of your
CUSTOM node. Positions are contiguous within one ubatch, so `pos[0]` +
`n_tokens` fully describes the window; you may assert
`pos[n-1] == pos[0] + n - 1`.

### 1.3 Deriving `wants_output` - get this right or you lose everything

`skip_compute = full_match && !wants_output`. Two failure modes:

- `wants_output = true` always -> **no window ever skips**; the tracker
  becomes dead code. This happens if you naively map it to "the Phase-1 API
  returns logits on every call".
- `wants_output = false` always -> the final window of a turn skips and
  **no logits are produced** (violates the I4 analog).

Correct source: llama.cpp marks output rows per ubatch (`ubatch.output`
flags / `n_outputs`). Wire that into your launch path:
`wants_output = (n_outputs_for_this_ubatch > 0)`. If your integration
cannot see ubatch metadata, an acceptable proxy for the llama.cpp decode
flow is: the host reads logits back after this call (your driver knows
whether a logits DMA/readback was requested). Do not guess beyond these two.

### 1.4 Firmware invariants required by Path A (verify before enabling)

- **F1 - persistence**: device KV persists across forward calls. No implicit
  clear on `pos_begin == 0`. (If your firmware auto-resets on a new turn,
  disable that; the tracker's NEW_TURN/divergence handling replaces it.)
- **F2 - absolute-position writes**: `forward(tokens, pos[])` writes KV rows
  at the given absolute positions, overwriting whatever was there
  (idempotent overwrite). Phase-1's contiguous layout (cell index ==
  position) satisfies this.
- **F3 - position-causal attention**: a row at position p attends only to
  KV entries at positions <= p. This is why stale entries beyond the
  current tail are harmless: they are never attended. Standard causal
  attention by position satisfies this.
- **F4 - mid-window start (optional)**: partial-match windows resume at
  `compute_from`, which is generally not ubatch-aligned. The RoPE kernel
  must honor arbitrary position offsets. If it cannot yet, recompute the
  whole window instead: pass `pos_begin/n_tokens` rather than
  `compute_from/n_compute` - equally correct by F2 (idempotent overwrite),
  just slower. Reproduce Test B's T4 pattern (divergence at 12'345, not a
  multiple of any ubatch size) on hardware before trusting mid-window start.

### 1.5 What the tracker does NOT do

- No multi-sequence support wired yet: Phase-1 uses `seq = 0`. When you add
  multi-seq (equal_seqs ubatches), instantiate per-seq streams by passing
  the real seq id; the class already keys on it.
- No host `seq_rm/seq_add` awareness: it observes only forward streams. If
  the host later starts trimming (llama-server cache_prompt), those show up
  as REWIND/NEW_TURN windows and are handled correctly - but K-shift
  (`seq_add`) is NOT: shifted positions change RoPE content, which
  position-based matching cannot see. Keep context-shift/`--cache-reuse`
  off (they are grade B; see doc 14's grade ladder) until there is a
  real shift command.

---

## 2. Path B - KvLedger in the host runtime

### 2.1 Standalone runner loop

```cpp
FakeDev -> your driver shim implementing INpuKvBackend
KvLedger led(cfg, &drv);

// per turn:
auto plan = led.plan_prefill(prompt);            // nullopt: no quiescent slot
for (auto & ub : plan->ubatches) {
    if (!led.on_ubatch_submitted(ub)) break;     // stale epoch: replan
    drv.submit(ub);                              // async ok
}
led.wait_quiescent(plan->slot);                  // completions arrive on driver thread
// decode:
for (;;) {
    token_t t = sample(logits);
    led.append_decoded(plan->slot, t);           // device wrote that KV entry
    ...
}
```

Driver completion callback (any thread - the ledger is thread-safe):

```cpp
void on_complete(desc)  { led.on_ubatch_committed(desc.slot, desc.pos_begin, desc.n, desc.epoch); }
void on_error(desc)     { led.on_ubatch_failed(desc.slot, desc.epoch); }
```

### 2.2 Driver/firmware protocol requirements (these ARE new commands)

- **kv_trim(slot, from_pos)**: real device command invalidating
  [from_pos, inf). Phase-1 firmware manages KV internally, so this command
  must be added; `kv_trim(slot, 0)` may be implemented as reset.
- **epoch echo**: every completion must carry the epoch it was submitted
  under. Wire format: put the 64-bit epoch (or a driver-side generation tag
  mapped 1:1 to it) in the submission descriptor and echo it verbatim in
  the completion. This is the fencing that makes async completions safe
  across trims - do not "optimize" it away; a stale completion applied
  after a trim silently corrupts the committed count (the mirror then
  over-claims, violating I1, and the next turn reuses garbage).
- **partial-failure semantics**: after a failed/partial ubatch the host
  sends `kv_trim(slot, committed)`; firmware must make that restore a
  consistent state regardless of how far the failed ubatch got.
- **contiguity**: submissions arrive in order with no position holes
  (planner guarantees it; firmware may assert `kv_tail == pos_begin`).
- `trim_granularity`: set to the device KV page size if the layout is
  paged; 1 for token-granular layouts.

### 2.3 Path B' - inside llama.cpp (`llama_kv_cache_dnpu`)

llama-server's `cache_prompt` already does host-level LCP (slot
`cache_tokens` + n_past), so do not run `plan_prefill` next to it. The
lasting value of KvLedger there is the **device-trim semantics + epoch
fencing** inside a `llama_memory_i` implementation. Mapping (verify current
interface names against the tree - the memory interface was refactored):

| `llama_memory_i` call | dNPU implementation | grade (doc 14) |
|---|---|---|
| `seq_rm(seq, p, -1)` (tail trim) | `kv_trim(slot, p)` + mirror truncate + epoch++ | A-with-trim |
| `seq_rm` middle range | return `false` -> server falls back to full reprocess (correct, slower) | - |
| `seq_cp`, `seq_add`, `seq_div` | unsupported; `get_can_shift() = false` | B/C |
| `state_write/read` | unsupported (report clearly) | D |
| `clear` | `kv_clear` + epoch++ | A |

Keep the epoch entirely internal to this class; llama.cpp above it must not
need to know.

---

## 3. Verification order (do not reorder)

1. **Host-only**: `make test` in this folder on your build machine. This is
   the executable spec; if you change any semantics, change the tests first.
2. **Driver shim**: reimplement `FakeDev` against the real driver
   (`INpuKvBackend` + a `run_ubatch` that submits and waits). Re-run Test A
   at its tiny sizes. The one assertion you cannot keep is
   `dev.kv == prompt` (KV is opaque): replace it with **logits equality vs
   the llama.cpp CPU backend** on the same prompt (golden reference; see
   doc 14 section 10). Everything else (committed/in_flight/epoch
   assertions) transfers unchanged.
3. **F4 check**: Test B's T4 pattern on hardware - divergence at a
   non-aligned position, verify logits match CPU whether you resume
   mid-window or recompute the window.
4. **Tracker in-backend**: enable Path A, replay Test C's stream shapes via
   llama-cli/llama-server with an agent-style workload (resend system
   prompt + history each turn). Verify (a) logits parity, (b) skip
   accounting close to Test C's numbers, (c) TTFT drop.
5. **Sustained soak**: long multi-turn session with random truncations and
   an injected driver failure per N turns; assert no divergence vs CPU
   reference afterward.

---

## 4. Pitfalls (each of these has bitten someone)

- **Token-id LCP vs retokenization drift**: reuse works on token ids, not
  text. Retokenized history must reproduce decode-time token boundaries;
  chat-template drift silently shortens LCP. Add a turn-2-style test
  against the real tokenizer before measuring reuse ratios.
- **Dynamic prompt segments** (timestamps injected early) cap reuse at the
  injection point. Place them late or normalize upstream. For A/B
  measurement record request payloads and replay original vs canonicalized;
  report R_t = (P_t - E_t)/P_t per turn.
- **Never skip the logits window** (I4 / `wants_output`). A "fully cached"
  turn must still evaluate at least the last token.
- **Do not simplify the two-watermark design** in the tracker to a single
  watermark. Truncating the mirror on a matching NEW_TURN window destroys
  the tail that later windows must match against; the separate
  `cursor`/`valid_end` is the fix, and Test C's growth case regresses
  immediately if you merge them.
- **Divergence truncation is correctness, not a policy**: with causal
  attention, KV past a divergence is invalid even if later token ids
  coincide. Do not "keep the tail just in case".
- **Epoch fencing is your atomicity guard** for async completion vs trim
  interleavings. If you add any new device mutation (a shift command, a
  device-side snapshot), it must bump the epoch too.
- **The `.github/workflows/ci.yml` here only runs if this folder is a repo
  root.** If you vendored the folder into a larger tree, wire `make test`
  into that tree's CI instead.

---

## 5. Suggested milestone tags

- M-pc1: Path A tracker integrated, `wants_output` wired, F1-F3 verified,
  logits parity on multi-turn resend.
- M-pc2: F4 verified on hardware (mid-window RoPE start) or explicitly
  waived to whole-window recompute.
- M-pc3: driver gains `kv_trim` + epoch echo; Test A green against real
  driver with logits-based end-check.
- M-pc4: `llama_kv_cache_dnpu` skeleton absorbing KvLedger; llama-server
  prefix reuse E2E with `cache_prompt`.
