# HANDOFF: NPU Backend Collaboration

This document is the **single entry point for an external collaborator (human or Claude) working in a separate environment** who continues the NPU backend integration on top of the documentation work in this repo.

Ground rules:

- Transfer is **one-way**: this repo -> collaborator. The collaborator's NPU code never comes back to this repo (security constraint).
- The collaborator receives a snapshot of this branch (clone or git bundle) and continues in their own environment.
- Reference branch: `claude/llaam-context-ggml-backend-qb6nqa`
- The documentation series itself (01~13) is written in Korean; this handoff document is in English.

**Status**: the collaborator has already **completed Phase 1** (single-sequence end-to-end path based on a single-node CUSTOM cgraph; see the confirmed profile in section 2). The center of gravity of this document is therefore not "getting started" but a **reference map for the next phase**: section 3 (doc map), section 4 (KV ownership decision), and section 5 (the host-visible KV implementation plan).

---

## 1. What this branch contains

The documentation series under `docs/my_doc/` is the entire deliverable (no code changes). Chapters 01~11 document the pipeline from GGUF loading to graph execution; 12~13 were added on this branch:

- **12** ([12-context-model-backend-ownership.md](12-context-model-backend-ownership.md)): relationships, ownership, and lifecycle of `llama_context` / `llama_model` / ggml-backend. Includes UML classification (association vs composition) and the context:model combination matrix (multi-context != multi-model, MTP case).
- **13** ([13-batch-ubatch-and-sequences.md](13-batch-ubatch-and-sequences.md)): execution data units. batch (logical) vs ubatch (physical), sequence vs context, request/slot/session terminology, annotated decode/prefill deep-dive diagrams (tensor shapes, KV cell layout), multi-sequence executables, and a KV-cache sizing appendix.

The commit history is the work timeline: `git log --oneline -- docs/my_doc` shows which question/decision landed in which order (all commits carry the `[DOC]` prefix).

## 2. dNPU integration status (Phase 1 complete - confirmed profile)

dNPU artifact structure: a **neural net constructor** (LLM topology + firmware ELF binary) + separate **weight files (.bin)** + **weight_info** (weight-usage metadata, json -> bin). These are mapped as GGUF tensors so they ride the llama_model loading path (mmap, buft, memory_breakdown) and integrate with ggml. Because the source of truth for the topology is the constructor/ELF rather than GGUF hparams, it is recommended to **embed a topology hash/version of the constructor into GGUF kv metadata and cross-validate at load time** (prevents silently loading a mismatched weight_info/ELF combination).

Confirmed decisions on the collaborator side, and what they imply:

| # | Confirmed | Implication in llama.cpp terms |
|---|---|---|
| 1 | NPU API: input = tokens/pos, output = logits. **No batch support (no mixing of sequences)** | Current profile is n_seq_max=1, single conversation. Adding per-token seq_id[] and output selection (equivalent of logits[]) to the API is the gateway to multi-sequence |
| 2 | **Variable ubatch width supported** | Prefill (hundreds of tokens) vs decode (1 token) width changes need no compiled-shape variants. Only the max width (n_ubatch cap) needs confirming |
| 3 | **KV cache managed inside firmware** (host-visible buffer is a future key feature) | `llama_memory_i` degenerates to a pass-through. seq_cp (prompt sharing) / seq_rm / context shift / session save-restore (`llama_state_seq_*`) / server slot-cache reuse are all unavailable. Once visible, the cell model of doc 13 applies directly under host management |
| 4 | cgraph = input (leaf) -> **one CUSTOM node** -> logits | build_arch_graph creates a single node. Graph rebuild cost ~0, can_reuse effectively meaningless, the only live gparam is the ubatch width |
| 5 | No cross-HW scheduling by sched (per-op support via a standard GGUF template is the Phase 2 plan) | Today sched only assigns the CUSTOM node and moves inputs/outputs (tokens in / logits out) |
| 6 | **lm_head runs inside the NPU** (output is logits) | lm_head = the `model.output` matrix projecting the final hidden state [n_embd] to logits [n_vocab] (4096x32000, ~262MB at F16). Transfer per output token is n_vocab floats (~128KB). The alternative (NPU stops at hidden state) cuts transfer 8x but moves a large matmul to the host |
| 7 | **Sampling on host** | llama.cpp sampler chain used as-is; only the backend-sampler feature is unused |
| 8 | supports_op = **GGML_OP_CUSTOM** | true only for the CUSTOM node. CPU fallback / partial offload only become meaningful with per-op support (Phase 2) |

One-line summary: **"a llama-cli / llama-simple grade single-sequence path in which build_graph collapses to one CUSTOM node."** The two gateways to multi-sequence are (1) seq_id support in the batch API, and (2) host-visible KV.

## 3. Next steps <-> reference document map

| Next step (key feature) | Directly relevant documents / diagrams |
|---|---|
| **seq_id support in the batch API** -> multi-sequence | Doc 13 ch.1~3: `llama_batch` seq_id/pos/logits layout (ch.2 table), ubatch split rules (split_simple), and in the annotated prefill/decode diagrams, **how KQ_mask implements sequence isolation** - the exact contract the NPU firmware must reproduce |
| **Host-visible KV** | **Full implementation plan in section 5 of this document.** Background: doc 13 ch.6 (cell model: K-vector width x cell count x streams), the KV state tables in the deep-dive diagrams (meaning of find_slot/apply), doc 12 buft/buffer ownership (who allocates/frees) |
| Features unlocked by visible KV (use as a verification checklist) | Doc 13 ch.5: seq_cp prompt sharing, session save/restore, llama-server slot-cache reuse |
| **Standard GGUF template + per-op support** (Phase 2) | Docs 01~05 (GGUF -> hparams -> build_graph -> cgraph pipeline), `tests/test-backend-ops.cpp` (per-op correctness) - CPU fallback / partial offload / op-level verification only come alive at this stage |

**Note on gateway ordering**: opening the batch API first yields "multiple conversations batched per pass but KV still in firmware", which forces the firmware to reproduce per-seq cell tagging and mask isolation internally (a firmware clone of doc 13's unified-KV model). Opening visible KV first lets llama.cpp's existing memory module be reused as-is, shifting the burden to host-side verification - **visible-KV-first reuses more llama.cpp infrastructure**.

## 4. KV-cache ownership: llama.cpp layer vs firmware layer

The essential difference is **who holds the cell ledger** (which cell belongs to which sequence at which position).

| Aspect | **llama.cpp layer** (`llama_memory_i`) | **firmware layer** (current dNPU) |
|---|---|---|
| Cell ledger | Held by host - every cell tagged with seq_id/pos, exposed as ggml tensors | Inside firmware, opaque to host |
| What the host sends | ubatch + slot plan (`init_batch`/`apply`) + KQ_mask | tokens/pos only - "append internally" |
| KV manipulation | `seq_rm/seq_cp/seq_keep`, shift, defrag - any range, any time | Only what the firmware exposes as API |
| Attention isolation | Host-built KQ_mask is the contract | Firmware-internal logic is the contract |
| Physical layout | Firmware must honor the buft-defined cell layout (doc 13's cell model) | Firmware free to tile/compress/page internally |

### Benefits of llama.cpp-side management

1. **Every feature in doc 13 comes for free** - this is the decisive advantage:
   - multi-sequence serving: unified pool + per-seq tagging -> `-np N`, server slots, continuous batching
   - prompt sharing: `seq_cp` adds tags without copying (prefill a system prompt once, share across N slots)
   - partial removal / rollback: `seq_rm(seq, p0, p1)` - without this, **speculative decoding is impossible** (rejected draft tokens must be evicted), and so is context shift
   - session save/restore: `llama_state_seq_*` requires KV to be visible as ggml tensors
   - defrag/optimization: the `memory_update(true)` retry path
   - quantized KV: `-ctk/-ctv q8_0` handled at the buft level
   With firmware management, each of these becomes one more firmware API (rollback API, copy API, save API, ...) whose semantics must be kept in sync with llama.cpp forever.
2. **Ecosystem compatibility**: llama-server slot-cache reuse (`-sps`), `llama-batched`/`llama-parallel`, and future upstream memory types (SWA/iSWA, MLA, hybrid, recurrent) work on the dNPU **without code changes**. With firmware management, every tool accumulates "not on dNPU" exceptions.
3. **Observability and debugging**: host can inspect cell state (`memory_breakdown()`, logs, the "cells 0..6 = seq 0" style verification drawn in the deep-dive diagrams). With firmware management, KV inconsistency bugs hide inside a black box.
4. **Single source of semantics**: KQ_mask/pos/isolation rules are defined once in llama.cpp and are identical to CPU/CUDA, so cross-verification (`test-backend-ops` style) is possible.

### Benefits of firmware-side management (for fairness)

- Simple host interface (tokens in / logits out) - optimal for getting Phase 1 working quickly, which is exactly what happened
- Layout freedom: tiling/compression/banking tuned to the NPU memory hierarchy without a host contract
- No host-device synchronization of cell metadata
- The only option when KV memory is not host-mappable

### Conclusion

> Firmware management is "**a simple interface, a wall of features**"; llama.cpp management is "**a layout contract, freedom of features**". Up to single-conversation E2E (the current point) firmware management is reasonable, but the roadmap destinations - multi-sequence serving, speculative decoding, slot caching - all stand on llama.cpp-managed KV. Hence "host-visible KV is the key feature" is the right call, and its substance is: **the firmware accepts externally injected KV buffer pointers plus the cell-layout contract (K-vector width x cell count x streams)**. As an interim compromise, adding just two firmware APIs first - per-seq rollback and full reset - unlocks speculative decoding and basic server operation.

## 5. Implementation plan: migrating to host-visible KV

This section is the concrete plan for the key feature identified in section 4. It is written so the collaborator can execute it without access to this conversation. A Korean technical companion covering the same ground in more depth (mechanism, contract, sequence diagram) is [doc 14](14-host-visible-kv.md).

### 5.0 Why - the tensor-role taxonomy and the "stateful node" problem

A ggml graph references three kinds of tensors:

| Role | Data lives in | Host fills it every step? | Examples |
|---|---|---|---|
| **True inputs** (`ggml_set_input`, copied by `set_inputs()`) | compute buffer | **yes** | `inp_tokens`, `inp_pos`, `KQ_mask`, `inp_out_ids`, **`k_idxs` / `v_idxs`** |
| **Persistent leafs** (exist outside the graph) | weight buffer / **KV buffer** | no - referenced by pointer | weights, **`cache_k_l` / `cache_v_l`** |
| Computed nodes | compute buffer | no - produced by the graph | Q/K/V, scores, FFN outputs |

In stock llama.cpp the KV cache is a **persistent, mutable leaf**: attention reads it through zero-copy views (`[d, n_kv]` window) and writes new entries through `ggml_set_rows(cache, cur, idxs)` nodes **inside the graph**. The "past" is never passed as data; only per-step **metadata** is passed as true inputs: `k_idxs`/`v_idxs` (which cells to write; built by `build_input_k_idxs`, src/llama-kv-cache.cpp:1294) and `KQ_mask` (which cells may be attended - the modern replacement of the old `n_past` parameter).

Today's dNPU CUSTOM node hides KV inside firmware, so the node is **stateful**: it is not a pure function of its inputs and leafs. That single property is what blocks rollback (speculative decoding), `seq_rm/seq_cp`, session save/restore, and multi-sequence slot management. The goal of this migration is to make the node **pure again**: KV becomes an explicit persistent-leaf operand; per-step state selection travels through `k_idxs`/`v_idxs`/`KQ_mask`.

### 5.1 Target architecture

Layout contract (per layer, identical to stock llama.cpp; see doc 13 ch.6 and the deep-dive axis notes):

```
cache_k_l{il} : [n_embd_k_gqa, kv_size, n_stream]   (ne0 contiguous = one cell = one row)
cache_v_l{il} : [n_embd_v_gqa, kv_size, n_stream]   (v_trans = false, i.e. FA-style layout)
dtype: F16 first (quantized KV is a later option and requires firmware dequant)
```

Per-step data flow after migration:

```
host (llama.cpp)                          NPU firmware
----------------                          ------------
memory->init_batch/apply                  -
  -> plans cells, builds k_idxs/v_idxs
set_inputs: tokens, pos,                  forward(tokens, pos, k_idxs, v_idxs,
  k_idxs, v_idxs, KQ_mask  ------------->         n_kv, KQ_mask):
                                            for each layer:
                                              write new K/V rows at cells k_idxs[i]
                                              attend over window [0, n_kv) under KQ_mask
                                          <- logits
```

Invariants the firmware MUST honor (these are the semantic contract, not suggestions):

1. **Mask-only isolation**: `seq_rm` only edits host-side cell metadata - cells are never zeroed. Firmware must never assume residual cell contents mean anything; only `KQ_mask` and the idxs define validity.
2. **Idxs may be non-contiguous** (after defrag/reuse) - no "append at tail" assumption.
3. **Same cell index across all layers** for a given token (one ledger, 2 x n_layer tensors).
4. `n_kv` is a padded window; masked-out padding cells must contribute nothing.
5. Host may write KV outside the graph (session restore via buffer `set_tensor`) and read it (session save) - firmware must tolerate external writes between forwards.

### 5.2 Milestones

- **M0 - decisions**: KV residency (recommended: NPU-local memory that the host can also address; if not host-mappable, buffer `set_tensor`/`get_tensor` must be implemented as DMA copies). Confirm firmware can consume an externally provided KV region table. **No shared memory is required**: in a PCIe split-memory setup, KV stays resident in NPU DRAM and steady-state decode moves only metadata + logits over PCIe - exactly the CUDA dGPU model. See doc 14 section 6 for the per-operation traffic table and sequence diagram.
- **M1 - NPU buffer_type for KV**: extend the existing NPU buft (already used for GGUF-mapped weights) with a mutable, host-writable buffer: `alloc_buffer`, `get_base`, `set_tensor`, `get_tensor`, `clear`. `get_tensor` is what makes session save possible; `set_tensor` makes restore possible.
- **M2 - firmware API**: (a) `register_kv_region(layer, k_base, v_base, cell_stride, n_cells, n_stream, dtype)` called once at context init - a descriptor table avoids passing 2 x n_layer tensors as node srcs (**`GGML_MAX_SRC` = 10**, ggml.h:224 - a single CUSTOM node cannot carry 64 cache operands; the table is the Phase-1-compatible workaround, per-layer nodes are the Phase-2 alternative). (b) extend `forward()` with `k_idxs/v_idxs/n_kv/KQ_mask`.
- **M3 - llama.cpp side**: use the **stock `llama_kv_cache`** with the NPU buft - no custom memory class should be needed. In the custom `build_arch_graph`, reuse llama.cpp's own helpers (`build_input_k_idxs`, the kq_mask builders) so semantics match CPU exactly; `set_inputs` then works unchanged.
- **M4 - parity verification** (in order):
  1. single-seq logit parity vs CPU backend on fixed prompts (tolerance ~1e-2 relative for F16);
  2. **rollback determinism**: decode N tokens, `llama_memory_seq_rm(tail)`, re-decode - outputs must match bit-for-bit with a fresh run;
  3. session roundtrip: `llama_state_seq_save_file` -> restore into a fresh context -> continue decoding, compare;
  4. `memory_breakdown()` reports KV on the NPU buft.
- **M5 - feature unlock ladder**: rollback test green -> speculative decoding usable; then batch-API seq_id (the other gateway) -> `llama-batched -np 2` -> `llama-parallel` -> llama-server slots with `-sps` cache reuse.

### 5.3 Pitfall checklist

- `GGML_MAX_SRC = 10`: do not try to make cache tensors node srcs on the monolithic node (see M2).
- Do not zero/scrub cells on `seq_rm` and do not rely on zeroed memory - mask-only isolation.
- Use the `!v_trans` (FA-style) layout; the transposed-V path (llama-kv-cache.cpp:1257 onward) exists only for non-FA CPU/GPU kernels.
- Synchronization: host `set_tensor` writes (session restore) must be visible to firmware before the next forward; wire this through the backend's `synchronize`.
- `n_ctx` is padded to 256 (llama-context.cpp:204); the mask width `n_kv` is padded separately - never derive one from the other.
- Cell metadata (which seq/pos owns a cell) lives **only on the host**; firmware sees indices and masks, never seq_ids.

### 5.4 Definition of done

Host-visible KV is "done" when: parity (M4.1) + rollback (M4.2) + session roundtrip (M4.3) all pass, and `llama-batched -np 2` runs correctly once the batch API gains seq_id support. At that point every feature in doc 13 ch.5 becomes available without further firmware changes.

## 6. Reading order (NPU backend perspective)

Not everything needs to be read. For NPU backend work:

1. **[11-hw-backend-abstraction.md](11-hw-backend-abstraction.md)** - abstracting one HW accelerator across ggml / ONNX Runtime / ExecuTorch. The big picture.
2. **[12-context-model-backend-ownership.md](12-context-model-backend-ownership.md)** - identity and ownership of the six types: reg / device / backend / buft / buffer / sched. Which layer the NPU must implement, and who creates/destroys what. See especially ch.1 and the class diagram.
3. **[13-batch-ubatch-and-sequences.md](13-batch-ubatch-and-sequences.md)** - the execution units a backend actually receives. **Compute-buffer size is determined by the n_ubatch width**; graph-reuse conditions; the tensor-shape flow in the [annotated deep-dive puml](13-sequence-decode-batch2-single-context-detailed.puml) bears directly on NPU memory design.
4. (For more background) 06-execution-and-scheduling.md, 08-hardware-divergence.md.

## 7. Code entry points

| File | Role |
|---|---|
| `ggml/include/ggml-backend.h` | Public API - what the backend consumer (llama.cpp side) sees |
| `ggml/src/ggml-backend-impl.h` | **The iface tables a backend implementor must fill** (device / buffer_type / buffer / backend) |
| `ggml/src/ggml-backend-reg.cpp` | Registry - a new backend must register here to appear in device enumeration |
| `ggml/src/ggml-blas/` | Smallest reference implementation (good starting point) |
| `ggml/src/ggml-cuda/` | Complete reference implementation (buffers/streams/graphs) |
| `tests/test-backend-ops.cpp` | **Standard backend verification tool** - per-op comparison against CPU. Run it for every new NPU kernel |

The contract with sched is central: a backend declares its capabilities via `supports_op` / `supports_buft` / `offload_op`, and **unsupported ops fall back to CPU automatically** (see the sched role notes in doc 12). An NPU therefore does not need every op on day one - incremental growth from mul_mat outward is the normal path (this applies to Phase 2; Phase 1 uses a single CUSTOM node, section 2).

Recommended smoke-test ladder: `test-backend-ops` (op correctness) -> `llama-cli -m tiny-model -ngl 99` (single-seq E2E) -> `llama-batched -np 2` (multi-seq) -> `llama-batched-bench` (throughput).

## 8. Working conventions (as practiced in this repo)

- **Read AGENTS.md first**: AI-generated PRs to upstream (ggml-org/llama.cpp) are prohibited. **Private forks / internal work are exempt**, so the NPU work itself is fine - but if any of it is ever aimed at upstream, the AGENTS.md process (a human must fully understand and be able to defend the change) applies.
- Documentation conventions: `NN-topic.md` numbering, PlantUML both inline (code block) and as separate `.puml` files, keep the README.md index updated, ASCII arrows (`->`).
- Commit conventions: `[DOC] one-line summary` + `Assisted-by: Claude` trailer (Co-authored-by is prohibited by AGENTS.md).

## 9. Transfer method (bundle recipe)

### If the collaborator can read this repo

Hand over the branch name and this document's path: clone -> start at `docs/my_doc/00-handoff-npu-backend.md`.

### Fully isolated environment: git bundle (validated recipe)

Assuming the collaborator has a public llama.cpp clone, build an incremental bundle containing only this branch's unique commits and carry it across as a file. This recipe was validated in this repo (bundle size ~138KB).

```bash
# [sender] basis = the upstream commit just before the doc series started
#   (the receiver is guaranteed to have it)
#   find it: parent of `git log --oneline --reverse -- docs/my_doc | head -1`
git bundle create handoff-$(date +%Y%m%d).bundle \
    6b80c74..claude/llaam-context-ggml-backend-qb6nqa
git bundle verify handoff-*.bundle   # "requires this ref: 6b80c74..." is expected
git tag shared/1 claude/llaam-context-ggml-backend-qb6nqa   # record the share point

# [receiver] inside a public llama.cpp clone
git fetch /path/to/handoff-YYYYMMDD.bundle \
    claude/llaam-context-ggml-backend-qb6nqa:refs/heads/npu-work
git checkout npu-work    # full docs/my_doc/ + commit history

# [sender] later increments contain only the delta
git bundle create incr-$(date +%Y%m%d).bundle shared/1..claude/llaam-context-ggml-backend-qb6nqa
git tag -f shared/2 claude/llaam-context-ggml-backend-qb6nqa
# receiver: git fetch /path/to/incr.bundle onto the same npu-work branch
```

Caveats:

- **If the sender's clone is shallow** (common in CI/remote environments), a full-history bundle (`git bundle create x.bundle <branch>`) will be created but the receiver's clone from it fails. Use the **basis-ranged incremental bundle** above, or run `git fetch --unshallow origin` before a full bundle.
- The basis commit must exist upstream (a commit with a PR number in its subject is safe).
- If only the files (no history) are needed, `git archive HEAD docs/my_doc -o docs.tar` is the minimal alternative - not recommended, since the commit-message timeline is lost.

## 10. For the collaborator: continuing to the next phase

1. From this document, jump via the section-3 map to the documents for the key feature being started (first time: full reading order in section 6, ~30 minutes). For the host-visible KV migration specifically, execute the plan in section 5.
2. Skim the decision timeline: `git log --oneline -- docs/my_doc`.
3. Do the NPU work **on a branch in your own environment** (never push to this repo). If you add documents, continue the numbering from 15 (14 is taken by the host-visible KV doc), and maintain your own README index - your documents will not come back to this repo either.
4. When this repo's work is updated, receive it via the incremental bundle of section 9. No conflict risk: this side only touches `docs/my_doc/`, the NPU code lives under `ggml/src/`.
