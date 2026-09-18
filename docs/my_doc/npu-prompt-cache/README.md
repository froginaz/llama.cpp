# npu-prompt-cache

Host-side prompt (prefix) caching for an LLM inference stack whose KV-cache
lives in NPU device-private memory (host non-visible). Two standalone C++17
modules, no external dependencies:

- `src/kv_prefix_cache.hpp/.cpp` - **KvLedger**: host mirror of device KV
  state + LCP prefill planner. Design invariants I1..I5 (committed-only
  accounting, epoch fencing for async completions, last-token re-eval,
  granularity-aligned trim) are commented in the source.
- `src/backend_stream_tracker.hpp` - **BackendStreamTracker**: graph-time,
  ordinal-free ubatch stream tracker. Classifies each forward window by
  absolute position only (NEW_TURN / CONTINUATION / REWIND / GAP) using two
  watermarks per sequence (`cursor`, `valid_end`), and decides skip/compute
  without knowing "ubatch #k of turn T".

Workload target: agent frameworks that resend `system prompt + tool schemas
+ history` every turn, so prefix reuse dominates TTFT. Relates to the
ledger/visible-KV material in [14-host-visible-kv.md](../14-host-visible-kv.md).

**Integrating this into the NPU ggml-backend?** Start with
[INTEGRATION.md](INTEGRATION.md) - it maps each module to its layer,
lists the firmware/driver invariants (F1-F4, trim + epoch echo), and gives
the verification order and pitfalls. A beginner-friendly Korean companion
is [INTEGRATION.ko.md](INTEGRATION.ko.md).

## Build and test

CMake, same flow as llama.cpp on every platform (on Windows run these from
a Visual Studio developer prompt / with MSVC, exactly as for llama.cpp):

```bash
cmake -B build
cmake --build build --config Release
ctest --test-dir build --build-config Release --output-on-failure
```

On POSIX a plain `make test` also works (GNU make + g++). Note: the
CMakeLists scrubs `NDEBUG` from Release flags on purpose - the tests are
assert-based and would otherwise pass vacuously.

Tests cover: small-scale ledger scenarios with failure injection and epoch
fencing, 25K-token plans across `n_ubatch x trim_granularity` combinations,
and stream-tracker skip accounting including decode steps and GAP handling.
