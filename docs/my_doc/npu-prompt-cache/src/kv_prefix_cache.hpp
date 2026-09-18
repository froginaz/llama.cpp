// kv_prefix_cache.hpp - host-side prefix-cache planner for a discrete NPU
// (dNPU) whose KV-cache lives in device-private memory (host non-visible).
//
// The host cannot read device KV, so it keeps a per-slot token mirror
// ("ledger") of what the device holds and plans which prompt suffix to
// prefill. See kv_prefix_cache.cpp for the I1..I5 design invariants.

#pragma once

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <vector>

namespace npu_pc {

using token_t = int32_t;
using pos_t   = int32_t;
using epoch_t = uint64_t;

struct LedgerConfig {
    int32_t n_ubatch         = 512;
    int32_t trim_granularity = 1;   // 1 = token-level; set to KV page size for paged layouts
    int32_t n_slots          = 4;
};

// Device abstraction: the only two mutations the planner issues.
struct INpuKvBackend {
    virtual ~INpuKvBackend() = default;
    virtual void kv_trim(int32_t slot, pos_t from_pos) = 0; // invalidate [from_pos, inf)
    virtual void kv_clear(int32_t slot) = 0;
};

struct UbatchDesc {
    int32_t         slot;
    pos_t           pos_begin;
    int32_t         n_tokens;
    const token_t * tokens;    // points into the caller's prompt buffer
    epoch_t         epoch;
};

struct PrefillPlan {
    int32_t slot;
    int32_t n_prompt;
    int32_t n_reuse;
    int32_t n_eval;
    bool    did_trim;
    epoch_t epoch;
    std::vector<UbatchDesc> ubatches;
};

struct SlotStats {
    int32_t  committed;
    int32_t  in_flight;
    epoch_t  epoch;
    uint64_t last_used;
};

class KvLedger {
public:
    KvLedger(const LedgerConfig & cfg, INpuKvBackend * dev);

    // Plan the prefill for a prompt. Selects a quiescent slot (max LCP, or
    // LRU eviction when no slot shares a prefix); nullopt when no quiescent
    // slot exists or the prompt is empty.
    std::optional<PrefillPlan> plan_prefill(const std::vector<token_t> & prompt);

    // Same, but on a caller-chosen slot. nullopt if the slot is invalid or
    // not quiescent.
    std::optional<PrefillPlan> plan_prefill_on_slot(int32_t slot, const std::vector<token_t> & prompt);

    // Submission lifecycle. Return false = stale epoch, caller must drop.
    bool on_ubatch_submitted(const UbatchDesc & d);
    bool on_ubatch_committed(int32_t slot, pos_t pos_begin, int32_t n, epoch_t epoch);
    void on_ubatch_failed(int32_t slot, epoch_t epoch);

    // Decode path: the device wrote this token's KV entry during decode.
    void append_decoded(int32_t slot, token_t tok);

    void clear_slot(int32_t slot);
    void wait_quiescent(int32_t slot);   // cv-wait until in_flight == 0
    SlotStats stats(int32_t slot) const;

    // (slot, n_prompt, n_reuse) - fired after each successful plan.
    std::function<void(int32_t, int32_t, int32_t)> on_plan_metrics;

private:
    struct Slot {
        std::vector<token_t> tokens;  // mirror, size == committed + in_flight
        int32_t  committed = 0;
        int32_t  in_flight = 0;
        epoch_t  epoch     = 0;
        uint64_t last_used = 0;
    };

    int32_t lcp_committed_locked(int32_t slot, const std::vector<token_t> & prompt) const;
    std::optional<PrefillPlan> plan_on_slot_locked(int32_t slot, const std::vector<token_t> & prompt);

    LedgerConfig    cfg_;
    INpuKvBackend * dev_;

    std::vector<Slot>       slots_;
    mutable std::mutex      mtx_;
    std::condition_variable cv_;
    uint64_t                use_tick_ = 0;
};

} // namespace npu_pc
