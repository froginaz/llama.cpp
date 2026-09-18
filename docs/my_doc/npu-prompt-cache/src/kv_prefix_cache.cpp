// kv_prefix_cache.cpp - see kv_prefix_cache.hpp
//
// Design invariants:
//   I1: `committed` counts only tokens GUARANTEED present in device KV.
//       Under-claiming after failures is allowed; over-claiming never.
//   I2: Reuse decisions use the COMMITTED region only; in-flight
//       (submitted-but-unacked) tokens are never reused.
//   I3: Every device-KV mutation (trim/clear/rollback) increments the slot
//       `epoch`. Async NPU completions echo the epoch they were submitted
//       under; stale-epoch completions are dropped (fencing).
//   I4: If the whole prompt matches the cache, still re-evaluate at least
//       the LAST token (backend must produce logits).
//   I5: Reuse/trim boundary is aligned DOWN to `trim_granularity`.

#include "kv_prefix_cache.hpp"

#include <algorithm>
#include <cassert>
#include <limits>

namespace npu_pc {

static int32_t align_down(int32_t v, int32_t g) {
    return g <= 1 ? v : (v / g) * g;
}

KvLedger::KvLedger(const LedgerConfig & cfg, INpuKvBackend * dev)
    : cfg_(cfg), dev_(dev), slots_(cfg.n_slots) {
    assert(dev_ != nullptr);
    assert(cfg_.n_ubatch >= 1 && cfg_.trim_granularity >= 1 && cfg_.n_slots >= 1);
}

int32_t KvLedger::lcp_committed_locked(int32_t slot, const std::vector<token_t> & prompt) const {
    // I2: compare against the committed region only
    const Slot & s = slots_[slot];
    const int32_t n = std::min<int32_t>((int32_t) prompt.size(), s.committed);
    int32_t i = 0;
    while (i < n && s.tokens[i] == prompt[i]) {
        i++;
    }
    return i;
}

std::optional<PrefillPlan> KvLedger::plan_prefill(const std::vector<token_t> & prompt) {
    std::optional<PrefillPlan> plan;
    {
        std::lock_guard<std::mutex> lk(mtx_);
        if (prompt.empty()) {
            return std::nullopt;
        }
        // slot selection: quiescent slots only, max LCP wins
        int32_t  best_slot = -1;
        int32_t  best_lcp  = -1;
        int32_t  lru_slot  = -1;
        uint64_t lru_tick  = std::numeric_limits<uint64_t>::max();
        for (int32_t i = 0; i < cfg_.n_slots; ++i) {
            const Slot & s = slots_[i];
            if (s.in_flight != 0) {
                continue;
            }
            const int32_t l = lcp_committed_locked(i, prompt);
            if (l > best_lcp) {
                best_lcp  = l;
                best_slot = i;
            }
            if (s.last_used < lru_tick) {
                lru_tick = s.last_used;
                lru_slot = i;
            }
        }
        if (best_slot < 0) {
            return std::nullopt;   // no quiescent slot
        }
        if (best_lcp == 0) {
            best_slot = lru_slot;  // eviction: no shared prefix anywhere
        }
        plan = plan_on_slot_locked(best_slot, prompt);
    }
    if (plan && on_plan_metrics) {
        on_plan_metrics(plan->slot, plan->n_prompt, plan->n_reuse);
    }
    return plan;
}

std::optional<PrefillPlan> KvLedger::plan_prefill_on_slot(int32_t slot, const std::vector<token_t> & prompt) {
    std::optional<PrefillPlan> plan;
    {
        std::lock_guard<std::mutex> lk(mtx_);
        if (prompt.empty() || slot < 0 || slot >= cfg_.n_slots) {
            return std::nullopt;
        }
        if (slots_[slot].in_flight != 0) {
            return std::nullopt;   // quiescence required
        }
        plan = plan_on_slot_locked(slot, prompt);
    }
    if (plan && on_plan_metrics) {
        on_plan_metrics(plan->slot, plan->n_prompt, plan->n_reuse);
    }
    return plan;
}

std::optional<PrefillPlan> KvLedger::plan_on_slot_locked(int32_t slot, const std::vector<token_t> & prompt) {
    Slot & s = slots_[slot];
    assert(s.in_flight == 0);
    assert((int32_t) s.tokens.size() == s.committed);

    const int32_t n_prompt = (int32_t) prompt.size();

    int32_t lcp = lcp_committed_locked(slot, prompt);
    // I4: a fully cached prompt still re-evaluates its last token for logits
    if (lcp >= n_prompt) {
        lcp = n_prompt - 1;
    }
    // I5
    const int32_t n_reuse = align_down(lcp, cfg_.trim_granularity);

    PrefillPlan plan;
    plan.slot     = slot;
    plan.n_prompt = n_prompt;
    plan.n_reuse  = n_reuse;
    plan.n_eval   = n_prompt - n_reuse;
    plan.did_trim = false;

    if (s.committed > n_reuse) {
        dev_->kv_trim(slot, n_reuse);
        s.tokens.resize(n_reuse);
        s.committed = n_reuse;
        s.epoch++;                 // I3
        plan.did_trim = true;
    }
    plan.epoch = s.epoch;

    for (pos_t pos = n_reuse; pos < n_prompt; pos += cfg_.n_ubatch) {
        const int32_t n = std::min<int32_t>(cfg_.n_ubatch, n_prompt - pos);
        plan.ubatches.push_back(UbatchDesc{slot, pos, n, prompt.data() + pos, s.epoch});
    }

    s.last_used = ++use_tick_;
    return plan;
}

bool KvLedger::on_ubatch_submitted(const UbatchDesc & d) {
    std::lock_guard<std::mutex> lk(mtx_);
    assert(d.slot >= 0 && d.slot < cfg_.n_slots);
    Slot & s = slots_[d.slot];
    if (d.epoch != s.epoch) {
        return false;              // I3: stale plan
    }
    // in-order contiguous submission
    assert(d.pos_begin == s.committed + s.in_flight);
    assert((int32_t) s.tokens.size() == s.committed + s.in_flight);
    s.tokens.insert(s.tokens.end(), d.tokens, d.tokens + d.n_tokens);
    s.in_flight += d.n_tokens;
    return true;
}

bool KvLedger::on_ubatch_committed(int32_t slot, pos_t pos_begin, int32_t n, epoch_t epoch) {
    std::lock_guard<std::mutex> lk(mtx_);
    assert(slot >= 0 && slot < cfg_.n_slots);
    Slot & s = slots_[slot];
    if (epoch != s.epoch) {
        return false;              // I3: stale completion dropped
    }
    assert(pos_begin == s.committed);
    // I1: committed grows only on acked device work
    s.committed += n;
    s.in_flight -= n;
    assert(s.in_flight >= 0);
    if (s.in_flight == 0) {
        cv_.notify_all();
    }
    return true;
}

void KvLedger::on_ubatch_failed(int32_t slot, epoch_t epoch) {
    std::lock_guard<std::mutex> lk(mtx_);
    assert(slot >= 0 && slot < cfg_.n_slots);
    Slot & s = slots_[slot];
    if (epoch != s.epoch) {
        return;                    // stale failure, already fenced
    }
    // I1: under-claim - roll the mirror back to the committed sync point.
    // Device state past `committed` is unknown after a partial failure, so
    // restore the known sync point on the device as well.
    s.tokens.resize(s.committed);
    s.in_flight = 0;
    dev_->kv_trim(slot, s.committed);
    s.epoch++;                     // I3
    cv_.notify_all();
}

void KvLedger::append_decoded(int32_t slot, token_t tok) {
    std::lock_guard<std::mutex> lk(mtx_);
    assert(slot >= 0 && slot < cfg_.n_slots);
    Slot & s = slots_[slot];
    assert(s.in_flight == 0);
    s.tokens.push_back(tok);
    s.committed += 1;              // the device wrote this KV entry during decode
}

void KvLedger::clear_slot(int32_t slot) {
    std::lock_guard<std::mutex> lk(mtx_);
    assert(slot >= 0 && slot < cfg_.n_slots);
    Slot & s = slots_[slot];
    dev_->kv_clear(slot);
    s.tokens.clear();
    s.committed = 0;
    s.in_flight = 0;
    s.epoch++;                     // I3
    cv_.notify_all();
}

void KvLedger::wait_quiescent(int32_t slot) {
    std::unique_lock<std::mutex> lk(mtx_);
    assert(slot >= 0 && slot < cfg_.n_slots);
    cv_.wait(lk, [&] { return slots_[slot].in_flight == 0; });
}

SlotStats KvLedger::stats(int32_t slot) const {
    std::lock_guard<std::mutex> lk(mtx_);
    assert(slot >= 0 && slot < cfg_.n_slots);
    const Slot & s = slots_[slot];
    return SlotStats{s.committed, s.in_flight, s.epoch, s.last_used};
}

} // namespace npu_pc
