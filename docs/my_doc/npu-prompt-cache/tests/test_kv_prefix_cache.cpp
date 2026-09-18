// Test A: KvLedger small-scale scenarios (see SPEC).
// cfg {n_ubatch=8, granularity=4, n_slots=2}

#include "kv_prefix_cache.hpp"

#include <cassert>
#include <cstdio>
#include <vector>

using namespace npu_pc;

struct FakeDev : INpuKvBackend {
    std::vector<std::vector<token_t>> kv;

    explicit FakeDev(int n_slots) : kv(n_slots) {}

    void kv_trim(int32_t slot, pos_t from) override {
        if ((size_t) from < kv[slot].size()) {
            kv[slot].resize(from);
        }
    }
    void kv_clear(int32_t slot) override { kv[slot].clear(); }

    void run_ubatch(const UbatchDesc & d) {
        assert(kv[d.slot].size() == (size_t) d.pos_begin && "device contiguity");
        kv[d.slot].insert(kv[d.slot].end(), d.tokens, d.tokens + d.n_tokens);
    }
};

static void run_plan(KvLedger & led, FakeDev & dev, const PrefillPlan & plan) {
    for (const auto & ub : plan.ubatches) {
        bool ok = led.on_ubatch_submitted(ub);
        assert(ok);
        dev.run_ubatch(ub);
        ok = led.on_ubatch_committed(ub.slot, ub.pos_begin, ub.n_tokens, ub.epoch);
        assert(ok);
    }
}

static void assert_turn_done(const KvLedger & led, const FakeDev & dev, int32_t slot,
                             const std::vector<token_t> & prompt) {
    assert(dev.kv[slot] == prompt);   // token-by-token
    const SlotStats st = led.stats(slot);
    assert(st.committed == (int32_t) prompt.size());
    assert(st.in_flight == 0);
}

static std::vector<token_t> iota_tokens(token_t base, int n) {
    std::vector<token_t> v(n);
    for (int i = 0; i < n; ++i) v[i] = base + i;
    return v;
}

int main() {
    const LedgerConfig cfg{/*n_ubatch=*/8, /*granularity=*/4, /*n_slots=*/2};
    FakeDev  dev(cfg.n_slots);
    KvLedger led(cfg, &dev);

    int32_t metrics_reuse = -1;
    led.on_plan_metrics = [&](int32_t, int32_t, int32_t n_reuse) { metrics_reuse = n_reuse; };

    // ---- T1: cold 20-token prompt -> n_reuse == 0, then 5 decode tokens
    std::vector<token_t> p1 = iota_tokens(100, 20);
    auto plan = led.plan_prefill(p1);
    assert(plan && plan->n_reuse == 0 && !plan->did_trim);
    assert(plan->n_eval == 20 && plan->ubatches.size() == 3);   // 8+8+4
    assert(metrics_reuse == 0);
    const int32_t slot = plan->slot;
    run_plan(led, dev, *plan);
    assert_turn_done(led, dev, slot, p1);

    for (token_t t = 900; t <= 904; ++t) {
        led.append_decoded(slot, t);
        dev.kv[slot].push_back(t);   // the device wrote this entry during decode
    }
    led.wait_quiescent(slot);        // trivially returns (in_flight == 0)
    assert(led.stats(slot).committed == 25);

    // ---- T2: prompt = T1 + decode tokens + 10 new -> LCP=25 -> align4 -> 24
    std::vector<token_t> p2 = p1;
    for (token_t t = 900; t <= 904; ++t) p2.push_back(t);
    for (token_t t : iota_tokens(200, 10)) p2.push_back(t);
    assert(p2.size() == 35);

    plan = led.plan_prefill(p2);
    assert(plan && plan->slot == slot);            // max-LCP slot selection
    assert(plan->n_reuse == 24 && plan->did_trim); // 25 -> align4 -> 24, trim 25->24
    assert(plan->n_eval == 11 && plan->ubatches.size() == 2);   // 8+3
    assert(dev.kv[slot].size() == 24);             // device trimmed
    run_plan(led, dev, *plan);
    assert_turn_done(led, dev, slot, p2);

    // ---- T3: identical resend -> LCP=35 -> I4 -> 34 -> align4 -> 32
    plan = led.plan_prefill(p2);
    assert(plan && plan->slot == slot);
    assert(plan->n_reuse == 32 && plan->did_trim);
    assert(plan->n_eval == 3 && plan->ubatches.size() == 1);
    run_plan(led, dev, *plan);
    assert_turn_done(led, dev, slot, p2);

    // ---- T4: divergence at position 2 -> LCP=2 -> align4 -> 0, full recompute
    std::vector<token_t> p3 = p2;
    p3[2] = 7777;
    plan = led.plan_prefill(p3);
    assert(plan && plan->slot == slot);
    assert(plan->n_reuse == 0 && plan->did_trim);
    assert(dev.kv[slot].empty());                  // device trimmed to 0
    assert(plan->n_eval == 35 && plan->ubatches.size() == 5);   // 8*4+3
    run_plan(led, dev, *plan);
    assert_turn_done(led, dev, slot, p3);

    // ---- T5: +20 tokens; inject failure on the 2nd ubatch, then retry
    std::vector<token_t> p4 = p3;
    for (token_t t : iota_tokens(300, 20)) p4.push_back(t);
    assert(p4.size() == 55);

    plan = led.plan_prefill(p4);
    assert(plan && plan->slot == slot);
    assert(plan->n_reuse == 32 && plan->did_trim); // LCP=35 -> align4 -> 32
    assert(plan->ubatches.size() == 3);            // 8+8+7

    const UbatchDesc ub0 = plan->ubatches[0];
    const UbatchDesc ub1 = plan->ubatches[1];

    assert(led.on_ubatch_submitted(ub0));
    dev.run_ubatch(ub0);
    assert(led.on_ubatch_committed(ub0.slot, ub0.pos_begin, ub0.n_tokens, ub0.epoch));

    assert(led.on_ubatch_submitted(ub1));
    // device wrote a partial ubatch before failing
    dev.kv[slot].insert(dev.kv[slot].end(), ub1.tokens, ub1.tokens + 3);
    led.on_ubatch_failed(slot, ub1.epoch);

    // rollback restored the sync point
    {
        const SlotStats st = led.stats(slot);
        assert(st.in_flight == 0);
        assert(st.committed == 40);                            // 32 + 8
        assert((size_t) st.committed <= dev.kv[slot].size());
        assert(dev.kv[slot].size() == 40);                     // partial write trimmed
    }
    // stale completion from the failed epoch is fenced off (I3)
    assert(!led.on_ubatch_committed(ub1.slot, ub1.pos_begin, ub1.n_tokens, ub1.epoch));
    // stale submission likewise
    assert(!led.on_ubatch_submitted(plan->ubatches[2]));

    // retry succeeds
    plan = led.plan_prefill(p4);
    assert(plan && plan->slot == slot);
    assert(plan->n_reuse == 40 && !plan->did_trim);            // resume from sync point
    run_plan(led, dev, *plan);
    assert_turn_done(led, dev, slot, p4);

    // ---- clear_slot sanity
    led.clear_slot(slot);
    assert(dev.kv[slot].empty());
    assert(led.stats(slot).committed == 0);

    printf("test_kv_prefix_cache: OK\n");
    return 0;
}
