// Test B: KvLedger at 25K scale, sweeping n_ubatch x trim_granularity.
// Validates every plan structurally and end-state device == prompt.

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

// deterministic prompt generator from the SPEC
static std::vector<token_t> gen_prompt(uint32_t seed, int n) {
    std::vector<token_t> v(n);
    for (int i = 0; i < n; ++i) {
        v[i] = (token_t) ((seed + 2654435761u * (uint32_t) i) & 0x7fffffff);
    }
    return v;
}

static int32_t align_down(int32_t v, int32_t g) { return g <= 1 ? v : (v / g) * g; }

static void validate_plan(const PrefillPlan & p, const std::vector<token_t> & prompt,
                          int32_t UB, int32_t G) {
    assert(p.n_prompt == (int32_t) prompt.size());
    assert(p.n_reuse % G == 0);
    assert(p.n_reuse < p.n_prompt);
    assert(p.n_eval == p.n_prompt - p.n_reuse);

    const int32_t want_chunks = (p.n_eval + UB - 1) / UB;   // ceil
    assert((int32_t) p.ubatches.size() == want_chunks);

    pos_t expect_pos = p.n_reuse;
    for (size_t i = 0; i < p.ubatches.size(); ++i) {
        const UbatchDesc & ub = p.ubatches[i];
        assert(ub.pos_begin == expect_pos);                 // contiguous
        const bool last = (i + 1 == p.ubatches.size());
        if (!last) {
            assert(ub.n_tokens == UB);                      // all-but-last full size
        } else {
            assert(ub.n_tokens == p.n_eval - (int32_t) (p.ubatches.size() - 1) * UB);
        }
        assert(ub.tokens == prompt.data() + ub.pos_begin);  // aliases the prompt
        assert(ub.epoch == p.epoch);
        expect_pos += ub.n_tokens;
    }
    assert(expect_pos == p.n_prompt);
}

static void run_turn(KvLedger & led, FakeDev & dev, const std::vector<token_t> & prompt,
                     int32_t UB, int32_t G, int32_t expect_reuse) {
    auto plan = led.plan_prefill(prompt);
    assert(plan);
    validate_plan(*plan, prompt, UB, G);
    assert(plan->n_reuse == expect_reuse);
    for (const auto & ub : plan->ubatches) {
        assert(led.on_ubatch_submitted(ub));
        dev.run_ubatch(ub);
        assert(led.on_ubatch_committed(ub.slot, ub.pos_begin, ub.n_tokens, ub.epoch));
    }
    assert(dev.kv[plan->slot] == prompt);
    const SlotStats st = led.stats(plan->slot);
    assert(st.committed == (int32_t) prompt.size() && st.in_flight == 0);
}

int main() {
    const int32_t N1 = 25000;
    const int32_t ubs[]   = {1024, 2048, 4096};
    const int32_t grans[] = {1, 256};

    for (int32_t UB : ubs) {
        for (int32_t G : grans) {
            const LedgerConfig cfg{UB, G, /*n_slots=*/2};
            FakeDev  dev(cfg.n_slots);
            KvLedger led(cfg, &dev);   // fresh ledger per combo

            // T1: cold 25K
            std::vector<token_t> p1 = gen_prompt(12345u, N1);
            run_turn(led, dev, p1, UB, G, /*expect_reuse=*/0);

            // T2: +2000 tokens (same generator, so the prefix is identical)
            std::vector<token_t> p2 = gen_prompt(12345u, N1 + 2000);
            run_turn(led, dev, p2, UB, G, align_down(N1, G));

            // T3: identical resend -> I4 caps LCP at size-1
            run_turn(led, dev, p2, UB, G, align_down((int32_t) p2.size() - 1, G));

            // T4: flip token at 12'345 (not a multiple of any tested UB:
            // forces a cut mid-ubatch relative to the previous layout)
            std::vector<token_t> p3 = p2;
            p3[12345] = (p3[12345] + 1) & 0x7fffffff;
            run_turn(led, dev, p3, UB, G, align_down(12345, G));

            // T5: +1234 tokens on the divergent prompt
            std::vector<token_t> p4 = p3;
            for (token_t t : gen_prompt(99999u, 1234)) p4.push_back(t);
            run_turn(led, dev, p4, UB, G, align_down((int32_t) p3.size(), G));
        }
    }

    printf("test_lcp_ubatch_split: OK\n");
    return 0;
}
