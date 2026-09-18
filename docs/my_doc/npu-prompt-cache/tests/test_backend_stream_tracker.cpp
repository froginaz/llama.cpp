// Test C: BackendStreamTracker - ordinal-free stream observation at UB=1024.
// Feeds full prompts as forward streams from pos 0 (host-side caching off);
// wants_output=true only on the last window of each stream.

#include "backend_stream_tracker.hpp"

#include <cassert>
#include <cstdio>
#include <vector>

using namespace npu_pc;

static std::vector<token_t> gen_prompt(uint32_t seed, int n) {
    std::vector<token_t> v(n);
    for (int i = 0; i < n; ++i) {
        v[i] = (token_t) ((seed + 2654435761u * (uint32_t) i) & 0x7fffffff);
    }
    return v;
}

// returns skipped token count; optionally captures per-window decisions
static int64_t feed(BackendStreamTracker & trk, const std::vector<token_t> & p, int32_t UB,
                    std::vector<ForwardDecision> * out = nullptr) {
    int64_t skipped = 0;
    const int32_t n = (int32_t) p.size();
    for (pos_t pos = 0; pos < n; pos += UB) {
        const int32_t len  = std::min<int32_t>(UB, n - pos);
        const bool    last = (pos + len >= n);
        const ForwardView v{/*seq=*/0, pos, len, p.data() + pos, /*wants_output=*/last};
        const ForwardDecision d = trk.observe(v);
        if (out) out->push_back(d);
        if (d.skip_compute) {
            assert(d.n_compute == 0);
            skipped += len;
        }
    }
    return skipped;
}

int main() {
    const int32_t UB = 1024;
    BackendStreamTracker trk;
    std::vector<ForwardDecision> ds;

    // ---- 25K cold: nothing to skip
    std::vector<token_t> p1 = gen_prompt(7u, 25000);
    ds.clear();
    assert(feed(trk, p1, UB, &ds) == 0);
    assert(ds[0].kind == UbatchKind::CONTINUATION);   // first-ever window: cursor==0
    assert(trk.committed(0) == 25000);

    // ---- 27K growth: 24 full windows match; window 24 spans the boundary
    std::vector<token_t> p2 = gen_prompt(7u, 27000);
    ds.clear();
    assert(feed(trk, p2, UB, &ds) == 24 * 1024);      // 24576
    assert(ds[0].kind == UbatchKind::NEW_TURN);
    assert(ds[24].kind == UbatchKind::CONTINUATION);  // partial-match path
    assert(!ds[24].skip_compute);
    assert(ds[24].compute_from == 25000);             // resume at old valid_end
    assert(ds[24].n_compute == 25600 - 25000);
    assert(trk.committed(0) == 27000);

    // ---- identical 27K resend: all match, only the last window computes
    ds.clear();
    assert(feed(trk, p2, UB, &ds) == 26 * 1024);      // 26624
    assert(ds.back().kind == UbatchKind::CONTINUATION);
    assert(!ds.back().skip_compute);                  // full match but wants output
    assert(ds.back().n_compute == 27000 - 26 * 1024); // 376
    assert(trk.committed(0) == 27000);                // full-match windows touch nothing

    // ---- flip token 12'345: windows 0..11 skip; window 12 computes from
    //      the divergence offset; everything after is recomputed
    std::vector<token_t> p3 = p2;
    p3[12345] = (p3[12345] + 1) & 0x7fffffff;
    ds.clear();
    assert(feed(trk, p3, UB, &ds) == 12 * 1024);      // 12288
    assert(!ds[12].skip_compute);
    assert(ds[12].compute_from == 12345);
    assert(ds[12].n_compute == 13312 - 12345);        // 967
    assert(trk.committed(0) == 27000);                // mirror now holds p3

    // ---- 3 decode steps: 1-token forwards at pos == cursor
    const token_t dec[3] = {901, 902, 903};
    for (int i = 0; i < 3; ++i) {
        const ForwardView v{0, 27000 + i, 1, &dec[i], /*wants_output=*/true};
        const ForwardDecision d = trk.observe(v);
        assert(d.kind == UbatchKind::CONTINUATION);
        assert(!d.skip_compute && d.n_compute == 1);
    }
    assert(trk.committed(0) == 27003);                // decode mirroring is automatic

    // ---- resend including the 3 decode tokens + 500 new
    std::vector<token_t> p5 = p3;
    p5.push_back(dec[0]);
    p5.push_back(dec[1]);
    p5.push_back(dec[2]);
    for (token_t t : gen_prompt(555u, 500)) p5.push_back(t);
    assert(p5.size() == 27503);
    ds.clear();
    assert(feed(trk, p5, UB, &ds) == 26 * 1024);      // 26624; decode tokens match via mirror
    assert(trk.committed(0) == 27503);

    // ---- GAP: forward past valid_end -> poisoned hole, computed
    {
        const pos_t gap_pos = trk.committed(0) + 100;
        const token_t gap_toks[4] = {1, 2, 3, 4};
        const ForwardView v{0, gap_pos, 4, gap_toks, /*wants_output=*/false};
        const ForwardDecision d = trk.observe(v);
        assert(d.kind == UbatchKind::GAP);
        assert(!d.skip_compute);
        assert(d.compute_from == gap_pos && d.n_compute == 4);
        assert(trk.committed(0) == gap_pos + 4);
    }

    printf("test_backend_stream_tracker: OK\n");
    return 0;
}
