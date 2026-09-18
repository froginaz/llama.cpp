// backend_stream_tracker.hpp - graph-time, ordinal-free ubatch stream
// tracker for a ggml-backend that does LCP detection at graph_compute time.
//
// At graph_compute the backend sees one ubatch window (inp_tokens) plus
// absolute positions (inp_pos), but no "ubatch #k of turn T" ordinal.
// The ordinal is unnecessary: absolute position is a stronger coordinate.
// Two watermarks per sequence:
//   cursor    = end position of the last observed window (stream continuity)
//   valid_end = extent of the token mirror backed by VALID device KV

#pragma once

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace npu_pc {

using token_t = int32_t;
using pos_t   = int32_t;

enum class UbatchKind : uint8_t { NEW_TURN, CONTINUATION, REWIND, GAP };

struct ForwardView {
    int32_t         seq;
    pos_t           pos_begin;
    int32_t         n_tokens;
    const token_t * tokens;
    bool            wants_output;
};

struct ForwardDecision {
    UbatchKind kind;
    bool       skip_compute;
    pos_t      compute_from;
    int32_t    n_compute;
};

class BackendStreamTracker {
public:
    // single-threaded per seq
    ForwardDecision observe(const ForwardView & v) {
        Seq & s = seqs_[v.seq];
        const pos_t win_end = v.pos_begin + v.n_tokens;

        // 1. classification by position only
        UbatchKind kind;
        if (v.pos_begin == s.cursor) {
            kind = UbatchKind::CONTINUATION;   // next prefill chunk, or a decode step
        } else if (v.pos_begin == 0 && s.cursor > 0) {
            kind = UbatchKind::NEW_TURN;       // host restarted eval at 0
        } else if (v.pos_begin < s.cursor) {
            kind = UbatchKind::REWIND;         // host trimmed mid-sequence
        } else if (v.pos_begin > s.valid_end) {
            // GAP fail-safe: poison the hole so it can never LCP-match later
            s.tokens.resize(v.pos_begin, token_t(-1));
            s.tokens.resize(win_end);
            std::copy(v.tokens, v.tokens + v.n_tokens, s.tokens.begin() + v.pos_begin);
            s.valid_end = s.cursor = win_end;
            return ForwardDecision{UbatchKind::GAP, false, v.pos_begin, v.n_tokens};
        } else {
            // cursor < pos_begin <= valid_end: forward jump within the valid
            // mirror; the match logic below stays correct
            kind = UbatchKind::CONTINUATION;
        }

        // 2. in-window match vs the valid mirror region
        const int32_t overlap = std::clamp<int32_t>(s.valid_end - v.pos_begin, 0, v.n_tokens);
        int32_t match = 0;
        while (match < overlap && s.tokens[v.pos_begin + match] == v.tokens[match]) {
            match++;
        }
        const bool full_match = (match == v.n_tokens);

        // 3. skip rule: the final logits-bearing window identifies itself
        // locally via wants_output (I4 analog), so no "last ubatch" ordinal
        // is needed
        const bool    skip         = full_match && !v.wants_output;
        const pos_t   compute_from = full_match ? v.pos_begin : v.pos_begin + match;
        const int32_t n_compute    = skip ? 0 : (full_match ? v.n_tokens : v.n_tokens - match);

        // 4. mirror update. On full_match touch NOTHING: the old tail beyond
        // this window is still valid and future windows must match against
        // it (a single-watermark design that truncates here is a known bug).
        // On divergence, truncation is a CORRECTNESS requirement: with
        // causal attention, KV past a divergence is invalid even if later
        // token ids coincide.
        if (!full_match) {
            s.tokens.resize(win_end);
            std::copy(v.tokens, v.tokens + v.n_tokens, s.tokens.begin() + v.pos_begin);
            s.valid_end = win_end;
        }
        s.cursor = win_end;

        // device KV at [pos_begin, pos_begin+match) stays valid; if the NPU
        // kernel cannot start mid-window, recomputing the whole window is
        // equally correct (idempotent KV overwrite)
        return ForwardDecision{kind, skip, compute_from, n_compute};
    }

    pos_t committed(int32_t seq) const {
        auto it = seqs_.find(seq);
        return it == seqs_.end() ? 0 : it->second.valid_end;
    }

private:
    struct Seq {
        std::vector<token_t> tokens;   // mirror; size == valid_end
        pos_t cursor    = 0;
        pos_t valid_end = 0;
    };
    std::unordered_map<int32_t, Seq> seqs_;
};

} // namespace npu_pc
