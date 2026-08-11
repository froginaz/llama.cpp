# SWA와 Hybrid 모델의 KV 캐시 운영

[13번](13-batch-ubatch-and-sequences.md)(셀 모델)과 [14번](14-host-visible-kv.md)(장부/mask 운영)이 전제한 것은 "표준" unified KV 캐시였다. 이 문서는 그 모델이 **변형되는 두 경우** - sliding window attention(SWA)과 hybrid(attention + recurrent) 모델 - 에서 KV 캐시가 어떻게 운영되는지를 다룬다.

PlantUML 원본:

- [15-swa-hybrid-structure.puml](15-swa-hybrid-structure.puml) - 메모리 구현 3종의 구조 비교 다이어그램

---

## 1. 왜 변형이 필요한가

표준 KV 캐시의 비용은 `2 x n_layer x n_ctx x (n_head_kv x d_head) x type_size` - **n_ctx에 선형**이다 (13번 문서 6장). 문맥이 128K로 늘면 KV만 수십 GB가 된다. 이를 줄이는 아키텍처 수준의 답이 두 갈래다:

- **SWA**: 일부 레이어의 attention을 "최근 n_swa 토큰"으로 제한 -> 그 레이어의 KV는 창 크기만 유지하면 됨 (Gemma 2/3, GPT-OSS 등)
- **recurrent/hybrid**: 일부 레이어를 아예 attention이 아닌 고정 크기 상태(state)로 대체 (Mamba, RWKV, Jamba, Qwen3-Next 등) -> 그 레이어는 KV 캐시 자체가 없음

llama.cpp는 이를 **`llama_memory_i` 구현체 교체**로 흡수한다 - context/모델 그래프 빌더의 코드는 그대로이고, `model.create_memory()`가 아키텍처에 맞는 구현을 고른다 (12번 문서의 팩토리 패턴).

## 2. iSWA: llama_kv_cache 두 개의 조합

SWA 모델은 보통 SWA 레이어와 full-attention 레이어를 섞어 쓴다(interleaved). `llama_kv_cache_iswa`는 새 캐시 구현이 아니라 **표준 `llama_kv_cache` 인스턴스 2개를 레이어 필터로 묶은 합성물**이다 (src/llama-kv-cache-iswa.h:11~12 주석, 78~79):

```cpp
std::unique_ptr<llama_kv_cache> kv_base;   // full-attention 레이어용
std::unique_ptr<llama_kv_cache> kv_swa;    // SWA 레이어용
```

생성자(src/llama-kv-cache-iswa.cpp:30~72)가 하는 일:

```cpp
// 레이어 분배: is_swa(il) 기준으로 필터 체인 구성
const layer_filter_cb filter_base = [&](il) { return !hparams.is_swa(il); };
const layer_filter_cb filter_swa  = [&](il) { return  hparams.is_swa(il); };

// 크기: base는 요청대로, SWA는 "창 + ubatch 여유분"만
const uint32_t size_base = kv_size;
uint32_t size_swa = GGML_PAD(std::min(size_base,
        hparams.n_swa*(unified ? n_seq_max : 1) + n_ubatch), 256);
```

**SWA 캐시 크기 공식**이 핵심이다: `n_swa x n_seq + n_ubatch` - 창을 유지할 만큼 + 이번 배치가 들어갈 여유분. 예: n_swa=1024, n_seq=4, n_ubatch=512면 SWA 레이어당 4,608셀(패딩 후) - n_ctx=131072짜리 base 캐시의 3.5%다. 14번 문서 생성 과정의 `filter` 콜백(5단계)이 바로 이 레이어 분배에 쓰인다.

모든 seq 연산은 두 캐시에 **팬아웃**된다 (`seq_rm/cp/keep/add/div` 모두 base와 swa에 순차 적용, iswa.cpp:80~107). `seq_pos_min/max`는 SWA 캐시 기준으로 답한다 - base가 SWA의 상위집합(superset)이므로 더 제한적인 쪽이 정답이다 (110~116 주석).

`--swa-full` 옵션(iswa.cpp:53~58)은 SWA 캐시도 풀사이즈로 만든다 - 4장의 rollback 제약을 없애는 대신 메모리 절감을 포기하는 트레이드오프다.

## 3. SWA 셀의 생애: 별도 프루닝 패스는 없다 (lazy 만료)

"창 밖으로 벗어난 셀을 누가 언제 지우는가?"에 대한 답이 우아하다: **아무도 지우지 않는다. find_slot이 "만료된 셀"을 빈 칸처럼 재사용할 뿐이다** (src/llama-kv-cache.cpp:979~986):

```cpp
if (!can_use && cells.seq_count(idx) == 1) {
    const llama_pos pos_cell = cells.pos_get(idx);
    const llama_seq_id seq_id_cell = cells.seq_get(idx);

    // SWA mask: 이 셀의 pos가 이미 창 밖이면 -> 덮어써도 됨
    if (llama_hparams::is_masked_swa(n_swa, swa_type,
            pos_cell, cells.seq_pos_max(seq_id_cell) + 1)) {
        can_use = true;
    }
}
```

즉 SWA 캐시는 **링 버퍼처럼 동작**한다: 새 토큰이 창 밖 셀 자리를 자연히 재활용한다. 14번 문서의 "삭제 = 잊기, 덮어쓰기는 lazy" 원리가 여기서는 시간 축으로 자동화된 셈이다.

덮어쓸 때는 불변식 하나가 지켜진다 (src/llama-kv-cache.cpp:1068~1085): **"각 seq의 [pos_min, pos_max] 사이 모든 pos는 캐시에 존재해야 한다"**. 셀 하나를 덮어쓰면 그보다 낮은 pos의 셀들을 함께 `seq_rm`으로 정리해서(purge), "중간에 구멍 난 창"이 생기지 않게 한다. mask 생성기(set_input_kq_mask)도 같은 `is_masked_swa` 판정을 쓰므로(1569행) 장부와 mask의 기준이 항상 일치한다.

## 4. 운영상 결과: rollback 제약과 체크포인트

SWA 캐시에는 창 밖의 과거가 **물리적으로 없다**(덮어써짐). 그 결과:

- **자유로운 rollback 불가**: `seq_rm(seq, p, -1)`으로 창 이전 지점까지 무르고 재-decode하는 것이 불가능하다 - 그 구간의 K/V가 이미 사라졌기 때문. 14번 문서 E1(speculative)의 전제가 창 안에서만 성립한다.
- **서버의 보완책 = 컨텍스트 체크포인트**: llama-server의 `-ctxcp`(`--ctx-checkpoints`, 기본 32개)가 프롬프트 처리 중 주기적으로 SWA 캐시 상태의 스냅샷을 host 메모리에 저장해 둔다 (tools/server/server-context.cpp:1129~, create_checkpoint 2033~). 되돌아갈 일이 생기면(prefix 재사용, speculative 등) 가장 가까운 체크포인트를 복원하고 그 지점부터 재-decode한다. 스냅샷은 `llama_state_seq_*`에 **`LLAMA_STATE_SEQ_FLAGS_SWA_ONLY`** 플래그(include/llama.h:873)를 줘서 SWA 캐시 부분만 덤프한다 - base 캐시는 rollback 가능하므로 저장할 필요가 없다.

정리하면 SWA는 "메모리 절감 <-> 시간 여행의 자유"를 교환하며, 그 균형점을 사용자가 고른다: 기본(창 크기 캐시 + 체크포인트) vs `--swa-full`(풀 캐시, 제약 없음).

## 5. Hybrid: attention 캐시 + recurrent 상태의 합성

Mamba/RWKV류 레이어를 섞은 모델은 `llama_memory_hybrid`를 쓴다 - iSWA와 같은 합성 패턴이지만 이번엔 **이종(異種) 메모리의 조합**이다 (src/llama-memory-hybrid.h:16~17 주석):

```
llama_memory_hybrid
 ├─ llama_kv_cache        <- attention 레이어 (filter_attn)
 └─ llama_memory_recurrent <- recurrent 레이어 (filter_recr)
```

recurrent 쪽의 상태 모델은 셀 모델과 근본적으로 다르다 (src/llama-memory-recurrent.h:88~113):

```cpp
struct mem_cell {
    llama_pos pos  = -1;
    int32_t   src  = -1;   // 상태를 어디서 복사해 올지
    int32_t   tail = -1;
    std::set<llama_seq_id> seq_id;
};
std::vector<mem_cell> cells;      // 셀 수 ~= 시퀀스 수 (토큰 수 아님!)
std::vector<ggml_tensor *> r_l;   // 레이어별 conv 상태
std::vector<ggml_tensor *> s_l;   // 레이어별 ssm 상태
```

| | attention KV 셀 | recurrent 상태 셀 |
|---|---|---|
| 셀 1개 = | **토큰 1개**의 K/V | **시퀀스 1개**의 압축 상태 전체 |
| 셀 수 | n_ctx (토큰 용량) | ~n_seq_max (시퀀스 수) |
| 크기 | pos에 선형 증가 | **고정** (문맥이 얼마든 동일) |
| 과거 접근 | 임의 pos 읽기 가능 (mask로 선택) | 불가 - 과거가 상태에 비가역 압축됨 |
| rollback | 자유 (장부 태그 제거) | **원리적 불가** (제한적 예외: `n_rs_seq` 부분 롤백, 지원 아키텍처 한정) |

이 표의 마지막 행이 hybrid 운영의 지배 요인이다: **전체 메모리의 rollback 능력은 가장 약한 구성원(recurrent)을 따른다.** 그래서 hybrid/recurrent 모델에서 서버는 SWA와 같은 체크포인트 메커니즘에 의존하고, speculative decoding도 체크포인트 경유로만 가능하다 (server-context.cpp:1046 "speculative decoding will use checkpoints").

ubatch 분할도 다르다 (13번 문서 1장의 분할 표): recurrent 상태는 "시퀀스당 하나"라서 한 ubatch 안에 같은 seq의 토큰들이 **연속으로 묶여** 있어야 한다 -> `split_equal`/`split_seq`가 강제되고 `split_simple`은 못 쓴다 (src/llama-memory-hybrid.cpp:77~86).

## 6. 세 구현의 기능 매트릭스

| 기능 | unified `llama_kv_cache` | `iswa` (SWA) | `hybrid` (recurrent 포함) |
|---|---|---|---|
| seq_rm rollback | 자유 | **창 안에서만** (밖은 체크포인트) | **불가** (체크포인트/n_rs_seq) |
| seq_cp 공유 | 태그만 (0 복사) | 두 캐시 팬아웃 | attn 태그 + recurrent 상태 복사 |
| K-shift (`seq_add`) | O | O (두 캐시) | attn만 의미 있음 |
| session save/restore | O | O (SWA_ONLY 부분 저장 지원) | O (상태 포함) |
| ubatch 분할 | split_simple | split_simple(unified) | **split_equal/seq 강제** |
| 메모리 비용 | n_ctx 선형 | full층 선형 + SWA층 상수 | attn층 선형 + recr층 **상수** |

## 7. dNPU 함의

- **합성 패턴이 곧 로드맵이다**: iSWA/hybrid는 새 계약이 아니라 표준 `llama_kv_cache` N개(+recurrent)의 조합이므로, 14번 문서의 host-visible KV 계약(장부/idxs/mask)을 지키면 **iSWA는 자동으로 얻어진다** - firmware 입장에서는 "레이어별로 참조하는 KV 영역과 n_kv가 다를 뿐"이다 (`register_kv_region`이 레이어별 디스크립터인 이유가 여기서도 유효).
- SWA 캐시의 lazy 만료는 firmware에 아무 요구도 하지 않는다 - 재사용 판정은 host의 find_slot이 하고, firmware는 여전히 idxs/mask만 소비한다.
- hybrid의 recurrent 상태는 KV 계약 밖의 **별도 상태 텐서**(r_l/s_l)다. dNPU가 hybrid 모델을 지원하려면 이 상태 텐서들도 NPU buft에 두고 그래프가 읽고-쓰는(read-modify-write) 계약이 추가로 필요하다 - KV보다 단순하지만(고정 크기, seq당 1개) rollback 불가 특성은 그대로 따라온다.

## 핵심 요약

1. **SWA/hybrid는 memory 구현 교체로 흡수된다** - context/그래프 빌더는 불변, `create_memory()` 팩토리가 아키텍처에 맞는 구현을 고른다.
2. **iSWA = 표준 캐시 2개 + 레이어 필터**: SWA 캐시 크기는 `n_swa x n_seq + n_ubatch`(창+여유분)로, n_ctx와 무관하게 상수다.
3. **SWA 만료는 lazy**: 프루닝 패스 없이 find_slot이 창 밖 셀을 빈 칸처럼 재사용한다(링 버퍼). purge 불변식이 "[pos_min, pos_max] 연속성"을 지킨다.
4. **SWA의 대가는 rollback 제약**: 창 밖 과거는 물리적으로 사라지므로, 서버는 `-ctxcp` 체크포인트(SWA_ONLY 부분 스냅샷)로 보완한다. `--swa-full`은 메모리로 제약을 되사는 옵션이다.
5. **recurrent 상태는 "시퀀스당 고정 크기 1개"** - 토큰당 셀이라는 KV 모델과 근본이 다르고, 과거가 비가역 압축되므로 rollback이 원리적으로 불가하다. hybrid의 운영 능력은 이 가장 약한 고리를 따른다.
6. **dNPU 관점**: host-visible KV 계약만 지키면 iSWA는 공짜, hybrid는 recurrent 상태 텐서에 대한 추가 계약(고정 크기 read-modify-write)이 필요하다.
