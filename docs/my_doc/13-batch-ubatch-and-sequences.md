# batch, ubatch, sequence: llama.cpp의 실행 데이터 단위

[12번 문서](12-context-model-backend-ownership.md)가 `llama_context` / `llama_model` / ggml-backend의 **객체 구조**(소유권, 생명주기)를 다룬다면, 이 문서는 그 위에서 **실행 데이터가 흐르는 단위** - batch(논리), ubatch(물리), sequence(대화) - 를 다룬다. 요청이 어떤 단위로 묶이고, 쪼개지고, 격리되는지를 정의하고, 두 개의 대화를 예로 decode 파이프라인 상세까지 따라간다.

PlantUML 원본:

- [13-sequence-decode-batch2-multi-context.puml](13-sequence-decode-batch2-multi-context.puml) - batch size 2 decode 상세 시퀀스 다이어그램 (멀티 컨텍스트)

---

## 1. 용어 정의: batch vs ubatch

"batch"와 "ubatch"는 prefill/decode라는 단계 구분이 아니라 **논리(logical) 단위 vs 물리(physical) 단위**의 구분이다. 공식 정의는 include/llama.h:338-339:

```c
uint32_t n_batch;   // logical maximum batch size that can be submitted to llama_decode
uint32_t n_ubatch;  // physical maximum batch size
```

### llama_batch (논리 배치)

API 사용자가 `llama_decode()` 한 번에 제출하는 작업 묶음. 최대 크기가 `n_batch`이며, prefill/decode 구분이 없다. 여러 시퀀스의 토큰을 섞어 담을 수 있다 - llama-server가 "슬롯 A의 프롬프트 조각 512토큰 + 슬롯 B의 생성 토큰 1개 + 슬롯 C의 생성 토큰 1개"를 한 batch에 담는 것이 정확히 이 용도다. batch 단위로는 어떤 연산도 일어나지 않는다 (스케줄링/API 경계일 뿐).

### llama_ubatch (물리 마이크로배치)

**그래프 1회 빌드 + `graph_compute()` 1회 실행이 실제로 처리하는 단위** (src/llama-batch.h:15). 최대 크기가 `n_ubatch`이고, 컴퓨트 버퍼(중간 활성값 메모리)와 worst-case 그래프 예약이 모두 이 폭 기준으로 잡힌다. 내부 구조도 단순 토큰 나열이 아니라 `n_seq_tokens x n_seqs`(시퀀스 세트별 토큰 수 x 세트 수)로 정규화되어 있다.

### 관계

```
1 llama_decode(batch) 호출 = 논리 batch 1개
  -> memory->init_batch()가 N개의 ubatch로 분할
  -> ubatch마다 그래프 빌드(또는 재사용) + 실행 1회
```

흔한 통념인 "ubatch는 prefill을 쪼개는 단위"는 결과적으로는 대부분 맞지만 정의가 아니다. **분할은 prefill/decode를 가리지 않고 모든 batch에 항상 적용된다.** decode 스텝은 보통 `n_tokens`(= 활성 시퀀스 수)가 `n_ubatch`보다 작아 ubatch 1개로 끝나기 때문에 분할이 눈에 띄지 않을 뿐이다.

예 (`n_batch=8192`, `n_ubatch=512`):

| 시나리오 | batch 내용 | ubatch 분할 결과 |
|---|---|---|
| prefill | 프롬프트 8192토큰 | 512토큰 x 16개 -> 그래프 실행 16회 |
| decode | 32개 시퀀스가 1토큰씩 (32토큰) | 1개 -> 그래프 실행 1회 |
| 혼합 (서버) | 프롬프트 조각 500토큰 + 생성 토큰 12개 | 512토큰 1개 -> 그래프 실행 1회 |

### 분할 방식은 memory 모듈이 결정

`memory->init_batch()`가 KV 캐시 구조에 따라 세 가지 분할 방법 중 하나를 선택한다 (src/llama-batch.h:102-110):

| 방식 | 동작 | 사용처 |
|---|---|---|
| `split_simple` | 순서대로 최대 `n_ubatch`개씩 자름 | 통합(unified) KV 캐시, 단일 스트림 (src/llama-kv-cache.cpp:644) |
| `split_equal` | 시퀀스 세트들을 같은 길이로 묶어서 자름 | 비통합 KV(시퀀스별 스트림), recurrent/hybrid 모델 |
| `split_seq` | ubatch 하나에 시퀀스 세트 하나만 | recurrent 모델 (src/llama-memory-recurrent.cpp:417) |

`encode()`는 예외적으로 항상 `split_simple(n_tokens)` - 전체 배치를 ubatch 1개로 처리한다 (src/llama-context.cpp:1354). 그래서 non-causal attention에는 `n_ubatch >= n_tokens` 제약이 있다 (src/llama-context.cpp:1702).

### 왜 두 단계로 나누는가

- **`n_batch`가 제한하는 것**: 논리적 제출 크기와 출력 버퍼 (`n_outputs_max`의 기본값 = `n_batch`). 커도 메모리 부담이 작다.
- **`n_ubatch`가 제한하는 것**: 활성값 메모리(컴퓨트 버퍼는 ubatch 폭에 비례)와 그래프 크기. 실제 VRAM 사용량을 좌우한다.
- 그래서 "크게 묶어 제출하되(`n_batch` 크게 -> 스케줄링 효율), 실제 계산은 감당 가능한 조각으로(`n_ubatch`)"가 가능하다. 부수 효과로 멀티 GPU **파이프라인 병렬**(ubatch들이 디바이스 간 파이프라인으로 흐름)도 이 분할이 전제다.
- `cparams.n_ubatch = min(n_batch, n_ubatch)`로 항상 `n_ubatch <= n_batch`가 강제된다 (src/llama-context.cpp:184).

요약: **batch = "무엇을 함께 제출할지"의 논리 단위 (prefill/decode 혼합 가능), ubatch = "한 번의 그래프 실행이 감당할 물리 단위" (모든 batch가 항상 이 단위로 분할되며, prefill에서 분할이 두드러질 뿐).**

---

## 2. 예제로 보는 context vs sequence와 논리/물리 배치

일상적으로 쓰는 "컨텍스트"(대화 문맥)와 llama.cpp의 `llama_context`(C++ 객체)는 다른 것이다. 두 개의 독립적인 질문을 예로 정리한다:

- 대화 1: `"what is capital of france?"`
- 대화 2: `"who is best soccer player?"`

### 용어 대응

| 용어 | 정체 | 이 예시에서 |
|---|---|---|
| **sequence** (`llama_seq_id`) | 독립적인 토큰 스트림 하나 = 대화 하나. 자기만의 위치(pos 0,1,2...)와 KV 캐시 영역을 가짐 | 대화 1 = seq 0, 대화 2 = seq 1 |
| **`llama_context`** | 실행 엔진 객체 (KV 캐시 풀 + 스케줄러 + 버퍼). **최대 `n_seq_max`개의 sequence를 동시에 수용하는 그릇** | 보통 1개면 충분 - 두 대화 모두 이 안에서 처리 |
| **context window** (`n_ctx`) | 용량(토큰 수). sequence들이 나눠 씀 (`n_ctx_seq = n_ctx / n_seq_max`, src/llama-context.cpp:209) | 두 대화가 쓸 수 있는 토큰 예산 |

일상어의 "컨텍스트 2개"는 llama.cpp 용어로는 대부분 **"llama_context 1개 안의 sequence 2개"**다. llama-server의 슬롯이 정확히 sequence에 해당한다.

### 시나리오 A (일반적): context 1개, sequence 2개

토큰화 결과를 seq 0 = 7토큰, seq 1 = 6토큰이라 하자 (실제 개수는 토크나이저에 따라 다름).

**1단계 - prefill: 논리 batch 1개에 두 대화를 함께 담는다** (`n_tokens = 13`):

| i | token | seq_id | pos | logits |
|---|---|---|---|---|
| 0 | "what" | **0** | 0 | 0 |
| 1 | "is" | **0** | 1 | 0 |
| ... | ... | **0** | ... | 0 |
| 6 | "?" | **0** | 6 | **1** |
| 7 | "who" | **1** | 0 | 0 |
| 8 | "is" | **1** | 1 | 0 |
| ... | ... | **1** | ... | 0 |
| 12 | "?" | **1** | 5 | **1** |

**pos가 seq마다 0부터 다시 시작**한다. batch는 "두 대화를 한 번에 제출한다"는 논리적 묶음일 뿐, 두 질문이 이어진 하나의 문장이 되는 것이 아니다.

**2단계 - 물리 배치(ubatch)로 분할**:

- `n_ubatch = 512`(기본)라면: 13 <= 512이므로 **ubatch 1개** -> 그래프 1회 실행으로 두 질문을 동시에 계산. 가중치를 한 번만 읽으면서 13토큰을 다 처리하는 것이 배칭의 이득이다.
- `n_ubatch = 8`이라면: `split_simple`이 [i=0..7] 8토큰, [i=8..12] 5토큰의 **ubatch 2개**로 잘라 그래프를 2회 실행한다. 두 번째 ubatch가 seq 1의 나머지를 처리해도 결과는 동일하다.

**3단계 - 격리 보장**: 같은 그래프 안에서 함께 계산돼도 seq 0의 K/V는 KV 캐시의 seq 0 슬롯에, seq 1의 K/V는 seq 1 슬롯에 기록되고, **attention mask가 seq_id 기준으로 교차 접근을 차단**한다. "france" 토큰이 "soccer" 토큰을 attend할 수 없으므로, 수학적으로는 두 질문을 따로 돌린 것과 같은 결과가 나온다.

**4단계 - decode: 매 스텝이 batch size 2**: prefill 후 logits 2행(각 seq의 마지막 토큰 위치)을 샘플링해서

```
batch = { token[0]: seq0의 새 토큰 (pos 7),
          token[1]: seq1의 새 토큰 (pos 6) }   // n_tokens = 2
```

이것이 [3장](#3-decode-상세-batch-size-2가-처리되는-과정)의 상황이다. 2 <= n_ubatch이므로 항상 ubatch 1개 -> 그래프 1회로 두 대화가 동시에 한 토큰씩 전진한다. 한쪽 답이 먼저 끝나면 다음 스텝부터 batch는 n_tokens=1이 된다.

### 시나리오 B (비교): llama_context를 2개 만드는 경우

ctx1에 대화 1(그 안에서는 seq 0), ctx2에 대화 2(역시 자기 seq 0)를 담는 방법:

- `llama_decode()`가 **각 context마다 따로** 호출되고 그래프도 각각 실행된다. **두 질문이 하나의 batch로 묶이는 일은 없다** - batch는 context 내부의 개념이기 때문이다.
- 대신 KV 캐시, 컴퓨트 버퍼, 출력 버퍼가 context마다 별도로 생기고(메모리 증가), 스레드 2개로 병렬 실행이 가능하다 ([12번 문서 5-3절](12-context-model-backend-ownership.md#5-3-시퀀스-다이어그램-멀티-컨텍스트-하나의-model을-여러-context가-공유) 참고).

### 선택 기준

- 같은 설정으로 많은 대화를 효율적으로 다중화 -> **시나리오 A (sequence)**: 배칭 효과, 메모리 절약
- 완전한 격리, 서로 다른 `n_ctx`/cparams, 스레드 병렬 -> **시나리오 B (context)**

---

## 3. decode 상세: batch size 2가 처리되는 과정

"batch size 2"는 `llama_batch.n_tokens == 2`를 뜻한다. 전형적인 예는 생성 스텝에서 두 시퀀스가 각각 1토큰씩 넣는 경우다 (`token[0]` -> seq 0, `token[1]` -> seq 1, 둘 다 `logits=1`) - 바로 [2장](#2-예제로-보는-context-vs-sequence와-논리물리-배치) 예제의 4단계 상황이다. 아래 다이어그램은 멀티 컨텍스트 환경(context 2개가 model을 공유)에서 ctx1의 decode 내부를 상세히, ctx2를 요약으로 보여준다.

`llama_context::decode()` (src/llama-context.cpp:1632) 내부에서 batch 하나가 처리되는 단계:

1. **배치 검증/변환**: `balloc->init()` (src/llama-context.cpp:1683) -> `n_tokens_all=2`, `n_outputs_all=2` 산출. `sched_reserve()`는 이미 예약돼 있으면 no-op.
2. **ubatch 분할 + KV 슬롯 예약**: `memory->init_batch(*balloc, n_ubatch)` (src/llama-context.cpp:1723)가 batch를 ubatch들로 쪼개고 KV 슬롯을 계획한 `mctx`를 돌려준다. `n_ubatch >= 2`이면 2토큰짜리 ubatch 1개, `n_ubatch == 1`이면 1토큰 ubatch 2개가 되어 처리 루프가 2회 돈다. 슬롯 부족(`FAILED_PREPARE`) 시 `memory_update(true)`로 캐시 최적화 후 1회 재시도.
3. **ubatch 처리 루프** (`process_ubatch`, src/llama-context.cpp:1257):
   - `mctx->apply()`로 KV 슬롯 확정
   - **그래프 재사용 판정** (`can_reuse`, src/llama-context.cpp:1271): 직전 decode와 토폴로지가 같으면 `gf_res_prev`를 재사용(`n_reused++`). 매 스텝 n_tokens=2가 반복되는 생성 루프에서는 대부분 이 경로를 탄다. 토폴로지가 바뀌었으면 `sched_reset` -> `model.build_graph()` -> `sched_alloc_graph`.
   - `res->set_inputs(ubatch)`: 토큰 id 2개, pos, KV mask를 입력 텐서로 복사
   - `graph_compute(gf, batched=true)`: `n_tokens > 1`이므로 batched=true -> `n_threads_batch` 사용. sched가 split별로 백엔드(자기 전용 스트림)에 비동기 실행.
   - logits 추출: `ggml_backend_tensor_get_async`로 `buf_output`에 2행(`2 x n_vocab`) 복사
4. **App에서 결과 사용**: `llama_get_logits_ith(0)`, `(1)` - 첫 접근 시 `synchronize()`로 async 복사 완료를 보장. seq0/seq1 각각 샘플링해서 다음 스텝의 2토큰 배치를 다시 구성.

멀티 컨텍스트 관점에서 중요한 점:

- ctx2도 같은 1~4 단계를 **자기 소유물(balloc, sched, 컴퓨트 버퍼, 스트림, KV, 출력 버퍼)로 독립 수행**하며, 공유하는 것은 model 가중치(읽기 전용)뿐이다.
- 같은 GPU에서 두 context가 동시에 돌면 각자의 backend 인스턴스(스트림)에 커널이 제출되고, 하드웨어 스케줄러가 이를 교차 실행한다. 소프트웨어 레벨 동기화는 필요 없다.
- batch 안의 두 시퀀스(seq0, seq1)는 **하나의 그래프에서 함께 계산**되지만 (가중치를 1회만 읽는 배칭 효과), K/V는 각자의 시퀀스 슬롯에 기록되고 attention mask가 시퀀스 간 접근을 차단한다.

```plantuml
@startuml
title 멀티 컨텍스트 추론 루프 상세: batch size 2 (n_tokens = 2) decode

participant "App\n(스레드 A / 스레드 B)" as App
box "llama_context #1 소유물" #F0F8FF
  participant "llama_context #1" as C1
  participant "batch_allocr #1" as BA1
  participant "llama_memory_i\n(KV cache #1)" as KV1
  participant "ggml_backend_sched #1" as S1
  participant "ggml_backend_t #1\n(CUDA0 stream 1)" as B1
end box
participant "llama_model\n(weights, 공유 1벌)" as M
box "llama_context #2 소유물" #F5FFF0
  participant "llama_context #2" as C2
  participant "KV cache #2" as KV2
end box

note over App
  "batch size 2" = llama_batch.n_tokens == 2
  예: 생성 스텝에서 두 시퀀스가 1토큰씩
    token[0] -> seq 0, pos p0, logits=1
    token[1] -> seq 1, pos p1, logits=1
end note

par 스레드 A: ctx1 decode 상세

  App -> C1 : llama_decode(batch{n_tokens=2})

  == 3-1. 배치 검증/변환 ==
  C1 -> BA1 : balloc->init(batch, vocab, memory, n_seq_max)
  BA1 --> C1 : n_tokens_all=2, n_outputs_all=2
  C1 -> C1 : sched_reserve()\n(이미 예약됨 -> no-op)
  C1 -> KV1 : memory_update(false)\n(보류된 shift/copy 처리)

  == 3-2. ubatch 분할 + KV 슬롯 예약 ==
  C1 -> KV1 : memory->init_batch(balloc, n_ubatch)
  KV1 --> C1 : mctx (ubatch 목록 + 슬롯 계획)
  note right of C1
    n_ubatch >= 2 이므로 ubatch 1개(2토큰)로 처리.
    n_ubatch == 1 이었다면 1토큰 ubatch 2개
    -> 아래 루프가 2회 돈다.
    슬롯 부족(FAILED_PREPARE) 시
    memory_update(true)로 캐시 최적화 후 1회 재시도.
  end note
  C1 -> C1 : output_reserve(2)\n(buf_output #1에 2행 확보)

  == 3-3. ubatch 처리 (여기서는 1회) ==
  loop mctx의 각 ubatch
    C1 -> KV1 : mctx->apply() - KV 슬롯 확정
    alt 그래프 재사용 (can_reuse: 직전 decode와 동일 토폴로지)
      C1 -> C1 : gf_res_prev 재사용, n_reused++
      note right : 매 스텝 n_tokens=2가 반복되는\n생성 루프에서는 대부분 이 경로
    else 토폴로지 변경 (첫 호출, n_tokens 변화 등)
      C1 -> S1 : ggml_backend_sched_reset()
      C1 -> M : build_graph(gparams) [const, 읽기 전용]
      M --> C1 : ggml_cgraph (2-token 폭 그래프)
      C1 -> S1 : ggml_backend_sched_alloc_graph(gf)
      S1 -> S1 : 노드별 backend 배정 + split\n+ 컴퓨트 버퍼 #1에 활성값 배치
    end
    C1 -> C1 : res->set_inputs(ubatch)\n(토큰 id 2개, pos, KV mask 입력 텐서에 복사)
    C1 -> S1 : graph_compute(gf, batched=true)
    note right : n_tokens > 1 이므로 batched=true\n-> n_threads_batch 사용
    S1 -> B1 : split별 비동기 실행 (ctx1 전용 스트림)
    B1 -> M : 가중치 읽기 (모든 context 공유)
    B1 -> KV1 : seq0, seq1의 K/V를 각자 슬롯에 기록\n+ attention에서 과거 KV 읽기
    S1 --> C1 : GGML_STATUS_SUCCESS
    C1 -> C1 : logits 추출: tensor_get_async\n-> buf_output #1 [2 x n_vocab]
  end

  C1 --> App : return 0
  App -> C1 : llama_get_logits_ith(0), (1)
  note right : 첫 접근 시 synchronize()로\nasync 복사 완료 보장
  App -> App : seq0/seq1 각각 샘플링\n-> 다음 스텝도 n_tokens=2 배치 구성

else 스레드 B: ctx2 동일 파이프라인 (요약)

  App -> C2 : llama_decode(batch'{n_tokens=2})
  C2 -> M : build_graph() 또는 그래프 재사용 [읽기 전용]
  C2 -> KV2 : 자기 KV만 기록/읽기 (ctx1과 무간섭)
  C2 --> App : logits -> buf_output #2
  note right of C2
    ctx1과 같은 3-1 ~ 3-3 단계를
    자기 소유물(balloc/sched/stream/KV)로
    독립 수행. model 가중치만 공유.
  end note

end

note over M
  두 context가 동시에 실행되어도:
  - model(가중치)은 읽기 전용 공유 -> 동기화 불필요
  - KV/컴퓨트 버퍼/출력 버퍼/스트림은 context별 소유 -> 충돌 없음
  - 같은 GPU에서는 두 스트림의 커널이 하드웨어 스케줄러에 의해 교차 실행
end note
@enduml
```

---

## 핵심 요약

1. **batch = 논리 단위**: `llama_decode()` 1회 제출 묶음. prefill/decode 구분 없이 여러 sequence의 토큰을 혼합할 수 있고, 그 자체로는 연산이 일어나지 않는다.
2. **ubatch = 물리 단위**: 그래프 1회 실행이 처리하는 폭. 컴퓨트 버퍼(VRAM)와 그래프 예약이 이 기준이며, 모든 batch는 항상 ubatch로 분할된다 (prefill에서 두드러질 뿐).
3. **sequence = 대화**: 일상어 "컨텍스트(대화 문맥)"에 해당하는 것은 `llama_context`가 아니라 sequence(`llama_seq_id`)다. `llama_context`는 여러 sequence를 수용하는 실행 엔진이다.
4. **격리는 seq_id로**: 같은 그래프에서 함께 계산돼도 KV 슬롯과 attention mask가 sequence 간 접근을 차단하므로 결과는 개별 실행과 동일하다.
5. **다중화는 sequence로, 격리는 context로**: 배칭 효율이 필요하면 한 context에 여러 sequence, 완전한 격리/개별 설정/스레드 병렬이 필요하면 context를 분리한다.
