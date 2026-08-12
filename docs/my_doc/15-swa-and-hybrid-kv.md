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

## 2. 배경: 왜 full-attention과 SWA 레이어를 섞어 쓰는가

구현으로 들어가기 전에, "혼합(interleaved)"이라는 설계 자체를 표준 모델과 비교해 이해해 둔다.

### 전제: 레이어란 무엇인가 - 동일한 기계 32개와 창발적 분업

흔한 오해부터 교정한다: **레이어는 "역할이 배정된 서로 다른 부품"이 아니다.** 32개 레이어는 완전히 똑같은 구조(attention + FFN)를 32번 쌓은 것이고, 모든 토큰이 전부를 순서대로 통과한다:

```
토큰 "Paris"의 hidden state (4096차원 벡터)의 여정:

임베딩 -> [레이어 1] -> [레이어 2] -> ... -> [레이어 32] -> lm_head -> logits
           같은 구조      같은 구조            같은 구조
```

레이어 하나가 하는 일은 **한 라운드의 "수집 + 가공"**이다:

- **attention**: 다른 토큰들을 둘러보고 필요한 정보를 내 벡터로 가져온다 - 이때 참조하는 것이 그 레이어의 KV 캐시. **레이어마다 K/V가 따로 있는 이유가 "레이어마다 attention을 한 번씩 하기 때문"**이다 (13번 문서의 x n_layer 항의 정체).
- **FFN**: 가져온 정보를 소화/변환한다.

설계상 동일한 기계들인데, **훈련이 끝나면 깊이에 따른 분업이 저절로 생긴다** (해석 가능성 연구의 경향적 관찰):

| 깊이 | 저절로 맡게 되는 일 | 이 일에 필요한 시야 |
|---|---|---|
| 초반 (1~8층) | 철자/형태소/품사, 인접 단어의 문법 관계 | **좁음** - 주변 몇~수십 토큰 |
| 중반 | 의미 관계, 개체 추적, "그것"이 뭘 가리키는지 | 중간 ~ 가끔 멂 |
| 후반 | 전체 맥락 통합, 다음 토큰 계획 | 넓을 수 있음 |

분업이 생기는 이유는 필연적이다: 레이어 k의 입력은 레이어 k-1의 출력이므로, **얕은 층은 원재료(단어)를, 깊은 층은 이미 가공된 의미를 다룰 수밖에 없다** - 조립 라인의 앞 공정/뒷 공정과 같다.

장치 하나를 더 알아야 한다: **residual stream**. 각 레이어는 4096차원 "공용 메모장"에서 읽고 거기에 덧붙이는 방식이라, **레이어 3이 수집한 정보는 레이어 30에서도 사용 가능**하다. 이것이 아래 혼합 설계에서 결정적 역할을 한다.

이 전제에서 혼합의 논리가 나온다: **레이어가 맡은 일이 곧 그 레이어에게 필요한 시야를 결정한다.** 문법/인접 관계를 다루는 다수 레이어에게 10만 토큰 전체를 볼 권한(= n_ctx 선형 KV 비용)을 주는 것은 낭비다 - 실측해 보면 attention 가중치가 어차피 근처에 몰려 있다. 반면 "3만 토큰 앞의 복선 회수"는 누군가는 전체를 직접 봐야 한다. 그래서 설계자가 "27개는 창만, 5개는 전체" 제약을 걸면, **전체를 볼 수 있는 유일한 창구인 full 레이어들이 장거리 회수 전문가로 저절로 특화**되고, full 레이어가 가져온 먼 정보는 residual stream을 타고 이후 SWA 레이어들에게 전달된다.

비유: 원고 하나가 편집자 32명을 차례로 통과하는 교정팀이다. 모두 같은 빨간펜을 들었지만 앞자리는 자연히 맞춤법(주변 몇 단어면 되는 일), 뒷자리는 논지 일관성을 맡게 된다. 표준 모델은 32명 전원에게 원고 **전체 열람권**을 주는 것 - 맞춤법 담당자는 쓰지도 않는 권한의 비용(KV)을 다 낸다. SWA 혼합은 27명에게 "직전 몇 페이지만" 주고 복선 담당 5명에게만 전체 열람권을 주며, 5명이 찾아온 정보는 회람 메모(residual stream)로 뒷사람들에게 전달된다.

### 수식으로 보는 레이어: 입력 토큰 -> attention -> FFN -> logits

위 "수집 + 가공"을 수식과 shape으로 정확히 따라가 본다. 단일 토큰 decode(14번 문서 예제의 "Paris", pos 7) 기준이며, 13번 문서 상세 주석판의 텐서 흐름(`[4096,1] -> [32000,1]`)의 수학적 실체다.

**0단계. 입력 토큰 -> 벡터**

```
x = tok_embd[ id("Paris") ]          x ∈ R^4096
```

`get_rows` 한 번 - 어휘표(32000 x 4096)에서 행 하나를 꺼낸다. 이 `x`가 residual stream(공용 메모장)의 시작값이고, 이후 32개 레이어가 여기에 계속 덧붙인다.

**왜 4096인가 - n_embd(모델의 폭)**

hidden state의 차원 = `n_embd`(embedding dimension)로, **토큰 하나의 의미/문맥을 담는 벡터의 크기 = residual stream의 폭**이다. 파이프라인 전체가 이 폭으로 통일되어 있어(임베딩 -> 모든 레이어 -> lm_head) 레이어를 자유롭게 쌓을 수 있는 규격 역할을 한다. llama.cpp가 정하는 값이 아니라 GGUF 메타데이터(`llama.embedding_length`)에서 읽는 모델 고유값이다 (14번 문서의 "hparams = GGUF 메타데이터" 원칙).

4096이라는 값은 세 제약의 교집합이다:

1. **헤드 분할 정합**: `n_embd = n_head x d_head` (4096 = 32 x 128). d_head=128이 사실상 표준이므로 폭은 128의 배수 중에서 골라진다.
2. **하드웨어 친화**: 2의 거듭제곱(4096 = 2^12)이면 GPU 타일/텐서코어 정렬에 최적.
3. **스케일링 경험칙**: 파라미터 수 ~= `12 x n_layer x n_embd^2` - 폭의 **제곱**에 비례한다. 훈련 예산이 정한 파라미터 규모를 폭과 깊이에 배분하는 관례가 익숙한 계열을 만든다:

| 모델 규모 | n_embd (폭) | n_layer (깊이) | n_head x d_head |
|---|---|---|---|
| ~1B | 2048 | 22 | 16 x 128 |
| **~7B** | **4096** | **32** | **32 x 128** |
| ~13B | 5120 | 40 | 40 x 128 |
| ~70B | 8192 | 80 | 64 x 128 |

패턴: **모델을 키울 때 d_head(128)는 고정하고 폭과 깊이를 함께 늘린다.** 검산: 12 x 32 x 4096^2 ~= 6.4B ~= "7B". 즉 4096은 법칙이 아니라 "7B급의 폭"이라는 설계 관례다.

폭의 직관: residual stream에는 여러 정보(문법 특징, 개체, 의미)가 **중첩(superposition)**되어 실려 다니는데, n_embd가 그 동시 수용량이다. 좁으면 병목, 넓으면 예산 대비 낭비. 주의할 구분 하나 - KV 캐시의 폭 1024는 n_embd가 아니라 **GQA로 축소된 별도 값**(n_head_kv x d_head)이다: residual stream은 4096으로 다니지만 캐시에 저장되는 K/V만 1024로 압축된다 (13번 문서 6장).

**레이어 l의 전반부: attention = "수집"**

```
                    ┌─────────────────── residual stream x ────────────────────┐
                    │                                                          │
  h = RMSNorm(x)    │   (1) 정규화                                             │
                    │                                                          ▼
  q = W_q·h         │   (2) 투영: q ∈ R^4096 (32헤드 x 128)               x + attn_out
  k = W_k·h         │            k ∈ R^1024 (8헤드 x 128)  <- GQA 축소        │
  v = W_v·h         │            v ∈ R^1024                                    │
                    │                                                          │
  q,k <- RoPE(q,k,pos=7)  (3) 위치 회전 (K에 구워짐 -> K-shift가 필요한 이유) │
                    │                                                          │
  K_cache[l][cell 7] = k    (4) KV append (set_rows 노드 - 14번 문서의 그 지점)│
  V_cache[l][cell 7] = v                                                       │
```

**(5) attention 본체** - 먼저 인덱스 정의:

```
i    = Q 헤드 번호,   i ∈ {1, ..., 32}      (n_head개 - "누가 묻는가", 채널 축)
       qᵢ ∈ R^128 = q(∈R^4096)의 i번째 128차원 조각

j    = KV 셀 번호,    j ∈ {0, ..., n_kv-1}  ("어느 과거를 보는가", 토큰/셀 축)

h(i) = 헤드 i가 쓰는 KV 헤드 번호 = ceil(i/4) ∈ {1, ..., 8}
       (GQA: Q헤드 4개가 KV헤드 1개를 공유 - 32/8 = 4)
       셀 j의 K 1024개 = [128 x 8]이고, 헤드 i는 그중 h(i)번째 128차원 조각만 사용
```

**주의 - n_kv != n_head_kv**: 이름이 비슷하지만 축이 다르다. `n_head_kv`(=8)는 K/V **헤드 개수**(채널 축, 모델 설계 상수)이고, `n_kv`는 attention 창의 **셀 개수**(토큰 축, 대화가 길어지면 커지는 런타임 값)다.

헤드 i마다:

```
                 qᵢ · K_cache[l][j, h(i)]
  score_i(j) = ──────────────────────────  +  mask[j]     j = 0 .. n_kv-1
                      √d_head(=128)                        ↑
                                                      KQ_mask가 수식에
  α_i = softmax_j(score_i)    <- α_i ∈ R^n_kv          들어오는 정확한 자리:
        (j축에 대한 softmax)                           보이는 셀 = +0
                                                      가려진 셀 = -inf -> α=0
  oᵢ = Σⱼ α_i(j) · V_cache[l][j, h(i)]   <- oᵢ ∈ R^128
```

mask[j]가 **j축에만** 걸리는 이유도 이 구분에서 자명하다 - 격리/causal/SWA는 전부 "어느 과거(j)를 볼 수 있는가"의 문제이지, "어느 헤드(i)가 보는가"의 문제가 아니다.

**(5)-보충: 텐서 shape으로 다시 보기** - 인덱스 i, j가 텐서의 어느 축인지 따라가면 추상 인덱스가 물리적 실체가 된다. batch 1 decode 기준, ggml 표기 `[ne0, ne1, ne2]`:

```
축 이름:  d = 128 (d_head, 내적 축)    i = 32 (n_head, "누가 묻는가")
          j = n_kv (셀/창, "어느 과거") t = 1  (n_tokens)

q  [4096, 1] ──reshape──> [128, 32, 1]              = [d, i, t]
                             │
K창 view      [128, 8, n_kv] = [d, h, j]   <- 캐시에서 zero-copy로 자른 창
              (GQA: h=8이 i=32에 4개씩 공유/브로드캐스트)
                             │
score = K·q               [n_kv, 32, 1]   = [j, i, t]
                             ▲
              ★ d축(128)이 내적으로 "소멸"하고, j축이 "탄생"
                             │
 + mask       [n_kv, 1]     = [j, t]      <- ★ i축이 없다!
                             │               "mask는 j축의 것"이라는 사실이
softmax (j축) [n_kv, 32, 1]  │               shape 자체로 증명됨
                             │
o = V·α                   [128, 32, 1]  = [d, i, t]
                             ▲
              ★ j축이 가중합으로 "소멸", d축이 V에서 복귀
                             │
reshape + W_o             [4096, 1]     = [n_embd, t]
                             ▲
              ★ i축(32)이 연결(concat)로 접혀 소멸 - attention 끝
```

축의 생사(生死) 추적표:

| 단계 | shape | 무슨 일이 | 소멸한 축 | 탄생한 축 |
|---|---|---|---|---|
| q reshape | `[128, 32, 1]` | 채널을 헤드로 분할 | - | i |
| **score = K·q** | `[n_kv, 32, 1]` | 내적 | **d** | **j** |
| + mask `[n_kv, 1]` | 동일 | j축만 검열 (i축엔 브로드캐스트) | - | - |
| softmax_j | 동일 | j축을 확률분포로 | - | - |
| **o = V·α** | `[128, 32, 1]` | 가중합 | **j** | d (복귀) |
| reshape + W_o | `[4096, 1]` | 헤드 연결 | **i** | - |

이 표에서 두 가지가 즉시 명확해진다:

1. **n_kv vs n_head_kv 혼동의 최종 해소**: n_kv(=j)는 score에서 태어나 o에서 죽는 **일시적 축**이고, n_head_kv(=8)는 K창 view의 붙박이 **저장 형식 축**이다. 같은 텐서 `[128, 8, n_kv]` 안에 서로 다른 자리로 동시에 존재한다.
2. **mask의 소속**: mask shape `[n_kv, n_tokens]`에는 헤드 축이 아예 없다 - 위 문단의 결론이 shape으로 증명된다. 13번 문서 상세 주석판의 `KQ_mask [n_kv, 2]`가 정확히 이것이다.

attention 전체를 한 문장으로: **"[d,i,t]가 캐시 [d,h,j]를 만나 j를 잠시 빌렸다가(score), 검열받고(mask), 되갚으면서(가중합) d로 돌아오는 여정"**이다.

여기서 이 문서 시리즈의 모든 논의가 수식 한 줄에 모인다: **mask[j]가 -inf면 softmax 후 가중치가 정확히 0** - seq 격리도, causal도, SWA 창도, 패딩 차단도 전부 이 덧셈 항 하나로 구현된다. **SWA 레이어란 "mask가 j ∈ [pos-n_swa, pos] 밖을 전부 -inf로 채우는 레이어"일 뿐이다.**

```
  attn_out = W_o · [o₁; o₂; ...; o₃₂]     (4096 <- 32 x 128 연결)
  x = x + attn_out                        (6) residual 덧셈: 메모장에 덧붙임
```

**레이어 l의 후반부: FFN = "가공"**

```
  h = RMSNorm(x)
  ffn_out = W_down · ( silu(W_gate·h) ⊙ W_up·h )      (SwiGLU)
             4096  <-        11008    ⊙  11008  <- 4096
  x = x + ffn_out                          residual 덧셈
```

attention이 "남의 정보를 가져오는" 유일한 단계라면, FFN은 **토큰 혼자서** 자기 벡터를 비선형 변환하는 단계다 (상세 주석판의 `[11008, 1]` 왕복).

**32층 통과 후: lm_head -> logits**

```
  x_final = RMSNorm(x)                     (32층의 덧붙임이 누적된 메모장)

  logits  = W_lm_head · x_final            R^32000 <- R^4096
            ^^^^^^^^^
            lm_head (language model head) = 최종 hidden state를
            어휘 분포로 투영하는 행렬. llama.cpp의 model.output 텐서

  P(다음 토큰) = softmax(logits) -> 샘플링 -> "is"
```

lm_head에 대해 알아둘 것:

- **크기**: `[n_embd, n_vocab]` = 4096 x 32000 ~= 1.3억 파라미터 (F16 ~262MB) - 단일 행렬로는 모델 최대급. 소형 모델은 `tok_embd`와 가중치를 공유(weight tying)하기도 한다 - 0단계의 어휘표를 방향만 바꿔 재사용하는 셈.
- **레이어의 attention/FFN과 달리 딱 1회만** 실행된다 - 32층의 반복 구조 밖에 있는 "출구"다.
- **출력 토큰에만 계산한다**: prefill에서 13토큰이 들어와도 `logits=1`인 토큰(inp_out_ids)만 lm_head를 통과한다 (13번 문서 prefill 주석판의 "get_rows에서 13 -> 2 축소" 후 lm_head가 실행되는 이유). 중간 토큰의 logits는 아예 만들지 않는다.
- logits의 i번째 값 = "다음 토큰이 어휘 i일 점수"이고, 여기에 softmax/샘플링(온도, top-k 등)을 적용하는 것은 host의 샘플러 몫이다.

**전체 조감도**

```
 "Paris"(id) ──get_rows──> x⁰ ∈ R⁴⁰⁹⁶
                            │
              ┌─ 레이어 1 ──┤  x¹ = x⁰ + Attn₁(x⁰) + FFN₁(...)
              │  ...        │        ▲          ▲
              │             │        │          └ 혼자 가공
              │             │        └ K/V_cache[1] 참조·기록 + mask
              ├─ 레이어 l ──┤  xˡ = xˡ⁻¹ + Attnₗ(xˡ⁻¹) + FFNₗ(...)
              │  ...        │        (레이어마다 자기 KV - x n_layer의 정체)
              └─ 레이어 32 ─┤  x³²
                            │
                     RMSNorm + lm_head(W_lm_head, 4096x32000, 1회만)
                            │
                     logits ∈ R³²⁰⁰⁰ ──softmax/샘플링──> "is"
```

**ggml 구현과의 1:1 대응** (상세 주석판의 노드들이 위 수식의 어느 항인가):

| 수식 | ggml op |
|---|---|
| `W_q·h` 등 투영, lm_head(`W_lm_head`) | `mul_mat` (가중치 leaf x 활성값) |
| RoPE | `rope` (K-shift 그래프도 같은 op) |
| KV append | `set_rows` (14번 문서의 "그래프 안의 KV update") |
| `q·Kᵀ/√d + mask -> softmax -> ·V` | `mul_mat` + `soft_max(mask)` + `mul_mat`, FA면 `flash_attn_ext` 1개로 융합 |
| residual 덧셈 | `add` |
| SwiGLU | `mul_mat` x3 + `silu` + `mul` |

이 수식 수준에서 "레이어를 섞는다"의 의미가 최종적으로 명확해진다: **레이어 간 차이는 오직 (5)의 mask[j]가 허용하는 j의 범위뿐**이고(전체 vs 최근 n_swa), 나머지 수식은 전 레이어 동일하다. 그 mask 범위 차이가 K/V_cache[l]의 필요 크기 차이(n_ctx 선형 vs 상수)로, 곧 3장의 2-캐시 구조로 이어진다.

### 표준 모델: 모든 레이어가 전체를 본다

표준 transformer(Llama 3 등)는 모든 레이어의 모든 토큰이 자기 이전의 **전체** 토큰을 attend한다:

```
표준 (full attention, 모든 레이어):
pos:      0  1  2  ...                            99999
토큰 100000이 보는 범위:  [0 ................. 99999]   <- 전부, 매 레이어마다
```

이 "전부 본다"의 대가가 1장의 비용 공식이다: KV와 attention 연산 모두 문맥 길이에 선형.

### SWA: 최근 것만 본다

```
SWA (n_swa = 4096):
토큰 100000이 보는 범위:        [95904 ....... 99999]   <- 최근 4096개만
```

창 밖 토큰의 K/V는 **그 레이어에서는** 다시 쓰일 일이 없으므로 버려도 된다 - 4장의 링 버퍼(lazy 만료)가 성립하는 근거이고, 그 레이어의 KV 비용은 문맥 길이와 무관한 상수가 된다.

### 왜 "섞는가": 순수 SWA의 한계와 수용 영역의 누적

**순수 SWA(전 레이어 창)의 한계**: 창 밖 정보에 직접 접근할 수 없어 "10만 토큰 앞에 심어둔 비밀번호를 말해줘" 같은 **장거리 정확 회수(retrieval)**가 무너진다. (초기 Mistral 7B가 전 레이어 SWA였고, 이 한계가 확인되며 순수 SWA는 퇴조했다.)

그래도 SWA 레이어가 쓸모 있는 이유는 **수용 영역(receptive field)이 층마다 누적**되기 때문이다:

```
레이어 1: 토큰 X는 [X-4096, X]를 직접 봄
레이어 2: "레이어 1이 [X-8192..X-4096]를 이미 요약해 둔 토큰들"을 봄
  -> 간접적으로 [X-8192, X]에 접근
레이어 k: 유효 수용 영역 ~= k x n_swa     (CNN의 수용 영역 누적과 같은 원리)
```

32층 x 4096창이면 이론상 13만 토큰까지 정보가 간접 전파된다. 다만 "요약의 요약"이라 흐릿해서 정밀한 장거리 조회에는 부족하다.

**그래서 혼합**: 대다수 레이어는 SWA로 두고, 몇 층에 하나씩 full 레이어를 끼워 전체 문맥으로의 "직통 회선"을 복원한다:

```
Gemma 3 (5:1):   [SWA SWA SWA SWA SWA FULL] [SWA SWA SWA SWA SWA FULL] ...
Gemma 2 (1:1):   [SWA FULL] [SWA FULL] ...
```

이 설계를 정당화하는 경험적 사실: **attention 질량의 대부분은 원래 지역적**이다 (문법, 지시어, 구문 관계는 근처 토큰으로 결정). 대다수 레이어를 창으로 잘라도 품질 손실이 미미하고, 소수의 full 레이어가 장거리 회수라는 특수 임무를 전담한다.

비유: 100명이 참여하는 하루짜리 회의 기록 - 표준 모델은 서기 32명 전원이 아침부터의 발언 전체를 계속 뒤적이는 방식이고, SWA 혼합은 서기 27명이 "최근 10분"만 들으며 흐름을 따라가고 5명만 전체 회의록 열람 권한을 갖는 방식이다. "오전에 뭐라고 했었죠?"는 그 5명이 답한다.

### 비교표

| | 표준 (전층 full) | 순수 SWA (전층 창) | **혼합 (interleaved)** |
|---|---|---|---|
| 토큰의 직접 시야 | 전체, 매 레이어 | 최근 n_swa, 매 레이어 | 레이어에 따라 다름 |
| 장거리 정확 회수 | 최상 | **불가** (수용 영역 전파는 흐릿) | full 레이어 전담 -> 거의 유지 |
| KV 비용 (n_ctx=128K) | 전 레이어 선형 -> 수십 GB | 전 레이어 상수 | full층만 선형: 5:1이면 **~1/6 + 소량 상수** |
| 대표 모델 | Llama 3, Qwen | (초기 Mistral) | Gemma 2/3, GPT-OSS, Ministral |

### llama.cpp와의 연결

- 어느 레이어가 SWA인지는 **모델 설계자가 정하고 GGUF 메타데이터로 전달**된다 - `llama.attention.sliding_window`(창 크기)와 레이어 패턴이 hparams의 `is_swa(il)`로 조회될 뿐, llama.cpp가 결정하지 않는다 (14번 문서의 "hparams = GGUF 메타데이터" 원칙).
- 이 "레이어 이질성"이 다음 장 iSWA 2-캐시 구조의 존재 이유다: full층 셀은 오래 살고 SWA층 셀은 링 버퍼로 돌기 때문에, **수명 정책이 다른 두 집단을 한 캐시에 섞을 수 없어** `is_swa(il)` 필터로 두 표준 캐시에 나눠 담는다.
- 창 판정의 세부 변형(`swa_type`: 창 경계를 자르는 방식)도 hparams로 전달되어 mask 생성과 셀 재사용 판정(`is_masked_swa`)에 동일하게 적용된다.

## 3. iSWA: llama_kv_cache 두 개의 조합

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

`--swa-full` 옵션(iswa.cpp:53~58)은 SWA 캐시도 풀사이즈로 만든다 - 5장의 rollback 제약을 없애는 대신 메모리 절감을 포기하는 트레이드오프다.

## 4. SWA 셀의 생애: 별도 프루닝 패스는 없다 (lazy 만료)

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

## 5. 운영상 결과: rollback 제약과 체크포인트

SWA 캐시에는 창 밖의 과거가 **물리적으로 없다**(덮어써짐). 그 결과:

- **자유로운 rollback 불가**: `seq_rm(seq, p, -1)`으로 창 이전 지점까지 무르고 재-decode하는 것이 불가능하다 - 그 구간의 K/V가 이미 사라졌기 때문. 14번 문서 E1(speculative)의 전제가 창 안에서만 성립한다.
- **서버의 보완책 = 컨텍스트 체크포인트**: llama-server의 `-ctxcp`(`--ctx-checkpoints`, 기본 32개)가 프롬프트 처리 중 주기적으로 SWA 캐시 상태의 스냅샷을 host 메모리에 저장해 둔다 (tools/server/server-context.cpp:1129~, create_checkpoint 2033~). 되돌아갈 일이 생기면(prefix 재사용, speculative 등) 가장 가까운 체크포인트를 복원하고 그 지점부터 재-decode한다. 스냅샷은 `llama_state_seq_*`에 **`LLAMA_STATE_SEQ_FLAGS_SWA_ONLY`** 플래그(include/llama.h:873)를 줘서 SWA 캐시 부분만 덤프한다 - base 캐시는 rollback 가능하므로 저장할 필요가 없다.

정리하면 SWA는 "메모리 절감 <-> 시간 여행의 자유"를 교환하며, 그 균형점을 사용자가 고른다: 기본(창 크기 캐시 + 체크포인트) vs `--swa-full`(풀 캐시, 제약 없음).

## 6. Hybrid: attention 캐시 + recurrent 상태의 합성

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

## 7. 세 구현의 기능 매트릭스

| 기능 | unified `llama_kv_cache` | `iswa` (SWA) | `hybrid` (recurrent 포함) |
|---|---|---|---|
| seq_rm rollback | 자유 | **창 안에서만** (밖은 체크포인트) | **불가** (체크포인트/n_rs_seq) |
| seq_cp 공유 | 태그만 (0 복사) | 두 캐시 팬아웃 | attn 태그 + recurrent 상태 복사 |
| K-shift (`seq_add`) | O | O (두 캐시) | attn만 의미 있음 |
| session save/restore | O | O (SWA_ONLY 부분 저장 지원) | O (상태 포함) |
| ubatch 분할 | split_simple | split_simple(unified) | **split_equal/seq 강제** |
| 메모리 비용 | n_ctx 선형 | full층 선형 + SWA층 상수 | attn층 선형 + recr층 **상수** |

## 8. dNPU 함의

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
