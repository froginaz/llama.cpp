# HANDOFF: NPU backend 연동 작업자를 위한 온보딩

이 문서는 **다른 환경에서 작업하는 협업자(사람 또는 Claude)가 가장 먼저 읽는 진입점**이다. 이 repo의 문서 작업을 넘겨받아 자체 NPU backend 연동을 진행하는 상황을 전제한다.

전제 조건:

- 전달은 **일방향**이다: 이 repo -> 협업자. 협업자의 작업(NPU 코드)은 보안상 이 repo로 돌아오지 않는다.
- 협업자는 이 브랜치의 스냅샷(clone 또는 git bundle)을 받아 자기 환경에서 이어간다.
- 기준 브랜치: `claude/llaam-context-ggml-backend-qb6nqa`

---

## 1. 이 브랜치에 담긴 것

`docs/my_doc/`의 문서 시리즈가 산출물의 전부다 (코드 변경 없음). 01~11은 GGUF 로딩부터 그래프 실행까지의 파이프라인 문서, 12~13은 이번 작업분이다:

- **12번** ([12-context-model-backend-ownership.md](12-context-model-backend-ownership.md)): `llama_context` / `llama_model` / ggml-backend의 관계, 소유권, 생명주기. UML 관계 분류(연관 vs composition), context:model 조합 매트릭스(multi-context != multi-model, MTP 사례) 포함.
- **13번** ([13-batch-ubatch-and-sequences.md](13-batch-ubatch-and-sequences.md)): 실행 데이터 단위. batch(논리) vs ubatch(물리), sequence vs context, request/slot/session 용어, decode 파이프라인 상세(텐서 shape, KV 레이아웃 주석판 puml), multi-sequence 실행 도구.

커밋 이력이 곧 작업 타임라인이다: `git log --oneline -- docs/my_doc`로 어떤 질문/결정이 어떤 순서로 쌓였는지 볼 수 있다 (모든 커밋이 `[DOC]` 접두사).

## 2. NPU backend 관점의 읽기 순서

전부 읽을 필요 없다. NPU backend 연동에 필요한 순서:

1. **[11-hw-backend-abstraction.md](11-hw-backend-abstraction.md)** - HW 가속기 하나를 ggml / ONNX Runtime / ExecuTorch에 얹을 때의 추상화 비교. NPU 작업의 큰 그림.
2. **[12-context-model-backend-ownership.md](12-context-model-backend-ownership.md)** - reg / device / backend / buft / buffer / sched **여섯 타입의 정체와 소유권**. NPU가 구현해야 할 인터페이스가 어느 계층인지, 누가 생성/소멸하는지가 여기 있다. 특히 "1. 전체 구조 개요"와 클래스 다이어그램.
3. **[13-batch-ubatch-and-sequences.md](13-batch-ubatch-and-sequences.md)** - backend가 실제로 받는 실행 단위. **컴퓨트 버퍼 크기가 n_ubatch 폭으로 결정**되는 것, 그래프 재사용 조건, [상세 주석판 puml](13-sequence-decode-batch2-single-context-detailed.puml)의 텐서 shape 흐름이 NPU 메모리 설계에 직접 관련된다.
4. (배경이 더 필요하면) 06-execution-and-scheduling.md, 08-hardware-divergence.md.

## 3. NPU backend 구현 진입점 (코드)

| 파일 | 역할 |
|---|---|
| `ggml/include/ggml-backend.h` | 공개 API - backend 사용자(llama.cpp 측)가 보는 것 |
| `ggml/src/ggml-backend-impl.h` | **backend 구현자가 채워야 할 iface 정의** (device / buffer_type / buffer / backend 4종의 함수 테이블) |
| `ggml/src/ggml-backend-reg.cpp` | 레지스트리 - 새 backend가 여기 등록되어야 디바이스 열거에 잡힘 |
| `ggml/src/ggml-blas/` | 가장 작은 참고 구현 (시작점으로 적합) |
| `ggml/src/ggml-cuda/` | 완전한 참고 구현 (buffer/stream/graph 전부) |
| `tests/test-backend-ops.cpp` | **backend 검증 표준 도구** - op 단위로 CPU와 결과 비교. NPU 커널 하나 붙일 때마다 여기로 정합성 확인 |

sched와의 계약이 핵심이다: backend는 `supports_op` / `supports_buft` / `offload_op`로 자기가 할 수 있는 일을 선언하고, **못 하는 op는 sched가 자동으로 CPU에 폴백**시킨다 (12번 문서의 sched 역할 참고). 따라서 NPU는 전체 op를 다 구현할 필요 없이 mul_mat부터 점진 확장이 가능하다.

스모크 테스트 순서 권장: `test-backend-ops` (op 정합성) -> `llama-cli -m tiny모델 -ngl 99` (단일 seq E2E) -> `llama-batched -np 2` (멀티 seq) -> `llama-batched-bench` (처리량).

## 4. 작업 규약 (이 repo에서 지키던 것)

- **AGENTS.md 필독**: upstream(ggml-org/llama.cpp)으로의 AI 생성 PR은 금지다. **private fork/사내 작업은 예외**이므로 NPU 작업 자체는 문제 없으나, 성과를 upstream에 낼 계획이 생기면 AGENTS.md 절차(사람이 이해/설명 가능해야 함)를 따라야 한다.
- 문서 규약: `NN-주제.md` 번호 체계, PlantUML은 인라인 코드블록 + 별도 `.puml` 파일 병행, README.md 색인 갱신, ASCII 화살표(`->`) 사용.
- 커밋 규약: `[DOC] 한 줄 요약` + `Assisted-by: Claude` 트레일러 (Co-authored-by 금지 - AGENTS.md).

## 5. 전달 방법 (bundle 레시피)

### 협업자가 이 repo를 읽을 수 있는 경우

브랜치명과 이 문서 경로만 전달하면 된다: clone -> `docs/my_doc/00-handoff-npu-backend.md`부터 읽기.

### 완전 격리 환경인 경우: git bundle (검증된 레시피)

협업자 환경에 **공개 llama.cpp clone이 있다는 전제**로, 이 브랜치의 고유 커밋만 담은 증분 bundle을 만들어 파일로 반입한다. 이 방식은 이 repo에서 실제 검증되었다 (bundle 크기 ~138KB).

```bash
# [송신측] basis = 문서 작업 직전의 upstream 커밋 (수신측도 반드시 보유)
#   확인: git log --oneline --reverse -- docs/my_doc | head -1  의 부모 커밋
git bundle create handoff-$(date +%Y%m%d).bundle \
    6b80c74..claude/llaam-context-ggml-backend-qb6nqa
git bundle verify handoff-*.bundle   # "requires this ref: 6b80c74..." 표시가 정상
git tag shared/1 claude/llaam-context-ggml-backend-qb6nqa   # 공유 지점 기록

# [수신측] 공개 llama.cpp clone 안에서
git fetch /path/to/handoff-YYYYMMDD.bundle \
    claude/llaam-context-ggml-backend-qb6nqa:refs/heads/npu-work
git checkout npu-work    # docs/my_doc/ 전체 + 커밋 이력 확보

# [송신측] 이후 추가 작업분은 증분으로
git bundle create incr-$(date +%Y%m%d).bundle shared/1..claude/llaam-context-ggml-backend-qb6nqa
git tag -f shared/2 claude/llaam-context-ggml-backend-qb6nqa
# 수신측: git fetch /path/to/incr.bundle npu-work 브랜치로 동일하게
```

주의사항:

- **송신측 clone이 shallow면** (CI/원격 환경에서 흔함) 전체 이력 bundle(`git bundle create x.bundle 브랜치명`)은 만들어져도 수신측 clone이 실패한다. 위처럼 **basis 범위를 명시한 증분 bundle**을 쓰거나, `git fetch --unshallow origin` 후 전체 bundle을 만들 것.
- basis 커밋은 반드시 upstream에 존재하는 커밋이어야 한다 (PR 번호가 달린 커밋이면 안전).
- 커밋 이력 없이 문서 파일만 필요하면 `git archive HEAD docs/my_doc -o docs.tar`가 최소 대안이지만, 작업 타임라인(커밋 메시지)이 사라지므로 권장하지 않는다.

## 6. 협업자에게: 시작하는 방법

1. 이 문서 -> 2절의 읽기 순서대로 3개 문서를 읽는다 (약 30분 분량).
2. `git log --oneline -- docs/my_doc`로 의사결정 타임라인을 훑는다.
3. NPU 작업은 **자기 환경의 브랜치**에서 진행한다 (이 repo로 push하지 않는다). 문서를 추가하게 되면 14번부터 번호를 이어 쓰되, 그 문서들도 이 repo로 돌아오지 않음을 전제로 자기 repo의 README 색인을 별도 관리할 것.
4. 이 repo 쪽 작업이 갱신되면 5절의 증분 bundle로 받는다. 충돌 걱정은 없다 - 이쪽은 `docs/my_doc/`의 문서만 만지고, NPU 코드는 `ggml/src/` 쪽이므로 파일이 겹치지 않는다.
