# microkv

초소형 KV 캐시 장부(ledger) 모형. llama.cpp의 `llama_kv_cells`를 교육용으로
축소한 순수 파이썬 구현이다. **unit-test 자동 생성 데모의 대상 프로젝트**로
설계되었다.

## 구조

```
microkv/
├── microkv/
│   ├── __init__.py
│   └── ledger.py        <- 유일한 소스 (CellLedger, CacheFullError)
├── tests/               <- 데모에서 AI가 채울 위치 (현재 비어 있음)
├── requirements.txt
└── README.md
```

## 설계 의도 (테스트 생성 친화)

- **순수 함수/결정론**: I/O, 시간, 난수, 외부 의존성 없음 - mock 불필요.
- **명시적 계약**: 모든 공개 메서드에 Args/Returns/Raises docstring.
- **예외 경로 내장**: `ValueError` 4종 + `CacheFullError` 1종.
- **경계값이 자연스러움**: n_cells=1, p1=-1(끝까지), 빈 seq, 셀 재사용 등.
- **작음**: 소스 1파일, 공개 메서드 5개 - 작은 컨텍스트에도 통째로 들어감.

## 실행

```bash
pip install -r requirements.txt
pytest -q          # 데모 전: 수집된 테스트 0개
                   # 데모 후: AI가 생성한 tests/test_ledger.py가 전부 통과해야 함
```
