# Temporal QA (Immediate-After) Demo

이 폴더는 논문 제출용 최소 재현 데모입니다. 주어진 샘플 컨텍스트(`submission/sample/demo.jsonl`)에 대해 TKG 규칙 + MCTS(LLM 훅 포함)의 하이브리드로 즉시 후 질의를 처리합니다. 기본 데모는 Dummy LLM으로 동작하며, 실제 GPT-4 호출로 대체할 수 있습니다.

## 구성
- `submission/sample/demo.jsonl`: 샘플 1건 (컨텍스트 내 트리플과 질문/정답)
- `submission/qa/*`: TKG 인덱스/파서 (간단 포맷)
- `submission/vendor_adapter/tkg_backend.py`: REKG-MCTS 인터페이스용 어댑터
- `submission/vendor/rekgmcts/*`: 최소화한 MCTS/프롬프트/유틸
- `submission/scripts/run_demo.py`: 데모 실행 스크립트
- `submission/requirements.txt`: 의존성

## 설치
```bash
python -m venv .venv && ./.venv/Scripts/activate  # Windows PowerShell
pip install -r submission/requirements.txt
```

## 실행
```bash
python submission/scripts/run_demo.py --input submission/sample/demo.jsonl
```
출력 예시:
```json
{"pred": "E6", "label": "E6", "correct": 1, "path": [["E6", "R11", "E76"]]}
{
  "total": 1,
  "answered": 1,
  "accuracy_on_answered": 1.0,
  "overall_accuracy": 1.0
}
```

## 실제 GPT-4로 대체 (선택)
- 데모는 기본적으로 DummyLLM으로 점수/프루닝을 시뮬레이션합니다.
- 실제 GPT-4 API를 사용하려면, `submission/scripts/run_demo.py`의 `DummyLLM`을 교체하고 OpenAI 키를 환경변수 `OPENAI_API_KEY`로 설정하세요.

## 입력 포맷(간단)
- 컨텍스트의 각 줄은 다음 형식입니다: `E<HEAD> R<REL> E<TAIL> [<START>,<END>]`
- 질문 예: `Find the entity that was the R11 of E76 immediately after E57 R11 E76`


