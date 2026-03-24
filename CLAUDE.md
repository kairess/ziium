# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

지음(ziium)은 한국어 문장 구조를 중심에 둔 프로그래밍 언어로, Rust로 구현되어 있다. 파일 확장자는 `.zm`이다.

## 주요 명령

```bash
# 파일 실행
cargo run -- path/to/program.zm

# 디버그 모드 (토큰/AST/HIR 확인)
cargo run -- tokens path/to/program.zm
cargo run -- ast path/to/program.zm
cargo run -- hir path/to/program.zm

# REPL
cargo run -- repl

# 테스트 전체 실행
cargo test

# 특정 테스트 실행
cargo test test_name
cargo test --test hir_examples

# 샘플 실행
cargo run -- samples/00_hello_everyone.zm
```

## 브라우저 데모 (WASM) 빌드

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
wasm-pack build --target web --out-dir web/pkg
python3 -m http.server 8000 --directory web
```

## PyTorch 래퍼 환경 설정

tch-rs가 `.venv`의 Python PyTorch를 사용하려면:

```bash
source .venv/bin/activate.fish
export LIBTORCH_USE_PYTORCH=1
export LD_LIBRARY_PATH="$(python -c 'import torch, pathlib; print(pathlib.Path(torch.__file__).parent / "lib")'):$LD_LIBRARY_PATH"
```

버전 불일치 오류 시: `export LIBTORCH_BYPASS_VERSION_CHECK=1`

Python 패키지 관리는 `uv`로 한다 (`pyproject.toml`).

## 파이프라인 구조

```
source -> lexer -> normalizer -> parser -> resolver -> hir lowering -> interpreter
```

- `src/lexer.rs`: 토큰화, 들여쓰기 토큰, 조사 1차 분리
- `src/normalizer.rs`: 한글 조사 모호성 문맥 보정
- `src/parser.rs`: surface AST 생성 (`Call`, `TransformCall`, `Property`, `KeywordMessage`, `Resultive` 등 surface syntax 중심)
- `src/resolver.rs`: 이름 해석, 스코프 검사, definite binding 검사
- `src/hir.rs`: surface AST를 `Send` 중심 HIR로 lowering
- `src/interpreter.rs`: HIR 실행, 내장 함수/메시지 처리, 런타임 진단
- `src/main.rs`: CLI/REPL 진입점

## HIR 구조

HIR의 `SendSelector`는 두 층으로 나뉜다:
- **열린 surface selector**: `Property(String)`, `Transform(String)`
- **닫힌 built-in selector**: `Word(WordMessage)`, `Keyword(KeywordMessage)`, `Resultive(ResultiveMessage)`

현재 built-in 메시지 집합: `길이`, `제곱`, `더하기/빼기/곱하기/나누기`, `추가`, `지우기`, `점찍기`, `사각형채우기`, `글자쓰기`, `맨위/맨뒤/맨앞 요소를 꺼낸`

## 테스트 구조

- `tests/cli_smoke.rs`: CLI 스모크 테스트
- `tests/hir_examples.rs`: HIR lowering 예제 테스트
- `tests/parser_case_docs.rs`: parser fixture 문서 개수 검증
- `tests/repl_session.rs`: REPL 세션 테스트
- `tests/resolver_examples.rs`: resolver 예제 테스트
- `tests/fixtures/parser_cases.md`: parser 문서 기반 fixture

## 문서 구조

- `docs/LANGUAGE.md`: 공식 언어 명세 (문법, 의미, 메시지 경계)
- `docs/GRAMMAR.ebnf`: 파서 구현용 형식 문법
- `docs/DECISIONS.md`: 주요 설계 결정 기록
- `docs/IMPLEMENTATION.md`: 구현 구조와 작업 방식
- `docs/PYTORCH.md`: PyTorch 래퍼 문서
- `AGENTS.md`: AI 에이전트 작업 규칙

## 절대 깨지면 안 되는 규칙

- 영어 키워드의 한글 번역판으로 되돌리지 않는다.
- 조사와 서술어를 장식으로 취급하지 않는다.
- surface syntax와 내부 표현을 섞지 않는다.
- 자연어 추론으로 parser를 확장하지 않는다.
- truthiness를 도입하지 않는다.
- 블록은 들여쓰기 기반이다. 중괄호나 `끝` 토큰을 임의로 도입하지 않는다.
- 속성 접근 표기는 `의`를 유지한다. 점 표기를 도입하지 않는다.

## 변경 시 동기화 규칙

- 표면 문법이나 실행 의미가 바뀌면 `docs/LANGUAGE.md`와 `docs/GRAMMAR.ebnf`를 함께 갱신한다.
- 새 메시지 패턴을 열 때는 구현보다 먼저 문서와 결정 기록을 갱신한다.
- 중요한 설계 판단이 생기면 `docs/DECISIONS.md`에 기록한다.
- 문법 추가나 진단 변경에는 회귀 테스트를 함께 추가한다.
