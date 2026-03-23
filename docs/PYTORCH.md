# PYTORCH Wrapper

이 문서는 지음의 PyTorch 스타일 래퍼를 정리한다. 구현은 `tch-rs`를 사용한다.

- 참고: [LaurentMazare/tch-rs](https://github.com/LaurentMazare/tch-rs)

## 지음식 래퍼 철학

- 래퍼 이름은 동작 중심의 한국어로 둔다.
  - 생성자는 불필요한 `만들기`를 기본 이름에서 뺀다.
  - 예: `숫자손글씨데이터셋(...)`, `데이터로더(...)`
- 학습 루프 의미를 드러내는 이름을 우선한다.
  - 예: `매개변수갱신(최적화기)` (`스텝`은 하위 호환 별칭으로만 유지)
- 모델/배치/텐서는 "속성 + 메시지" 흐름으로 읽히게 만든다.
  - 예: `배치의 이미지들`, `모델의 매개변수`, `손실의 값`
- 시간/측정도 한국어 내장 함수로 제공한다.
  - 예: `현재시간초()`

## 자동 장치 동작

- 모델/배치는 내부에서 자동으로 장치를 고른다.
  - CUDA 가능: 자동 CUDA
  - CUDA 불가: 자동 CPU
- 평가모드로바꾸기를 호출하면 순전파는 자동으로 no-grad로 실행된다.

## 자동 데이터 준비

- `숫자손글씨데이터셋({ 학습용: ... })`은 `./data/MNIST/raw`만 사용한다.
- 이 경로에서 MNIST를 읽지 못하면, 자동으로 MNIST 원본 4개 파일을 `./data/MNIST/raw`에 내려받아 압축을 풀고 다시 로드한다.

## 주요 내장 함수

- 모델 구성: `평탄화`, `선형층`, `렐루`, `순차신경망`
- 데이터: `숫자손글씨데이터셋`, `데이터로더`, `배치가져오기`
- 학습: `교차엔트로피손실`, `아담`, `기울기초기화`, `역전파`, `매개변수갱신`
- 추론/평가: `순전파`, `최대인덱스`, `같은값개수`, `학습모드로바꾸기`, `평가모드로바꾸기`
- 시간 측정: `현재시간초`

> 하위 호환: `숫자손글씨데이터셋만들기`, `데이터로더만들기`, `스텝`도 현재는 동작한다.

## MNIST 예제

- 전체 예제:
  - `samples/15_minst_torch.zm`
- 실행:

```bash
cargo run -- samples/15_minst_torch.zm
```

### 시간 측정 예시

```zm
시작시간초는 현재시간초()이다.
# ... 학습/평가 ...
종료시간초는 현재시간초()이다.
걸린시간초는 종료시간초 - 시작시간초이다.
{ 걸린시간초 }을 출력한다.
```

## 환경 변수 설정

`tch-rs`가 `.venv`의 Python PyTorch를 사용하도록 아래 설정을 권장한다.

```bash
source .venv/bin/activate
export LIBTORCH_USE_PYTORCH=1
export LD_LIBRARY_PATH="$(python -c 'import torch, pathlib; print(pathlib.Path(torch.__file__).parent / "lib")'):$LD_LIBRARY_PATH"
```

그 다음 실행:

```bash
cargo run -- samples/15_minst_torch.zm
```

버전 검사 오류가 나면(예: `tch`와 Python `torch`의 minor 버전 차이) 임시로 다음을 추가할 수 있다.

```bash
export LIBTORCH_BYPASS_VERSION_CHECK=1
```

## 현재 제약

- 자동 다운로드에는 네트워크 연결이 필요하다.
- 웹 타깃(`wasm32`)에서는 PyTorch 래퍼를 사용할 수 없다.
