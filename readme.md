# ROM Ultimate: Offline/Online 모드 선택형 실행 아키텍처 제안

데이터가 주어졌을 때,
1) 모드 구성 방식(POD 등) 선택,
2) Offline 학습 방식(RBF / NN / Projection-based) 선택,
3) Online 실행,
을 일관되게 연결하기 위한 **확장 가능한 코드 구조** 예시입니다.

---

## 1. 핵심 설계 원칙

- **전략 분리(Strategy Pattern)**: `mode_builder`, `offline_trainer`, `online_runner`를 인터페이스로 분리.
- **실험 재현성**: 모든 설정은 `configs/*.yaml`로 관리.
- **파이프라인 일원화**: `run_pipeline.py` 하나에서 `build -> train -> deploy -> infer` 실행.
- **플러그인 확장**: 새 기법 추가 시 기존 코드 최소 수정(등록만 추가).

---

## 2. 추천 디렉토리 구조

```text
rom_ultimate/
├─ pyproject.toml
├─ readme.md
├─ configs/
│  ├─ dataset/
│  │  └─ default.yaml
│  ├─ mode/
│  │  ├─ pod.yaml
│  │  └─ dmd.yaml
│  ├─ offline/
│  │  ├─ rbf.yaml
│  │  ├─ nn.yaml
│  │  └─ projection.yaml
│  └─ experiment/
│     └─ baseline.yaml
├─ data/
│  ├─ raw/
│  ├─ processed/
│  └─ splits/
├─ artifacts/
│  ├─ modes/
│  ├─ models/
│  ├─ scalers/
│  └─ reports/
├─ src/
│  └─ rom/
│     ├─ __init__.py
│     ├─ interfaces/
│     │  ├─ mode_builder.py
│     │  ├─ offline_trainer.py
│     │  └─ online_runner.py
│     ├─ registry/
│     │  ├─ mode_registry.py
│     │  ├─ trainer_registry.py
│     │  └─ runner_registry.py
│     ├─ modes/
│     │  ├─ pod_builder.py
│     │  └─ dmd_builder.py
│     ├─ trainers/
│     │  ├─ rbf_trainer.py
│     │  ├─ nn_trainer.py
│     │  └─ projection_trainer.py
│     ├─ runners/
│     │  ├─ static_runner.py
│     │  └─ streaming_runner.py
│     ├─ data/
│     │  ├─ dataset.py
│     │  ├─ preprocess.py
│     │  └─ split.py
│     ├─ core/
│     │  ├─ config.py
│     │  ├─ pipeline.py
│     │  ├─ io.py
│     │  └─ metrics.py
│     └─ cli/
│        ├─ run_pipeline.py
│        ├─ train_offline.py
│        └─ run_online.py
└─ tests/
   ├─ test_registry.py
   ├─ test_pipeline_smoke.py
   └─ test_online_runner.py
```

---

## 3. 각 계층의 책임

### A. Mode Builder (`modes/*`)
- 입력: 고차원 스냅샷/시계열 데이터
- 출력: reduced basis(모드 행렬), 모드 메타데이터
- 예: POD면 SVD 기반으로 rank/truncation 정책 적용

### B. Offline Trainer (`trainers/*`)
- 입력: 모드 좌표(latent coefficient), 조건 변수(파라미터)
- 출력: 매핑 모델(예: `u -> a` 혹은 `param -> latent`)
- 선택지:
  - **RBF**: 작은~중간 데이터, 빠른 구축
  - **NN**: 비선형 복잡도 높고 데이터 많은 경우
  - **Projection-based**: 물리/연산 구조를 보존하고 싶을 때

### C. Online Runner (`runners/*`)
- 입력: 온라인 입력(새 파라미터/초기조건/센서)
- 동작: offline 산출물 로드 → 추론/시간적 업데이트
- 출력: 재구성된 상태, latency/오차 로그

---

## 4. 인터페이스 예시 (최소 스켈레톤)

```python
# src/rom/interfaces/mode_builder.py
from abc import ABC, abstractmethod

class ModeBuilder(ABC):
    @abstractmethod
    def fit(self, snapshots):
        ...

    @abstractmethod
    def transform(self, snapshots):
        ...

    @abstractmethod
    def save(self, path):
        ...
```

```python
# src/rom/interfaces/offline_trainer.py
from abc import ABC, abstractmethod

class OfflineTrainer(ABC):
    @abstractmethod
    def fit(self, x_train, y_train):
        ...

    @abstractmethod
    def predict(self, x):
        ...

    @abstractmethod
    def save(self, path):
        ...
```

```python
# src/rom/interfaces/online_runner.py
from abc import ABC, abstractmethod

class OnlineRunner(ABC):
    @abstractmethod
    def load_artifacts(self, mode_path, model_path):
        ...

    @abstractmethod
    def step(self, online_input):
        ...
```

---

## 5. Registry 기반 선택 로직

설정 파일에서 문자열만 바꾸면 구현체가 선택되도록 만듭니다.

```python
# src/rom/registry/mode_registry.py
REGISTRY = {
    "pod": "rom.modes.pod_builder:PODBuilder",
    "dmd": "rom.modes.dmd_builder:DMDBuilder",
}
```

`trainer_registry.py`도 동일하게:
- `rbf` -> `RBFTrainer`
- `nn` -> `NNTrainer`
- `projection` -> `ProjectionTrainer`

이 구조를 쓰면, 새 기법 추가 시:
1) 구현 파일 추가,
2) 레지스트리 1줄 등록,
3) config 이름만 변경
으로 끝납니다.

---

## 6. 파이프라인 오케스트레이션

```python
# src/rom/core/pipeline.py (개념)
def run_pipeline(cfg):
    # 1) 데이터 로드/전처리
    data = load_dataset(cfg.dataset)

    # 2) 모드 생성
    mode_builder = build_mode(cfg.mode.name, cfg.mode)
    mode_builder.fit(data.snapshots)
    latent = mode_builder.transform(data.snapshots)

    # 3) offline 학습
    trainer = build_trainer(cfg.offline.name, cfg.offline)
    trainer.fit(data.params, latent)

    # 4) 산출물 저장
    mode_builder.save(cfg.paths.mode_artifact)
    trainer.save(cfg.paths.model_artifact)

    # 5) 평가/리포트
    evaluate_and_report(...)
```

---

## 7. 설정 파일 예시

```yaml
# configs/experiment/baseline.yaml
dataset:
  name: default
mode:
  name: pod
  rank: 32
offline:
  name: rbf
  kernel: gaussian
online:
  runner: static
paths:
  mode_artifact: artifacts/modes/pod_rank32.pkl
  model_artifact: artifacts/models/rbf_gaussian.pkl
```

핵심은 `mode.name`, `offline.name`, `online.runner` 세 값만 바꿔 실험 축을 구성하는 것입니다.

---

## 8. 추천 개발 순서

1. `interfaces/*` 먼저 정의
2. `PODBuilder + RBFTrainer + StaticRunner` 최소 조합 구현
3. `pipeline.py` 연결 후 smoke test 작성
4. `NNTrainer` 추가
5. `ProjectionTrainer` 추가
6. 실험 로그/메트릭 비교 자동화

---

## 9. 실무 팁

- **아티팩트 버전 관리**: 파일명에 데이터 버전/하이퍼파라미터 포함.
- **수치 안정성**: POD truncation 기준(에너지 누적률 등)을 config에 명시.
- **평가 분리**: offline 정확도와 online 안정성(rollout error) 별도 추적.
- **실시간 요구 대응**: OnlineRunner는 모델 로드 비용 최소화(초기 1회 로드).

---

## 10. 한 줄 결론

`ModeBuilder / OfflineTrainer / OnlineRunner`를 독립 인터페이스 + registry + config 기반으로 설계하면,
POD/RBF에서 시작해 NN/Projection 방식으로 자연스럽게 확장 가능한 ROM 파이프라인을 만들 수 있습니다.
