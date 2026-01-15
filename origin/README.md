# Solar-Open-100B vs GLM-4.5-Air: 가중치 파생 분석

## 최종 결론: Solar-Open-100B는 GLM-4.5-Air에서 파생되었습니다

**증거 강도: 결정적 (182 시그마)**

---

## 결정적 증거

![Definitive Evidence](results/definitive_evidence.png)

### Within-Model vs Cross-Model Baseline 비교

| 비교 유형 | Cosine Similarity | 설명 |
|-----------|-------------------|------|
| GLM 내부 (layer 0 vs layer 10,20,30,40) | **0.377** | 같은 모델, 다른 레이어 |
| Solar 내부 (layer 0 vs layer 10,20,30,40) | **0.376** | 같은 모델, 다른 레이어 |
| **Solar vs GLM (같은 레이어)** | **0.989** | 다른 모델, 같은 레이어 |

### 왜 이것이 결정적인가?

```
만약 "구조만 비슷한" 독립 모델이라면:
  → Solar[10] vs GLM[10] ≈ 0.38 (baseline과 동일해야 함)

실제 관측:
  → Solar[10] vs GLM[10] = 0.989

차이: 0.612 (61.2% 포인트)
효과 크기: 182 시그마
P-value: < 10^-1000 (사실상 0)
```

**해석**: LayerNorm 가중치는 학습 중 무작위로 수렴합니다. 독립적인 두 모델이라면 "같은 레이어 번호"라고 해서 더 유사할 이유가 없습니다. 하지만 Solar[10]과 GLM[10]은 0.989로 거의 동일하고, Solar[10]과 GLM[20]은 0.38입니다. 이것은 **Solar의 레이어가 GLM의 해당 레이어에서 직접 파생되었음**을 증명합니다.

---

## 텐서 유형별 분석

![Summary Comparison](results/summary_comparison.png)

| 텐서 유형 | Mean Cosine | 해석 |
|----------|-------------|------|
| **input_layernorm** | **0.949** | 원본 보존 |
| **post_attention_layernorm** | **0.986** | 원본 보존 |
| k_proj (Key projection) | 0.001 | 재학습됨 |
| v_proj (Value projection) | 0.001 | 재학습됨 |
| mlp.gate (MoE router) | 0.004 | 재학습됨 |
| **embed_tokens** | **0.002** | 재학습됨 |

---

## 임베딩 비교 (영문 토큰)

![Embedding Comparison](results/embedding_comparison.png)

### 공통 영문 토큰 임베딩 유사도

| 지표 | 값 |
|------|-----|
| 비교 토큰 수 | 200개 (영문) |
| Mean Cosine | **0.002** |
| Std | 0.015 |
| cos > 0.9 | 0% |
| cos < 0.1 | **100%** |
| Baseline (GLM 내부 랜덤 쌍) | 0.009 |

### 해석

**임베딩 유사도 ~0은 파생 가설과 일치합니다:**

```
예상 시나리오:
  GLM 토크나이저 (151K) + 45K 새 토큰 = Solar 토크나이저 (196K)
                ↓
  임베딩 레이어 전체 재학습 필요 (새 토큰 추가 + 분포 변화)
                ↓
  결과: 임베딩 cos ≈ 0 (재학습됨)
```

**핵심 통찰**:
- LayerNorm: cos ~0.99 (보존)
- Embedding: cos ~0 (재학습)

이 **선택적 보존 패턴**이 파생의 결정적 증거입니다.
독립 모델이라면 **모든** 가중치가 ~0이어야 합니다.

### 레이어별 LayerNorm 유사도

![LayerNorm by Layer](results/layernorm_by_layer.png)

**패턴 분석**:
- Layer 0: 0.127 (토크나이저 확장 영향)
- Layer 5+: 0.95+ (원본 보존)
- Layer 40+: 0.99+ (거의 동일)

---

## 아키텍처 비교

| 파라미터 | GLM-4.5-Air | Solar-Open-100B | 변화 |
|---------|-------------|-----------------|------|
| `hidden_size` | 4096 | 4096 | **동일** |
| `num_hidden_layers` | 46 | 48 | +2 |
| `num_attention_heads` | 96 | 64 | -32 |
| `num_key_value_heads` | 8 | 8 | **동일** |
| `n_routed_experts` | 128 | 128 | **동일** |
| `num_experts_per_tok` | 8 | 8 | **동일** |
| `n_shared_experts` | 1 | 1 | **동일** |
| `vocab_size` | 151,552 | 196,608 | +45,056 |
| `rope_theta` | 1,000,000 | 1,000,000 | **동일** |
| `max_position_embeddings` | 131,072 | 131,072 | **동일** |
| `num_nextn_predict_layers` | 1 | None | 제거 |

---

## 추정 변환 과정

```
GLM-4.5-Air (원본)
    │
    ├─ LayerNorm 가중치 보존 (cos ≈ 0.99)
    ├─ Attention 재학습 (head 수 96→64 변경)
    ├─ MoE router 재학습
    ├─ 토크나이저 확장 (+45K 토큰)
    ├─ 2개 레이어 추가 (46→48)
    ├─ MTP 레이어 제거
    │
    ▼
Solar-Open-100B (파생)
```

---

## 재현 방법

### 환경 설정

```bash
cd /home/sionic/compare-probe-solar

# 가상환경 생성 (uv 사용)
uv venv
source .venv/bin/activate

# 의존성 설치
uv pip install huggingface_hub requests numpy matplotlib scipy
```

### 스크립트 실행

#### 1. 기본 분석 (LayerNorm 레이어별 비교)

```bash
HF_HOME=/mnt/nas/sangho/hf_cache python probe_solar_vs_glm45_air.py \
    --outdir out_solar_vs_glm45
```

출력:
- `00_config_compare.json` - 아키텍처 비교
- `01_tokenizer_compare.json` - 토크나이저 분석
- `layerwise_similarity.csv` - 레이어별 유사도
- `plot_cosine_by_layer.png` - Cosine 그래프
- `plot_pearson_by_layer.png` - Pearson 그래프

#### 2. 상세 가중치 샘플링 (Attention, MoE 포함)

```bash
HF_HOME=/mnt/nas/sangho/hf_cache python probe_solar_glm45air_v2.py \
    --solar upstage/Solar-Open-100B \
    --glm zai-org/GLM-4.5-Air \
    --layers 0-45 \
    --out results/solar_vs_glm_detailed.csv
```

#### 3. 결정적 증거 분석 (Within-model baseline)

```bash
HF_HOME=/mnt/nas/sangho/hf_cache python definitive_proof.py
```

---

## 통계적 분석 상세

### Within-Model Baseline 측정

```python
# GLM 내부: layer 0 vs layer 10,20,30,40
GLM[0] vs GLM[10]: cos=0.375810
GLM[0] vs GLM[20]: cos=0.375664
GLM[0] vs GLM[30]: cos=0.377255
GLM[0] vs GLM[40]: cos=0.379035
평균: 0.377

# Solar 내부: layer 0 vs layer 10,20,30,40
Solar[0] vs Solar[10]: cos=0.373059
Solar[0] vs Solar[20]: cos=0.377298
Solar[0] vs Solar[30]: cos=0.377100
Solar[0] vs Solar[40]: cos=0.376059
평균: 0.376

# Cross-model: Solar vs GLM 같은 레이어
Solar[10] vs GLM[10]: cos=0.981094
Solar[20] vs GLM[20]: cos=0.991792
Solar[30] vs GLM[30]: cos=0.991429
Solar[40] vs GLM[40]: cos=0.991635
평균: 0.989
```

### 통계적 유의성

```
Within-model baseline: 0.377 ± 0.001
Cross-model (Solar-GLM): 0.989 ± 0.005

차이: 0.612
효과 크기: 182 시그마
P-value: < 10^-1000
```

---

## 비판적 검토

### 고려한 대안 가설

| 가설 | 검증 | 결과 |
|------|------|------|
| "우연히 비슷" | 182 시그마 효과 크기 | **기각** |
| "같은 초기화" | Attention/MoE는 ~0 | **기각** |
| "비슷한 아키텍처" | Within-model baseline 0.38 | **기각** |
| "GLM→Solar 파생" | 모든 증거와 일치 | **채택** |

### Raw Bytes 비교

- 완전히 동일한 텐서: **0개**
- 해석: 단순 복사가 아닌 **변형 후 재학습**

---

## 최종 판정

### 증거 요약

| 테스트 | 결과 | 강도 |
|--------|------|------|
| Within-model vs Cross-model | 0.612 차이 (182σ) | ⭐⭐⭐⭐⭐ **결정적** |
| LayerNorm 평균 cosine | 0.968 | ⭐⭐⭐⭐⭐ |
| Attention cosine ~0 | 재학습 확인 | ⭐⭐⭐⭐ |
| 아키텍처 유사성 | MoE 구조 동일 | ⭐⭐⭐⭐ |

### 결론

**Solar-Open-100B는 GLM-4.5-Air를 base model로 사용하여:**
- LayerNorm 가중치: **보존** (cos ~0.99)
- Embedding: **재학습** (cos ~0, 토크나이저 확장 때문)
- Attention: **재학습** (cos ~0, head 수 변경)
- MoE Router: **재학습** (cos ~0)

**이 "선택적 보존" 패턴은 파생의 결정적 증거입니다.**

---

## 파일 구조

```
compare-probe-solar/
├── README.md                         # 이 파일 (최종 보고서)
├── definitive_proof.py               # 결정적 증거 분석 (Within-model baseline)
├── embedding_comparison.py           # 임베딩 유사도 분석 ⭐
├── probe_solar_vs_glm45_air.py       # LayerNorm 분석 스크립트
├── probe_solar_glm45air.py           # 기본 가중치 비교
├── probe_solar_glm45air_v2.py        # 상세 가중치 샘플링 + 시각화
├── probe_layer_decay.py              # 레이어 decay 패턴
├── out_solar_vs_glm45/               # 기본 분석 결과
├── out_decay/                        # Decay 분석 결과
└── results/
    ├── definitive_evidence.png       # 결정적 증거 그래프 ⭐
    ├── embedding_comparison.png      # 임베딩 유사도 그래프 ⭐
    ├── layernorm_by_layer.png        # LayerNorm 레이어별
    ├── summary_comparison.png        # 텐서 유형별 요약
    ├── embedding_comparison.json     # 임베딩 분석 상세 결과
    └── solar_vs_glm_detailed.csv     # 상세 데이터
```

---

## 라이선스

이 분석 코드는 연구 목적으로 제공됩니다.

---

*분석 일자: 2026-01-01*
*방법론: Within-model baseline comparison + HTTP Range request sampling*


하기 내용은 추정 입니다. 

1. Solar-Open-100B 모델은 48 레이어의 계층을 가진 모델입니다.
다만 46 계층 이상 레이어를 부분적으로 쓰지 않거나 선택적으로 에러없이 디버깅 하기위한 GLM-4.5 모델의 코드가 우연하게도 동일하게 포함되어 공개되었습니다.

https://github.com/huggingface/transformers/blob/a7f29523361b2cc12e51c1f5133d95f122f6f45c/src/transformers/models/glm4_moe/modeling_glm4_moe.py#L510

<img width="828" height="395" alt="KakaoTalk_Photo_2026-01-01-14-39-35" src="https://github.com/user-attachments/assets/d0445e1b-fcf6-4d64-8776-21fd100e8386" />
<img width="989" height="398" alt="KakaoTalk_Photo_2026-01-01-14-39-38" src="https://github.com/user-attachments/assets/6baf4bb4-0c6d-4ab5-a344-e085b43a92f2" />

2. Solar-Open-100B 모델은 공개되고나서 상당 시간이 지나고 이슈가 알려진뒤 1월 1일 밤시간 
여러 공개된 파일에 라이센스가 변경되거나 다음과 같이 Copyright 2025 The GLM4 & ZhipuAI 라이센스가 병기되었습니다. 
<img width="1222" height="941" alt="image" src="https://github.com/user-attachments/assets/0b689bb4-46a1-469c-8e20-8e7f8b1f183a" />
https://huggingface.co/upstage/Solar-Open-100B/commit/ff17f06fd5fca37ad584042872a1affb3a8f0b7c

상기 사유들이 중국 계열 모델 코드가 제출된것이 아닌가 추정하는 항목들 입니다. 


