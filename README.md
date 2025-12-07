# 🎬 Synopsify - 영화 장르 분류 프로젝트

영화 **줄거리(plot)** 를 입력하면, 해당 영화의 **장르(genre)** 를 예측하는  
**분류 모델** 비교 프로젝트

- **프레임워크**: PyTorch, Transformers
- **모델**: LSTM, BERT-base
- **장르 수**: 27개 클래스

## 📋 프로젝트 개요

이 프로젝트는 영화 줄거리 텍스트를 입력받아 27개 장르 중 하나로 분류하는 **Single-label 분류** 모델입니다.

- **입력**: 영화 줄거리 텍스트 (영어)
- **출력**: 27개 장르 중 1개 예측
- **장르 종류**: action, comedy, drama, thriller, horror, sci-fi, romance 등


## 📊 데이터셋

### 데이터 출처
- **Kaggle**: [Genre Classification Dataset (IMDB)](https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb)

### 데이터 구성
- **Train 데이터**: 54,214개 샘플
- **Test 데이터**: 54,200개 샘플
- **장르 수**: 27개 클래스
- **데이터 형식**: `id ::: title ::: genre ::: plot`

### 데이터 특징
- 장르 불균형 존재 (drama, comedy 등이 많음)
- 텍스트 길이 다양 (최대 256 토큰으로 제한)

### 데이터 전처리
1. **텍스트 정제**: 소문자 변환, 구두점 제거, 공백 정리
2. **토큰화**: 문장을 단어 단위로 분리
3. **Vocab 생성**: 자주 쓰인 상위 30,000개 단어만 사용 (LSTM)
4. **길이 통일**: 모든 문장을 256 토큰으로 맞춤 (부족하면 padding, 넘치면 truncation)
5. **데이터 분할**: Train 데이터를 8:2 비율로 train/validation 분할

## 🏗️ 모델 구조

### 1. LSTM 모델

#### 구조
```
Input → Embedding → Bidirectional LSTM → Dropout → FC Layer → Output
```

#### 주요 특징
- **Embedding**: 단어를 128차원 벡터로 변환
- **Bidirectional LSTM**: 앞→뒤, 뒤→앞 양방향으로 문맥 이해
  - NUM_LAYERS: 1
  - HIDDEN_DIM: 128
- **Dropout**: 0.4 (과적합 방지)
- **FC Layer**: 27개 장르 분류

#### 하이퍼파라미터
| 항목 | LSTM | LSTM CW |
|------|-----------|----------|
| Vocab Size | 30,000 | 30,000 |
| Embedding Dim | 128 | 128 |
| Hidden Dim | 128 | 128 |
| NUM_LAYERS | 1 | 1 |
| Bidirectional | ✅ | ✅ |
| Dropout | 0.4 | 0.4 |
| Max Seq Length | 256 | 256 |
| Optimizer | Adam | Adam |
| Learning Rate | 5e-4 | 5e-4 |
| Batch Size | 64 | 64 |
| Epochs | 10 | 10 |
| Class Weight | ❌ | ✅ |

**LSTM CW**: Class Weight를 적용하여 드문 장르에 더 높은 가중치 부여

---

### 2. BERT 모델

#### 구조
```
Input → Embedding (Token + Position) → Transformer (12 layers) → Classification Head → Output
```

#### 주요 특징
- **Pretrained BERT-base**: 이미 학습된 언어 이해 모델 사용
  - NUM_LAYERS: 12
  - HIDDEN_DIM: 768
- **Classification Head**: 27개 장르 분류용 레이어 추가
- **Fine-tuning**: 영화 데이터로 추가 학습

#### 하이퍼파라미터
| 항목 | 값 |
|------|-----|
| Model | bert-base-uncased |
| Transformer Layers | 12 (사전 설정) |
| Hidden Size | 768 (사전 설정) |
| Max Length | 256 |
| Optimizer | AdamW |
| Learning Rate | 2e-5 |
| Batch Size | 16 |
| Epochs | 3 |
| Warmup Ratio | 0.1 |

## 📈 실험 결과

### 성능 비교

| 모델 | Accuracy | F1-macro | F1-micro | Loss |
|------|----------|----------|----------|------|
| **LSTM** | 0.5376 | 0.1847 | 0.5376 | 1.6593 |
| **LSTM CW** | 0.4856 | 0.2646 | 0.4856 | 2.2267 |
| **BERT** | **0.6947** | **0.4838** | **0.6947** | **1.0699** |

### 주요 발견사항

1. **BERT가 모든 지표에서 최고 성능**
   - Accuracy: +15.7%p (vs LSTM High)
   - F1-macro: +29.9%p (vs LSTM High)

2. **LSTM CW의 Class Weight 효과**
   - F1-macro는 향상 (0.1847 → 0.2646)
   - Accuracy는 하락 (0.5376 → 0.4856)

3. **Pretrained 모델의 효과 확인**
   - BERT의 사전 학습된 언어 이해 능력이 장르 분류에 유리
   - 적은 Epochs(3)로도 좋은 성능 달성

## 🚀 사용 방법

### 1. 데이터 준비

1. Kaggle에서 데이터셋 다운로드
   - [Genre Classification Dataset (IMDB)](https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb)

2. `data/` 폴더에 파일 복사
   ```
   data/
   ├── train_data.txt
   ├── test_data.txt
   └── test_data_solution.txt
   ```

### 2. 모델 학습

#### LSTM 모델 학습
```bash
# LSTM (Class Weight 없음)
jupyter notebook lstm_train.ipynb

# LSTM CW(Class Weight 적용)
jupyter notebook lstm_CW_train.ipynb
```

#### BERT 모델 학습
```bash
jupyter notebook bert_train.ipynb
```

### 3. 예측

학습된 모델로 장르 예측:
```bash
jupyter notebook predict.ipynb
```

**사용 예시**:
```python
# LSTM 모델 사용
예측 장르: comedy (confidence=0.118)

# BERT 모델 사용
예측 장르: fantasy (confidence=0.331)
```

## 📁 프로젝트 구조

```
Synopsify/
├── data/                      # 데이터 파일
│   ├── train_data.txt
│   ├── test_data.txt
│   └── test_data_solution.txt
├── model/                     # 학습된 모델
│   ├── lstm.pt
│   ├── lstm_CW.pt
│   └── bert/
├── lstm_train.ipynb          # LSTM 학습 코드
├── lstm_CW_train.ipynb       # LSTM CW 학습 코드
├── bert_train.ipynb          # BERT 학습 코드
├── predict.ipynb             # 예측 코드
├── README.md
└── .gitignore
```


## 🔧 환경 설정

### 필수 라이브러리
```
python
torch
transformers
pandas
numpy
scikit-learn
tqdm
```

### 설치 방법
```bash
pip install torch transformers pandas numpy scikit-learn tqdm
```

## 📝 주요 실험 변경사항

1. **Loss Function**: CrossEntropyLoss → Weighted CrossEntropyLoss (LSTM CW)
2. **Pretrained Model**: None → BERT-base (BERT)
3. **Optimizer**: Adam → AdamW (BERT)
4. **Scheduler**: None → Linear Warmup (BERT)

## 🎯 결론

### 주요 성과
- ✅ **BERT-base 모델이 최고 성능 달성** (Accuracy: 69.47%)
- ✅ **Pretrained 모델의 효과 확인**
- ✅ **Class Weight 적용으로 F1-macro 개선** 

### 한계점 및 개선 방향
- F1-macro가 여전히 낮음 (0.48) → 장르 불균형 문제
- 27개 클래스 분류는 어려운 태스크
- **향후 개선**: 더 큰 BERT 모델, Data Augmentation, Ensemble