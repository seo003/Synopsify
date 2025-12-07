# 🎬 Synopsify

영화 **줄거리(plot)** 를 입력하면, 해당 영화의 **장르(genre)** 를 예측하는  
멀티라벨(Multi-label) 분류 모델

- 데이터셋: [Kaggle - Genre Classification Dataset (IMDB)](https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb)
- 프레임워크: **PyTorch**
- 실행 환경: **Google Colab**


## 1. 프로젝트 개요

이 프로젝트의 목표는:

1. 영화의 **제목(title)** 과 **줄거리(plot)** 를 입력으로 받아
2. 영화가 속한 **하나 이상의 장르**(예: `Drama`, `Comedy`, `Crime`, ...)를
3. **멀티라벨 분류(Multi-Label Classification)** 형태로 예측하는 것입니다.


## 2. 데이터셋 설명

Kaggle에서 제공하는 압축 파일을 풀면 보통 다음과 같은 파일들이 포함됩니다.

- `train_data.txt`
- `test_data.txt`
- `test_data_solution.txt` (레이블이 포함된 정답 파일)

이 중 **학습에 사용하는 파일은 `train_data.txt`**이며, 각 줄은 아래와 같은 형식입니다.

```text
id ::: title ::: genre ::: plot
```

또는 버전에 따라:

```text
title ::: genre ::: plot
```

- `title`: 영화 제목 (예: `Oscar et la dame rose (2009)`)
- `genre`: 장르 (예: `Drama`, 또는 `Drama, Romance` 처럼 여러 개)
- `plot`: 영화 줄거리 텍스트

코드에서는 다음과 같이 전처리합니다.

- `title` 뒤에 붙은 `(2009)` 같은 연도 표기 제거
- `text = title + ". " + plot` 형태로 합쳐서 **모델 입력**으로 사용
- `genre` 컬럼을 `,` 또는 `|` 로 split 해서 **장르 리스트**로 사용  
  예: `"Drama, Romance"` → `["drama", "romance"]`

---

## 3. 모델 구조

PyTorch로 구현한 간단한 텍스트 분류기입니다.

1. **토큰화 & 정수 인코딩**
   - 전체 텍스트를 소문자로 변환, 특수문자 제거
   - 공백 기준 토큰화
   - 등장 빈도 상위 `MAX_VOCAB_SIZE`개의 단어만 사용 (기본 20,000)
   - `<pad> = 0`, `<unk> = 1` 인덱스 사용

2. **입력 시퀀스 구성**
   - 각 샘플을 정수 시퀀스로 변환
   - 길이는 `MAX_SEQ_LEN` (기본 256)으로 **pad 또는 truncate**

3. **모델 아키텍처**

```text
[입력 토큰 ID 시퀀스] → [임베딩] → [패딩 제외 평균] → [Linear + ReLU + Dropout] → [Linear] → [장르별 로짓]
```

- `nn.Embedding(vocab_size, EMBED_DIM)`
- 토큰 임베딩의 **마스크드 평균(pooling)** 사용
- `nn.Linear(EMBED_DIM, HIDDEN_DIM)` + `ReLU` + `Dropout(0.3)`
- `nn.Linear(HIDDEN_DIM, NUM_LABELS)`
- 손실 함수: `BCEWithLogitsLoss` (multi-label용)

4. **출력**
   - 각 장르에 대해 **0~1 사이의 확률(sigmoid)** 로 해석
   - 기준 임계값(threshold, 기본 0.5 또는 0.3) 이상이면 해당 장르로 예측

---

## 4. Colab에서 실행 방법

### 4.1. 데이터 준비

1. Kaggle에서 데이터셋 다운로드  
   (예: `genre-classification-dataset-imdb.zip`)

2. Colab 환경에 업로드 방법 (둘 중 하나 선택)

   - **방법 A: 수동 업로드**
     - Colab 왼쪽 사이드바 → 폴더 아이콘 → `Upload` 버튼 누르고 zip 파일 업로드
     - 업로드 후 아래 명령으로 압축 해제:
       ```bash
       !unzip "/content/genre-classification-dataset-imdb.zip" -d "/content/"
       ```
       압축을 풀면 `/content/Genre Classification Dataset` 폴더가 생겼다고 가정.

   - **방법 B: 구글 드라이브 마운트 후 사용**
     - 드라이브에 zip 업로드 → Colab에서 마운트 → 동일하게 `unzip` 후 경로만 수정

3. 코드에서 `DATA_DIR` 경로 설정

```python
DATA_DIR = "/content/Genre Classification Dataset"
TRAIN_FILE = os.path.join(DATA_DIR, "train_data.txt")
```

데이터 폴더 이름이 다르다면 여기만 실제 이름으로 수정하면 됩니다.

---

### 4.2. Colab 코드 구조

README와 함께 제공된 Colab 코드는 다음 순서의 셀로 구성되어 있습니다.

1. **라이브러리 임포트 & 디바이스 설정**
2. **데이터 경로 & 하이퍼파라미터 설정**
3. **데이터 로드 & 기본 전처리**
   - `train_data.txt` 읽기, 컬럼 이름 정리
4. **장르 전처리 (멀티라벨)**
   - 장르 문자열 → 리스트 → multi-hot 벡터
5. **텍스트 정제 & 토큰화, Vocab 생성**
6. **Dataset / DataLoader 정의**
7. **PyTorch 모델 정의**
8. **학습 루프 및 검증(F1 점수)**
9. **임의 문장에 대해 장르 예측 함수**

각 셀을 **위에서부터 순서대로 실행**하면 학습이 진행됩니다.

---

## 5. 주요 하이퍼파라미터

코드 상단에서 쉽게 수정할 수 있는 주요 값들:

```python
MAX_VOCAB_SIZE = 20000   # 사용할 최대 단어 수
MAX_SEQ_LEN = 256        # 입력 문장의 최대 토큰 길이
BATCH_SIZE = 128
EMBED_DIM = 128
HIDDEN_DIM = 128
NUM_EPOCHS = 5
LR = 1e-3
```

- GPU 성능이 부족하면 `BATCH_SIZE`를 줄이고,
- 문장이 길면 `MAX_SEQ_LEN`을 늘려볼 수 있습니다.
- 학습이 덜 되었다고 느껴지면 `NUM_EPOCHS`를 늘리세요.

---

## 6. 학습 & 평가

학습 셀을 실행하면 매 epoch마다 다음과 같이 출력됩니다.

```text
[Epoch 1] Train loss: 0.25xx
Val loss: 0.21xx | F1(macro): 0.44xx
...
```

- `Train loss`: 학습 데이터에 대한 BCEWithLogitsLoss 평균
- `Val loss`: 검증 데이터에 대한 loss
- `F1(macro)`: 모든 장르에 대해 macro average F1 score  
  (각 장르의 F1을 평균낸 값, 데이터 불균형에 조금 더 공정)

---

## 7. 예측 사용 예시

학습이 끝난 후, 아래와 같이 함수를 호출하여 새 줄거리에 대해 장르를 예측할 수 있습니다.

```python
sample_plot = (
    "A young boy grows up in a small town and faces many challenges with his family and friends."
)
print("Input:", sample_plot)
print("Predicted genres:", predict_genres(sample_plot, top_k=5))
```

출력 예시:

```text
Predicted genres: [('drama', 0.81), ('family', 0.42)]
```

- `top_k`: 상위 몇 개 장르를 보고 싶은지
- `threshold`: 이 값보다 높은 확률만 결과에 포함

---

## 8. 구조 확장 아이디어

현재 모델은 **간단한 평균 임베딩 + MLP** 기반입니다. 더 좋은 성능을 위해:

1. **RNN 계열 적용**
   - `nn.LSTM`, `nn.GRU` 등으로 시퀀스 정보를 활용
2. **CNN 기반 텍스트 분류**
   - `nn.Conv1d` + Global Max Pooling 등 적용
3. **Transformer / BERT 기반 모델**
   - Hugging Face `transformers` 라이브러리 사용
   - `bert-base-uncased` 등으로 파인튜닝
4. **데이터 증강**
   - 문장 랜덤 삭제 / synonym replacement 등

이 README에서 설명한 구조는 **PyTorch 멀티라벨 텍스트 분류의 기본 뼈대**로,  
위 아이디어들을 얹어서 쉽게 확장할 수 있습니다.

---

## 9. 라이선스 & 출처

- 데이터셋 출처:  
  [Kaggle - Genre Classification Dataset (IMDB)](https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb)
- 이 코드는 교육 및 연구용으로 자유롭게 수정하여 사용할 수 있습니다.
