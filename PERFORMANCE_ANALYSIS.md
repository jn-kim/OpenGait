# DenoisingGait CCPG 성능 재현 분석

## 현재 성능 vs 논문 성능

| 평가 조건 | 논문 수치 | 실제 재현 결과 | 편차 |
|---------|---------|-------------|------|
| CL (Full Cloth-changing) | 91.8% | 82.72% | **-9.08%** |
| UP (Up-changing) | 95.8% | 88.04% | **-7.76%** |
| DN (Pant-changing) | 96.4% | 89.69% | **-6.71%** |
| BG (Bag-changing) | 98.7% | 92.49% | **-6.21%** |
| Mean (전체 평균) | 95.7% | 88.24% | **-7.46%** |

## 현재 설정 요약

### 모델 설정
- `diffusion_ckpt`: `./pretrained_LVMs/segmind/tiny-sd`
- `r`: 3
- `p`: 0.5
- `threshold`: 0.5
- `in_channel`: 8 (diffusion feature + silhouette)

### 학습 설정
- `total_iter`: 60000
- `lr`: 0.1
- `milestones`: [20000, 40000, 50000]
- `batch_size`: [8, 4] (ID, sequences)
- `frames_num_fixed`: 16
- `frames_skip_num`: 30
- `sample_type`: `fixed_allordered`

### 평가 설정
- `batch_size`: 4
- `sample_type`: `all_ordered`
- `frames_all_limit`: 720

## 가능한 원인 분석

### 1. 학습 Iteration 부족
- 현재: 60000 iterations
- 가능성: 논문에서 더 많은 iteration 사용했을 수 있음
- 확인 필요: 논문의 정확한 iteration 수

### 2. Frame Sampling 방법
- 현재: `frames_num_fixed: 16`, `frames_skip_num: 30`
- 가능성: 논문에서 다른 frame sampling 전략 사용
- 확인 필요: 논문의 frame sampling 설정

### 3. Diffusion Model Checkpoint
- 현재: `segmind/tiny-sd`
- 가능성: 논문에서 다른 checkpoint 사용했을 수 있음
- 확인 필요: 논문에서 사용한 정확한 checkpoint

### 4. 학습률 스케줄
- 현재: milestones [20000, 40000, 50000]
- 가능성: 다른 스케줄 사용했을 수 있음
- 확인 필요: 논문의 정확한 learning rate schedule

### 5. 데이터 전처리 차이
- 현재: timestep=700으로 diffusion feature 추출
- 가능성: 논문에서 다른 timestep 사용했을 수 있음
- 확인 필요: 논문의 전처리 설정

### 6. 하이퍼파라미터 차이
- `p`: 0.5 (data augmentation probability)
- `threshold`: 0.5
- `r`: 3 (flow radius)
- 확인 필요: 논문의 정확한 하이퍼파라미터 값

## 개선 방안

### 즉시 확인 가능한 사항
1. **논문/공식 코드 확인**: 정확한 설정 값 확인
2. **학습 로그 분석**: loss 수렴 여부 확인
3. **체크포인트 확인**: 최종 iteration의 loss 값 확인

### 실험적 개선 방안
1. **더 많은 iteration 학습**: 80000 또는 100000 iterations
2. **Frame 수 증가**: `frames_num_fixed`를 30으로 증가
3. **학습률 조정**: 초기 learning rate를 0.05로 낮춤
4. **Batch size 증가**: 더 큰 batch size 사용 (메모리 허용 시)

## 다음 단계

1. 논문/공식 코드에서 정확한 설정 값 확인
2. 학습 로그에서 loss 수렴 여부 확인
3. 필요시 추가 학습 또는 하이퍼파라미터 튜닝

