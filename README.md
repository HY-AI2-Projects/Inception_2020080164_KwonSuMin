# Inception_2020080164_KwonSuMin
* 한양대학교 산업융합학부 정보융합전공 2020080164 권수민  
* 본 게시물은 인공지능2 기말 프로젝트로, Inception 논문을 기반으로 작성된 문서입니다.  
* 논문 원본 : https://ffighting.net/deep-learning-paper-review/vision-model/inception/
---
##  서론
### 1. Inception 개념
2015년 ImageNet 콘테스트에서 우승한 모델로, 다양한 크기의 컨볼루션 필터를 병렬로 연결하고 다양한 스케일의 특징을 동시에 학습할 수 있는 능력을 갖춘 모델입니다. 이러한 구조는 모델이 더욱 유연하게 특징을 추출할 수 있게 해주며 이를 통해 성능을 크게 향상시킬 수 있습니다. 또한, 1X1 컨볼루션을 통해 차원을 축소하여 모델의 크기와 계산량을 효과적으로 줄일 수 있습니다.  
### 2. 기존 방식의 문제점
  * #### Overfitting(과적합)  
   네트워크가 깊어지면서 표현력이 좋아지고 이에 따라 모델이 학습데이터에서만 최적화하는 문제가 발생하였습니다.
  * #### Computation Cost(계산 복잡도)
   네트워크가 깊어지면서 학습해야할 파라미터 개수가 늘어나고 계산량이 많아지는 문제가 발생하였습니다.  
   
---
## 제안방법
### 1. 다양한 사이즈의 필터 사용
![image](https://github.com/HY-AI2-Projects/Inception_2020080164_KwonSuMin/assets/146939941/e6f560e9-0758-496d-9dfe-a7249af1e265)
  
1X1, 3X3, 5X5 Convolution을 사용하여 다양한 사이즈의 필터를 병렬로 사용합니다.  
기존 Convolution 연산방법과 비교하였을 때 파라미터 개수가 훨씬 줄어들었습니다.  

### 2. 1X1 Convolution 사용
![image](https://github.com/HY-AI2-Projects/Inception_2020080164_KwonSuMin/assets/146939941/b9032dbf-c0ac-4b80-8324-0e4fbc398b90)
  
이 그림은 Inception Module이라고 부릅니다. 이는 3X3, 5X5 Convolution 전에 1X1 Convolution 을 사용함으로써 채널 수와 학습할 파라미터 개수가 줄어들었습니다. 

### 3. Average Pooling 사용
![image](https://github.com/HY-AI2-Projects/Inception_2020080164_KwonSuMin/assets/146939941/096760d6-5d58-4ddb-b71b-0f2cf777c063)
  

네트워크의 마지막 부분에서 Fully Connected 연결 대신 Average Pooling을 사용합니다.  
이는 학습 파라미터 개수 감소 및 Overfiting을 방지할 수 있습니다.

### 4. Auxiliary Classifier 사용
저자들은  Auxiliary Classifier(보조분류기)를 사용을 추가 제안합니다. 네트워크가 깊어짐으로 생긴 Gradient Vanishing 문제는 해결되지 않습니다. 이를 해결하기 위해 네트워크 중간중간에 Classifier을 추가하여 학습 Loss를 흐르게 만들어주는 방법입니다.

### 5. 전체 구조
앞서 설명드린 제안방법을 반영한 전체구조는 다음과 같습니다.  
![image](https://github.com/HY-AI2-Projects/Inception_2020080164_KwonSuMin/assets/146939941/8d79f88e-7271-4e61-8a85-305cc238fa8b)

--- 
## 결론
