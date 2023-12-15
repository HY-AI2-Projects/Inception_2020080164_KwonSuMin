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
<p align = center><img src = ![image](https://github.com/HY-AI2-Projects/Inception_2020080164_KwonSuMin/assets/146939941/9f21e8f5-8a6a-4b11-91eb-fef114b62663)
 width="50%" height="50%" title="이미지 조절"></p>
![image](https://github.com/HY-AI2-Projects/Inception_2020080164_KwonSuMin/assets/146939941/fd6d99a9-20c7-472e-97b9-352bbfabefc0)
