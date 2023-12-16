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
## 코드설명
```pythion
import torch
import torch.nn as nn
import torch.nn.functional as F
```
* torch 모듈을 Import 합니다.

```python
  #Module class 정의
class InceptionModule(nn.Module):
    def __init__(self, in_channels):
        super(InceptionModule, self).__init__()
        
        # 1x1 Convolution
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=1)
        
        # 1x1 Conv followed by 3x3 Conv
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.Conv2d(64, 128, kernel_size=3, padding=1)
        )
        
        # 1x1 Conv followed by 5x5 Conv
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.Conv2d(64, 128, kernel_size=5, padding=2)
        )
        
        # 3x3 MaxPooling followed by 1x1 Conv
        self.pool = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, 32, kernel_size=1)
        )
        
    def forward(self, x):
        conv1_out = self.conv1(x)
        conv3_out = self.conv3(x)
        conv5_out = self.conv5(x)
        pool_out = self.pool(x)
        
        # Concatenate along channel dimension
        output = torch.cat([conv1_out, conv3_out, conv5_out, pool_out], dim=1)
        
        return output
```
* `__init__` 메서드  
    class 초기화 메서드로, Incetpion 모듈의 각 요소를 정의합니다.
    1x1 컨볼루션, 1x1 컨볼루션 뒤에 3x3 컨볼루션, 1x1 컨볼루션 뒤에 5x5 컨볼루션, 3x3 맥스 풀링 뒤에 1x1 컨볼루션 등이 있습니다.  
* `forward` 메서드  
    순전파를 정의하는 메서드로, 입력 데이터를 각각의 컴포넌트를 통과시킨 후 채널 차원을 따라 이들을 연결(concatenate)하여 최종 출력을 생성합니다.

```python
class SimpleInception(nn.Module):
    def __init__(self):
        super(SimpleInception, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.inception1 = InceptionModule(64)
        self.fc = nn.Linear(64*4*32*32, 10)  # Assuming input size is [32, 32, 3]
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.inception1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
```
* Inception 모듈을 사용하여 간단한 모델인 `SimpleInception`을 정의합니다.  

* `__init__` 메서드  
    클래스의 초기화 메서드로, 모델의 구조를 정의합니다.  
    첫 번째 레이어는 3채널의 입력 이미지에 64개의 3x3 필터를 사용하는 컨볼루션 레이어입니다.  
    두 번째 레이어는 앞서 정의한 'InceptionModule' 클래스의 인스턴스인 'inception1`'니다.  
    세 번째 레이어는 fully connected 레이어로, 입력 크기를 64*4*32*32로 가정하고 10개의 출력을 갖습니다.
  
* `forward` 메서드  
    순전파를 정의하는 메서드로, 입력 데이터가 각 레이어를 통과하도록 구성됩니다.  
    먼저 입력 데이터는 3x3 컨볼루션 레이어를 통과하고, 그 다음으로는 Inception 모듈을 통과합니다.
    그 후에는 펼쳐진(flatten) 형태로 변환된 다음, fully connected 레이어를 통과하여 최종 출력을 생성합니다.
  
* 이 모델은 단순한 형태의 Inception 아키텍처를 사용하여 이미지 분류를 수행하는 모델로 보입니다.

```python
# Create the model
model = SimpleInception()
```

* 이 코드는 앞서 정의한 `SimpleInception` 클래스를 사용하여 모델을 생성하는 부분입니다.  
* `SimpleInception()`  
    'SimpleInception' 클래스의 인스턴스를 생성하는 코드입니다.
    이 모델은 이미지 분류 작업을 위해 정의된 단순한 Inception 아키텍처를 사용합니다.
    이제 `model` 변수에는 이 인스턴스가 저장되어 있으며, 이 모델을 사용하여 이미지 데이터에 대한 예측을 수행할 수 있습니다.  
* 이 모델을 훈련하거나 추론을 수행하기 위해 데이터를 전달하고 적절한 손실 함수 및 옵티마이저를 선택하여 훈련 단계를 진행할 수 있습니다.

--- 
## 결론
기존 AlexNet에 비해 약 10% 정도의 성능 개선 및 절반 정도의 파라미터 개수가 감소되었으며, 이 모델의 의의는 다음과 같습니다.
 * 다양한 스케일의 특징 동시 학습
 * 네트워크 내 네트워크 개념 활용
 * 여러버전을 발표하며 중요한 아이디어 제안

본 게시물에서는 Inception 모델의 개념과 핵심 구조, 성능 등 기초지식에 대해 다루어보았습니다. 
Inception 모델은 여러 크기의 Convolution 필터와 Pooling 레이어를 병렬로 연결함으로써, 물체의 크기와 상관없이 다양한 스케일을 동시학습 할 수 있습니다. 주로, 이미지 분류, 객체탐지, 이미지 세그멘테이션 등에서 활용되고 있으며, 딥러닝의 지속적인 발전 덕분에 Inception 은 널리 사용되고 있습니다.  
앞으로의 딥러닝 발전을 기대하며, 본 문서가 딥러닝 초보자들에게 유익한 자료가 되길 바랍니다. 

### 별첨
논문리뷰자료
Jupyter notebook 자료
