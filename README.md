<div align="center">
<img src="https://raw.githubusercontent.com/kolonist26/jax-kr/main/images/jax_logo_250px.png" alt="logo"></img>
</div>

# JAX: Autograd와 XLA

![Continuous integration](https://github.com/google/jax/actions/workflows/ci-build.yaml/badge.svg)
![PyPI version](https://img.shields.io/pypi/v/jax)

[**빠른 시작**](#빠른-시작-클라우드로-colab)
| [**변환**](#변환)
| [**설치 가이드**](#설치)
| [**신경망 라이브러리**](#신경망-라이브러리)
| [**변경 로그**](https://jax.readthedocs.io/en/latest/changelog.html)
| [**참고 문서**](https://jax.readthedocs.io/en/latest/)


## JAX란 무엇인가?

JAX는 [Autograd](https://github.com/hips/autograd) 와 [XLA](https://www.tensorflow.org/xla)를 결합하여 고성능 수치 계산을 위해 개발되었으며, 대규모 머신러닝 연구에 사용됩니다.

JAX는 [Autograd](https://github.com/hips/autograd)의 업데이트된 버전을 통해 네이티브 Python 및 Numpy 함수를 자동으로 미분할 수 있습니다. 이는 반복문, 분기, 재귀, 클로저를 통해 미분할 수 있으며, 미분의 미분의 미분까지 계산할 수 있습니다. (역전파로 알려진) 역방향-모드 미분을 [`grad`](#grad를-사용한-자동-미분)를 통해 지원할 뿐만 아니라 순방향-모드 미분도 지원하며, 이 두 방식은 임의의 순서로 자유롭게 조합될 수 있습니다.

새로운 점은 JAX가 [XLA](https://www.tensorflow.org/xla)를 사용하여 Numpy 프로그램을 GPU와 TPU에서 컴파일하고 실행한다는 것입니다. 컴파일은 기본적으로 배후에서 일어나며, 라이브러리 호출은 즉시 컴파일되어 실행됩니다. 그러나 JAX는 단일 함수 API인 [`jit`](#jit을-이용한-컴파일)을 사용하여 자신의 Python 함수를 XLA 최적화 커널로 즉시 컴파일할 수도 있습니다. 컴파일과 자동 미분은 임의로 구성될 수 있어, 복잡한 알고리즘을 표현하고 Python을 벗어나지 않고 최대 성능을 얻을 수 있습니다. 심지어 [`pmap`](#pmap을-사용한-spmd-프로그래밍)을 사용하여 여러 GPU나 TPU 코어를 한 번에 프로그래밍하고, 전체를 통해 미분할 수도 있습니다.

조금 더 깊이 파고들면, JAX가 실제로는 [조합 가능한 함수 변환](#변환)을 위한 확장 가능한 시스템이라는 것을 알 수 있습니다. [`grad`](#grad를-사용한-자동-미분)와 [`jit`](#jit을-이용한-컴파일) 모두 그러한 변환의 예입니다. 다른 예로는 자동 벡터화를 위한 [`vmap`](#vmap을-사용한-자동-벡터화)과 여러 가속기의 단일 프로그램 다중 데이터(Single-Program Multiple-Data, SPMD) 병렬 프로그래밍을 위한 [`pmap`](#pmap을-사용한-spmd-프로그래밍)이 있으며, 더 많은 기능이 추가될 예정입니다.

이는 공식적인 Google 제품이 아닌 연구 프로젝트입니다. 버그와 [위험 요소들](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html)을 예상하세요. [버그 신고](https://github.com/google/jax/issues) 등을 통해 여러분의 생각을 알려주시면 도움이 됩니다!

```python
import jax.numpy as jnp
from jax import grad, jit, vmap

def predict(params, inputs):
  for W, b in params:
    outputs = jnp.dot(inputs, W) + b
    inputs = jnp.tanh(outputs)  # 다음 계층으로의 입력
  return outputs                # 마지막 계층에서는 활성화 함수를 적용하지 않음

def loss(params, inputs, targets):
  preds = predict(params, inputs)
  return jnp.sum((preds - targets)**2)

grad_loss = jit(grad(loss))  # 컴파일된 기울기 평가 함수
perex_grads = jit(vmap(grad_loss, in_axes=(None, 0, 0)))  # 예제별 기울기를 빠르게 계산
```

### Contents
* [빠른 시작: 클라우드로 Colab](#빠른-시작-클라우드로-colab)
* [변환](#변환)
* [현재 주의할 점](#현재-주의할-점)
* [설치](#설치)
* [신경망 라이브러리](#신경망-라이브러리)
* [JAX 인용](#jax-인용)
* [참고 문서](#참고-문서)

## 빠른 시작: 클라우드로 Colab
브라우저에서 노트북을 사용하여 Google Cloud GPU에 연결해서 바로 시작하세요.
다음은 몇 가지 입문용 노트북입니다:
- [기본기: 가속기에서의 NumPy, 미분을 위한 grad, 컴파일을 위한 jit, 벡터화를 위한 vmap](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html)
- [TensorFlow 데이터셋 데이터 로딩을 사용한 간단한 신경망 훈련](https://colab.research.google.com/github/google/jax/blob/main/docs/notebooks/neural_network_with_tfds_data.ipynb)

**JAX는 이제 클라우드 TPU에서 실행됩니다.** 미리보기는 [클라우드 TPU
Colabs](https://github.com/google/jax/tree/main/cloud_tpu_colabs)를 확인하세요.

JAX에 대해 더 깊이 알아보고자 한다면:
- [자동 미분 쿡북, 파트 1: JAX에서 쉽고 강력한 자동 미분](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html)
- [주의해야 할 점과 위험 요소들](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html)
- [노트북 전체 목록](https://github.com/google/jax/tree/main/docs/notebooks)

## 변환

기본적으로, JAX는 수치 함수를 변환하기 위한 확장 가능한 시스템입니다. 여기에는 주요 관심사인 네 가지 변환: `grad`, `jit`, `vmap`, 그리고 `pmap`이 있습니다.

### `grad`를 사용한 자동 미분

JAX는 [Autograd](https://github.com/hips/autograd)와 대략적으로 같은 API를 가지고 있습니다.
가장 인기 있는 함수는 역방향-모드 기울기를 위한 [`grad`](https://jax.readthedocs.io/en/latest/jax.html#jax.grad)입니다:

```python
from jax import grad
import jax.numpy as jnp

def tanh(x):  # 함수 정의
  y = jnp.exp(-2.0 * x)
  return (1.0 - y) / (1.0 + y)

grad_tanh = grad(tanh)  # 기울기 함수 구하기
print(grad_tanh(1.0))   # x = 1.0에서 평가
# 0.4199743 출력
```

`grad`를 사용하여 어떤 차수까지든 미분할 수 있습니다.

```python
print(grad(grad(grad(tanh)))(1.0))
# 0.62162673 출력
```

더 고급 자동미분을 위해, 역방향-모드 벡터-야코비안 곱셈을 위한 [`jax.vjp`](https://jax.readthedocs.io/en/latest/jax.html#jax.vjp)와 순방향-모드 야코비안-벡터 곱셈을 위한 [`jax.jvp`](https://jax.readthedocs.io/en/latest/jax.html#jax.jvp)를 사용할 수 있습니다. 이 두 가지는 서로, 그리고 다른 JAX 변환들과 임의로 조합될 수 있습니다. 여기 [전체 헤세 행렬](https://jax.readthedocs.io/en/latest/_autosummary/jax.hessian.html#jax.hessian)을 효율적으로 계산하는 함수를 만드는 하나의 방법이 있습니다:

```python
from jax import jit, jacfwd, jacrev

def hessian(fun):
  return jit(jacfwd(jacrev(fun)))
```

[Autograd](https://github.com/hips/autograd)와 마찬가지로, 파이썬 제어 구조와 함께 미분을 자유롭게 사용할 수 있습니다:

```python
def abs_val(x):
  if x > 0:
    return x
  else:
    return -x

abs_val_grad = grad(abs_val)
print(abs_val_grad(1.0))   # 1.0 출력
print(abs_val_grad(-1.0))  # -1.0 출력 (abs_val 재평가)
```

더 자세한 정보는 [자동 미분에 관한 참고 문서](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation) [JAX 자동 미분 쿡북](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html)을 참고하세요.

### `jit`을 이용한 컴파일

XLA를 사용하면 `@jit` 데코레이터나 고차 함수로 사용되는 [`jit`](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)으로, 엔드-투-엔드 방식으로 함수를 컴파일할 수 있습니다.

```python
import jax.numpy as jnp
from jax import jit

def slow_f(x):
  # 요소별 연산은 결합으로부터 큰 이점을 얻습니다
  return x * x + x * 2.0

x = jnp.ones((5000, 5000))
fast_f = jit(slow_f)
%timeit -n10 -r3 fast_f(x)  # ~ 4.5 ms / Titan X에서 반복문
%timeit -n10 -r3 slow_f(x)  # ~ 14.5 ms / 반복문 (또한, GPU에서 JAX)
```

원하는대로 `jit` 과 `grad` 및 JAX의 다른 변환을 같이 사용할 수 있습니다.

`jit`을 사용하면 함수가 사용할 수 있는 Python 제어 흐름의 종류에 제약이 생깁니다; 자세한 내용은 
jit을 사용하면 함수가 사용할 수 있는 Python 제어 흐름의 종류에 제약이 생깁니다; 자세한 내용은 [Gotchas
Notebook](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#python-control-flow-+-JIT)을 참고하세요.

### `vmap`을 사용한 자동 벡터화

[`vmap`](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)은 벡터화 맵입니다.
배열 축을 따라 함수를 매핑하는 친숙한 의미를 가지고 있지만, 반복문을 바깥에 유지하는 대신,
더 나은 성능을 위해 함수의 기본 연산으로 반복문을 내부로 밀어 넣습니다.

`vmap`을 사용하면 코드에서 배치 차원을 직접 다룰 필요가 없어질 수 있습니다. 예를 들어, 다음과 같은 간단한 *비배치* 신경망 예측 함수를 고려해보세요:

```python
def predict(params, input_vec):
  assert input_vec.ndim == 1
  activations = input_vec
  for W, b in params:
    outputs = jnp.dot(W, activations) + b  # `activations`은 오른쪽에!
    activations = jnp.tanh(outputs)        # 다음 층으로의 입력
  return outputs                           # 마지막 층에는 활성화 없음
```

우리는 종종 배치 차원을 `activations`의 왼쪽에 허용하기 위해 `jnp.dot(activations, W)`를 사용하지만, 이 특정 예측 함수는 단일 입력 벡터에만 적용되도록 작성되었습니다. 만약 이 함수를 한 번에 입력 배치에 적용하고 싶다면, 의미적으로 우리는 단순히 다음과 같이 작성할 수 있습니다

```python
from functools import partial
predictions = jnp.stack(list(map(partial(predict, params), input_batch)))
```

하지만 한 번에 하나의 예제를 네트워크를 통해 밀어넣는 것은 느릴 것입니다! 계산을 벡터화하여 모든 층에서 행렬-행렬 곱셈을 수행하는 것이 행렬-벡터 곱셈을 수행하는 것보다 낫습니다.

`vmap` 함수가 우리를 위해 그 변환을 수행합니다. 즉, 우리가 다음과 같이 작성한다면

```python
from jax import vmap
predictions = vmap(partial(predict, params))(input_batch)
# 또는
predictions = vmap(predict, in_axes=(None, 0))(params, input_batch)
```

그러면 `vmap` 함수는 외부 루프를 함수 내부로 밀어넣고, 마치 우리가 수동으로 배치 처리를 한 것처럼 행렬-행렬 곱셈을 실행하게 됩니다.

`vmap` 없이 간단한 신경망을 수동으로 배치 처리하는 것은 충분히 쉽지만, 다른 경우에는 수동 벡터화가 비현실적이거나 불가능할 수 있습니다. 예를 들어, 효율적으로 예제별 기울기를 계산하는 문제를 생각해봅시다: 즉, 고정된 파라미터 세트에 대해, 배치 내 각 예제에서 별도로 평가된 손실 함수의 기울기를 계산하고자 합니다. `vmap`을 사용하면 쉽습니다:

```python
per_example_gradients = vmap(partial(grad(loss), params))(inputs, targets)
```

물론, `vmap`은 `jit`, `grad`, 및 JAX의 다른 변환과 임의로 구성될 수 있습니다! 우리는 `vmap`을 전방- 및 역방향-모드 자동 미분과 함께 사용하여 `jax.jacfwd`, `jax.jacrev`, 그리고 `jax.hessian`에서 빠른 야코비안 및 헤세 행렬 계산을 수행합니다.

### `pmap`을 사용한 SPMD 프로그래밍

여러 개의 가속기, 예를 들어 여러 GPU를 병렬 프로그래밍하기 위해, [`pmap`](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)을 사용합니다.
`pmap`을 사용하면 빠른 병렬 집합 통신 및 연산을 포함하여 단일 프로그램 다중 데이터(SPMD) 프로그램을 작성하게 됩니다. `pmap`을 적용하면 작성하는 함수가 XLA에 의해 컴파일되고(`jit`과 유사하게), 여러 장치들에 복제되어 병렬로 실행됩니다.

8개 GPU에서의 예시는 다음과 같습니다:

```python
from jax import random, pmap
import jax.numpy as jnp

# GPU당 하나씩 8개의 랜덤 5000 x 6000 행렬 생성
keys = random.split(random.PRNGKey(0), 8)
mats = pmap(lambda key: random.normal(key, (5000, 6000)))(keys)

# 각 장치에서 병렬로 로컬 matmul 실행(데이터 전송 없음)
result = pmap(lambda x: jnp.dot(x, x.T))(mats)  # result.shape은 (8, 5000, 5000)

# 각 장치에서 병렬로 평균을 계산하고 결과를 출력
print(pmap(jnp.mean)(result))
# [1.1566595 1.1805978 ... 1.2321935 1.2015157] 출력
```

순수한 맵을 표현하는 것 외에도, 장치 간에 빠른 [집합 통신 연산](https://jax.readthedocs.io/en/latest/jax.lax.html#parallel-operators)을 사용할 수 있습니다:

```python
from functools import partial
from jax import lax

@partial(pmap, axis_name='i')
def normalize(x):
  return x / lax.psum(x, 'i')

print(normalize(jnp.arange(4.)))
# [0.         0.16666667 0.33333334 0.5       ] 출력
```

더 정교한 통신 패턴을 위해 [`pmap` 함수들을 중첩](https://colab.research.google.com/github/google/jax/blob/main/cloud_tpu_colabs/Pmap_Cookbook.ipynb#scrollTo=MdRscR5MONuN)할 수도 있습니다.

모든 것이 조합되므로, 병렬 계산을 통해 미분하는 것이 자유롭습니다:

```python
from jax import grad

@pmap
def f(x):
  y = jnp.sin(x)
  @pmap
  def g(z):
    return jnp.cos(z) * jnp.tan(y.sum()) * jnp.tanh(x).sum()
  return grad(lambda w: jnp.sum(g(w)))(x)

print(f(x))
# [[ 0.        , -0.7170853 ],
#  [-3.1085174 , -0.4824318 ],
#  [10.366636  , 13.135289  ],
#  [ 0.22163185, -0.52112055]]

print(grad(lambda x: jnp.sum(f(x)))(x))
# [[ -3.2369726,  -1.6356447],
#  [  4.7572474,  11.606951 ],
#  [-98.524414 ,  42.76499  ],
#  [ -1.6007166,  -1.2568436]]
```

`pmap`를 역방향-모드로 미분할 때(예: `grad`를 사용하여), 계산의 역방향 전달도 전방 전달처럼 병렬화됩니다.

더 자세한 정보는 [SPMD 쿡북](https://colab.research.google.com/github/google/jax/blob/main/cloud_tpu_colabs/Pmap_Cookbook.ipynb)과 [처음부터 시작하는 SPMD MNIST 분류기](https://github.com/google/jax/blob/main/examples/spmd_mnist_classifier_fromscratch.py)를 참고하세요.

## 현재 주의할 점

예시와 설명을 포함하여 현재 주의할 점에 대한 보다 철저한 조사를 원하신다면, [Gotchas
Notebook](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html)을 읽어보시길 강력히 추천합니다. 몇 가지 주요 사항은 다음과 같습니다:

1. JAX 변환은 [순수 함수](https://en.wikipedia.org/wiki/Pure_function)에서만 작동하는데, 이는
   부작용이 없고 [참조 투명성](https://en.wikipedia.org/wiki/Referential_transparency)을 존중하는 함수를 말합니다(즉, `is`를 사용한 객체 식별 테스트는 보장되지 않습니다). 순수하지 않은 파이썬 함수에 JAX 변환을 사용하면 `Exception: Can't lift Traced...` 또는 `Exception: Different traces at same level`과 같은 오류를 볼 수 있습니다.
1. `x[i] += y`와 같은 [배열의 in-place
   mutating 업데이트](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#in-place-updates)는 지원되지 않습니다. 하지만 [함수적 대안들](https://jax.readthedocs.io/en/latest/jax.ops.html)이 있습니다. `jit` 아래에서, 해당 함수적 대안들은 자동으로 버퍼를 재사용합니다.
1. [무작위 숫자는
   다릅니다](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#random-numbers). 하지만 [좋은 이유](https://github.com/google/jax/blob/main/docs/jep/263-prng.md)로 그렇습니다.
1. 만약 [합성곱 연산자](https://jax.readthedocs.io/en/latest/notebooks/convolutions.html)를 찾고
   있다면, 그것들은 `jax.lax` 패키지 안에 있습니다.
1. JAX는 기본적으로 single-precision (32-bit, 예: `float32`) 값을 강제하며, 
  [doube-precision을
   활성화하기](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision)
   (64-bit, 예: `float64`) 위해서는 시작할 때 `jax_enable_x64` 변수를 설정해야 합니다(또는 환경 변수 `JAX_ENABLE_X64=True`를 설정해야 합니다).
   TPU에서, JAX는 `jax.numpy.dot` 및 `lax.conv`와 같은 'matmul 유사' 연산자의 내부 임시 변수를 _제외한_ 모든 것에 대해 기본적으로 32비트 값을 사용합니다.
   해당 연산자들은 `precision` 매개변수를 가지고 있으며, 이를 사용하여 실제 32비트를 시뮬레이션할 수 있지만, 실행 시간이 느려질 수 있는 비용이 발생합니다.
1. NumPy의 dtype 확장 규칙 중 일부는 Python 스칼라와 NumPy 타입을 혼합하여 사용할 때 보장되지 않습니다.
   `np.add(1, np.array([2], np.float32)).dtype`은 `float64`가 아니라 `float32`가 됩니다.
1. 일부 변환은 `jit`과 같이, [Python 제어 흐름을
   사용하는 방법을 제한합니다](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#control-flow). 뭔가 잘못되면 항상 큰 오류가 발생합니다.
   [`jit`의 `static_argnums`
   파라미터](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit), [`lax.scan`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html#jax.lax.scan)과 같은 [구조화된 제어 흐름 기본 요소](https://jax.readthedocs.io/en/latest/jax.lax.html#control-flow-operators)를 사용하거나 더 작은 하위 기능에 `jit`을 사용해야 할 수도 있습니다.

## 설치

### 지원되는 플랫폼

|            | Linux x86_64 | Linux aarch64 | Mac x86_64   | Mac ARM      | Windows x86_64 | Windows WSL2 x86_64 |
|------------|--------------|---------------|--------------|--------------|----------------|---------------------|
| CPU        | 예          | 예           | 예          | 예          |예            | 예                 |
| NVIDIA GPU | 예          | 예           | 아니오           | n/a          | 아니오             | 실험적        |
| Google TPU | 예          | 해당사항 없음           | 해당사항 없음          | 해당사항 없음          | 해당사항 없음            | 해당사항 없음                 |
| AMD GPU    | 실험적 | 아니오            | 아니오           | 해당사항 없음          | 아니오             | 아니오                  |
| Apple GPU  | 해당사항 없음          | 아니오            | 실험적 | 실험적 | 해당사항 없음            | 해당사항 없음                 |


### 지침

| 하드웨어   | 지침                                                                                                    |
|------------|-----------------------------------------------------------------------------------------------------------------|
| CPU        | `pip install -U "jax[cpu]"`                                                                                       |
| NVIDIA GPU on x86_64 | `pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`        |
| Google TPU | `pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html`                 |
| AMD GPU    | [Docker](https://hub.docker.com/r/rocm/jax)를 사용하거나 [소스로부터 빌드](https://jax.readthedocs.io/en/latest/developer.html#additional-notes-for-building-a-rocm-jaxlib-for-amd-gpus)하세요. |
| Apple GPU  | [Apple의 지침](https://developer.apple.com/metal/jax/)을 따르세요.                                          |

기타 설치 방법에 대한 정보는 [문서](https://jax.readthedocs.io/en/latest/installation.html)를 참고하세요. 소스에서 컴파일하기, Docker로 설치하기, 다른 버전의 CUDA 사용하기, 커뮤니티에서 지원하는 conda 빌드, 자주 묻는 질문에 대한 답변을 포함하고 있습니다.



## 신경망 라이브러리

여러 구글 연구 그룹들이 JAX에서 신경망을 훈련하기 위한 라이브러리를 개발하고 공유합니다. 신경망 훈련을 위한 완전한 기능의 라이브러리, 예제 및 사용 방법 가이드를 원한다면 [Flax](https://github.com/google/flax)를 시도해보세요.

Google X는 신경망 라이브러리 [Equinox](https://github.com/patrick-kidger/equinox)를 관리합니다. 이것은 JAX 생태계의 여러 다른 라이브러리의 기반이 되고 있습니다.

또한, DeepMind는 기울기 처리와 최적화를 위한 [Optax](https://github.com/deepmind/optax), RL 알고리즘을 위한 [RLax](https://github.com/deepmind/rlax), 안정적인 코드와 테스팅을 위한 [chex](https://github.com/deepmind/chex)를 포함하여 [JAX 주변의 라이브러리 생태계](https://deepmind.com/blog/article/using-jax-to-accelerate-our-research)를 오픈 소스화하였습니다. (NeurIPS 2020에서 DeepMind의 JAX 생태계에 대한 강연은 [여기](https://www.youtube.com/watch?v=iDxJxIyzSiM)에서 시청하세요)

## JAX 인용

이 레포지토리를 인용하기 위해서는:

```
@software{jax2018github,
  author = {James Bradbury and Roy Frostig and Peter Hawkins and Matthew James Johnson and Chris Leary and Dougal Maclaurin and George Necula and Adam Paszke and Jake Vander{P}las and Skye Wanderman-{M}ilne and Qiao Zhang},
  title = {{JAX}: composable transformations of {P}ython+{N}um{P}y programs},
  url = {http://github.com/google/jax},
  version = {0.3.13},
  year = {2018},
}
```

위의 bibtex 항목에서, 이름들은 알파벳 순서로 정렬되어 있으며, 버전 번호는 [jax/version.py](../main/jax/version.py)에서 의도된 것입니다. 또한, 연도는 프로젝트의 오픈 소스 출시에 해당합니다.

JAX의 초기 버전은 자동 미분과 XLA로의 컴파일만을 지원하였으며, [SysML 2018에서 발표된 논문](https://mlsys.org/Conferences/2019/doc/2018/146.pdf)에서 설명되었습니다. 현재, JAX의 아이디어와 기능을 더 포괄적이고 최신의 논문으로 다루기 위해 작업 중입니다.

## 참고 문서

JAX API에 대한 자세한 내용은, [참고 문서](https://jax.readthedocs.io/)를 확인하세요.

JAX 개발자로 시작하는 방법에 대해서는, [개발자 문서](https://jax.readthedocs.io/en/latest/developer.html)를 참고하세요.
