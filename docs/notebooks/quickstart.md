---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.0
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

+++ {"id": "xtWX4x9DCF5_"}

# JAX 빠른 시작

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google/jax/blob/main/docs/notebooks/quickstart.ipynb) [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/google/jax/blob/main/docs/notebooks/quickstart.ipynb)

**JAX는 CPU, GPU, 및 TPU에서 작동하는 NumPy이며, 고성능 머신러닝 연구를 위한 훌륭한 자동 미분 기능을 제공합니다.**

[Autograd](https://github.com/hips/autograd)의 업데이트된 버전을 통해 JAX는 네이티브 파이썬 및 NumPy 코드를 자동으로 미분할 수 있습니다.
이는 파이썬의 다양한 기능, 예를 들어 반복문, 조건문, 재귀, 클로저 등을 통한 미분이 가능하며, 미분의 미분, 그 미분의 미분까지도 계산할 수 있습니다.
역방향-모드 뿐만 아니라 순방향-모드 미분을 지원하며, 두 방식은 임의의 순서로 자유롭게 조합될 수 있습니다.

새로운 점은 JAX가 [XLA](https://www.tensorflow.org/xla)를 사용하여 NumPy 코드를 GPU와 TPU와 같은 가속기에서 컴파일하고 실행한다는 것입니다.
컴파일은 기본적으로 백그라운드에서 일어나며, 라이브러리 호출은 즉석에서 컴파일되어 실행됩니다. 하지만 JAX는 당신이 자신의 파이썬 함수를 XLA 최적화 커널로 즉석에서 컴파일할 수 있게 하는 단일 함수 API도 제공합니다.
컴파일과 자동 미분은 임의로 조합될 수 있어, 복잡한 알고리즘을 표현하고 파이썬을 떠나지 않고도 최대의 성능을 얻을 수 있습니다.

```{code-cell} ipython3
:id: SY8mDvEvCGqk

import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
```

+++ {"id": "FQ89jHCYfhpg"}

## 행렬 곱하기

+++ {"id": "Xpy1dSgNqCP4"}

다음 예시에서는 무작위 데이터를 생성할 것입니다. NumPy와 JAX 사이의 큰 차이점 중 하나는 난수를 생성하는 방법입니다. 자세한 내용은 [JAX에서 흔히 발생하는 문제]를 참고하세요.

[JAX에서 흔히 발생하는 문제]: https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#%F0%9F%94%AA-Random-Numbers

```{code-cell} ipython3
:id: u0nseKZNqOoH
:outputId: 03e20e21-376c-41bb-a6bb-57431823691b

key = random.PRNGKey(0)
x = random.normal(key, (10,))
print(x)
```

+++ {"id": "hDJF0UPKnuqB"}

바로 들어가서 두 개의 큰 행렬을 곱해보겠습니다.

```{code-cell} ipython3
:id: eXn8GUl6CG5N
:outputId: ffce6bdc-86e6-4af0-ab5d-65d235022db9

size = 3000
x = random.normal(key, (size, size), dtype=jnp.float32)
%timeit jnp.dot(x, x.T).block_until_ready()  # GPU에서 실행
```

+++ {"id": "0AlN7EbonyaR"}

JAX는 기본적으로 비동기 실행을 사용하기 때문에 ({ref}`async-dispatch`를 참고) `block_until_ready`를 추가했습니다.

JAX NumPy 함수는 일반 NumPy 배열에서 작동합니다.

```{code-cell} ipython3
:id: ZPl0MuwYrM7t
:outputId: 71219657-b559-474e-a877-5441ee39f18f

import numpy as np
x = np.random.normal(size=(size, size)).astype(np.float32)
%timeit jnp.dot(x, x.T).block_until_ready()
```

+++ {"id": "_SrcB2IurUuE"}

그것은 매번 데이터를 GPU로 전송해야 하기 때문에 더 느립니다. {func}`~jax.device_put`을 사용하여 NDArray가 장치 메모리에 의해 지원되는지 확인할 수 있습니다.

```{code-cell} ipython3
:id: Jj7M7zyRskF0
:outputId: a649a6d3-cf28-445e-c3fc-bcfe3069482c

from jax import device_put

x = np.random.normal(size=(size, size)).astype(np.float32)
x = device_put(x)
%timeit jnp.dot(x, x.T).block_until_ready()
```

+++ {"id": "clO9djnen8qi"}

{func}`~jax.device_put`의 출력은 여전히 NDArray처럼 동작하지만, 출력, 그래프 그리기, 디스크에 저장하기, 분기 등 필요할 때만 CPU로 값이 복사됩니다. {func}`~jax.device_put`의 동작은 `jit(lambda x: x)` 함수와 동일하지만, 더 빠릅니다.

+++ {"id": "ghkfKNQttDpg"}

GPU(또는 TPU!)가 있다면, 이러한 호출은 가속기에서 실행되며 CPU보다 훨씬 빠를 수 있는 잠재력을 가지고 있습니다.

NumPy와 JAX의 성능 특성 비교에 대해서는 {ref}`faq-jax-vs-numpy`를 참고하세요.

+++ {"id": "iOzp0P_GoJhb"}

JAX는 GPU-backed NumPy보다 훨씬 더 많은 것을 제공합니다. 수치 코드를 작성할 때 유용한 몇 가지 프로그램 변환도 함께 제공됩니다. 현재, 주요한 것은 세 가지입니다:

{func}`~jax.jit`, 코드 속도 향상을 위해
{func}`~jax.grad`, 미분을 위해
{func}`~jax.vmap`, 자동 벡터화 또는 배치를 위해.

이것들을 하나씩 살펴보겠습니다. 우리는 이것들을 흥미로운 방법으로 조합하는 것으로 마무리할 것입니다.

+++ {"id": "bTTrTbWvgLUK"}

## {func}`~jax.jit`을 사용하여 함수 속도 향상

+++ {"id": "YrqE32mvE3b7"}

JAX는 GPU 또는 TPU에서 투명하게 실행됩니다 (만약 가지고 있지 않다면 CPU로 대체됩니다). 그러나 위의 예에서, JAX는 한 번에 하나의 연산을 GPU에 전달하고 있습니다. 연산자들의 시퀀스가 있다면, [XLA](https://www.tensorflow.org/xla)를 사용하여 여러 연산을 함께 컴파일하기 위해 `@jit` 데코레이터를 사용할 수 있습니다. 해봅시다.

```{code-cell} ipython3
:id: qLGdCtFKFLOR
:outputId: 870253fa-ba1b-47ec-c5a4-1c6f706be996

def selu(x, alpha=1.67, lmbda=1.05):
  return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

x = random.normal(key, (1000000,))
%timeit selu(x).block_until_ready()
```

+++ {"id": "a_V8SruVHrD_"}

`@jit`을 사용하면 속도를 높일 수 있으며, `selu`가 처음 호출될 때 jit-컴파일 되고 이후에는 캐시될 것입니다.

```{code-cell} ipython3
:id: fh4w_3NpFYTp
:outputId: 4d56b4f2-5d58-4689-ecc2-ac361c0245cd

selu_jit = jit(selu)
%timeit selu_jit(x).block_until_ready()
```

+++ {"id": "HxpBc4WmfsEU"}

## {func}`~jax.grad`를 사용한 미분 계산

수치 함수를 평가하는 것 외에도, 우리는 그것들을 변환하고 싶습니다. 한 가지 변환은 [자동 미분](https://en.wikipedia.org/wiki/Automatic_differentiation)입니다. JAX에서는 [Autograd](https://github.com/HIPS/autograd)에서와 마찬가지로, {func}`~jax.grad` 함수를 사용하여 기울기를 계산할 수 있습니다.

```{code-cell} ipython3
:id: IMAgNJaMJwPD
:outputId: 6646cc65-b52f-4825-ff7f-e50b67083493

def sum_logistic(x):
  return jnp.sum(1.0 / (1.0 + jnp.exp(-x)))

x_small = jnp.arange(3.)
derivative_fn = grad(sum_logistic)
print(derivative_fn(x_small))
```

+++ {"id": "PtNs881Ohioc"}

유한 차분을 사용하여 우리의 결과가 정확한지 검증해봅시다.

```{code-cell} ipython3
:id: JXI7_OZuKZVO
:outputId: 18c1f913-d5d6-4895-f71e-e62180c3ad1b

def first_finite_differences(f, x):
  eps = 1e-3
  return jnp.array([(f(x + eps * v) - f(x - eps * v)) / (2 * eps)
                   for v in jnp.eye(len(x))])


print(first_finite_differences(sum_logistic, x_small))
```

+++ {"id": "Q2CUZjOWNZ-3"}

미분을 하는 것은 {func}`~jax.grad`를 호출하는 것만큼 쉽습니다. {func}`~jax.grad`와 {func}`~jax.jit`는 조합되어 임의로 혼합될 수 있습니다. 위의 예에서 우리는 `sum_logistic`을 jit한 다음 그 미분을 취했습니다. 우리는 더 나아갈 수 있습니다:

```{code-cell} ipython3
:id: TO4g8ny-OEi4
:outputId: 1a0421e6-60e9-42e3-dc9c-e558a69bbf17

print(grad(jit(grad(jit(grad(sum_logistic)))))(1.0))
```

+++ {"id": "yCJ5feKvhnBJ"}

보다 고급 자동 미분을 위해서, 역방향-모드 벡터-야코비안 곱셈을 위한 {func}`jax.vjp`와 순방향-모드 야코비안-벡터 곱셈을 위한 {func}`jax.jvp`를 사용할 수 있습니다. 두 가지는 서로와 다른 JAX 변환과 임의로 조합될 수 있습니다. 이들을 조합하여 헤세 행렬을 효율적으로 계산하는 함수를 만들 수 있는 한 가지 방법이 있습니다:

```{code-cell} ipython3
:id: Z-JxbiNyhxEW

from jax import jacfwd, jacrev
def hessian(fun):
  return jit(jacfwd(jacrev(fun)))
```

+++ {"id": "TI4nPsGafxbL"}

## {func}`~jax.vmap`을 사용한 자동 벡터화

+++ {"id": "PcxkONy5aius"}

JAX에는 API에서 유용하게 사용할 수 있는 또 다른 변환이 있습니다: {func}`~jax.vmap`, 맵 벡터화입니다. 함수를 배열 축을 따라 매핑하는 친숙한 의미를 가지고 있지만, 반복문을 외부에 유지하는 대신 함수의 원시 연산으로 반복문을 내리는 방식으로, 성능을 향상시킵니다. {func}`~jax.jit`와 함께 사용하면, 수동으로 배치 차원을 추가하는 것만큼 빠를 수 있습니다.

+++ {"id": "TPiX4y-bWLFS"}

우리는 간단한 예제로 작업하고, {func}`~jax.vmap`를 사용하여 행렬-벡터 곱을 행렬-행렬 곱으로 확장시킬 것입니다. 이 경우에는 수동으로 하기 쉽지만, 같은 기술을 더 복잡한 함수에 적용할 수 있습니다.

```{code-cell} ipython3
:id: 8w0Gpsn8WYYj

mat = random.normal(key, (150, 100))
batched_x = random.normal(key, (10, 100))

def apply_matrix(v):
  return jnp.dot(mat, v)
```

+++ {"id": "0zWsc0RisQWx"}

`apply_matrix`와 같은 함수가 주어졌을 때, 파이썬에서 배치 차원을 따라 반복할 수 있지만, 보통 그런 작업은 성능이 좋지 않습니다.

```{code-cell} ipython3
:id: KWVc9BsZv0Ki
:outputId: bea78b6d-cd17-45e6-c361-1c55234e77c0

def naively_batched_apply_matrix(v_batched):
  return jnp.stack([apply_matrix(v) for v in v_batched])

print('Naively batched')
%timeit naively_batched_apply_matrix(batched_x).block_until_ready()
```

+++ {"id": "qHfKaLE9stbA"}

이 작업을 수동으로 배치하는 방법을 알고 있습니다. 이 경우, `jnp.dot`은 추가 배치 차원을 투명하게 처리합니다.

```{code-cell} ipython3
:id: ipei6l8nvrzH
:outputId: 335cdc4c-c603-497b-fc88-3fa37c5630c2

@jit
def batched_apply_matrix(v_batched):
  return jnp.dot(v_batched, mat.T)

print('Manually batched')
%timeit batched_apply_matrix(batched_x).block_until_ready()
```

+++ {"id": "1eF8Nhb-szAb"}

그러나, 배치 지원이 없는 더 복잡한 함수가 있다고 가정해 봅시다. 우리는 {func}`~jax.vmap`을 사용하여 자동으로 배치 지원을 추가할 수 있습니다.

```{code-cell} ipython3
:id: 67Oeknf5vuCl
:outputId: 9c680e74-ebb5-4563-ebfc-869fd82de091

@jit
def vmap_batched_apply_matrix(v_batched):
  return vmap(apply_matrix)(v_batched)

print('Auto-vectorized with vmap')
%timeit vmap_batched_apply_matrix(batched_x).block_until_ready()
```

+++ {"id": "pYVl3Z2nbZhO"}

물론, {func}`~jax.vmap`은 {func}`~jax.jit`, {func}`~jax.grad`, 그리고 다른 JAX 변환과 임의로 조합될 수 있습니다.

+++ {"id": "WwNnjaI4th_8"}

이것은 JAX가 할 수 있는 것의 일부일 뿐입니다. 여러분이 이를 사용하여 무엇을 할지 정말 기대됩니다!
