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
  name: python3
---

+++ {"id": "LQHmwePqryRU"}

# JAX에서 생각하는 방법

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google/jax/blob/main/docs/notebooks/thinking_in_jax.ipynb) [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/google/jax/blob/main/docs/notebooks/thinking_in_jax.ipynb)

JAX는 가속화된 수치 코드를 작성하기 위한 간단하면서도 강력한 API를 제공하지만, JAX에서 효과적으로 작업하기 위해서는 추가적인 고려가 필요할 때가 있습니다. 이 문서는 JAX가 어떻게 작동하는지 바닥부터 이해를 돕기 위해 준비되었으므로, 보다 효과적으로 사용할 수 있습니다.

+++ {"id": "nayIExVUtsVD"}

## JAX vs. NumPy

**핵심 개념:**

- JAX는 편의를 위해 NumPy에서 영감을 받은 인터페이스를 제공합니다.
- 덕 타이핑을 통해, JAX 배열은 종종 NumPy 배열의 대체재로 사용될 수 있습니다.
- NumPy 배열과 달리, JAX 배열은 항상 불변입니다.

NumPy는 수치 데이터를 다루기 위한 잘 알려지고 강력한 API를 제공합니다. 편의를 위해 JAX는 `jax.numpy`를 제공하는데, 이는 numpy API를 밀접하게 반영하며 JAX로의 쉬운 입문을 제공합니다. `numpy`로 할 수 있는 거의 모든 것이 `jax.numpy`로도 할 수 있습니다:

```{code-cell} ipython3
:id: kZaOXL7-uvUP
:outputId: 7fd4dd8e-4194-4983-ac6b-28059f8feb90

import matplotlib.pyplot as plt
import numpy as np

x_np = np.linspace(0, 10, 1000)
y_np = 2 * np.sin(x_np) * np.cos(x_np)
plt.plot(x_np, y_np);
```

```{code-cell} ipython3
:id: 18XbGpRLuZlr
:outputId: 3d073b3c-913f-410b-ee33-b3a0eb878436

import jax.numpy as jnp

x_jnp = jnp.linspace(0, 10, 1000)
y_jnp = 2 * jnp.sin(x_jnp) * jnp.cos(x_jnp)
plt.plot(x_jnp, y_jnp);
```

+++ {"id": "kTZcsCJiuPG8"}

`np`를 `jnp`로 교체하는 것을 제외하고 코드 블록은 동일하며, 결과도 같습니다. 볼 수 있듯이, JAX 배열은 종종 NumPy 배열을 대신하여 플로팅과 같은 것들에 직접 사용될 수 있습니다.

배열 자체는 다른 파이썬 타입으로 구현됩니다:

```{code-cell} ipython3
:id: PjFFunI7xNe8
:outputId: d3b0007e-7997-45c0-d4b8-9f5699cedcbc

type(x_np)
```

```{code-cell} ipython3
:id: kpv5K7QYxQnX
:outputId: ba68a1de-f938-477d-9942-83a839aeca09

type(x_jnp)
```

+++ {"id": "Mx94Ri7euEZm"}

파이썬의 [덕-타이핑](https://en.wikipedia.org/wiki/Duck_typing)은 많은 곳에서 JAX 배열과 NumPy 배열을 서로 교환 가능하게 사용할 수 있게 합니다.

그러나 JAX 배열과 NumPy 배열 사이에는 하나의 중요한 차이가 있습니다: JAX 배열은 불변입니다, 즉 생성된 후 그 내용을 변경할 수 없다는 의미입니다.

다음은 NumPy에서 배열을 변경하는 예입니다:

```{code-cell} ipython3
:id: fzp-y1ZVyGD4
:outputId: 6eb76bf8-0edd-43a5-b2be-85a79fb23190

# NumPy: 가변 배열
x = np.arange(10)
x[0] = 10
print(x)
```

+++ {"id": "nQ-De0xcJ1lT"}

JAX에서는 배열이 불변이기 때문에 동일한 작업을 수행하면 오류가 발생합니다:

```{code-cell} ipython3
:id: l2AP0QERb0P7
:outputId: 528a8e5f-538f-4739-fe95-1c3605ba8c8a

%xmode minimal
```

```{code-cell} ipython3
:id: pCPX0JR-yM4i
:outputId: c7bf4afd-8b7f-4dac-d065-8189679861d6
:tags: [raises-exception]

# JAX: 불변 배열
x = jnp.arange(10)
x[0] = 10
```

+++ {"id": "yRYF0YgO3F4H"}

개별 요소를 업데이트하기 위해, JAX는 업데이트된 복사본을 반환하는 [인덱스 업데이트 문법](https://jax.readthedocs.io/en/latest/jax.ops.html#indexed-update-operators)을 제공합니다:

```{code-cell} ipython3
:id: 8zqPEAeP3UK5
:outputId: 20a40c26-3419-4e60-bd2c-83ad30bd7650

y = x.at[0].set(10)
print(x)
print(y)
```

+++ {"id": "886BGDPeyXCu"}

## NumPy, lax & XLA: JAX API 계층화

**핵심 개념:**

- `jax.numpy`는 익숙한 인터페이스를 제공하는 고수준 래퍼입니다.
- `jax.lax`는 보다 엄격하고 종종 더 강력한 저수준 API입니다.
- 모든 JAX 작업은 [XLA](https://www.tensorflow.org/xla/) – 가속화된 선형 대수 컴파일러에서의 작업에 대한 용어로 구현됩니다.

+++ {"id": "BjE4m2sZy4hh"}

`jax.numpy`의 소스를 살펴보면, 모든 연산이 결국 `jax.lax`에 정의된 함수들의 용어로 표현되는 것을 볼 수 있습니다. `jax.lax`를 다차원 배열을 다루기 위한 보다 엄격하지만 종종 더 강력한 API로 생각할 수 있습니다.

예를 들어, `jax.numpy`는 혼합 데이터 유형 간의 연산을 허용하기 위해 인자를 암시적으로 승격시킬 수 있지만, `jax.lax`는 그렇지 않습니다:

```{code-cell} ipython3
:id: c6EFPcj12mw0
:outputId: 827d09eb-c8aa-43bc-b471-0a6c9c4f6601

import jax.numpy as jnp
jnp.add(1, 1.0)  # jax.numpy API는 혼합 타입을 암시적으로 승격합니다.
```

```{code-cell} ipython3
:id: 0VkqlcXL2qSp
:outputId: 7e1e9233-2fe1-46a8-8eb1-1d1dbc54b58c
:tags: [raises-exception]

from jax import lax
lax.add(1, 1.0)  # jax.lax API는 명시적인 타입 승격을 요구합니다.
```

+++ {"id": "aC9TkXaTEu7A"}

`jax.lax`를 직접 사용하는 경우, 다음과 같은 상황에서는 타입 승격을 명시적으로 수행해야 합니다:

```{code-cell} ipython3
:id: 3PNQlieT81mi
:outputId: 4bd2b6f3-d2d1-44cb-f8ee-18976ae40239

lax.add(jnp.float32(1), 1.0)
```

+++ {"id": "M3HDuM4x2eTL"}

이러한 엄격함과 함께, `jax.lax`는 NumPy가 지원하는 것보다 더 일반적인 연산들에 대해 효율적인 API를 제공합니다.

예를 들어, 1D 컨볼루션을 생각해 보세요. NumPy에서는 다음과 같이 표현할 수 있습니다:

```{code-cell} ipython3
:id: Bv-7XexyzVCN
:outputId: d570f64a-ca61-456f-8cab-6cd643cb8ea1

x = jnp.array([1, 2, 1])
y = jnp.ones(10)
jnp.convolve(x, y)
```

+++ {"id": "0GPqgT7S0q8r"}

내부적으로, 이 NumPy 연산은 [`lax.conv_general_dilated`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv_general_dilated.html)에 의해 구현된 훨씬 더 일반적인 컨볼루션으로 변환됩니다:

```{code-cell} ipython3
:id: pi4f6ikjzc3l
:outputId: 0bb56ae2-7837-4c04-ff8b-6cbc0565b7d7

from jax import lax
result = lax.conv_general_dilated(
    x.reshape(1, 1, 3).astype(float),  # note: explicit promotion
    y.reshape(1, 1, 10),
    window_strides=(1,),
    padding=[(len(y) - 1, len(y) - 1)])  # equivalent of padding='full' in NumPy
result[0, 0]
```

+++ {"id": "7mdo6ycczlbd"}

이것은 배치 처리된 컨볼루션 작업으로, 딥 뉴럴 네트워크에서 종종 사용되는 타입의 컨볼루션에 대해 효율적으로 설계되었습니다. 훨씬 더 많은 보일러플레이트가 필요하지만, NumPy에서 제공하는 컨볼루션보다 훨씬 더 유연하고 확장 가능합니다 (JAX에서의 컨볼루션에 대한 자세한 내용은 [JAX의 컨볼루션](https://jax.readthedocs.io/en/latest/notebooks/convolutions.html)을 참조하세요).

본질적으로, 모든 `jax.lax` 연산은 XLA의 연산들에 대한 파이썬 래퍼입니다; 예를 들어, 여기서 컨볼루션 구현은 [XLA:ConvWithGeneralPadding](https://www.tensorflow.org/xla/operation_semantics#convwithgeneralpadding_convolution)에 의해 제공됩니다.
모든 JAX 연산은 궁극적으로 이러한 기본적인 XLA 연산들의 용어로 표현되며, 이것이 바로 즉시 실행 (JIT) 컴파일을 가능하게 합니다.

+++ {"id": "NJfWa2PktD5_"}

## JIT을 할지 말지

**핵심 개념:**

- 기본적으로 JAX는 연산을 하나씩 순차적으로 실행합니다.
- 즉시 실행 (JIT) 컴파일 데코레이터를 사용하면, 연산의 시퀀스를 함께 최적화하고 한 번에 실행할 수 있습니다.
- 모든 JAX 코드를 JIT 컴파일할 수 있는 것은 아니며, 배열 형태가 정적이며 컴파일 시간에 알려져 있어야 합니다.

모든 JAX 연산이 XLA 용어로 표현된다는 사실은 JAX가 XLA 컴파일러를 사용하여 코드 블록을 매우 효율적으로 실행할 수 있게 합니다.

예를 들어, `jax.numpy` 연산에 대해 표현된 2D 행렬의 행을 정규화하는 함수를 고려해 보세요:

```{code-cell} ipython3
:id: SQj_UKGc-7kQ

import jax.numpy as jnp

def norm(X):
  X = X - X.mean(0)
  return X / X.std(0)
```

+++ {"id": "0yVo_OKSAolW"}

`jax.jit` 변환을 사용하여 함수의 즉시 컴파일된 버전을 만들 수 있습니다:

```{code-cell} ipython3
:id: oHLwGmhZAnCY

from jax import jit
norm_compiled = jit(norm)
```

+++ {"id": "Q3H9ig5GA2Ms"}

이 함수는 원본과 동일한 결과를 표준 부동 소수점 정확도까지 반환합니다:

```{code-cell} ipython3
:id: oz7zzyS3AwMc
:outputId: ed1c796c-59f8-4238-f6e2-f54330edadf0

np.random.seed(1701)
X = jnp.array(np.random.rand(10000, 10))
np.allclose(norm(X), norm_compiled(X), atol=1E-6)
```

+++ {"id": "3GvisB-CA9M8"}

그러나 컴파일(연산의 융합, 임시 배열 할당 방지, 그 밖의 여러 트릭 포함)로 인해, JIT 컴파일된 경우 실행 시간이 몇 단계 빨라질 수 있습니다(JAX의 [비동기 디스패치](https://jax.readthedocs.io/en/latest/async_dispatch.html)를 고려하여 `block_until_ready()` 사용에 주의하세요):

```{code-cell} ipython3
:id: 6mUB6VdDAEIY
:outputId: 1050a69c-e713-44c1-b3eb-1ef875691978

%timeit norm(X).block_until_ready()
%timeit norm_compiled(X).block_until_ready()
```

+++ {"id": "B1eGBGn0tMba"}

`jax.jit`에는 제한 사항이 있습니다: 특히, 모든 배열이 정적인 형태를 가져야 한다는 것입니다. 이는 일부 JAX 연산이 JIT 컴파일과 호환되지 않음을 의미합니다.

예를 들어, 이 연산은 op-by-op 모드에서 실행될 수 있습니다:

```{code-cell} ipython3
:id: YfZd9mW7CSKM
:outputId: 6fdbfde4-7cde-447f-badf-26e1f8db288d

def get_negatives(x):
  return x[x < 0]

x = jnp.array(np.random.randn(10))
get_negatives(x)
```

+++ {"id": "g6niKxoQC2mZ"}

그러나 jit 모드에서 실행하려고 하면 오류를 반환합니다:

```{code-cell} ipython3
:id: yYWvE4rxCjPK
:outputId: 9cf7f2d4-8f28-4265-d701-d52086cfd437
:tags: [raises-exception]

jit(get_negatives)(x)
```

+++ {"id": "vFL6DNpECfVz"}

이는 함수가 컴파일 시간에 알려지지 않은 형태의 배열을 생성하기 때문입니다: 출력의 크기는 입력 배열의 값에 따라 달라지므로, JIT과 호환되지 않습니다.

+++ {"id": "BzBnKbXwXjLV"}

## JIT 메커니즘: 트레이싱과 정적 변수

**핵심 개념:**

- JIT 및 기타 JAX 변환은 함수가 특정 모양과 유형의 입력에 대해 미치는 영향을 결정하기 위해 트레이싱하는 방식으로 작동합니다.

- 트레이스되지 않기를 원하는 변수는 *정적(static)*으로 표시할 수 있습니다.

`jax.jit`을 효과적으로 사용하기 위해서는 그 작동 방식을 이해하는 것이 유용합니다. JIT 컴파일된 함수 내부에 몇 개의 `print()` 문을 넣고 그 함수를 호출해 봅시다:

```{code-cell} ipython3
:id: TfjVIVuD4gnc
:outputId: 9f4ddcaa-8ab7-4984-afb6-47fede5314ea

@jit
def f(x, y):
  print("Running f():")
  print(f"  x = {x}")
  print(f"  y = {y}")
  result = jnp.dot(x + 1, y + 1)
  print(f"  result = {result}")
  return result

x = np.random.randn(3, 4)
y = np.random.randn(4)
f(x, y)
```

+++ {"id": "Ts1fP45A40QV"}

print 문이 실행되지만, 함수에 전달한 데이터를 출력하는 대신, 그것들을 대신하는 *트레이서(tracer)* 객체를 출력합니다.

이러한 트레이서 객체는 `jax.jit`이 함수에 의해 지정된 연산 시퀀스를 추출하는 데 사용하는 것입니다. 기본 트레이서는 배열의 **모양**과 **dtype**을 인코딩하는 대리자이지만, 값에 대해서는 무지합니다. 이렇게 기록된 연산 시퀀스는 파이썬 코드를 다시 실행할 필요 없이 XLA 내에서 동일한 모양과 dtype을 가진 새로운 입력에 효율적으로 적용될 수 있습니다.

일치하는 입력에 대해 컴파일된 함수를 다시 호출할 때, 재컴파일은 필요 없으며 아무것도 출력되지 않습니다. 왜냐하면 결과는 파이썬이 아닌 컴파일된 XLA에서 계산되기 때문입니다:

```{code-cell} ipython3
:id: xGntvzNH7skE
:outputId: 43aaeee6-3853-4b00-fb2b-646df695204a

x2 = np.random.randn(3, 4)
y2 = np.random.randn(4)
f(x2, y2)
```

+++ {"id": "9EB9WkRX7fm0"}

추출된 연산 시퀀스는 JAX 표현식, 또는 짧게는 *jaxpr*에 인코딩됩니다. `jax.make_jaxpr` 변환을 사용하여 jaxpr을 볼 수 있습니다:

```{code-cell} ipython3
:id: 89TMp_Op5-JZ
:outputId: 48212815-059a-4af1-de82-cd39ecac264a

from jax import make_jaxpr

def f(x, y):
  return jnp.dot(x + 1, y + 1)

make_jaxpr(f)(x, y)
```

+++ {"id": "0Oq9S4MZ90TL"}

이것의 한 가지 결과로, 배열의 내용에 대한 정보 *없이* JIT 컴파일이 수행되기 때문에, 함수 내의 제어 흐름 문장은 추적된 값에 의존할 수 없습니다. 예로, 이것은 실패합니다:

```{code-cell} ipython3
:id: A0rFdM95-Ix_
:outputId: e37bf04e-6a6a-4536-e423-f082f52d5f11
:tags: [raises-exception]

@jit
def f(x, neg):
  return -x if neg else x

f(1, True)
```

+++ {"id": "DkTO9m8j-TYI"}

추적하고 싶지 않은 변수가 있다면, JIT 컴파일을 위해 static으로 표시할 수 있습니다:

```{code-cell} ipython3
:id: K1C7ZnVv-lbv
:outputId: e9d6cce3-b036-43da-ad99-887af9625ab0

from functools import partial

@partial(jit, static_argnums=(1,))
def f(x, neg):
  return -x if neg else x

f(1, True)
```

+++ {"id": "dD7p4LRsGzhx"}

다른 static 인수로 JIT 컴파일된 함수를 호출하면 재컴파일이 발생하므로, 함수는 여전히 예상대로 작동합니다:

```{code-cell} ipython3
:id: sXqczBOrG7-w
:outputId: 5fb7c278-b87e-4a6b-ef50-5e4e9c765b52

f(1, False)
```

+++ {"id": "ZESlrDngGVb1"}

어떤 값과 연산이 static이 될지, 추적될지를 이해하는 것은 `jax.jit`을 효과적으로 사용하는 데 있어 중요한 부분입니다.

+++ {"id": "r-RCl_wD5lI7"}

## 정적 연산 vs 추적 연산

**핵심 개념:**

- 값이 정적이거나 추적될 수 있듯이, 연산도 정적이거나 추적될 수 있습니다.

- 정적 연산은 파이썬에서 컴파일 시간에 평가되며; 추적 연산은 XLA에서 컴파일 및 실행 시간에 평가됩니다.

- 정적으로 처리하고 싶은 연산에는 `numpy`를 사용하고, 추적되길 원하는 연산에는 `jax.numpy`를 사용하세요.

정적 및 추적 값 사이의 이러한 구분은 정적 값이 정적으로 유지되도록 하는 방법에 대해 생각하게 만듭니다. 이 함수를 고려해 보세요:

```{code-cell} ipython3
:id: XJCQ7slcD4iU
:outputId: 3646dea0-f6b6-48e9-9dc0-c4dec7816b7a
:tags: [raises-exception]

import jax.numpy as jnp
from jax import jit

@jit
def f(x):
  return x.reshape(jnp.array(x.shape).prod())

x = jnp.ones((2, 3))
f(x)
```

+++ {"id": "ZO3GMGrHBZDS"}

이 함수는 트레이서가 정수 타입의 1D 값 시퀀스 대신 발견되었다는 오류로 실패합니다. 이가 왜 발생하는지 이해하기 위해 함수에 몇 가지 출력문을 추가해 보겠습니다:

```{code-cell} ipython3
:id: Cb4mbeVZEi_q
:outputId: 30d8621f-34e1-4e1d-e6c4-c3e0d8769ec4

@jit
def f(x):
  print(f"x = {x}")
  print(f"x.shape = {x.shape}")
  print(f"jnp.array(x.shape).prod() = {jnp.array(x.shape).prod()}")
  # 이 오류를 피하기 위해 이 부분을 주석 처리하세요:
  # return x.reshape(jnp.array(x.shape).prod())

f(x)
```

+++ {"id": "viSQPc3jEwJr"}

`x`가 추적되고 있음에도 불구하고, `x.shape`는 정적인 값입니다. 그러나 이 정적인 값을 `jnp.array`와 `jnp.prod`에 사용할 때, 추적되는 값이 되며, 이 시점에서는 정적 입력을 요구하는 `reshape()` 같은 함수에서 사용될 수 없게 됩니다 (기억하세요: 배열 형태는 정적이어야 합니다).

유용한 패턴은 정적인 연산(즉, 컴파일 시간에 수행되어야 함)에는 `numpy`를 사용하고, 추적되어야 하는 연산(즉, 실행 시간에 컴파일되고 실행되어야 함)에는 `jax.numpy`를 사용하는 것입니다. 이 함수의 경우, 다음과 같이 보일 수 있습니다:

```{code-cell} ipython3
:id: GiovOOPcGJhg
:outputId: 5363ad1b-23d9-4dd6-d9db-95a6c9de05da

from jax import jit
import jax.numpy as jnp
import numpy as np

@jit
def f(x):
  return x.reshape((np.prod(x.shape),))

f(x)
```

+++ {"id": "C-QZ5d1DG-dv"}

이러한 이유로, JAX 프로그램에서는 표준 컨벤션으로 `import numpy as np`와 `import jax.numpy as jnp`를 사용하여, 연산이 정적 방식(`numpy`를 사용하여 컴파일 시간에 한 번) 또는 추적 방식(`jax.numpy`를 사용하여 실행 시간에 최적화)으로 수행될지에 대해 더 세밀한 제어가 가능하도록 두 인터페이스를 모두 사용할 수 있습니다.
