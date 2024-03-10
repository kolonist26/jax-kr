JAX 자주 묻는 질문 (FAQ)
====================================

.. comment RST primer for Sphinx: https://thomas-cokelaer.info/tutorials/sphinx/rest_syntax.html
.. comment Some links referenced here. Use `JAX - The Sharp Bits`_ (underscore at the end) to reference

.. _JAX - The Sharp Bits: https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html

여기에 자주 묻는 질문들에 대한 답변을 수집하고 있습니다.
기여는 언제든 환영입니다!

``jit`` 은 함수의 동작을 변경합니다
--------------------------------------------

만약 :func:`jax.jit` 을 쓰고나서 파이썬 함수의 동작에 변화가 있다면, 아마도 그 함수는 전역 변수를 사용하거나 부수 효과를 가지고 있을 것입니다.
다음 코드에서, ``impure_func`` 은 전역 변수 ``y`` 를 사용하고 ``print`` 라는 부수 효과를 가지고 있습니다.::

    y = 0

    # @jit   # Different behavior with jit
    def impure_func(x):
      print("Inside:", y)
      return x + y

    for y in range(3):
      print("Result:", impure_func(y))

``jit`` 이 없을 때의 출력::

    Inside: 0
    Result: 0
    Inside: 1
    Result: 2
    Inside: 2
    Result: 4

``jit`` 이 있을 때의 출력::

    Inside: 0
    Result: 0
    Result: 1
    Result: 2

:func:`jax.jit` 은, 처음에 파이썬 인터프리터를 사용하여 한 번 실행되며,
이때 ``Inside`` 출력이 발생하고, ``y`` 의 첫 번째 값이 관찰됩니다.
그 후, 함수는 컴파일되어 캐시되고, 다른 ``x`` 값들로 여러 번 실행되지만, ``y`` 는 첫 번째 값과 동일한 값으로 실행됩니다.

추가 읽을 자료:

  * `JAX - The Sharp Bits`_

.. _faq-jit-numerics:

``jit`` 는 출력의 정확한 숫자를 변경합니다
---------------------------------------------
사용자는 때때로 함수를 :func:`jit` 으로 래핑하는 것이 함수의 출력을 변경할 수 있다는 사실에 놀랍니다.
예를 들어:

>>> from jax import jit
>>> import jax.numpy as jnp
>>> def f(x):
...   return jnp.log(jnp.sqrt(x))
>>> x = jnp.pi
>>> print(f(x))
0.572365

>>> print(jit(f)(x))
0.5723649

이러한 약간의 출력 차이는 XLA 컴파일러 내부의 최적화에서 비롯됩니다:
컴파일 중에, XLA는 전반적인 계산을 더 효율적으로 만들기 위해 특정 연산을 재배열하거나 생략하기도 합니다.

이 경우, XLA는 로그의 성질을 활용하여 ``log(sqrt(x))`` 를 수학적으로 동일한 표현인 ``0.5 * log(x)`` 로 대체하는데,
이는 원래보다 더 효율적으로 계산될 수 있습니다. 출력의 차이는 부동 소수점 연산이 실제 수학의 근사치에 불과하기 때문에,
같은 표현을 계산하는 서로 다른 방식이 미묘하게 다른 결과를 낼 수 있다는 사실에서 비롯됩니다.

다른 경우에는, XLA의 최적화가 훨씬 더 극적인 차이를 초래할 수도 있습니다.
다음 예를 고려해봅시다:

>>> def f(x):
...   return jnp.log(jnp.exp(x))
>>> x = 100.0
>>> print(f(x))
inf

>>> print(jit(f)(x))
100.0

JIT이 적용되지 않은 op-by-op 모드에서는 결과가 inf인데, 이는 ``jnp.exp(x)`` 가 오버플로우되어 ``inf`` 를 반환하기 때문입니다.
하지만 JIT 아래에서, XLA는 ``log`` 가 ``exp`` 의 역함수임을 인식하고 컴파일된 함수에서 해당 연산을 제거하여 단순히 입력을 반환합니다.
이 경우, JIT 컴파일은 실제 결과의 더 정확한 부동 소수점 근사값을 생성합니다.

불행히도 XLA의 대수적 단순화 전체 목록은 잘 문서화되어 있지 않지만,
C++에 익숙하고 XLA 컴파일러가 어떤 종류의 최적화를 하는지 궁금하다면, 소스 코드에서 이를 볼 수 있습니다: 
`algebraic_simplifier.cc`_.

.. _faq-slow-compile:

``jit`` 으로 데코레이트된 함수는 컴파일하는데 매우 느립니다.
--------------------------------------------------

만약 ``jit`` 으로 데코레이트된 함수가 처음 호출될 때 수십 초(혹은 그 이상!) 걸리지만,
다시 호출될 때는 빠르게 실행된다면, JAX가 코드를 추적하거나 컴파일하는데 오랜 시간이 걸리고 있는 것입니다.

이는 보통 함수 호출이 JAX의 내부 표현에서 대량의 코드를 생성한다는 신호인데,
이는 주로 ``for`` 루프와 같은 파이썬 제어 흐름을 많이 사용하기 때문입니다.
소수의 반복에 대해 파이썬은 괜찮지만,
*많은* 반복이 필요하다면 코드를 재작성하여 `JAX의 구조화된 제어 흐름 기본 요소 <https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#Structured-control-flow-primitives>`_
(예: :func:`lax.scan`)를 사용하거나, 루프를 ``jit`` 으로 래핑하지 않는 것이 좋습니다(루프 *내부*에서 ``jit`` 데코레이트된 함수는 여전히 사용 가능합니다).

이 문제가 확실하지 않다면, 함수에 대해 :func:`jax.make_jaxpr` 를 실행해 보는 것이 좋습니다.
출력이 수백 또는 수천 줄에 달한다면 컴파일이 느릴 것으로 예상할 수 있습니다.

코드를 파이썬 루프를 피하도록 재작성하는 방법이 명확하지 않은 경우가 있는데, 이는 코드가 다양한 모양의 많은 배열을 사용하기 때문일 수 있습니다.
이 경우 추천되는 해결책은 :func:`jax.numpy.where` 와 같은 함수를 사용하여 고정된 형태의 패딩된 배열에서 계산을 수행하는 것입니다.

만약 당신의 함수가 다른 이유로 인해 컴파일하는데 느리다면, Github 이슈를 생성해주세요.

.. _faq-jit-class-methods:

``jit`` 를 메소드와 함께 사용하는 방법은?
--------------------------------
:func:`jax.jit` 대부분의 예시는 독립적인 파이썬 함수를 데코레이트하는 것과 관련이 있지만,
클래스 내의 메소드를 데코레이트하는 것은 약간의 복잡성을 도입합니다.
예를 들어, 다음과 같은 간단한 클래스를 고려해봅시다.
여기서 우리는 표준 :func:`~jax.jit` 주석을 메소드에 사용했습니다::

    >>> import jax.numpy as jnp
    >>> from jax import jit
     
    >>> class CustomClass:
    ...   def __init__(self, x: jnp.ndarray, mul: bool):
    ...     self.x = x
    ...     self.mul = mul
    ... 
    ...   @jit  # <---- How to do this correctly?
    ...   def calc(self, y):
    ...     if self.mul:
    ...       return self.x * y
    ...     return y

그러나, 이 방법을 사용하여 이 메소드를 호출하려고 하면 오류가 발생합니다::

    >>> c = CustomClass(2, True)
    >>> c.calc(3)  # doctest: +SKIP
    ---------------------------------------------------------------------------
    TypeError                                 Traceback (most recent call last)
      File "<stdin>", line 1, in <module
    TypeError: Argument '<CustomClass object at 0x7f7dd4125890>' of type <class 'CustomClass'> is not a valid JAX type.

문제는 함수의 첫 번째 인수가 ``self`` 이며, 그 타입이 ``CustomClass`` 인데, JAX가 이 타입을 처리하는 방법을 모른다는 것입니다.
이 경우 우리가 사용할 수 있는 세 가지 기본 전략이 있으며, 이에 대해 아래에서 논의할 것입니다.

전략 1: JIT-컴파일된 도우미 함수
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
가장 간단한 접근법은 클래스 외부에 도우미 함수를 생성하고, 이를 보통 방식대로 JIT-데코레이트하는 것입니다. 예를 들어::

    >>> from functools import partial
    
    >>> class CustomClass:
    ...   def __init__(self, x: jnp.ndarray, mul: bool):
    ...     self.x = x
    ...     self.mul = mul
    ... 
    ...   def calc(self, y):
    ...     return _calc(self.mul, self.x, y)
    
    >>> @partial(jit, static_argnums=0)
    ... def _calc(mul, x, y):
    ...   if mul:
    ...     return x * y
    ...   return y

결과는 예상대로 작동할 것이다::

    >>> c = CustomClass(2, True)
    >>> print(c.calc(3))
    6

이러한 접근법의 장점은 단순하고 명시적이며, ``CustomClass`` 타입의 객체를 처리하는 방법을 JAX에 가르칠 필요가 없다는 것이다.
그러나, 모든 메소드 로직을 같은 장소에 유지하고 싶을 수도 있습니다.

전략 2: ``self`` 를 static으로 표시하기
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
또 다른 일반적인 방법은 ``self`` 인자를 static으로 표시하기 위해 ``static_argnums`` 를 사용하는 것입니다.
그러나 이는 예상치 못한 결과를 피하기 위해 주의해서 수행되어야 합니다.
다음과 같이 단순히 이 작업을 수행하고 싶을 수 있습니다::

    >>> class CustomClass:
    ...   def __init__(self, x: jnp.ndarray, mul: bool):
    ...     self.x = x
    ...     self.mul = mul
    ...  
    ...   # WARNING: this example is broken, as we'll see below. Don't copy & paste!
    ...   @partial(jit, static_argnums=0)
    ...   def calc(self, y):
    ...     if self.mul:
    ...       return self.x * y
    ...     return y

메소드를 호출하면 더 이상 오류가 발생하지 않습니다::

    >>> c = CustomClass(2, True)
    >>> print(c.calc(3))
    6

그러나 한 가지 문제가 있습니다: 첫 번째 메소드 호출 후 객체를 변형시키면,
다음 메소드 호출이 잘못된 결과를 반환할 수 있습니다::

    >>> c.mul = False
    >>> print(c.calc(3))  # Should print 3
    6

이유는 무엇일까요? 객체를 static으로 표시하면, 이는 JIT의 내부 컴파일 캐시에서 사전 키로 사용될 것이며,
이는 해당 객체의 해시(즉, ``hash(obj)``) 동등성(즉, ``obj1 == obj2``) 및 객체 식별성(즉, ``obj1 is obj2``)이 일관된 행동을 할 것으로 가정합니다.
사용자 정의 객체의 기본 ``__hash__`` 는 그 객체 ID이므로, JAX는 변형된 객체가 재컴파일을 트리거해야 한다는 것을 알 방법이 없습니다.

이 문제는 적절한 ``__hash__`` 및 ``__eq__`` 메서드를 정의함으로써 부분적으로 해결할 수 있습니다; 예를 들면::

    >>> class CustomClass:
    ...   def __init__(self, x: jnp.ndarray, mul: bool):
    ...     self.x = x
    ...     self.mul = mul
    ... 
    ...   @partial(jit, static_argnums=0)
    ...   def calc(self, y):
    ...     if self.mul:
    ...       return self.x * y
    ...     return y
    ... 
    ...   def __hash__(self):
    ...     return hash((self.x, self.mul))
    ... 
    ...   def __eq__(self, other):
    ...     return (isinstance(other, CustomClass) and
    ...             (self.x, self.mul) == (other.x, other.mul))

(``__hash__`` 를 오버라이딩할 때의 요구 사항에 대한 더 많은 토론을 위해 :meth:`object.__hash__` 문서를 참조하세요).

이는 객체를 **절대 변형시키지 않는 한** JIT 및 기타 변형과 올바르게 작동해야 합니다.
해시 키로 사용되는 객체의 변형은 여러 가지 미묘한 문제를 일으키며,
예를 들어 가변 파이썬 컨테이너(예: :class:`dict`, :class:`list`)는 ``__hash__`` 를 정의하지 않는 반면,
그들의 불변 대응물(예: :class:`tuple`)은 합니다.

클래스가 내부 변형(예: 메소드 내에서 ``self.attr = ...`` 설정)에 의존하는 경우,
그 객체는 실제로 "static"이 아니며 이를 그렇게 표시하는 것은 문제를 일으킬 수 있습니다.
다행히, 이 경우에는 다른 옵션이 있습니다.

전략 3: ``CustomClass`` 를 PyTree로 만들기
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
클래스 메서드를 올바르게 JIT 컴파일하는 가장 유연한 접근 방법은 해당 타입을 사용자 정의 PyTree 객체로 등록하는 것입니다; :ref:`extending-pytrees` 를 참조하세요.
이를 통해 클래스의 어떤 구성 요소를 정적으로 처리해야 하고 어떤 것을 동적으로 처리해야 하는지 정확히 지정할 수 있습니다. 다음은 그 예시입니다::

    >>> class CustomClass:
    ...   def __init__(self, x: jnp.ndarray, mul: bool):
    ...     self.x = x
    ...     self.mul = mul
    ... 
    ...   @jit
    ...   def calc(self, y):
    ...     if self.mul:
    ...       return self.x * y
    ...     return y
    ... 
    ...   def _tree_flatten(self):
    ...     children = (self.x,)  # arrays / dynamic values
    ...     aux_data = {'mul': self.mul}  # static values
    ...     return (children, aux_data)
    ...
    ...   @classmethod
    ...   def _tree_unflatten(cls, aux_data, children):
    ...     return cls(*children, **aux_data)
    
    >>> from jax import tree_util
    >>> tree_util.register_pytree_node(CustomClass,
    ...                                CustomClass._tree_flatten,
    ...                                CustomClass._tree_unflatten)

이 방법은 확실히 더 복잡하지만, 위에서 사용된 더 간단한 접근 방법과 관련된 모든 문제를 해결합니다::

    >>> c = CustomClass(2, True)
    >>> print(c.calc(3))
    6

    >>> c.mul = False  # mutation is detected
    >>> print(c.calc(3))
    3

    >>> c = CustomClass(jnp.array(2), True)  # non-hashable x is supported
    >>> print(c.calc(3))
    6

당신의 ``tree_flatten`` 및 ``tree_unflatten`` 함수가 클래스의 모든 관련 속성을 올바르게 처리한다면,
특별한 주석 없이도 이 타입의 객체를 JIT 컴파일된 함수의 인수로 직접 사용할 수 있어야 합니다.

.. _faq-data-placement:

장치에서 데이터 및 계산 배치 제어하기
-----------------------------------------------------

먼저 JAX에서 데이터 및 계산 배치의 원리를 살펴보겠습니다.

JAX에서 계산은 데이터 배치를 따릅니다. JAX 배열에는 두 가지 배치 속성이 있습니다:
1) 데이터가 저장되는 장치; 그리고 2) 데이터가 해당 장치에 **고정** 되었는지 여부(데이터가 때때로 장치에 *sticky* 되어 있다고 언급됩니다).

기본적으로, JAX 배열은 기본 장치(``jax.devices()[0]``)에 고정되지 않은 상태로 배치됩니다.
기본적으로 첫 번째 GPU 또는 TPU입니다. GPU 또는 TPU가 없는 경우, ``jax.devices()[0]`` 은 CPU입니다.
기본 장치는 :func:`jax.default_device` 컨텍스트 관리자를 사용하여 일시적으로 덮어쓸 수 있거나,
환경 변수 ``JAX_PLATFORMS`` 또는 absl 플래그 ``--jax_platforms`` 를 "cpu", "gpu", 또는 "tpu"로 설정함으로써
전체 프로세스에 대해 설정할 수 있습니다(``JAX_PLATFORMS`` 은 플랫폼 목록일 수도 있으며, 이는 우선 순위 순서대로 사용 가능한 플랫폼을 결정합니다).

>>> from jax import numpy as jnp
>>> print(jnp.ones(3).devices())  # doctest: +SKIP
{CudaDevice(id=0)}

고정되지 않은 데이터를 포함하는 계산은 기본 장치에서 수행되며 결과는 기본 장치에 고정되지 않은 상태로 남습니다.

:func:`jax.device_put` 을 ``device`` 매개변수와 함께 사용하여 데이터를 명시적으로 장치에 배치할 수도 있습니다.
이 경우 데이터는 해당 장치에 **고정**됩니다:

>>> import jax
>>> from jax import device_put
>>> arr = device_put(1, jax.devices()[2])  # doctest: +SKIP
>>> print(arr.devices())  # doctest: +SKIP
{CudaDevice(id=2)}

일부 고정 입력을 포함하는 계산은 고정 장치에서 발생하며 결과는 같은 장치에 고정됩니다.
한 개 이상의 장치에 고정된 인수에 대해 작업을 호출하면 오류가 발생합니다.

``device`` 매개변수 없이 :func:`jax.device_put` 을 사용할 수도 있습니다.
데이터가 이미 장치에 있으면(고정되었든 아니든) 그대로 유지됩니다.
데이터가 어떤 장치에도 없는 경우, 즉, 일반 Python 또는 NumPy 값인 경우 기본 장치에 고정되지 않은 상태로 배치됩니다.

JIT된 함수는 다른 원시 연산처럼 동작합니다. 데이터를 따르며,
한 개 이상의 장치에 고정된 데이터에 대해 호출될 경우 오류를 표시합니다.

(2021년 3월 `PR #6002 <https://github.com/google/jax/pull/6002>`_ 이전에는
``jax.device_put(jnp.zeros(...), jax.devices()[1])`` 와 같은 경우 실제로
``jax.devices()[1]`` 에 0의 배열을 생성하는 대신 기본 장치에서 배열을 생성한 다음 이동하는
일부 지연 생성이 있었습니다. 하지만 이 최적화는 구현을 단순화하기 위해 제거되었습니다.)

(2020년 4월 현재, :func:`jax.jit` 은 장치 배치에 영향을 미치는 `device` 매개변수를 가지고 있습니다.
그 매개변수는 실험적이며 제거되거나 변경될 가능성이 높으며, 사용하는 것은 권장되지 않습니다.)

실제 예를 통해 알아보고 싶다면,
`multi_device_test.py <https://github.com/google/jax/blob/main/tests/multi_device_test.py>`_ 의 ``test_computation_follows_data`` 를 읽어보는 것을 추천합니다.

.. _faq-benchmark:

JAX 코드 벤치마킹
---------------------

당신은 방금 NumPy/SciPy에서 JAX로 복잡한 함수를 이식했습니다. 이 작업이 실제로 속도를 높였을까요?

JAX를 사용하여 코드의 속도를 측정할 때 NumPy와의 다음과 같은 중요한 차이점을 염두하세요:

1. **JAX 코드는 Just-In-Time (JIT)으로 컴파일**됩니다.
   JAX로 작성된 대부분의 코드는 JIT 컴파일을 지원하는 방식으로 작성될 수 있으며, 이는 코드를 훨씬 *더 빠르게* 실행할 수 있게 합니다(`To JIT or not to JIT`_ 참조).
   JAX에서 최대 성능을 얻으려면, 가장 바깥쪽 함수 호출에 :func:`jax.jit` 를 적용해야 합니다.
   JAX 코드를 처음 실행할 때는 컴파일되기 때문에 느릴 것임을 명심하세요.
   이는 자신의 코드에서 ``jit`` 를 사용하지 않더라도 마찬가지입니다. 왜냐하면 JAX의 내장 함수도 JIT 컴파일되기 때문입니다.
2. **JAX는 비동기 디스패치를 가집니다**. 이는 계산이 실제로 일어났는지를 보장하기 위해 ``.block_until_ready()`` 를 호출해야 함을 의미합니다(:ref:`async-dispatch` 참조).
3. **JAX는 기본적으로 32비트 데이터 타입만을 사용**합니다.
   공정한 비교를 위해 NumPy에서 명시적으로 32비트 데이터 타입을 사용하거나 JAX에서 64비트 데이터 타입을 활성화할 수 있습니다(`Double (64 bit) precision`_ 참조).
4. **CPU와 가속기 사이의 데이터 전송에는 시간이 걸립니다**.
   함수를 평가하는 데 걸리는 시간만을 측정하고 싶다면, 먼저 데이터를 실행하고자 하는 장치로 전송해야 할 수도 있습니다(:ref:`faq-data-placement` 참조).

JAX 대 NumPy를 비교하기 위한 마이크로벤치마크를 모든 이러한 요령을 적용하여 구성하는 예시는 다음과 같습니다.
IPython의 편리한 `%time and %timeit magics`_ 을 사용합니다::

    import numpy as np
    import jax.numpy as jnp
    import jax

    def f(x):  # function we're benchmarking (works in both NumPy & JAX)
      return x.T @ (x - x.mean(axis=0))

    x_np = np.ones((1000, 1000), dtype=np.float32)  # same as JAX default dtype
    %timeit f(x_np)  # measure NumPy runtime

    %time x_jax = jax.device_put(x_np)  # measure JAX device transfer time
    f_jit = jax.jit(f)
    %time f_jit(x_jax).block_until_ready()  # measure JAX compilation time
    %timeit f_jit(x_jax).block_until_ready()  # measure JAX runtime

Colab_\ 에서 GPU와 함께 실행했을 때, 확인할 수 있습니다:

- NumPy는 CPU에서 평가당 16.2 ms가 걸립니다
- JAX는 NumPy 배열을 GPU로 복사하는 데 1.26 ms가 걸립니다
- JAX는 함수를 컴파일하는 데 193 ms가 걸립니다
- JAX는 GPU에서 평가당 485 µs가 걸립니다

이 경우, 데이터가 전송되고 함수가 컴파일된 후, JAX는 GPU에서 반복 평가에 대해 약 30배 빠르다는 것을 볼 수 있습니다.

이것이 공정한 비교일까요? 아마도 그렇습니다.
궁극적으로 중요한 성능은 전체 애플리케이션을 실행할 때의 것이며, 이는 어느 정도 데이터 전송과 컴파일을 포함하게 됩니다.
또한, 우리는 JAX/가속기 대 NumPy/CPU의 증가된 오버헤드를 상쇄하기에 충분히 큰 배열(1000x1000)과 충분히 집중적인 계산(``@`` 연산자는 행렬-행렬 곱셈을 수행함)을 선택하는 데 주의를 기울였습니다.
예를 들어, 이 예제를 10x10 입력으로 전환하면, JAX/GPU는 NumPy/CPU보다 10배 느리게 실행됩니다(100 µs 대 10 µs).

.. _To JIT or not to JIT: https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html#to-jit-or-not-to-jit
.. _Double (64 bit) precision: https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision
.. _`%time and %timeit magics`: https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-time
.. _Colab: https://colab.research.google.com/

.. _faq-jax-vs-numpy:

JAX가 NumPy보다 빠른가??
~~~~~~~~~~~~~~~~~~~~~~~~~
사용자들이 종종 벤치마크를 통해 시도해보려는 질문 중 하나는 JAX가 NumPy보다 빠른가 하는 것입니다; 패키지의 차이로 인해 간단한 대답은 없습니다.

대체적으로:

- NumPy 연산은 즉시 실행되며, 동기적으로만 처리되고 CPU에서만 실행됩니다.
- JAX 연산은 즉시 실행되거나 컴파일 후에 실행될 수 있습니다(:func:`jit` 내부에 있을 때);
  이들은 비동기적으로 디스패치되며(:ref:`async-dispatch` 참조), CPU, GPU, 또는 TPU에서 실행될 수 있으며, 각각은 매우 다른 지속적으로 발전하는 성능 특성을 가집니다.

이러한 구조적 차이로 인해 NumPy와 JAX 사이의 직접적인 벤치마크 비교를 의미있게 만들기 어렵습니다.

또한, 이러한 차이들은 패키지 사이의 다른 엔지니어링 초점을 이끌어냈습니다: 예를 들어, NumPy는 개별 배열 연산에 대한 호출당 디스패치 오버헤드를 줄이는 데 상당한 노력을 기울였습니다.
왜냐하면 NumPy의 계산 모델에서는 그 오버헤드를 피할 수 없기 때문입니다. 반면 JAX는 디스패치 오버헤드를 피할 수 있는 여러 방법(JIT 컴파일, 비동기 디스패치, 배치 변환 등)을 가지고 있으므로,
호출당 오버헤드를 줄이는 것이 덜 우선순위였습니다.

이 모든 것을 염두에 두고, 요약하자면: CPU에서 개별 배열 연산의 마이크로벤치마크를 수행하는 경우, NumPy는 호출당 디스패치 오버헤드가 더 낮기 때문에 일반적으로 JAX보다 더 나은 성능을 기대할 수 있습니다.
GPU나 TPU에서 코드를 실행하거나, CPU에서 더 복잡한 JIT 컴파일된 연산 시퀀스를 벤치마킹하는 경우, 일반적으로 JAX가 NumPy보다 더 나은 성능을 기대할 수 있습니다.

.. _faq-different-kinds-of-jax-values:

JAX 값의 다양한 종류
-----------------------------

함수를 변환하는 과정에서 JAX는 일부 함수 인자들을 특별한 트레이서(tracer) 값으로 대체합니다.

이것은 ``print`` 문을 사용하면 볼 수 있습니다::

  def func(x):
    print(x)
    return jnp.cos(x)

  res = jax.jit(func)(0.)

위 코드는 정확한 값을 ``1.`` 로 반환하지만,
``x`` 의 값에 대해 ``Traced<ShapedArray(float32[])>`` 라고 출력합니다.
일반적으로 JAX는 이러한 트레이서 값들을 내부적으로 투명하게 처리합니다.
예를 들어, ``jax.numpy`` 함수를 구현하는 데 사용되는 수치적 JAX 원시 연산에서 그렇습니다.
이것이 위 예제에서 ``jnp.cos`` 가 작동하는 이유입니다.

더 정확히 말하자면, **트레이서** 값은 JAX 변환된 함수의 인자에 대해 도입됩니다.
단, :func:`jax.jit` 의 ``static_argnums`` 나 :func:`jax.pmap` 의 ``static_broadcasted_argnums`` 와 같은 특별한 매개변수로 식별된 인자들을 제외합니다.
일반적으로 적어도 하나의 트레이서 값을 포함하는 계산은 트레이서 값을 생성합니다.
트레이서 값 외에도 **일반** 파이썬 값이 있습니다. 즉, JAX 변환 외부에서 계산되거나, 특정 JAX 변환의 앞서 언급한 정적 인자로부터 발생하거나,
오로지 다른 일반 파이썬 값들로부터만 계산된 값들입니다. JAX 변환의 부재에서는 어디에서나 사용되는 값들입니다.

트레이서 값은 배열의 형태와 dtype에 대한 정보를 포함하는 **추상적** 값, 예를 들어, ``ShapedArray`` 를 운반합니다.
이런 트레이서들을 **추상 트레이서**라고 합니다. 일부 트레이서들,
예를 들어, 자동 미분 변환의 인자들에 대해 도입된 것들은 실제 배열 데이터를 포함하는 ``ConcreteArray`` 추상 값들을 운반하며
예를 들어, 조건문을 해결하는 데 사용됩니다.
이런 트레이서들을 **구체적 트레이서**라고 합니다. 이러한 구체적 트레이서로부터 계산된 트레이서 값들은 정규 값들과 결합할 수 있습니다.
**구체적 값**은 일반 값이거나 구체적 트레이서입니다.

대부분의 경우 트레이서 값에서 계산된 값들은 자체적으로 트레이서 값입니다.
완전히 추상 값에 의해 계산될 수 있는 몇 안 되는 예외가 있습니다.
이 경우 결과는 일반 값일 수 있습니다. 예를 들어, ``ShapedArray`` 추상 값이 있는 트레이서의 형태를 얻는 것입니다.
또 다른 예는 구체적 트레이서 값을 명시적으로 일반 유형으로 캐스팅할 때입니다. 예를 들어, ``int(x)`` 또는 ``x.astype(float)`` 입니다.
``bool(x)`` 의 경우도 마찬가지이며, 구체성이 가능할 때 파이썬 bool을 생성합니다. 이 경우는 제어 흐름에서 자주 발생하기 때문에 특히 중요합니다.

변환들이 추상 또는 구체적 트레이서를 어떻게 도입하는지:

* :func:`jax.jit``: ``static_argnums`` 로 지정된 것을 제외한 모든 위치 인자에 대해 **추상적 트레이서**를 도입합니다. 이들은 정규 값으로 남아 있습니다.
* :func:`jax.pmap``: ``static_broadcasted_argnums`` 로 지정된 것을 제외한 모든 위치 인자에 대해 **추상적 트레이서**를 도입합니다.
* :func:`jax.vmap`, :func:`jax.make_jaxpr`, :func:`xla_computation`: 모든 위치 인자에 대해 **추상적 트레이서**를 도입합니다.
* :func:`jax.jvp` 와 :func:`jax.grad` 는 모든 위치 인자에 대해 **구체적 트레이서**를 도입합니다.
  예외는 이러한 변환이 외부 변환 내에 있고 실제 인자들이 자체적으로 추상적 트레이서인 경우입니다. 이 경우, 자동 미분 변환에 의해 도입된 트레이서들도 추상 트레이서입니다.
* 모든 고차 제어 흐름 원시 연산(:func:`lax.cond`, :func:`lax.while_loop`, :func:`lax.fori_loop`, :func:`lax.scan`)은 함수적을 처리할 때 JAX 변환이 진행 중이든 아니든 추상 트레이서를 도입합니다.

이 모든 것은 데이터를 기반으로 조건부 제어 흐름을 가지는 코드,
즉 오직 정규 파이썬 값으로만 작동할 수 있는 코드에 관련됩니다::

    def divide(x, y):
      return x / y if y >= 1. else 0.

:func:`jax.jit`를 적용하려면 ``y``가 정규 값으로 유지되도록 ``static_argnums=1`` 을 명시해야 합니다.
이는 ``y >= 1.`` 의 부울 표현식이 구체적 값(정규 또는 트레이서)을 요구하기 때문입니다. 명시적으로 ``bool(y >= 1.)``, ``int(y)``, 또는 ``float(y)`` 를 작성하는 경우에도 마찬가지입니다.

흥미롭게도, ``jax.grad(divide)(3., 2.)`` 는 작동합니다. 왜냐하면 :func:`jax.grad` 는 구체적 트레이서를 사용하고 ``y`` 의 구체적 값을 사용하여 조건을 해결하기 때문입니다.

.. _faq-donation:

버퍼 기부
---------------

JAX가 계산을 수행할 때 모든 입력과 출력에 대해 장치상의 버퍼를 사용합니다.
계산 후에 하나의 입력이 더 이상 필요 없고 그것이 출력 중 하나의 형태와 요소 유형과 일치한다면,
해당 입력 버퍼를 출력을 위해 기부하고자 한다는 것을 지정할 수 있습니다. 이는 기부된 버퍼의 크기만큼 실행에 필요한 메모리를 줄입니다.

다음과 같은 패턴이 있는 경우, 버퍼 기부를 사용할 수 있습니다::

   params, state = jax.pmap(update_fn, donate_argnums=(0, 1))(params, state)

이것은 불변인 JAX 배열에 대한 메모리 효율적인 함수적 업데이트를 수행하는 방법으로 생각할 수 있습니다.
계산의 경계 내에서 XLA는 이 최적화를 대신 수행할 수 있지만,
jit/pmap 경계에서는 기부하는 함수를 호출한 후에 기부된 입력 버퍼를 사용하지 않을 것이라고 XLA에게 보장해야 합니다.

이는 ``donate_argnums`` 매개변수를 :func:`jax.jit`, :func:`jax.pjit`, 및 :func:`jax.pmap` 함수에 사용함으로써 달성됩니다.
이 매개변수는 위치 인수 목록으로 들어가는 인덱스(0 기준)의 시퀀스입니다::

   def add(x, y):
     return x + y

   x = jax.device_put(np.ones((2, 3)))
   y = jax.device_put(np.ones((2, 3)))
   # Execute `add` with donation of the buffer for `y`. The result has
   # the same shape and type as `y`, so it will share its buffer.
   z = jax.jit(add, donate_argnums=(1,))(x, y)

현재 키워드 인수로 함수를 호출할 때 이 기능이 작동하지 않는다는 점에 유의하세요!
다음 코드는 어떠한 버퍼도 기부하지 않을 것입니다::

   params, state = jax.pmap(update_fn, donate_argnums=(0, 1))(params=params, state=state)

버퍼가 기부된 인수가 트리 구조인 경우, 그 구성 요소의 모든 버퍼가 기부됩니다::

   def add_ones(xs: List[Array]):
     return [x + 1 for x in xs]

   xs = [jax.device_put(np.ones((2, 3))), jax.device_put(np.ones((3, 4)))]
   # Execute `add_ones` with donation of all the buffers for `xs`.
   # The outputs have the same shape and type as the elements of `xs`,
   # so they will share those buffers.
   z = jax.jit(add_ones, donate_argnums=0)(xs)

계산에서 이후에 사용되는 버퍼를 기부하는 것은 허용되지 않으며,
JAX는 `y` 에 대한 버퍼가 기부된 후 유효하지 않게 되었기 때문에 오류를 발생시킬 것입니다::

   # Donate the buffer for `y`
   z = jax.jit(add, donate_argnums=(1,))(x, y)
   w = y + 1  # Reuses `y` whose buffer was donated above
   # >> RuntimeError: Invalid argument: CopyToHostAsync() called on invalid buffer

기부된 버퍼가 사용되지 않는 경우, 예를 들어 출력에 사용될 수 있는 기부된 버퍼보다 더 많은 경우 경고를 받게 됩니다::

   # Execute `add` with donation of the buffers for both `x` and `y`.
   # One of those buffers will be used for the result, but the other will
   # not be used.
   z = jax.jit(add, donate_argnums=(0, 1))(x, y)
   # >> UserWarning: Some donated buffers were not usable: f32[2,3]{1,0}

기부가 출력의 형태와 일치하지 않는 경우에도 기부가 사용되지 않을 수 있습니다::

   y = jax.device_put(np.ones((1, 3)))  # `y` has different shape than the output
   # Execute `add` with donation of the buffer for `y`.
   z = jax.jit(add, donate_argnums=(1,))(x, y)
   # >> UserWarning: Some donated buffers were not usable: f32[1,3]{1,0}

기울기에는 ``where`` 를 사용할 때 `NaN` 이 포함될 수 있습니다.
------------------------------------------------

정의되지 않은 값을 피하기 위해 ``where`` 를 사용하여 함수를 정의하는 경우,
주의를 기울이지 않으면 역방향 미분에 대해 ``NaN`` 을 얻을 수 있습니다::

  def my_log(x):
    return jnp.where(x > 0., jnp.log(x), 0.)

  my_log(0.) ==> 0.  # Ok
  jax.grad(my_log)(0.)  ==> NaN

간단히 설명하면, ``grad`` 계산 중에 정의되지 않은 ``jnp.log(x)`` 에 해당하는 수반(adjoint)은 ``NaN`` 이고
``jnp.where`` 의 수반(adjoint)에 누적됩니다. 이러한 함수를 작성하는 올바른 방법은
부분적으로 정의된 함수 *내부에* ``jnp.where`` 가 있는지 확인하여 수반(adjoint)이 항상 유한한지 확인하는 것입니다::

  def safe_for_grad_log(x):
    return jnp.log(jnp.where(x > 0., x, 1.))

  safe_for_grad_log(0.) ==> 0.  # Ok
  jax.grad(safe_for_grad_log)(0.)  ==> 0.  # Ok

원래의 것 외에도 내부 ``jnp.where`` 가 필요할 수 있습니다. 예를 들어::

  def my_log_or_y(x, y):
    """Return log(x) if x > 0 or y"""
    return jnp.where(x > 0., jnp.log(jnp.where(x > 0., x, 1.), y)


추가 읽을 자료:

  * `Issue: gradients through jnp.where when one of branches is nan <https://github.com/google/jax/issues/1052#issuecomment-514083352>`_.
  * `How to avoid NaN gradients when using where <https://github.com/tensorflow/probability/blob/master/discussion/where-nan.pdf>`_.


입력이 정렬 순서에 따라 다르게 처리되는 함수에 대해 왜 기울기가 0인가?
---------------------------------------------------------

입력의 상대적 순서에 의존하는 연산(예: ``max``, ``greater``, ``argsort`` 등)을 사용하여 입력을 처리하는 함수를 정의하면,
기울기가 어디에서나 0임을 발견하고 놀랄 수 있습니다.
예를 들어, `x` 가 음수일 때 `0` 을 반환하고, `x` 가 양수일 때 `1` 을 반환하는 단계 함수 `f(x)` 를 정의하는 경우가 있습니다::

  import jax
  import numpy as np
  import jax.numpy as jnp

  def f(x):
    return (x > 0).astype(float)

  df = jax.vmap(jax.grad(f))

  x = jnp.array([-1.0, -0.5, 0.0, 0.5, 1.0])

  print(f"f(x)  = {f(x)}")
  # f(x)  = [0. 0. 0. 1. 1.]

  print(f"df(x) = {df(x)}")
  # df(x) = [0. 0. 0. 0. 0.]

처음 보기에 기울기가 어디에서나 0인 것이 혼란스러울 수 있습니다:
결국 출력은 입력에 따라 변화하니, 기울기가 어떻게 0일 수 있는가? 그러나, 이 경우 0이 올바른 결과입니다.

왜 그럴까요? 미분이 측정하는 것은 ``x`` 에 대한 무한히 작은 변화에 따른 ``f`` 의 변화량을 측정하는 것을 기억하세요.
``x=1.0`` 일 때, ``f`` 는 ``1.0`` 을 반환합니다. 우리가 ``x`` 를 약간 더 크거나 작게 만든다면, 이것은 출력을 변경시키지 않으므로, 정의에 따라 :code:`grad(f)(1.0)` 은 0이어야 합니다.
이 같은 논리는 ``f`` 의 모든 값이 0보다 클 때 모두 적용됩니다: 무한히 작은 입력을 변화시켜도 출력은 변하지 않으므로, 기울기는 0입니다.
마찬가지로, ``x`` 의 모든 값이 0보다 작을 때, 출력은 0입니다. ``x`` 를 변화시켜도 이 출력은 변하지 않으므로, 기울기는 0입니다.
이것은 ``x=0`` 인 까다로운 경우에도 보입니다. 분명히, 만약 당신이 ``x`` 를 위로 변화시킨다면,
출력을 변경시킬 것입니다, 그러나 이것은 문제가 됩니다: ``x`` 에 대한 무한히 작은 변화가 함수 값에 유한한 변화를 일으키는 것을 의미하며, 이는 기울기가 정의되지 않았음을 의미합니다.
다행히도, 이 경우에 기울기를 측정하는 또 다른 방법이 있습니다: 우리는 함수를 아래로 변화시키는데, 이 경우 출력은 변하지 않으므로 기울기는 0입니다.
JAX와 다른 자동 미분 시스템들은 이런 방식으로 불연속성을 처리하는 경향이 있습니다: 양의 기울기와 음의 기울기가 일치하지 않지만, 하나는 정의되어 있고 다른 하나는 그렇지 않다면, 우리는 정의된 것을 사용합니다.
이 기울기의 정의에 따라, 수학적으로나 수치적으로 이 함수의 기울기는 어디에서나 0입니다.

문제는 함수가 ``x = 0`` 에서 불연속성을 가지고 있다는 데에서 비롯됩니다.
여기서 ``f`` 는 본질적으로 `Heaviside Step Function`_ 입니다, 그리고 우리는 `Sigmoid Function`_ 을 매끄러운 대체물로 사용할 수 있습니다.
시그모이드는 x가 0에서 멀 때 헤비사이드 함수와 대략적으로 동일하지만, ``x = 0`` 에서의 불연속성을 매끄럽고 미분 가능한 곡선으로 대체합니다.
:func:`jax.nn.sigmoid` 를 사용함으로써, 우리는 잘 정의된 기울기를 가진 유사한 계산을 얻습니다:

  def g(x):
    return jax.nn.sigmoid(x)

  dg = jax.vmap(jax.grad(g))

  x = jnp.array([-10.0, -1.0, 0.0, 1.0, 10.0])

  with np.printoptions(suppress=True, precision=2):
    print(f"g(x)  = {g(x)}")
    # g(x)  = [0.   0.27 0.5  0.73 1.  ]

    print(f"dg(x) = {dg(x)}")
    # dg(x) = [0.   0.2  0.25 0.2  0.  ]

:mod:`jax.nn` 하위 모듈은 또한 다른 일반적인 순위 기반 함수들의 매끄러운 버전을 가지고 있습니다.
예를 들어 :func:`jax.nn.softmax` 은 :func:`jax.numpy.argmax` 의 사용을 대체할 수 있고,
:func:`jax.nn.soft_sign` 은 :func:`jax.numpy.sign` 의 사용을 대체할 수 있으며,
:func:`jax.nn.softplus` 또는 :func:`jax.nn.squareplus` 는 :func:`jax.nn.relu` 의 사용을 대체할 수 있습니다. +@

JAX Tracer를 NumPy 배열로 어떻게 변환할 수 있나요?
------------------------------------------------
런타임에 변환된 JAX 함수를 검사하면, 배열 값들이 :class:`~jax.core.Tracer` 객체로 대체된 것을 확인할 수 있습니다::

  @jax.jit
  def f(x):
    print(type(x))
    return x

  f(jnp.arange(5))

이는 다음과 같이 출력됩니다::

  <class 'jax.interpreters.partial_eval.DynamicJaxprTracer'>

자주 나오는 질문은 이러한 트레이서를 일반 NumPy 배열로 어떻게 되돌릴 수 있는가입니다.
간단히 말해서, **Tracer를 NumPy 배열로 변환하는 것은 불가능**합니다.
왜냐하면 트레이서는 주어진 형태와 dtype을 가진 모든 가능한 값의 추상적 표현이며, NumPy 배열은 그 추상 클래스의 구체적인 멤버이기 때문입니다.
JAX 변환 내에서 트레이서가 어떻게 작동하는지에 대한 더 자세한 토론을 원한다면, `JIT mechanics`_ 을 참조하세요.

Tracer를 배열로 되돌리려는 문제는 일반적으로 런타임에 계산의 중간 값을 접근하는 것과 관련된 다른 목표 내에서 제기됩니다. 예를 들어:

- 런타임에 디버깅 목적으로 추적된 값을 출력하고 싶다면, :func:`jax.debug.print` 를 사용하는 것을 고려해볼 수 있습니다.
- 변환된 JAX 함수 내에서 비-JAX 코드를 호출하고 싶다면, :func:`jax.pure_callback` 을 사용하는 것을 고려해볼 수 있으며, 이에 대한 예시는 `Pure callback example`_ 에서 확인할 수 있습니다.
- 런타임에 배열 버퍼를 입력하거나 출력하고 싶다면 (예를 들어, 파일에서 데이터를 로드하거나, 배열의 내용을 디스크에 로깅하고 싶은 경우),
  :func:`jax.experimental.io_callback` 을 사용하는 것을 고려해볼 수 있으며, 이에 대한 예시는 `IO callback example`_ 에서 찾을 수 있습니다.

런타임 콜백 및 그 사용 예시에 대한 자세한 정보를 원한다면, `External callbacks in JAX`_ 을 참조하세요.

.. _JIT mechanics: https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html#jit-mechanics-tracing-and-static-variables
.. _External callbacks in JAX: https://jax.readthedocs.io/en/latest/notebooks/external_callbacks.html
.. _Pure callback example: https://jax.readthedocs.io/en/latest/notebooks/external_callbacks.html#example-pure-callback-with-custom-jvp
.. _IO callback example: https://jax.readthedocs.io/en/latest/notebooks/external_callbacks.html#exploring-jax-experimental-io-callback
.. _Heaviside Step Function: https://en.wikipedia.org/wiki/Heaviside_step_function
.. _Sigmoid Function: https://en.wikipedia.org/wiki/Sigmoid_function
.. _algebraic_simplifier.cc: https://github.com/tensorflow/tensorflow/blob/v2.10.0/tensorflow/compiler/xla/service/algebraic_simplifier.cc#L3266
