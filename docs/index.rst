JAX: 고성능 배열 컴퓨팅
=====================================

JAX는 고성능 수치 계산을 위해 `Autograd`_와 `XLA`_를 결합한 것입니다.

If you're looking to train neural networks, use Flax_ and start with its documentation.
Some associated tools are Optax_ and Orbax_.
For an end-to-end transformer library built on JAX, see MaxText_.

.. grid:: 3
   :margin: 0
   :padding: 0
   :gutter: 0

   .. grid-item-card:: 익숙한 API
      :columns: 12 6 6 4
      :class-card: sd-border-0
      :shadow: None

      JAX는 연구자와 엔지니어들이 쉽게 적응할 수 있도록 NumPy 스타일의 익숙한 API를 제공합니다.

   .. grid-item-card:: 변환
      :columns: 12 6 6 4
      :class-card: sd-border-0
      :shadow: None

      JAX는 컴파일, 배치 처리, 자동 미분 및 병렬화를 위한 조립 가능한 함수 변환 기능이 포함되어 있습니다.

   .. grid-item-card:: 어디서든 실행
      :columns: 12 6 6 4
      :class-card: sd-border-0
      :shadow: None

      동일한 코드가 CPU, GPU, 및 TPU를 포함한 여러 백엔드에서 실행됩니다.

.. grid:: 3

    .. grid-item-card:: :material-regular:`rocket_launch;2em` 시작하기
      :columns: 12 6 6 4
      :link: beginner-guide
      :link-type: ref
      :class-card: getting-started

    .. grid-item-card:: :material-regular:`library_books;2em` 사용자 가이드
      :columns: 12 6 6 4
      :link: user-guides
      :link-type: ref
      :class-card: user-guides

    .. grid-item-card:: :material-regular:`laptop_chromebook;2em` 개발자 문서
      :columns: 12 6 6 4
      :link: contributor-guide
      :link-type: ref
      :class-card: developer-docs


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: 시작하기

   installation
   notebooks/quickstart
   notebooks/thinking_in_jax
   notebooks/Common_Gotchas_in_JAX
   faq

.. toctree::
   :hidden:
   :maxdepth: 1

   jax-101/index


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: 더 많은 자료

   user_guides
   advanced_guide
   contributor_guide
   building_on_jax
   notes
   jax


.. toctree::
   :hidden:
   :maxdepth: 1

   changelog
   glossary


.. _Autograd: https://github.com/hips/autograd
.. _XLA: https://openxla.org/xla
.. _Flax: https://flax.readthedocs.io/
.. _Orbax: https://orbax.readthedocs.io/
.. _Optax: https://optax.readthedocs.io/
.. _MaxText: https://github.com/google/maxtext/
