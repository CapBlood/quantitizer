# quantitizer
Пакет quantitizer позволяет сжимать модели эмбеддингов посредством квантования матриц.

## Особенности
- Доступна реализация квантования с использованием CUDA (при помощи [RAPIDS](https://rapids.ai/)).
- Частичная интеграция с библиотекой [gensim](https://radimrehurek.com/gensim/) (fasttext и word2vec).
- Загрузка уже ранее сжатых моделей и использование их в своих проектах.

## Установка
```
pip install quantitizer
```

Для работы с реализацией квантования
при помощи CUDA необходимо также установить [cuML](https://github.com/rapidsai/cuml). 


## Примеры

### Пример сжатия массива numpy
На CPU:
```python
import numpy as np

from quantitizer import quantitize 

matrix = np.random.random((50000, 1000))
qmatrix = quantitize(matrix, sub_size=5)
matrix.nbytes // 1024 // 1024 # В мегабайтах
>> 381

qmatrix.nbytes // 1024 // 1024
>> 2
```

На GPU:
```python
import numpy as np

from quantitizer.cuda import quantitize_cuda 

matrix = np.random.random((50000, 1000))
qmatrix = quantitize_cuda(matrix, sub_size=5)
matrix.nbytes // 1024 // 1024 # В мегабайтах
>> 381

qmatrix.nbytes // 1024 // 1024
>> 2
```

### Пример сжатия модели fasttext
```python
from gensim.models import FastText

from quantitizer.integration.gensim.fasttext import quantitize_ft

ft = FastText.load_fasttext_format("../data/cc.en.300.bin").wv
ft_compressed = quantitize_ft(ft, 2)
```

### Пример использования сжатой модели fasttext
```python
from quantitizer.pretrain import load
from quantitizer.integration.gensim.fasttext import load_ft

load("fasttext-compressed-en-100")
ft = load_ft("fasttext_compressed_en_100")
vec = ft.get_vector("word")
```

### Пример использования сжатой модели word2vec

```python
from quantitizer.pretrain import load
from quantitizer.integration.gensim.word2vec import load_wv

load("w2v-compressed-en-100")
wv = load_wv("data/w2v-compressed-en-100")
vec = wv.get_vector("computer")
```

## Доступные модели

| Название                     |                       Описание                       |  Размер  | Совпадение с оригиналом, % |
|:-----------------------------|:----------------------------------------------------:|:--------:|:--------------------------:|
| `fasttext-compressed-en-100` | Английская версия fasttext, сжатая с разбиением 100. | ~ 423 Мб |            ~ 94            |
| `w2v-compressed-en-100`      | Английская версия word2vec, сжатая с разбиением 100. | ~ 348 Мб |            ~ 95            |



## Эксперименты

### Fasttext
Здесь описывается эксперимент сжатия модели эмбеддингов fasttext
английского языка доступной по ссылке https://fasttext.cc/docs/en/crawl-vectors.html, оригинальный размер которой ~ 7215 Мб.

Конфигурации:

| Тип вычислительного устройства |     Характеристики     |
|:------------------------------:|:----------------------:|
|              CPU               |      Apple M1 Pro      |
|              GPU               | NVIDIA RTX 3070 (8 GB) |

Результаты экспериментов:

| Количество разбиений |          CPU          |          GPU          | Объём памяти в RAM, Мб |   Точность   |
|:--------------------:|:---------------------:|:---------------------:|:----------------------:|:------------:|
|          2           | ~ 1 минута 4 секунды  |      ~ 14 секунд      |        ~ 269.36        |    ~ 0.40    |
|         100          | ~ 61 минута 30 секунд | ~ 4 минуты 51 секунда |        ~ 642.82        |    ~ 0.94    |


## Разработка
### Развертывание среды
Для установки всех модулей, необходимых для разработки данного пакета, необходимо использовать пакетные менеджер 
[Poetry](https://python-poetry.org/). Затем необходимо использовать команду `poetry install` для установки
всех необходимых зависимостей.

### Тестирование
Для запуска всех тестов используется команда `poetry run test_all`.

### Сборка
Сборка пакета осуществляется командой `poetry build`.

## Ссылки
- [Подробнее](http://mccormickml.com/2017/10/13/product-quantizer-tutorial-part-1/) о product quantization
- [Пример](http://ethen8181.github.io/machine-learning/deep_learning/multi_label/product_quantization.html#Computing-Query-Distance)
- О [сжатии](https://habr.com/ru/post/489474/) моделей
