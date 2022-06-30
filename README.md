# quantitizer
Пакет quantitizer позволяет проводить сжимать модели машинного обучения, 
например, модели эмбеддингов посредством квантования матриц.

## Особенности
- Доступна реализация квантования с использованием CUDA (при помощи [RAPIDS](https://rapids.ai/)).
- Интеграция с библиотекой [gensim](https://radimrehurek.com/gensim/) (на данный момент только модель fasttext).

## Примеры
```python
>> matrix = np.random.random((50000, 1000))
>> qmatrix = quantitize(matrix, sub_size=5)
>> matrix.nbytes // 1024 // 1024 # В мегабайтах
>> 381
>> qmatrix.nbytes // 1024 // 1024
>> 2
```

## Эксперименты

### Fasttext
Здесь описывается эксперимент сжатия модели эмбеддингов fasttext английского языка доступной по ссылке https://fasttext.cc/docs/en/crawl-vectors.html.

Конфигурации:

| Тип вычислительного устройства |     Характеристики     |
|:------------------------------:|:----------------------:|
|              CPU               |      Apple M1 Pro      |
|              GPU               | NVIDIA RTX 3070 (8 GB) |

Результаты экспериментов:

| Количество разбиений |          CPU          |          GPU          | Объём памяти, Мб |   Точность   |
|:--------------------:|:---------------------:|:---------------------:|:----------------:|:------------:|
|          2           | ~ 1 минута 4 секунды  |      ~ 14 секунд      |     ~ 269.36     |    ~ 0.40    |
|         100          | ~ 61 минута 30 секунд | ~ 4 минуты 51 секунда |     ~ 642.82     |    ~ 0.94    |


## TODO
- [x] Сделать установку пакета с помощью pip.
- [ ] Реализовать классы-замены для подмены numpy array и pytorch tensor.
- [x] Ускорить quantitize.
- [x] Привести примеры использования для квантизации моделей fasttext и [сравнить](https://vasnetsov93.medium.com/shrinking-fasttext-embeddings-so-that-it-fits-google-colab-cd59ab75959e) качество.

## Ссылки
- [Подробнее](http://mccormickml.com/2017/10/13/product-quantizer-tutorial-part-1/) о product quantization
- [Пример](http://ethen8181.github.io/machine-learning/deep_learning/multi_label/product_quantization.html#Computing-Query-Distance)
- О [сжатии](https://habr.com/ru/post/489474/) моделей