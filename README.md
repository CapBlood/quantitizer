# quantitizer
Пакет quantitizer позволяет проводить сжимать модели машинного обучения, 
например, модели эмбеддингов посредством квантования матрицы.


## Examples
```python
>> matrix = np.random.random((50000, 1000))
>> qmatrix = quantitize(matrix, sub_size=5)
>> matrix.nbytes // 1024 // 1024 # В мегабайтах
>> 381
>> qmatrix.nbytes // 1024 // 1024
>> 2
```

## TODO
- [ ] Сделать установку пакета с помощью pip.
- [ ] Реализовать классы-замены для подмены numpy array и pytorch tensor.
- [ ] Ускорить quantitize.
- [ ] Привести примеры использования для квантизации моделей fasttext и [сравнить](https://vasnetsov93.medium.com/shrinking-fasttext-embeddings-so-that-it-fits-google-colab-cd59ab75959e) качество.

## References
- [Подробнее](http://mccormickml.com/2017/10/13/product-quantizer-tutorial-part-1/) о product quantization
- [Пример](http://ethen8181.github.io/machine-learning/deep_learning/multi_label/product_quantization.html#Computing-Query-Distance)
- О [сжатии](https://habr.com/ru/post/489474/) моделей