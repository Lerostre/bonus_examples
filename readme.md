Я вообще хотел бы накидать индивидуальных заданий для всех, кто захочет бонуса, мне кажется, таких немного, а те, что меня спросили, ребята шарящие - мы по мо1 знакомы. И я бы оформил всё красиво в ТЗ, что я хочу видеть. В общем случае это хотя бы пара-тройка вещей, которые проходятся по курсу, и хотя бы какая-то вещь, которой в курсе нет. Пока у меня были такие мысли

```
postponed
```

1. Написать несколько алгоритмов для поиска с расчётом, чтобы у них:
   - был базовый или абстрактный класс, который умеет хоть что-то, что уже умеет генсим - вывод самых похожих слов, подсчёт расстояния через метод, такие штуки. Придётся перегружать операторы как минимум
   - был одинаковый интерфейс, вне зависимости от имплементации, придётся поиграться с наследованием
   - было хотя бы 2 простеньких алгоритма, чтобы был какой-нибудь осмысленный код, над которым можно посидеть, пооптимизировать и подокапываться. Здесь для этого придётся в спарс матрицы лезть
   - в идеале чтобы был ещё миксин, потому что функции подсчёта расстояния из Tfidf для BM25 не годятся, зато нормально смотрятся в чём-то типа коллаборативной фильтрации или факторизационной машины
   - вполне возможно, что пособирать мусор тоже придётся, чтобы класс весил не так много
  
2. Написать кастомный w2v с большинством функций генсима, например, адаптивным lr, правильным neg sampling'ом и так далее. В идеале это должно быть продолжение 1, но мне кажется, это будет довольно сложно.
   - есть базовый класс, который умеет всё то же, что базовый класс в 1
   - есть кастомный даталоадер, да, там тоже не всё так просто, в идеале, он бы ещё с файла умел читать, ещё и с генераторами придётся чуть-чуть повозиться
   - есть оптимизация кода, хотя бы распараллеливание, мне кажется, вещь очень важная, было бы прикольно, чтобы в ней ращобрались
   - ну и сам w2v на самом деле не так легко реализовать эффективно, там ещё и градиентный спуск придётся выводить, кстати тоже можно отдельным классом

```
preprocessor
```
  
3. Трансформер для препроцессинга текстов. В идеале это должен быть класс, совместимый с `ColumnTransformer` из `sklearn`, чтобы он:
   - применялся к одной или нескольким колонкам, как того требует `ColumnTransformer`
   - наследовался от склёрновских миксинов, но там несложно
   - умел в многопоточность, иначе на больших данных будет грустно
   - поддерживал не только русский, можно наверное, тоже сделать базовый класс и класс по языкам
   - имел синтаксис такой же, как склёрновский трансформер
   - умел принимать не только датафрейм, но и матрицу, тут либо украсть `check_X_y` из склёрна, что тоже познавательно, его ещё найти надо, либо трай эксептом овладеть

```
rfe
```

4. Есть у меня один feature selector, который надо довести до ума. Он должен уметь:
   - корректно работать, как алгоритм Recursive Feature Elimination
   - уметь работать с любой метрикой, которую определит сам пользователь (склёрн, например, не умеет), нужно будет ещё обработать случаи `predict`, `predict_proba`
   - уметь работать с любой моделью, вот тут совсем сложно, потому что у бустингов сильно разные кварги, с ними ещё как-то можно справиться, но на обычные модели из склёрна они совершенно не похожи, там недо будет думать, скорее всего наследоваться
   - записывал логи и красиво рисовал, ну это не сложно
   - умел принимать любой набор параметров для фита и feature importance, тоже несложно, но надо маленько подумать
  
Ну и что-нибудь ещё тоже можно придумать. Для ценителей нейросетей можно скинуть задачку закодить бэкпроп, там можно хорошо так посидеть. Можно попробовать какую-нибудь библиотеку просто сделать с нуля, по крайней мере какой-то ключевой функционал, как тот же генсим вверху. Но надо думать, а я даже не уверен, что мои идеи вообще кому-то нужны
