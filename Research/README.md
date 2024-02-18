### Оркестрация векторной базы данных для применения в LLM моделях на примере FAQ бота с использованием базы знаний по библиотеке Langchain.

<div align="center">
<table>
 <tr>
    <td><b style="font-size:30px">Бараненков Сергей</b></td>
    <td><b style="font-size:30px">Демочкина Полина</b></td>
    <td><b style="font-size:30px">Тихобаев Илья</b></td>
     <td><b style="font-size:30px">Яхтин Леонид</b></td>
 </tr>
 <tr>
    <td>Высшая Школа Экономики</td>
    <td>Высшая Школа Экономики</td>
    <td>Высшая Школа Экономики</td>
    <td>Высшая Школа Экономики</td>
 </tr>
<tr>
    <td>Нижний Новгород</td>
    <td>Нижний Новгород</td>
    <td>Нижний Новгород</td>
    <td>Нижний Новгород</td>
</tr>
<tr>
    <td>ssbaranenkov@edu.hse.ru</td>
    <td>pvdemochkina@edu.hse.ru</td>
    <td>istikhobaev@edu.hse.ru</td>
    <td>yakhtin.leonid@yandex.ru</td>
</tr>
</table>
</div>

**Аннотация**

В данной исследовательской работе рассматривается оркестрация векторных баз данных для применения в больших лингвистических моделях (LLM) на примере разработки чат-бота FAQ, использующего базу знаний библиотеки Langchain. Основная цель исследования - создание чат-бота, интегрированного в платформу Telegram, который способен отвечать на вопросы, связанные с библиотекой Langchain. Этот бот использует векторную базу данных в сочетании с LLM для предоставления точных и контекстуально релевантных ответов. Ключевые технические особенности проекта включают использование GigaChat для генерации окончательных ответов и применение дополнительного контекста из векторной базы данных в процессе генерации. Проект следует систематическому подходу, начиная с тщательного изучения современных методов векторизации документов и особенностей современных векторных баз данных. Практическая реализация включает настройку LLM, развертывание векторной базы данных и тестирование качества семантического поиска в этой базе данных. В статье представлен обширный обзор современных методов векторизации документов, включая усреднение эмбеддингов слов, предобученные модели семейства BERT, модели семейства GPT и другие методы на основе нейронных сетей. Также рассматриваются тонкости современных векторных баз данных, таких как FAISS, с акцентом на различные типы индексов, включая Flat Index и HNSW Index. В разделе о практической реализации описывается создание пайплайна с использованием LangChain, детализируя шаги от подготовки данных до развертывания Телеграм-бота. Исследование завершается обзором дополнительных исследований, сравнивающих эмбеддинги и их эффективность в контексте релевантного поиска. Эта работа представляет новый подход к улучшению LLM с помощью внешних баз знаний, специально для русскоязычных наборов данных, и предлагает взгляд на эффективное использование векторных баз данных в приложениях, основанных на искусственном интеллекте.

**Ключевые слова**: LLM, GigaChat, LangChain, chat bot

### **1. Введение**

<img src="https://lh7-us.googleusercontent.com/OcnOGQ5TXUQ7ByzGpWuYC3vfmQUJekuPRkS4MpiD5Cqg5cEJd73NQlwlL6QEqJMnKDrZJtZnkhkyyV1MVcErcEwDi7vLLVSR43-iU_kOrcLmqWiPA9pg-h_WRK6Sx2YgR60XJVgMm-WBxSfSdWbbKHI" width="300">

**1.1. Набор данных**  
https://github.com/PolinaDemochkina/LangChain_bot/blob/main/Final/langchain_qa_dataset.csv

**1.2. Этапы выполнения работы**

1.  Исследование предметной области:  
    a. Современные методы векторизации документов  
    b. Особенности современных векторных баз данных  
     
2.  Практическая реализация прототипа системы:  
    a. Получение доступа к LLM  
    b. Разворачивание векторной базы данных  
    c. Тестирование качества семантического поиска в векторной базе данных  
    d. Создание демонстрационного приложения в виде Телеграм бота  
     

### **2. Обзор литературы**

Одним из современных трендов в области исследования больших лингвистических моделей (LLM) является персонализация ответа за счет генерации запроса с расширенным поиском (retrieval augmented generation) \[1\]. Для обеспечения более персонализированного ответа в доменной области необходимо решить проблему поиска семантически похожих сегментов текста из дополнительной базы знаний. Для этого предлагается использовать векторную базу данных \[2\].

  
**2.1. Современные методы векторизации документов**

1.  Усреднение эмбеддингов слов.  
    Эмбеддинг, полученный данным методом, является усредненением эмбеддингов всех слов в предложении. Сам метод является очень простым в реализации, однако он не позволяет уловить более сложные нюансы из контекста.
2.  Предобученные модели из семейства BERT моделей.  
    В отличие от предыдущего метода модели из семейства BERT учитывают контекст каждого слова в предложении, что приводит к созданию богатых и контекстно-зависимых вложений. Примером векторайзера из этой категории является SentenceBERT \[3\]. Его особенностью является механизм внимания, который позволяет создавать векторы, опираясь только на наиболее важные части фрагмента.
3.  Предобученные модели из семейства GPT моделей.  
    На данный момент векторизер ADA от OpenAI является state-of-the-art методом для получения эмбеддингов из текста. Его плюсами является расширенный контекст (8192 токенов) и уменьшенная размерность результирующего вектора (1536) \[5\]. Эта модель понимает специфический синтаксис, характерный для таких доменных областей как программирование (код), поэтому она идеально подойдет для нашей практической реализации проекта.
4.  Методы, основанные на других нейронных сетях  
    Особенностью векторайзеров из этой группы является то, что они обучены предсказывать окружающие предложения, что помогает им хорошо понимать семантику предложений. Примерами являются векторайзеры Skip-Thought vectors и InferSent \[4\]. Последний обучается в ходе решения задачи логического вывода, поэтому он отлично подходит для решения задачи перефразирования, семантической схожести и анализа сентимента. Однако минусом этого векторайзера является то, что он не всегда хорошо работает с данными в узкой доменной области.

**2.2. Особенности современных векторных баз данных**

2.2.1.  FAISS - популярная библиотека для оптимального индексирования эмбеддингов и поиска ближайшего соседа. В ней представлено большое количество различных вариантов индексов. \[10\]\[11\]  
2.2.2.  Flat Index  
    Представляет собой самый простой вид индекса, в котором эмбеддинги хранятся как есть. При получении запроса на поиск ближайшего эмбеддинга перебираются все имеющиеся точки. \[9\]  
    <img src="https://lh7-us.googleusercontent.com/-U1saP5xMGqJCYy6u8E6WfgMhD4kqWrkinAYNwfg-h6CfUZPpKimQfH7Q836rLX0eb-h-KfziSvwTYhYcGbqjMdabX46__r3PvfO8XRNB1cIG3KdSKnJdcHhQS7WwCow6mTmWOd2TzejBSfkiV4njz8" width="300"><img src="https://lh7-us.googleusercontent.com/WJuvTjmejvmKI4x6MjpK6PF9k2Iw_TyZGsdY48jF_RgKXoWMYQNvGLdMWLp_COIba87mVhgqVwIm3SVE3GQLuRiK_C5tx0dK-3nrsLuUE5K7dyzT0nESIOwz88vUvhhLFvy7KZm3fEQuFzu_7C2u2pk" width="300"><img src="https://lh7-us.googleusercontent.com/sP_o2b59sxs8rG1SXB81Ytb_ODQ0KrOHSYUj7If3oBFQ8NHrSo6d2LHmnVhdOQX3lSCW4Sfsmf7EAbVdBHwXKfLXBEKIsjXCEluME7vhltbwObqE5g8MH1EvZkwPu8FE_GQxvnE9vsy5eVRTS4P6kU4" width="300">  
2.2.3.  Inverted File Index  
    Более продвинутый индекс - в отличие от flat индекса пространство дополнительно разбивается на кластера, называемые voronoi ceils. Из-за этого сужается область поиска ближайших соседей, так как мы сначала определяем кластер и только после этого перебираем все эмбеддинги этого кластера. Индекс может давать неточные результаты если поисковый эмбеддинг находится где-то близко к пересечению кластеров. Степень ошибки можно регулировать, перебирая соседей целевого кластера. \[9\]  
    <img src="https://lh7-us.googleusercontent.com/PJfTRD9bwqplfkBJDyFgIlYdOZza8Eo9jsLj1qPvIA-6Ltq8q2R-Qch6NdhaMo4a2poG5-96uL2YlR-pTR5xunFLO7bZA9pCkga6_oEgnlFXVfTXruEgdAbX9gAssph6QuHDMgCPeSJkHrSyvmUEQnk" width="250"><img src="https://lh7-us.googleusercontent.com/BcgCFiX_DTOHBzQ36KUvIrdIaS5nbY55V_TNDuFVGYTMlEy6MKce8-kCRLkNXpXllnMKmlpK4hqBk4TvAe8tXJSPdphQIHbUz-XfnOrNuiqnQY-RjIFt81KfWhRDNrwZv7X_OOUZBRxY6P-uNoKAAmk" width="250"><img src="https://lh7-us.googleusercontent.com/3RzUYwMpr1QRhLqIN-p5RYa1BLBfFFn46lJ2f0BnpREPog12ihqXcxpWkPi_NczyMDJLw7NOIhp2lL3JFUX6qfa_vO609kRonggVngQI2Bp_f2-XqeBJcKku9bMA0Qz1DbvKKi_8JqEl9d-n8ZJdk-A" width="250">  
2.2.4.  HNSW Index  
    Является одним из наиболее продвинутых и популярных индексов, используемых в различных векторных и не только базах данных. Основан на свойствах тесного мира, которыми обладают большинство естественно созданных сетевых структур, что позволяет искать ближайшего соседа за логарифм. Для еще большего ускорения поиска к графу добавляются дублирующие уровни похожим на структуру данных skip list образом. Эта оптимизация позволяет существенно ограничить область поиска. \[6\]\[7\]\[8\] \[9\]  
    <img src="https://lh7-us.googleusercontent.com/OHd0BAaxLtV7X67GLq3Ge1UbykMmr5wxeL1baMTGtWHE_HZU_oRhN_o_Q2rhVCjzYG6S8AimhVj8r0Nw_gHsLpP2Jj29KdlZIwlEPoFEH4Z6NrvU2SrHf_PO3dYkzecrwjz8_kAAg29qhOCzLf6TLlc" width="250"><img src="https://lh7-us.googleusercontent.com/Z_hiJWgfFlzBqINvnRxiMo9FksbGAp9tOTwsmUKt_veUmwd0pIaGjp1dB5JVgD8SlLjfGErVZQL27xDA9QXKvJrOpIh-Ol7iomSldqXivC6eWh1xa0RNLNPSOfVKuY9Ewzx6HPQpCHDaidQReU3VQEI" width="250"><img src="https://lh7-us.googleusercontent.com/m3d30zV_fyPXQlKvsRl1h_XWa1XAMl8_KBd8xoFI1GxOnSgsuSGCCe9AvxA3BrBlkupxilr6URBc_5-hcvTda04xr6cqfKmGGVQzsf3Tq8gml_RwD65guXxlHGCjmi7iNVWLAlJElHShtf9chYoVmVE" width="250">  
2.2.5.  Product Quantization  
    Метод уменьшения размерности как размера эмбеддинга так и типа, необходимого для хранения каждого признака. Исходный эмбеддинг разбивается на m частей, создается m подпространств с заданным числом центроидов. После этого каждому из m эмбеддингов назначается id ближайшего центроида в своем подпространстве. \[9\]  
    <img src="https://lh7-us.googleusercontent.com/9djhLeBi98jxx3GNvfb9QDsAKO-vKvpFoRG2I2umrl50fy0CDNG-3i56wpqLmOmXrW5bw7dNRfI7KuoKaI02uTDFhb5pMa8I_VIhn2v0c8JhIB4lGeLbZmk_lGviNpRn1aNjxJ2RUjF6p3qKem79fHI" width="300"><img src="https://lh7-us.googleusercontent.com/zfyWG3XFwzl1CDuHNxKM_97KrKt69toKW1w_MeWeslLohbzeQEqm_cjGlyi9xE1OK-DChregBmRTuqPHQs7WvhBO9NGvFm1Ug6t8l2LQzbpX5gx5aXntmyzGl0RxxmtUfqS4yf8QJtGFMqu97Zaihhs" width="300">  
2.2.6.  Composite Indices  
    Кроме большинства популярных индексов в faiss возможно их смешивать между собой, регулируя таким образом качество, скорость поиска и количество занимаемой памяти. \[9\]  
    <img src="https://lh7-us.googleusercontent.com/ICS3XCHUXfqV7Y0V70ayFb4li5DvAyjKb4LcyPa09z4uceMseuFooeHrM9zuiNw4_JWyQ7fNRlvQCBNTApWBjPj3LSBxJZjB1oHagw0AsVNVlI6l-nFujkmap2Xm5U13VIQQ6yHSxpoAeoOxm0ky7Hc" width="300"><img src="https://lh7-us.googleusercontent.com/PafVOaLZD_AWqxnANI-7jvU9ivuaTP2aG8Ff8CpIRkrF5iRSnFA0YEaM6kvwNDRGIokQ0NVSyslfhO59SvxwTeNR2gfG6I6uiufnsB6YikHf3qYS1h1fAgmhmVs6YS9xYCkD4hWRIfaosJ-6vWaUZVw" width="300">  

### 3. Практическая реализация проекта

**3.1. Создание пайплайна с помощью LangChain**  
Для создания пайплайна проекта мы выбрали фреймворк LangChain, позволяющий реализовать адаптер для GigaChat, а также генерацию модифицированного промпта для лингвистической модели. На этапе подготовки данных необходимо разделить текст на фрагменты необходимой длины (512-2048 токенов оптимально для "подмешивания" в промпт). Так как все тексты в нашем датасете уже имели длину меньше 2048 токенов, их не пришлось разбивать на меньшие фрагменты. Однако стоит отметить что при необходимости можно воспользоваться функцией `RecursiveCharacterTextSplitter` из библиотеки LangChain. Далее с помощью API OpenAI были получены эмбеддинги текстов, включающие в себя как вопрос так и ответ на этот вопрос. Эти эмбеддинги были загружены в векторную базу данных и сохранены на диск для дальнейшего переиспользования в виде HNSW индекса. 

Стоит отметить, что система может работать только с датасетами вида вопрос-ответ, но и с любыми другими датасетами - для этого необходимо незначительно поменять код загрузки датасета. Однако в силу специфики использования GigaChat рекомендуется использовать русскоязычные датасеты.

Этап получения контекста для аугментации промпта состоит из нескольких шагов:
1. Опредеряем ретривер - интерфейс получения k ближайших векторов в векторной бд для искомого вектора. В нашем случае наиболее оптимальными гиперпараметрами оказались k = 5 и threshold = 0.3.
2. Определяем шаблон промпта - он должен включать вопрос пользователя и релевантный вопросу контекст. Также в промпте для LLM прописываем необходимость использования данных из представленного контекста.
![image](https://github.com/PolinaDemochkina/LangChain_bot/assets/50075840/5aa87e46-be75-4cf7-9241-631244261219)
3. Так как GigaChat имеет ограничение на количество токенов в запросе, использовались `LengthBasedExampleSelector` и `FewShotPromptTemplate`. С их помощью из итогового промпта автоматически вырезалось минимально возможное количество наименее релевантного контекста для удовлетворения ограничениям API.
4. Из-за того что для подстановки переменных в промпт LangChain под капотом использует f-strings дополнительно пришлось обрабатывать контекст и промпт, в котором имеются фигурные скобки, заменяя `{}` на `{{}}`.
5. Для соответствия rate-limit OpenAI в 3 запроса в минуту пришлось добавлять таймер в 20 секунд после каждого запроса.

**3.2. Telegram бот**  
Для взаимодействия с полученной системой был разработан Telegram бот, которому можно отправить вопрос, связанный с предварительно загруженным контекстом и получить релевантный ответ. Для улучшения читабельности кода был написан обработчик, форматирующий код в ответе под формат Telegram.

**3.3. Дополнительные исследования**  
Дополнительно было проведено сравнительное исследование эмбеддингов. В контексте релевантного поиска ada эмбеддинги оказались лучше некоторых других популярных мультиязычных эмбеддингов - при поиске ближайших соседей подходящий контекст находился чаще. Также было проведено небольшое тестирование точности ada эмбеддингов, для этого в векторной бд искался контекст как по точным, так и по переформулированным вопросам. В обоих случаев accuracy нахождения нужного контекста в топ-5 релевантных документах составляла 100%.

### Список литературы

\[1\] P. Liu, et al., Pre-train, prompt, and predict: A systematic survey of prompting methods in natural language processing (2023), ACM Computing Surveys, 55(9), 1-35.  
\[2\] J. Chen, et al., When large language models meet personalization: Perspectives of challenges and opportunities (2023), arXiv preprint arXiv:2307.16376.  
\[3\] N. Reimers, et al., Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks (2019), Conference on Empirical Methods in Natural Language Processing.  
\[4\] A. Conneau, et al., Supervised Learning of Universal Sentence Representations from Natural Language Inference Data (2017), In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pages 670–680.  
\[5\] New and improved embedding model, https://openai.com/blog/new-and-improved-embedding-model  
\[6\] Y. Malkov et al., Approximate Nearest Neighbor Search Small World Approach (2011), International Conference on Information and Communication Technologies & Applications  
\[7\] Y. Malkov et al., Scalable Distributed Algorithm for Approximate Nearest Neighbor Search Problem in High Dimensional General Metric Spaces (2012), Similarity Search and Applications, pp. 132-147  
\[8\] Y. Malkov, D. Yashunin, Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs (2016), IEEE Transactions on Pattern Analysis and Machine Intelligence  
\[9\] J.Briggs, The Missing Manual, https://www.pinecone.io/learn/series/faiss/hnsw/  
\[10\] Faiss, https://github.com/facebookresearch/faiss  
\[11\] FAISS: Быстрый поиск лиц и клонов на многомиллионных данных, https://habr.com/ru/companies/okkamgroup/articles/509204/
