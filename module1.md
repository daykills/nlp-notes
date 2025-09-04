Sentiment Analysis
 - Feature Extraction: words | ngrams to feature space
    - bag of words: 1. bag-of-ngrams; 2. tf-idf(term frequency inverse document frequency)
    - example: The movie was great. so "2-grams" is "the movie", "movie was", and "was great".
 - preprocessing: string to words | ngrams
    - tokenization
       - whitespace tokenization. example: was great! -> {was great !}
       - [sometimes] stop word removal.stopwords are generally function words like the, an and etc..
