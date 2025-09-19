# Word Embeddings

### Word Embeddings： A word embedding is a way of representing words as dense numerical vectors so that a machine learning model can understand and work with them.
Key Idea： 
- Each word is mapped to a vector of real numbers (for example, 100-dimensional).
- Words that appear in similar contexts in text end up with vectors that are close together in this space.
- This means that relationships between words can be learned. For example: vector("king") - vector("man") + vector("woman") ≈ vector("queen").


# Skip-gram
- predict the context words from the target

Defination:  

Input: a single word (the “center” word).
Output: the probability of each word in the vocabulary being a “context word” (a neighbor within a fixed window around the center word).
So instead of predicting the center word from context (CBOW), Skip-gram does the opposite: predicts context words given the center word.

Procedure:

1. Prepare training data
   - Choose a window size k
   - For each word in the corpus, collect all surrounding words within that window.
   - Create training pairs of (center_word, context_word).
2. Input representation
   - Represent the input word as a one-hot vector (size = vocab size).
3. Embedding lookup
   - Multiply the one-hot vector by an embedding matrix W.
   - This selects the row of W that corresponds to the word → the embedding vector.
4. Prediction layer
   - Multiply the embedding by another weight matrix W' to get scores for all vocabulary words.
   - Apply softmax to turn scores into probabilities.
5. Loss function
   - Use cross-entropy loss.
   - For each training pair, maximize the probability of the true context word.
   - Often negative sampling or hierarchical softmax is used for efficiency.
6. Training
   - Repeat for many epochs over the corpus.
   - Adjust weights so that words appearing in similar contexts end up with similar embeddings.
     
Skip-gram Example (tiny corpus)
```

Sentence: "I like cats and dogs"
Vocabulary: [I, like, cats, dogs, and]
Vocab size = 5
Window size = 1

Step 1: Generate training pairs
(I, like)
(like, I), (like, cats)
(cats, like), (cats, and)
(and, cats), (and, dogs)
(dogs, and)
```
