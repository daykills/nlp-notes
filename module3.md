# Word Embeddings

### Word Embeddings： A word embedding is a way of representing words as dense numerical vectors so that a machine learning model can understand and work with them.
Key Idea： 
- Each word is mapped to a vector of real numbers (for example, 100-dimensional).
- Words that appear in similar contexts in text end up with vectors that are close together in this space.
- This means that relationships between words can be learned. For example: vector("king") - vector("man") + vector("woman") ≈ vector("queen").


### Skip-gram
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


### CBOW Model
- The CBOW model predicts a target word given its context words.

```
Example sentence:
"the cat sat on the mat"
If target word = "sat" (index=2)
Context window size k=2
Context words = ["the", "cat", "on", "the"]

Steps:
Convert context words → embeddings.
Average (or sum) the embeddings → get context vector.
Feed into a linear layer → softmax over vocab.
Train model to predict the target word "sat".

🔹 Difference from Skip-gram

CBOW: context → predict target
Skip-gram: target → predict context

👉 Example:

CBOW: “the ___ sat on” → predict “cat”
Skip-gram: “cat” → predict “the”, “sat”
```


### Applying embeddings in the deep networks
- Map from word indices to imbeddings at the first layer of the network
     ```
     句子: "the cat sat"

        ┌────────────┐
        │   单词表   │   词汇表大小 V=5
        │ "the": 0   │
        │ "cat": 1   │
        │ "sat": 2   │
        │ "on" : 3   │
        │ "mat": 4   │
        └────────────┘
                │
                   ▼
   单词 → 索引: ["the", "cat", "sat"] → [0, 1, 2]
                   │
                   ▼
        One-hot 向量 (长度=V)
        "cat" → [0, 1, 0, 0, 0]
                   │
                   ▼
    ┌─────────────────────────┐
    │    Embedding 矩阵 E     │   大小 = (V × d)，比如 (5 × 3)
    │ "the": [ 0.2, -0.1, 0.5]│
    │ "cat": [-0.3,  0.8, 0.1]│
    │ "sat": [ 0.7,  0.4,-0.6]│
    │ "on" : [-0.2, -0.5, 0.9]│
    │ "mat": [ 0.1,  0.3, 0.7]│
    └─────────────────────────┘
                   │
                   ▼
   Embedding Lookup
   "cat"(索引=1) → [-0.3, 0.8, 0.1]
   "sat"(索引=2) → [ 0.7, 0.4,-0.6]
   "the"(索引=0) → [ 0.2,-0.1, 0.5]
   
   cat" (index=1) 直接取第 1 行 → [-0.3, 0.8, 0.1]。
   这样就把词变成了一个稠密向量。
   ---
   
   最终得到的句子向量序列:
   这些向量会送进后续网络 (RNN, CNN, Transformer 等)。
   在训练过程中，embedding 矩阵里的行会更新，从而让相似词的向量靠近。
   [ [0.2,-0.1,0.5], [-0.3,0.8,0.1], [0.7,0.4,-0.6] ]
   ```
- Approach 1: Learn these embeddings as parameters from your data（就像让学生从零学起，所有知识都要自己学）
  -  Randomly initialize all parameters and learn with backpropagation.
  -  Can work reasonably well.
- Approach 2: Init word embeddings using GloVe, keep fixed.（就像给学生一本现成的字典（GloVe），他们直接有基础知识，再在具体任务上应用）
  - Faster because no need to update these parameters.
- Approach 3: initialize word embeddings GloVe, fine-tune
   -  Works best for some tasks
 
### Deep Averaging Networks (DANs) 
   - 向量取平均发生在 把单词 embedding 拿出来之后、送入全连接层之前，向量取平均之后，梯度会被均分，再传回去给每个单词 embedding。
   - Instead of using RNNs (which read words sequentially) or CNNs (which look for n-gram patterns), DANs just average the embeddings of all words in the sentence.
   - Then, they pass this averaged vector through one or more feed-forward (fully connected) layers (a “deep” network).
   - Finally, a softmax layer produces a prediction (e.g., sentiment class, topic class).
   



