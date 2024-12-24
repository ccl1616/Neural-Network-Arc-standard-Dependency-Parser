# Neural Network Arc-standard Dependency Parser
## Introduction
This project implements a feed-forward neural network to predict transitions in an arc-standard dependency parser. The parser learns to construct dependency trees from sentences by predicting a sequence of shift/reduce operations and dependency relation labels. The implementation includes the input representation for the neural network, network architecture, training pipeline, and decoding algorithm.

## Algorithm & Techniques Used
1. Neural Network Architecture

    Feed-forward neural network with:

    Embedding layer (128 dimensions)
    Hidden layer (128 units) with ReLU activation
    Output layer (91 units) for transition predictions
![IMG_5798203FB80C-1](https://hackmd.io/_uploads/r1JYA5OSyl.jpg)


2. CrossEntropyLoss for training
Adagrad optimizer with learning rate 0.01

3. Parser Features

    Input representation based on top 3 tokens on stack and next 3 tokens in buffer
    Special token handling for numbers (<CD>), proper names (<NNP>), unknown words (<UNK>), root (<ROOT>), and padding (<NULL>)
    One-hot encoded transition outputs (91 classes = 45 dependency relations * 2 + 1)

4. Parsing Algorithm

    [Arc-standard transition-based dependency parsing](https://web.stanford.edu/~jurafsky/slp3/old_oct19/15.pdf) is a method where a sentence's syntactic structure is built incrementally using a sequence of transitions (shift/reduce operations). The parser maintains a stack and buffer, and at each step decides whether to shift a word onto the stack or create a dependency arc between words on the stack. The resulting structure is a directed tree where nodes are the words of the sentence, and edges represent the grammatical relationships between those words.
    
    Greedy decoding includes following validity constraints:
    Arc operations require non-empty stack
    Root node cannot be target of left-arc
    Buffer must be empty before final reduce
    ![](https://miro.medium.com/v2/resize:fit:1400/1*mfcStHLTzMZC1evPaSJaag.png)
[Resource](https://towardsdatascience.com/natural-language-processing-dependency-parsing-cf094bbbe3f7)

## Workflow
1. Extract vocabulary from training data
2. Generate input/output matrices for training
3. Train neural network model
4. Use trained model for greedy parsing
5. Evaluate on development/test data

## How to Run
### Prerequisites
```
    python -m pip install torch numpy
```
### Training Steps
1. Generate vocabulary:
```
    python get_vocab.py data/train.conll data/words.vocab data/pos.vocab
```
2. Extract training matrices:
```
    python extract_training_data.py data/train.conll data/input_train.npy data/target_train.npy
    python extract_training_data.py data/dev.conll data/input_dev.npy data/target_dev.npy
```
3. Train model:
```
    python train_model.py data/input_train.npy data/target_train.npy data/model.pt
```
    
### Parsing & Evaluation
- Parse sentences:
```
    python decoder.py data/model.pt data/dev.conll
```
- Evaluate parser:
```
    python evaluate.py data/model.pt data/dev.conll
```

## Result
The parser achieves an average LAS (~70%) and generate resonable sentences:

* Labeled Attachment Score (LAS): 71.87%
* Unlabeled Attachment Score (UAS): 77.32%
* Macro Avg. LAS: 72.26%
* Macro Avg. UAS: 77.92%

While these scores are lower than state-of-the-art parsers (~97% LAS), they demonstrate reasonable performance for a simple feed-forward neural network implementation. The parser successfully learns basic syntactic patterns and dependency relationships, though it struggles with more complex linguistic constructions.
    
The model was evaluated on 5,039 sentences from the development set. Performance could potentially be improved by:

* Adding more sophisticated features
* Using a deeper/wider network architecture
* Implementing beam search decoding
* Incorporating character-level representations
* Adding POS tag embeddings