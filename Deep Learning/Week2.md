# Neural Network Programming: Basics & Notation

### Key Concepts:
1. Efficiently processing training sets without explicit loops.
2. Understanding **forward propagation** and **backward propagation** steps in neural networks.
3. Using **logistic regression** as an introductory algorithm to explain these concepts.

---

## Binary Classification with Logistic Regression
Logistic regression is an algorithm used for **binary classification**. Let's set up the problem:

- **Input**: An image (e.g., a picture of a cat).
- **Output**: A label indicating whether the image is a cat (1) or not a cat (0).

The label is denoted as \( y \). Let's explore how an image is represented in a computer.

### Image Representation
An image is stored in the computer as **three separate matrices**, representing the red, green, and blue color channels.

For example:
- **Image size**: 64 x 64 pixels
- **Matrices**: Three 64 x 64 matrices for red, green, and blue pixel values

To convert these matrices into a feature vector:
1. **Unroll** all pixel values from the red channel.
2. **Unroll** all pixel values from the green channel.
3. **Unroll** all pixel values from the blue channel.

If the image is 64 x 64 pixels:
- Total feature vector size = `64 * 64 * 3 = 12,288`
- We denote this as `n = n_x = 12288`

---

## Notation for Training Sets
In binary classification, our goal is to learn a classifier that inputs a feature vector \( x \) and predicts whether the label \( y \) is 1 or 0.

### Training Example Representation
A single training example is represented as a pair:
`(x, y)`
Where:
- `x` = Input feature vector (dimension `n_x`)
- `y` = Label (either 0 or 1)

The **training set** consists of `m` examples:
```
(x^(1), y^(1)), (x^(2), y^(2)), ..., (x^(m), y^(m)) 
```

We use:
- `m` or `m_train` to specify the number of training examples.
- `m_test` to specify the number of test examples.

---

## Organizing Training Data
To efficiently handle training data, we organize inputs and outputs into matrices.

### Input Matrix `X`
The input feature vectors are stacked as columns in a matrix `X`:
```
X = [x^(1), x^(2}, ..., x^(m)]
```
Where:
- `X` has `m` columns where `m` is number of training examples.
- `X` has `n_x` rows

In Python, you can verify this with:
```python
X.shape  # Output: (n_x, m)
```

### Output Matrix `Y`
The output labels are also stacked as columns in a matrix `Y`:
```
Y = [y^(1), y^(2), ..., y^(m)]
```
Where:
- `Y` has dimentions `(1, m)`

---

