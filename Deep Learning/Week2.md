# Neural Network Programming Basics

## Introduction
- Neural networks involve key techniques for implementation:
  - Avoid explicit `for` loops to process the training set.
  - Use **forward propagation** for prediction and **backward propagation** for learning.
- Concepts will be introduced using **logistic regression**, a simpler algorithm for **binary classification**.

---
## Logistic Regression Overview
### Problem Setup
- Binary classification: Input an image and output:
  - **1** if the image is a cat.
  - **0** otherwise.
- **Output label (y):** Denotes the classification result.

---
### Image Representation in Computers
- An image is stored as **three matrices** for:
  - Red, Green, and Blue (RGB) color channels.
- For a 64x64 pixel image:
  - Each channel is a 64x64 matrix.
  - Total pixels: $64 \times 64 \times 3 = 12,288$.
  - Represented as a **feature vector** $x$ of dimension $n_x = 12,288$.
### Input Feature Vector ($x$)
- Unroll RGB pixel values into a long vector.
- **Unrolling** (also called **flattening**) means converting a multi-dimensional array (like a matrix or tensor) into a **single-dimensional vector** by arranging all the elements sequentially.
- Dimension: $n_x = 64 \times 64 \times 3 = 12,288$. (i.e. 1 col and 12,288 rows)

---
## Training and Notation
### Training Example
- A single training example: $(x, y)$
  - $x$: Input feature vector (size $n_x$ dimensional feature vector). 
	  - "Dimensional" in this context  means the vector $x$ has $n_x$ entries, or $n_x$ dimensions.
  - $y$: Output label (0 or 1).
### Training Set
- **Training set**: $m$ examples, denoted as:

$$
\{(x^{(1)}, y^{(1)}), (x^{(2)}, y^{(2)}), \dots, (x^{(m)}, y^{(m)})\}
$$

- Number of training examples: $m_{\text{train}}$ or just $m$
- Number of test examples: $m_{\text{test}}$.

---
### Matrix Representation
1. **Inputs (X):**
   - Stack input vectors column-wise:

$$
X = \begin{bmatrix}
x^{(1)} & x^{(2)} & \dots & x^{(m)}
\end{bmatrix}
$$

   - Dimension: $n_x \times m$.
   - In Python: `X.shape` gives $(n_x, m)$.

2. **Outputs (Y):**
   - Stack output labels column-wise:

$$
Y = \begin{bmatrix}
y^{(1)} & y^{(2)} & \dots & y^{(m)}
\end{bmatrix}
$$
  
   - Dimension: $1 \times m$.
   - In Python: `Y.shape` gives $(1, m)$.

### Implementation Tip
- Use **column-wise stacking** for both $X$ and $Y$ to simplify neural network implementation.

---

## Recap
- Key Notation:
  - $n_x$: Dimension of input features.
  - $m$: Number of training examples.
  - $X$: Input matrix of size $n_x \times m$.
  - $Y$: Output matrix of size $1 \times m$.
- **Forward propagation**: Pass data through the network to make predictions.
- **Backward propagation**: Compute gradients to update model parameters.

---

## Resources
- A notation guide is available on the course website for quick reference.
