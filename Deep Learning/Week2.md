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

# Logistic Regression

## Overview
- **Logistic Regression** is used for **binary classification problems** where output labels $Y$ are either 0 or 1.
- Given input features $X$ (e.g., an image), the goal is to predict $\hat{Y}$, the probability that $Y=1$, i.e., $\hat{Y} = P(Y=1|X)$.

## Parameters
- **$W$**: Weight vector, dimension = same as $X$.
- **$b$**: Bias term, a real number.
- Together, these parameters define the logistic regression model.

## Model Definition
1. **Linear function**: Compute $Z = W^T X + b$
   - Not suitable for probabilities directly because $Z$ can take values outside $[0, 1]$.
2. **Sigmoid function**: Transform $Z$ to ensure $\hat{Y} \in [0, 1]$:

$$
\hat{Y} = \sigma(Z) = \frac{1}{1 + e^{-Z}}
$$
   
   - $\sigma(Z)$ (sigmoid) squashes the output to the range $[0, 1]$.
## Properties of the Sigmoid Function
- Formula: $\sigma(Z) = \frac{1}{1 + e^{-Z}}$
- Behavior:
  - **$Z \to \infty$**: $\sigma(Z) \to 1$
  - **$Z \to -\infty$**: $\sigma(Z) \to 0$
  - $\sigma(0) = 0.5$ (crosses the y-axis at 0.5)
- Plot:
  - Horizontal axis: $Z$
  - Vertical axis: $\sigma(Z)$
  - Smooth curve transitioning from 0 to 1.
## Notation and Conventions
- **Standard Approach**:
  - Keep $W$ and $b$ as separate parameters.
- **Alternative Notation**:
  - Define an extra feature $x_0 = 1$, making $X \in \mathbb{R}^{n+1}$.
  - Combine $W$ and $b$ into a single parameter vector $\theta$, where:

$$
\hat{Y} = \sigma(\theta^T X)
$$

  - This approach is not used in this course.

## Key Takeaways
- Logistic regression outputs $\hat{Y}$, the probability that $Y=1$.
- Sigmoid ensures the output $\hat{Y}$ is a valid probability.
- Parameters $W$ and $b$ are learned to make $\hat{Y}$ a good estimate of $P(Y=1|X)$.

---

## Example: Computing $Z$ and $\hat{Y}$ in Logistic Regression

#### Scenario
You are building a logistic regression model to predict whether a given email is spam ($Y = 1$) or not spam ($Y = 0$). Suppose:
- The feature vector $X = [0.5, 1.2]$ represents two features (e.g., word frequency and length of the email).
- The weight vector $W = [2.0, -1.5]$ and bias term $b = 0.5$ are parameters of the model.

---

### Step 1: Compute $Z = W^T X + b$

The linear combination of $X$ with weights $W$ is calculated as:

$$
Z = W^T X + b
$$

#### Calculation:
$$
Z = (2.0 \times 0.5) + (-1.5 \times 1.2) + 0.5
$$
$$
Z = 1.0 - 1.8 + 0.5
$$
$$
Z = -0.3
$$

---
### Step 2: Apply the Sigmoid Function to Compute $\hat{Y}$

Use the formula for the sigmoid function:

$$
\hat{Y} = \sigma(Z) = \frac{1}{1 + e^{-Z}}
$$

#### Calculation:
Substitute $Z = -0.3$ into the sigmoid function:

$$
\hat{Y} = \frac{1}{1 + e^{0.3}}
$$

First, compute $e^{0.3} \approx 1.3499$:

$$
\hat{Y} = \frac{1}{1 + 1.3499}
$$

$$
\hat{Y} = \frac{1}{2.3499} \approx 0.425
$$

---
### Interpretation
- **$Z = -0.3$**: The raw output of the linear function.
- **$\hat{Y} \approx 0.425$**: The predicted probability that the email is spam is **42.5%**.
---
