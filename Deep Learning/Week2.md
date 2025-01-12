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
# Cost Function and Loss Function

### Notation
- Superscript $(i)$: Denotes the $i$-th training example (e.g., $X^{(i)}, y^{(i)}$).
- $z^{(i)} = W^T X^{(i)} + b$: The linear combination of weights, inputs, and bias for the $i$-th example.
---
## Loss Function
- Measures how well the model's prediction matches the true label for a single training example.
- **Loss Function ($L(\hat{y}, y)$)**:

$$L(\hat{y}, y) = -\left[ y \log(\hat{y}) + (1 - y) \log(1 - \hat{y}) \right]$$

### Intuition
1. **Case 1**: $y = 1$  
   - Loss: $L(\hat{y}, 1) = -\log(\hat{y})$
   - Goal: Maximize $\hat{y}$ (closer to 1), as $\hat{y} \leq 1$.
2. **Case 2**: $y = 0$  
   - Loss: $L(\hat{y}, 0) = -\log(1 - \hat{y})$
   - Goal: Minimize $\hat{y}$ (closer to 0), as $0 \leq \hat{y}$.

- This loss function ensures predictions $\hat{y}$ are close to the true labels $y$.

---
## Cost Function
- Aggregates the loss across the entire training set to evaluate model performance.
- **Cost Function ($J(W, b)$)**:

$$J(W, b) = \frac{1}{m} \sum_{i=1}^m L(\hat{y}^{(i)}, y^{(i)})$$

* Where:

$$L(\hat{y}^{(i)}, y^{(i)}) = -\left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]$$
* Therefore:

$$J(W, b) = - \frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]$$

  - $m$: Total number of training examples.

### Optimization Objective
- Minimize the cost function:

$$\min_{W, b} J(W, b)$$

---
## Why Not Use Squared Error Loss?
- Squared error loss:

$$L(\hat{y}, y) = \frac{1}{2} (\hat{y} - y)^2$$

- Challenges:
  - Leads to a **non-convex optimization problem**.
  - Gradient descent may struggle with multiple local minima.
- Logistic regression uses a **convex loss function** to ensure a single global minimum.

---
## Key Takeaways
- Loss function $L(\hat{y}, y)$ measures performance on a single example.
- Cost function $J(W, b)$ evaluates performance across the entire training set.
- Optimization involves minimizing $J(W, b)$ to find the best $W$ and $b$.
- Logistic regression is a foundational concept for understanding neural networks.
---
# Logistic Regression and Gradient Descent

## Recap of Logistic Regression
- **Model**: Logistic regression uses parameters $w$ and $b$ to predict outputs.
- **Loss Function**: Measures performance on a single training example.
- **Cost Function** ($J(w, b)$): Measures overall performance across the training set.

### Cost Function Formula

$$
J(w, b) = \frac{1}{m} \sum_{i=1}^m \text{Loss}(\hat{y}^{(i)}, y^{(i)})
$$

- $\hat{y}^{(i)}$: Predicted value for the $i$-th example.
- $y^{(i)}$: Actual label for the $i$-th example.
- **Goal**: Minimize $J(w, b)$ to find the optimal parameters.

---
## Gradient Descent Algorithm
- **Purpose**: Find the parameters $w$ and $b$ that minimize $J(w, b)$.
- **Visualization**:
  - Imagine $J(w, b)$ as a surface above the parameter space.
  - Gradient descent finds the "lowest point" (global minimum) on this surface.
### Properties of the Cost Function
- **Convexity**: $J(w, b)$ is a convex function (bowl-shaped).
  - Ensures a single global minimum.
  - This property simplifies optimization.
### Steps in Gradient Descent
1. **Initialization**:
   - Typically $w = 0$ and $b = 0$.
   - Random initialization also works but is uncommon for logistic regression.
2. **Iterative Update**:
   - Update $w$:

$$w := w - \alpha \frac{\partial J(w, b)}{\partial w}$$

   - Update $b$:
$$b := b - \alpha \frac{\partial J(w, b)}{\partial b}$$

   - Repeat until convergence.
- **Learning Rate ($\alpha$)**:
  - Controls the step size for updates.
  - Proper tuning is critical for effective optimization.
---
## Gradient Descent Intuition
- **Slope**:
  - $\frac{\partial J(w)}{\partial w}$ gives the slope at a point on the curve.
  - Positive slope: Decrease $w$.
  - Negative slope: Increase $w$.
- **Algorithm Behavior**:
  - Moves $w$ and $b$ iteratively toward the global minimum.
---
## Notation and Partial Derivatives
- **Derivatives**:
  - $\frac{\partial J(w, b)}{\partial w}$ measures the slope of $J(w, b)$ with respect to $w$.
  - $\frac{\partial J(w, b)}{\partial b}$ measures the slope with respect to $b$.
- **Partial Derivative Symbol** ($\partial$):
  - Used when $J$ depends on two or more variables.
  - Example: $\frac{\partial J(w, b)}{\partial w}$ instead of $\frac{d J(w)}{d w}$.
  - In practice, both represent the slope.

---
## Implementation Notes
- **Code Variables**:
  - `dw`: Gradient (slope) with respect to $w$.
  - `db`: Gradient with respect to $b$.
  - Updates:
    ```python
    w = w - alpha * dw
    b = b - alpha * db
    ```

---
## Key Takeaways
1. Gradient descent minimizes the convex cost function $J(w, b)$ by updating $w$ and $b$ iteratively.
2. The learning rate $\alpha$ determines the size of the updates.
3. Partial derivatives guide the direction of the steepest descent.
4. Intuitive understanding of calculus is sufficient to implement and use gradient descent effectively.

# Intuitive Understanding of Derivatives

## Overview
- **Goal:** Provide an intuitive understanding of calculus and derivatives for deep learning.
- **Key Message:** A deep understanding of calculus isn't mandatory for applying deep learning effectively, but basic intuition helps.
---
## Why Learn Calculus for Deep Learning?
- Essential concepts are encapsulated in **forward functions** and **backward functions**.
- Intuition about derivatives helps in understanding how algorithms optimize.
---
## **Introduction to Derivatives**
- **Definition:** Derivative = slope of a function.
- **Simplified Concept:** Whenever you hear "derivative," think **slope**.
---
## Example: Linear Function
### Function Definition
- $f(a) = 3a$ (a straight line).
### Calculating the Derivative
1. At $a = 2$:
   - $f(2) = 3 \cdot 2 = 6$.
   - Increase $a$ slightly to $2.001$:  
     $f(2.001) = 3 \cdot 2.001 = 6.003$.
   - **Slope**:

$$\text{slope} = \frac{\Delta f(a)}{\Delta a} = \frac{6.003 - 6}{2.001 - 2} = \frac{0.003}{0.001} = 3.$$

   - **Interpretation**: A small nudge in $a$ (e.g., $+0.001$) results in $f(a)$ increasing 3 times as much.

2. At $a = 5$:
   - $f(5) = 3 \cdot 5 = 15$.
   - Increase $a$ slightly to $5.001$:  
     $f(5.001) = 3 \cdot 5.001 = 15.003$.
   - **Slope**:

$$
\text{slope} = \frac{15.003 - 15}{5.001 - 5} = \frac{0.003}{0.001} = 3.
$$

### General Observation
- The slope (derivative) is constant ($= 3$) across the function.
---
## Notation for Derivatives
1. $\frac{df(a)}{da}$: Slope of $f(a)$ with respect to $a$.
2. $\frac{d}{da} f(a)$: Equivalent notation.
3. **Meaning**: A small change in $a$ ($\Delta a$) results in $\Delta f(a) \approx 3 \cdot \Delta a$.
---
## Formal Definition of Derivative
- **Infinitesimal Change:**  
  Derivatives formally consider an infinitely small nudge in $a$.
- Definition:  

$$
\frac{df(a)}{da} = \lim_{\Delta a \to 0} \frac{\Delta f(a)}{\Delta a}.
$$
---
## Key Insights
- **Constant Slope:** For $f(a) = 3a$, the slope is always $3$ regardless of $a$.
- **Geometric Interpretation:**
  - Triangle ratio ($\text{height} / \text{width}$) = $3$ everywhere.
  - This represents a uniform slope.
---
## Next Steps
- Explore functions with varying slopes in the next video.

> [!tip] **Practical Takeaway**
> Understanding derivatives as "slopes" simplifies their application in deep learning tasks.

# Understanding Derivatives: Lecture Notes

## Key Concepts
- **Derivative**: Measures the slope of a function at a specific point.
- **Slope**: Can vary at different points on non-linear functions.
---
## Example 1: Function $f(a) = a^2$
1. **Point $a = 2$**:
   - $f(a) = 2^2 = 4$
   - Nudge $a$ to $2.001$: $f(2.001) \approx 4.004$
   - Approximation: $f(a)$ increases $4 \times$ the nudge size.
   - Derivative: $\frac{d}{da}f(a) = 4$ at $a = 2$.

2. **Point $a = 5$**:
   - $f(a) = 5^2 = 25$
   - Nudge $a$ to $5.001$: $f(5.001) \approx 25.010$
   - Approximation: $f(a)$ increases $10 \times$ the nudge size.
   - Derivative: $\frac{d}{da}f(a) = 10$ at $a = 5$.

3. **General Formula**:

$$ 
\frac{d}{da}(a^2) = 2a 
$$

   - Verified:
     - At $a = 2$: $2 \cdot 2 = 4$
     - At $a = 5$: $2 \cdot 5 = 10$

---
## Example 2: Function $f(a) = a^3$
1. **General Formula**:

$$
\frac{d}{da}(a^3) = 3a^2
$$

2. **Point $a = 2$**:
   - $f(a) = 2^3 = 8$
   - Nudge $a$ to $2.001$: $f(2.001) \approx 8.012$
   - Approximation: $f(a)$ increases $12 \times$ the nudge size.
   - Verified:
     - $3 \cdot 2^2 = 3 \cdot 4 = 12$

---
## Example 3: Function $f(a) = \log(a)$ (Base $e$)
1. **General Formula**:

$$
\frac{d}{da}(\log(a)) = \frac{1}{a}
$$

2. **Point $a = 2$**:
   - $f(a) = \log(2) \approx 0.69315$
   - Nudge $a$ to $2.001$: $f(2.001) \approx 0.69365$
   - Approximation:
     - Increase in $f(a) \approx 0.0005$
     - Verified: $\frac{1}{2} \cdot 0.001 = 0.0005$

---
## Visualizing Derivatives
- Slope ($\frac{\Delta y}{\Delta x}$) varies for curves:
  - For $f(a) = a^2$: Slope at $a=2$ is steeper than at $a=1$.
  - For $f(a) = \log(a)$: Slope decreases as $a$ increases.

---
## Takeaways
1. **Definition**:
   - Derivative = Slope of a function at a specific point.
   - Slopes vary for non-linear functions (e.g., $f(a) = a^2$, $\log(a)$).

2. **Finding Derivatives**:
   - Use calculus formulas or tables:
     - $\frac{d}{da}(a^2) = 2a$
     - $\frac{d}{da}(a^3) = 3a^2$
     - $\frac{d}{da}(\log(a)) = \frac{1}{a}$

---
> [!NOTE] Infinitesimal Nudges
> Derivatives assume **infinitesimally small** nudges to $a$, ensuring accuracy of calculations.
---
# Computation Graph and Forward/Backward Propagation

## Overview
The computations of a neural network are structured in two key steps:
1. **Forward Pass (Forward Propagation):** Computes the output of the network.
2. **Backward Pass (Back Propagation):** Computes gradients or derivatives.

The **computation graph** is a visual tool that organizes these computations, helping to understand the flow of operations.
---
## Example: Function Computation Using a Graph

We aim to compute a function $J$, defined as:

$$
J = 3(a + b \cdot c)
$$

### Steps to Compute $J$
1. Compute $u = b \cdot c$.
2. Compute $V = a + u$.
3. Compute $J = 3 \cdot V$.
### Example Inputs
- $a = 5$
- $b = 3$
- $c = 2$
#### Calculations
- $u = b \cdot c = 3 \cdot 2 = 6$
- $V = a + u = 5 + 6 = 11$
- $J = 3 \cdot V = 3 \cdot 11 = 33$

---
## Computation Graph Representation

- **Nodes** represent variables and computations.
- **Edges (arrows)** represent the flow of data.

### Computation Graph for $J$

1. Start with variables $a$, $b$, and $c$.
2. Compute $u = b \cdot c$.
3. Compute $V = a + u$.
4. Compute $J = 3 \cdot V$.

---

## Forward and Backward Pass

### Forward Pass (Blue Arrows)
- Computes $J$ by moving left-to-right through the graph.

### Backward Pass (Red Arrows)
- Computes derivatives by moving right-to-left through the graph.

---

## Applications
- In **logistic regression**, $J$ often represents the cost function to minimize.
- Forward pass computes the cost.
- Backward pass calculates gradients for optimization.

---

## Key Takeaways
- The computation graph is a structured way to represent and organize calculations.
- **Forward pass:** Computes the output value.
- **Backward pass:** Computes derivatives for optimization.

---

# Derivatives and Backpropagation in Computation Graphs

## Overview
This lecture discusses how to compute derivatives in a computation graph using **backpropagation**. Key points include:
- **Derivative calculations** for a function $J$.
- **Chain rule** and its application in computation graphs.
- **Notation** used in programming for backpropagation.

---

## Computation Graph Example: Derivatives of $J$
- **Given function:** $J = 3v$
  - $v = 11$, so $J = 33$.
- **Goal:** Compute $\frac{\partial J}{\partial v}$.
  - If $v$ increases by $0.001$ to $11.001$, then $J$ increases to $33.003$.
  - **Result:** $\frac{\partial J}{\partial v} = 3$ because $J$ increases 3 times as much as $v$.

---

## Chain Rule in Computation Graphs
1. **Definition:**
   - If $a$ affects $v$, and $v$ affects $J$, then the total derivative is:

$$
\frac{\partial J}{\partial a} = \frac{\partial J}{\partial v} \cdot \frac{\partial v}{\partial a}
$$

2. **Example:**
   - $v = a + u$, where $a = 5$ and $u = 6$.
   - $\frac{\partial v}{\partial a} = 1$.
   - $\frac{\partial J}{\partial v} = 3$.
   - **Result:** $\frac{\partial J}{\partial a} = 3 \cdot 1 = 3$.

---

## Backpropagation Steps
1. Compute **$\frac{\partial J}{\partial v}$** (start from the final output variable).
2. Use $\frac{\partial J}{\partial v}$ to compute $\frac{\partial J}{\partial a}$ and $\frac{\partial J}{\partial u}$.
3. Apply the **chain rule** iteratively for intermediate variables.

---

## Derivative Examples in the Graph
1. **$\frac{\partial J}{\partial u}$:**
   - $u = b \cdot c$, where $b = 3$, $c = 2$, and $u = 6$.
   - $\frac{\partial u}{\partial b} = c = 2$.
   - $\frac{\partial J}{\partial u} = 3$.
   - **Result:** $\frac{\partial J}{\partial b} = \frac{\partial J}{\partial u} \cdot \frac{\partial u}{\partial b} = 3 \cdot 2 = 6$.

2. **$\frac{\partial J}{\partial c}$:**
   - Similarly, $\frac{\partial u}{\partial c} = b = 3$.
   - **Result:** $\frac{\partial J}{\partial c} = \frac{\partial J}{\partial u} \cdot \frac{\partial u}{\partial c} = 3 \cdot 3 = 9$.

---

## Notation in Code
- **Variable Names:**
  - Use `dvar` to denote $\frac{\partial J}{\partial \text{var}}$.
  - Example: `dv` for $\frac{\partial J}{\partial v}`, `db` for $\frac{\partial J}{\partial b}`.
- Simplifies coding and avoids lengthy variable names like `dFinalOutputVar/dvar`.

---

## Key Takeaways
1. **Efficient Derivative Calculation:**
   - Perform calculations from **right to left** in the computation graph.
   - Use intermediate derivatives (e.g., $\frac{\partial J}{\partial v}$) to compute others.
2. **Chain Rule Application:**
   - Break down derivatives into smaller parts for computation.
3. **Practical Implementation:**
   - Use clear and consistent variable naming in code to represent derivatives.

---

## Formula Summary
- **Chain Rule:** 

$$
\frac{\partial J}{\partial a} = \frac{\partial J}{\partial v} \cdot \frac{\partial v}{\partial a}
$$

- **Examples:**
  - $\frac{\partial J}{\partial v} = 3$
  - $\frac{\partial J}{\partial a} = 3 \cdot 1 = 3$
  - $\frac{\partial J}{\partial b} = 3 \cdot 2 = 6$
  - $\frac{\partial J}{\partial c} = 3 \cdot 3 = 9$

---

# Gradient Descent for Logistic Regression

## Overview
This lecture explains how to compute derivatives and implement **gradient descent** for logistic regression. Key points:
- **Goal:** Minimize the loss function to optimize parameters ($W_1$, $W_2$, $B$).
- **Methodology:** Use a computation graph for forward and backward propagation.

---

## Logistic Regression Setup
1. **Prediction:**

$$
\hat{Y} = \sigma(Z) \quad \text{where } Z = W_1 X_1 + W_2 X_2 + B
$$

   - $\sigma(Z)$ is the sigmoid function.

2. **Loss Function:**

$$
L = -Y \log(\hat{Y}) - (1 - Y) \log(1 - \hat{Y})
$$

   - $Y$: Ground truth label.
   - $\hat{Y}$: Model prediction.

3. **Goal:** Minimize $L$ by updating $W_1$, $W_2$, and $B$.

---

## Forward Propagation Steps
1. Compute $Z = W_1 X_1 + W_2 X_2 + B$.
2. Apply sigmoid: $\hat{Y} = \sigma(Z)$.
3. Compute loss: $L$.

---

## Backpropagation: Derivatives of $L$
### Step 1: Compute $\frac{\partial L}{\partial A}$
- **Formula:** 

$$
\frac{\partial L}{\partial A} = -\frac{Y}{A} + \frac{1 - Y}{1 - A}
$$

- **Notation in code:** `dA`.

---

### Step 2: Compute $\frac{\partial L}{\partial Z}$
- Use the **chain rule:**

$$
\frac{\partial L}{\partial Z} = \frac{\partial L}{\partial A} \cdot \frac{\partial A}{\partial Z}
$$

  - $\frac{\partial A}{\partial Z} = A(1 - A)$.
  - Combine terms: 

$$
\frac{\partial L}{\partial Z} = A - Y
$$

- **Notation in code:** `dZ`.

---

### Step 3: Compute $\frac{\partial L}{\partial W_1}, \frac{\partial L}{\partial W_2}, \frac{\partial L}{\partial B}$
- **Formulas:**
  - $\frac{\partial L}{\partial W_1} = X_1 \cdot \frac{\partial L}{\partial Z} = X_1 \cdot dZ$
  - $\frac{\partial L}{\partial W_2} = X_2 \cdot \frac{\partial L}{\partial Z} = X_2 \cdot dZ$
  - $\frac{\partial L}{\partial B} = \frac{\partial L}{\partial Z} = dZ$
- **Notations in code:** `dW1`, `dW2`, `dB`.

---

## Gradient Descent Updates
For a single training example:
- Update formulas:

$$
W_1 := W_1 - \alpha \cdot dW1
$$

$$
W_2 := W_2 - \alpha \cdot dW2
$$

$$
B := B - \alpha \cdot dB
$$

  - $\alpha$: Learning rate.

---

## Multiple Training Examples
To train with $M$ examples:
1. Compute the average gradients across all examples:
   - $\text{Mean}(dW1)$, $\text{Mean}(dW2)$, $\text{Mean}(dB)$.
2. Apply gradient descent using these averaged gradients.

---

## Key Takeaways
1. **Computation Graph:**
   - Forward propagation computes predictions and loss.
   - Backward propagation computes derivatives for parameter updates.
2. **Gradient Descent:**
   - Efficiently updates parameters using computed gradients.
3. **Single Example vs. Full Dataset:**
   - Extend single-example gradient descent to multiple examples by averaging gradients.

---
# Lecture Notes: Vectorization in Deep Learning

## Introduction to Vectorization
- **Definition**: Vectorization is the process of eliminating explicit `for` loops in your code, replacing them with built-in functions or operations that take advantage of parallel processing.
- **Importance**: 
  - Crucial in deep learning due to large datasets.
  - Reduces computation time significantly.
  - Improves efficiency by utilizing optimized parallelization techniques.

---

## Example: Logistic Regression
### Formula:

$$
Z = W^T X + B
$$

- **Variables**:
  - $W$: Weight vector ($n_x$-dimensional).
  - $X$: Input vector ($n_x$-dimensional).
  - $B$: Bias (scalar).

### Non-Vectorized Implementation:
```python
Z = 0
for i in range(len(X)):  # Iterate through elements of X
    Z += W[i] * X[i]
Z += B  # Add bias
````

- **Disadvantages**:
    - Computationally expensive.
    - Slow for large vectors.

### Vectorized Implementation:

```python
Z = np.dot(W, X) + B  # Efficient matrix multiplication and addition
```

- **Advantages**:
    - Faster execution.
    - Utilizes optimized libraries like NumPy.

---
## Vectorization Demo

### Code to Compare Vectorized and Non-Vectorized Implementations:

```python
import numpy as np
import time

# Create random vectors
a = np.random.rand(10**6)
b = np.random.rand(10**6)

# Vectorized version
tic = time.time()
c_vectorized = np.dot(a, b)
toc = time.time()
print(f"Vectorized version: {(toc - tic) * 1000:.2f} ms")

# Non-vectorized version
c_non_vectorized = 0
tic = time.time()
for i in range(10**6):
    c_non_vectorized += a[i] * b[i]
toc = time.time()
print(f"Non-vectorized version: {(toc - tic) * 1000:.2f} ms")

# Verify results
print(f"Results match: {c_vectorized == c_non_vectorized}")
```

### Results:

- **Vectorized Execution Time**: ~1.5 ms.
- **Non-Vectorized Execution Time**: ~480 ms.
- **Speedup**: ~300x faster.

---

## Key Concepts
- **SIMD (Single Instruction, Multiple Data)**:
    - Enables parallel processing of data using built-in functions (e.g., `np.dot`).
    - Efficient on both CPUs and GPUs.
    - GPUs excel in parallel operations but CPUs also benefit significantly.
- **Rule of Thumb**:
    - **Avoid explicit `for` loops** whenever possible.
    - Use built-in vectorized functions for faster computations.

---

## Applications in Deep Learning
- **Training Efficiency**:
    - Vectorization accelerates model training.
    - Enables faster experimentation and iteration.
- **Scalable Implementations**:
    - Essential for leveraging GPU/CPU parallelism.
    - Optimizes performance of large-scale models.

---
## Summary

- **Key Takeaway**: Always aim to vectorize your code in deep learning and computational tasks.
- **Benefits**:
    - Drastically reduces runtime.
    - Takes advantage of hardware parallelism.
- **Next Steps**:
    - Explore more examples of vectorization.
    - Apply vectorization techniques to logistic regression.

> **Note**: Practice using NumPy functions and libraries that support vectorization to deepen your understanding.

# Advanced Vectorization Examples in Deep Learning

## Key Principle: Avoid Explicit `for` Loops
- **Rule of Thumb**: Always avoid `for` loops when possible by using built-in functions or vectorized operations.
- **Benefits**:
  - Significantly faster computation.
  - Cleaner and more concise code.

---

## Example 1: Matrix-Vector Multiplication
### Mathematical Definition:

$$
u_i = \sum_j A_{ij} \cdot v_j
$$

- **Goal**: Compute $u = A \cdot v$.

### Non-Vectorized Implementation:
```python
u = np.zeros(n)
for i in range(len(A)):
    for j in range(len(v)):
        u[i] += A[i][j] * v[j]
````

### Vectorized Implementation:

```python
u = np.dot(A, v)
```

- **Advantage**: Eliminates two nested loops, resulting in much faster execution.

---

## Example 2: Element-wise Exponential

### Problem:

Compute $u = [e^{v_1}, e^{v_2}, \dots, e^{v_n}]$ for a vector $v$.

### Non-Vectorized Implementation:

```python
u = np.zeros(len(v))
for i in range(len(v)):
    u[i] = np.exp(v[i])
```

### Vectorized Implementation:

```python
u = np.exp(v)
```

- **Advantage**: Eliminates the loop and uses a single NumPy function.

### Other Useful NumPy Functions:

- **Element-wise Operations**:
    - `np.log(v)`: Natural logarithm.
    - `np.abs(v)`: Absolute value.
    - `np.maximum(v, 0)`: Element-wise maximum with 0.
    - `v**2`: Element-wise square.
    - `1 / v`: Element-wise inverse.
- **Takeaway**: Always check for NumPy built-in functions before writing loops.

---

## Example 3: Logistic Regression Gradient Descent

### Problem:

Eliminate redundant loops in the gradient computation for logistic regression.

### Initial Non-Vectorized Code:

```python
dW = [0, 0, ..., 0]  # For each feature
for i in range(m):  # Loop over training examples
    for j in range(nx):  # Loop over features
        dW[j] += x[i][j] * dz[i]
dW /= m
```

### Improved Vectorized Code:

1. Replace feature-wise loop with a vector operation:

```python
dW = np.zeros((nx, 1))
for i in range(m):
    dW += x[i].reshape(-1, 1) * dz[i]
dW /= m
```

2. Benefits:
    - Reduced to a single loop over training examples.
    - Faster and more scalable.

---

## Towards Full Vectorization

- **Next Steps**:
    - Fully eliminate the loop over training examples.
    - Process the entire dataset in one step.
- **Preview**:
    - By leveraging advanced vectorization techniques, we can process entire training sets simultaneously.
    - Example: Writing logistic regression gradient computations without any loops.

---

## Summary

- **Vectorization Principles**:
    1. Avoid explicit `for` loops where possible.
    2. Use NumPy built-in functions for common operations.
    3. Replace iterative computations with vectorized operations.
- **Key Benefits**:
    
    - Faster execution.
    - Clean and maintainable code.
- **Looking Ahead**:
    - Explore full vectorization for entire training sets in the next lecture.

---

#  Vectorizing Logistic Regression

## Overview
- **Goal**: Implement forward propagation for logistic regression on an entire training set in a fully vectorized manner, eliminating explicit `for` loops.
- **Benefits**:
  - Process all training examples simultaneously.
  - Drastically improve computational efficiency.

---

## Forward Propagation Steps

### Step 1: Compute $Z$ for All Training Examples
#### Formula:

$$
Z = W^T X + B
$$

- **Variables**:
  - $X$: Input matrix of shape $(n_x, m)$, where $n_x$ is the number of features and $m$ is the number of training examples.
  - $W$: Weight vector of shape $(n_x, 1)$.
  - $B$: Bias scalar, broadcasted to shape $(1, m)$.

#### Explanation:
- **Matrix Dimensions**:
  - $W^T$: Row vector of shape $(1, n_x)$.
  - $X$: Matrix of shape $(n_x, m)$.
  - $W^T X$: Resulting matrix of shape $(1, m)$.
  - Adding $B$: Python automatically broadcasts $B$ to $(1, m)$.

- **Implementation in Python**:
```python
Z = np.dot(W.T, X) + B
````

- **Result**: $Z$ is a $(1, m)$ matrix containing $Z_1, Z_2, \dots, Z_m$.

#### Key Concept: Broadcasting

- Python automatically broadcasts scalars or smaller arrays to match the required shape during operations.
- Example: Adding a scalar $B$ to a $(1, m)$ matrix expands $B$ to a $(1, m)$ vector.

---

### Step 2: Compute $A$ (Activations)

#### Formula:

A=σ(Z)A = \sigma(Z)

- **Variables**:
    - $\sigma$: Sigmoid function, defined as $\sigma(z) = \frac{1}{1 + e^{-z}}$.
    - $Z$: Matrix of shape $(1, m)$.

#### Explanation:

- **Vectorized Sigmoid**:
    
    - A vector-valued sigmoid function can efficiently compute $A$ for all $Z$ values simultaneously.
- **Implementation in Python**:
    

```python
A = sigmoid(Z)  # Assuming `sigmoid` is implemented as a vectorized function
```

- **Result**: $A$ is a $(1, m)$ matrix containing $A_1, A_2, \dots, A_m$.

---

## Summary of Vectorized Forward Propagation

- **Steps**:
    1. Compute $Z$ for all examples:
        
        ```python
        Z = np.dot(W.T, X) + B
        ```
        
    2. Compute $A$ for all examples:
        
        ```python
        A = sigmoid(Z)
        ```
        
- **Key Points**:
    - Fully vectorized implementation processes all $m$ training examples simultaneously.
    - No explicit loops required.
    - Results in significant computational speedup.

---

# Vectorizing Gradient Computations in Logistic Regression

## Overview
- **Goal**: Perform gradient computations for all $m$ training examples simultaneously, eliminating explicit `for` loops.
- **Outcome**: Achieve a highly efficient implementation of gradient descent for logistic regression.

---

## Gradient Computations

### Step 1: Compute $dZ$
#### Formula:

$$
dZ = A - Y
$$

- **Variables**:
  - $A$: Activations (predicted outputs), shape $(1, m)$.
  - $Y$: True labels, shape $(1, m)$.
  - $dZ$: Difference between predictions and true labels, shape $(1, m)$.

#### Implementation in Python:
```python
dZ = A - Y
````

- **Result**: Vector $dZ$ contains all $dZ_1, dZ_2, \dots, dZ_m$.

---

### Step 2: Compute Gradients for $W$ and $B$

#### Gradient for $W$:

$$
dW=1m⋅X⋅dZTdW = \frac{1}{m} \cdot X \cdot dZ^T
$$

- **Variables**:
    - $X$: Input matrix, shape $(n_x, m)$.
    - $dZ^T$: Transpose of $dZ$, shape $(m, 1)$.
    - $dW$: Weight gradients, shape $(n_x, 1)$.

#### Implementation in Python:

```python
dW = (1 / m) * np.dot(X, dZ.T)
```

---

#### Gradient for $B$:

dB=1m⋅sum(dZ)dB = \frac{1}{m} \cdot \text{sum}(dZ)

- **Variables**:
    - $dZ$: Row vector of shape $(1, m)$.
    - $dB$: Scalar (bias gradient).

#### Implementation in Python:

```python
dB = (1 / m) * np.sum(dZ)
```

---

## Vectorized Gradient Descent

### Full Implementation of One Iteration

1. **Forward Propagation**:
    
    - Compute $Z$: Z=WTX+BZ = W^T X + B
        
        ```python
        Z = np.dot(W.T, X) + B
        ```
        
    - Compute $A$ using the sigmoid function: A=σ(Z)A = \sigma(Z)
        
        ```python
        A = sigmoid(Z)
        ```
        
2. **Backward Propagation**:
    
    - Compute $dZ$:
        
        ```python
        dZ = A - Y
        ```
        
    - Compute $dW$:
        
        ```python
        dW = (1 / m) * np.dot(X, dZ.T)
        ```
        
    - Compute $dB$:
        
        ```python
        dB = (1 / m) * np.sum(dZ)
        ```
        
3. **Gradient Descent Updates**:
    
    - Update $W$ and $B$:
        
        ```python
        W = W - learning_rate * dW
        B = B - learning_rate * dB
        ```
        

---

## Key Insights

- **Efficiency**:
    
    - All $m$ training examples are processed simultaneously using vectorized operations.
    - Eliminates `for` loops for both forward and backward propagation.
- **Outer Loop**:
    
    - A `for` loop is still required for multiple iterations of gradient descent:
        
        ```python
        for _ in range(num_iterations):
            # Forward and Backward Propagation
        ```
        

---

## Summary

- **Steps Achieved**:
    - Fully vectorized computation for forward propagation.
    - Fully vectorized computation for gradients ($dW$ and $dB$).
    - Efficient parameter updates without explicit loops over training examples.
- **Next Topic**:
    - Explore broadcasting, a powerful NumPy feature that optimizes certain operations by expanding dimensions automatically.
---

#  Broadcasting in Python

## Overview
- **What is Broadcasting?**:
  - A technique in Python and NumPy that allows operations on arrays of different shapes by automatically expanding dimensions to match.
  - **Benefits**:
    - Faster computations.
    - Cleaner, more concise code.

---

## Example: Nutritional Matrix
- **Problem**:
  - Given a matrix `A` representing the calories from carbs, proteins, and fats in 100g of 4 foods, calculate the percentage of calories from each source.

### Matrix `A`:
|          | Apples | Beef | Eggs | Potatoes |
|----------|--------|------|------|----------|
| **Carbs** | 56     | 0    | 0    | 220      |
| **Proteins** | 1.2   | 104  | 52   | 8        |
| **Fats**    | 1.8   | 135  | 99   | 1        |

### Steps:
1. **Sum Columns**:
   - Compute the total calories for each food.
   - **Code**:
     ```python
     cal = A.sum(axis=0)
     ```
   - Result: `[59, 239, 151, 229]` (Total calories for Apples, Beef, Eggs, and Potatoes).

2. **Calculate Percentages**:
   - Divide each element of `A` by the total calories of its column.
   - **Code**:
     ```python
     percentage = (A / cal.reshape(1, 4)) * 100
     ```
   - Result: A matrix of percentages.

---

## Key Concepts of Broadcasting

### How Broadcasting Works:
1. Align shapes of arrays by expanding dimensions as needed.
2. Perform element-wise operations on the expanded arrays.

### Examples:
1. **Scalar Addition**:
   - Adding `100` to a `(4, 1)` vector:
     ```
     [1]      [1+100]  
     [2]  +   [2+100]  
     [3]      [3+100]  
     [4]      [4+100]
     ```

2. **Matrix Addition**:
   - Adding a `(1, n)` row vector to a `(m, n)` matrix:
     - The row vector is copied `m` times to match the shape of the matrix.

3. **Column Vector Addition**:
   - Adding a `(m, 1)` column vector to a `(m, n)` matrix:
     - The column vector is copied `n` times to match the shape of the matrix.

4. **General Rule**:
   - For an operation between shapes `(m, n)` and `(1, n)`, the smaller array is expanded to `(m, n)`.
   - Similarly, for shapes `(m, n)` and `(m, 1)`, the smaller array is expanded to `(m, n)`.

---

## Code Example: Division with Broadcasting
1. **Matrix `A`**:

```python
   A = np.array([
       [56, 0, 0, 220],  # Carbs
       [1.2, 104, 52, 8],  # Proteins
       [1.8, 135, 99, 1]  # Fats
   ])
```

2. **Total Calories**:
    
    ```python
    cal = A.sum(axis=0)
    ```
    
3. **Percentages**:
    
    ```python
    percentage = (A / cal.reshape(1, 4)) * 100
    ```
    

---

## Applications in Neural Networks

- **Broadcasting Use Cases**:
    
    - Adding bias terms to activations.
    - Scaling gradients.
    - Vectorized computations for forward and backward propagation.
- **Comparison to MATLAB/Octave**:
    
    - MATLAB’s `bsxfun` performs a similar role as broadcasting in Python.

---

## Summary

- **Advantages of Broadcasting**:
    
    - Simplifies operations with fewer lines of code.
    - Enhances computational efficiency.
    - Reduces the need for explicit loops.
- **Key Operations**:
    
    - Element-wise addition, subtraction, multiplication, and division.
    - Automatic expansion of smaller arrays to match larger ones.
- **Best Practices**:
    
    - Use `reshape` to ensure matrices have the expected dimensions.
    - Familiarize yourself with NumPy's broadcasting rules to avoid bugs.

---

# Avoiding Bugs in Python with Better Vector Practices

## Overview
- **Strength of Python/NumPy**:
  - Flexible and expressive; allows complex operations in a single line.
  - Enables efficient computations with features like broadcasting.
- **Challenge**:
  - Flexibility can lead to subtle, hard-to-diagnose bugs, especially with non-intuitive array behaviors.
- **Objective**:
  - Provide tips to simplify code and reduce bugs when using Python and NumPy.

---

## Rank-1 Arrays: The Problem

### Definition:
- A **rank-1 array** has a shape like `(n,)`, meaning it’s neither a row vector `(1, n)` nor a column vector `(n, 1)`.
- **Example**:

```python
  a = np.random.randn(5)  # Creates a rank-1 array
  print(a.shape)  # Output: (5,)
```

### Issues with Rank-1 Arrays:

1. **Ambiguity**:
    - It doesn’t behave consistently as a row or column vector.
    - Example:
        
        ```python
        a.T  # Transpose looks the same as the original array
        ```
        
2. **Unexpected Results**:
    - Inner and outer products may yield surprising outcomes:
        
        ```python
        np.dot(a, a.T)  # Returns a scalar, not a matrix
        ```
        
3. **Non-Intuitive Behavior**:
    - Operations may lead to subtle bugs due to inconsistent behavior.

---

## Recommended Practices

### 1. Always Specify Vector Shape

- Use either:
    
    - **Column Vector**: `(n, 1)`
        
        ```python
        a = np.random.randn(5, 1)  # Explicit column vector
        ```
        
    - **Row Vector**: `(1, n)`
        
        ```python
        a = np.random.randn(1, 5)  # Explicit row vector
        ```
        
- **Benefits**:
    
    - Eliminates ambiguity.
    - Makes operations (e.g., transpose, dot products) intuitive.

---

### 2. Use Assertions to Validate Shapes

- Add assertions to ensure arrays have the expected dimensions.
- **Example**:
    
    ```python
    assert a.shape == (5, 1), "a is not a column vector!"
    ```
    
- **Advantages**:
    - Cheap to execute.
    - Acts as documentation for expected dimensions.
    - Prevents bugs early.

---

### 3. Use `reshape()` When Necessary

- Convert arrays to the desired shape explicitly.
- **Example**:
    
    ```python
    a = a.reshape(5, 1)  # Convert to column vector
    a = a.reshape(1, 5)  # Convert to row vector
    ```
    
- **Key Point**:
    - Reshaping is computationally inexpensive and clarifies code.

---

## Comparison of Rank-1 Arrays vs Explicit Vectors

|**Operation**|**Rank-1 Array** (Shape `(n,)`)|**Column Vector** `(n, 1)`|**Row Vector** `(1, n)`|
|---|---|---|---|
|**Shape**|Ambiguous|Explicitly vertical|Explicitly horizontal|
|**Transpose**|No visible change|Becomes row vector|Becomes column vector|
|**Inner Product**|Scalar|Matrix|Matrix|
|**Outer Product**|Error or unexpected behavior|Matrix|Matrix|

---

## Summary of Recommendations

1. **Avoid Rank-1 Arrays**:
    - Never use arrays with shape `(n,)`.
2. **Always Use Explicit Shapes**:
    - Use `(n, 1)` for column vectors and `(1, n)` for row vectors.
3. **Add Assertions**:
    - Validate array shapes with assertions for clarity and error prevention.
4. **Use `reshape()`**:
    - Adjust array shapes explicitly when needed.

---

## Key Takeaways

- Eliminating rank-1 arrays simplifies code and reduces bugs.
- Explicitly defined vector shapes make matrix operations more predictable.
- Assertions and reshaping are effective tools for debugging and ensuring consistency.

By following these practices, you can write cleaner, more reliable Python/NumPy code, especially when implementing machine learning algorithms or neural networks.

---
# Introduction to Jupyter Notebooks on Coursera

## Overview
- **Objective**: Familiarize yourself with Jupyter (iPython) notebooks on Coursera for programming assignments.
- **Key Features**:
  - Interactive code execution.
  - Markdown for instructions and explanations.
  - Convenient environment for learning and experimenting with algorithms.

---

## Key Features of Jupyter Notebooks

### 1. **Code Blocks**
- **Appearance**: Long, light gray blocks represent sections where you can write code.
- **Guidelines**:
  - Look for markers: `# START CODE HERE` and `# END CODE HERE`.
  - Write your code between these markers.
- **Example**:

```python
  # START CODE HERE
  print("Hello World")
  # END CODE HERE
```

- **Execution**:
    - Run code by pressing **`Shift + Enter`** or using the menu: `Cell > Run Cell`.
    - The output will appear below the code block.

---

### 2. **Markdown Blocks**

- **Purpose**: Contain instructions or explanations in a readable format.
- **Editing**:
    - Double-clicking on a markdown block shows raw markdown text.
    - Run the cell again (**`Shift + Enter`**) to render the text back into its formatted appearance.

---

### 3. **Kernel**

- **Definition**: The backend process that runs your code in the notebook.
- **Tips**:
    - If the kernel crashes (e.g., due to a large job or a connection issue):
        - Go to the menu: `Kernel > Restart Kernel`.
    - Restarting the kernel resets the state, so you may need to re-run all previous code blocks.

---

### 4. **Executing Multiple Code Blocks**

- Ensure you execute all required code blocks sequentially, even if some blocks don’t require edits.
    - Example: Blocks importing libraries or initializing variables:
        
        ```python
        import numpy as np
        ```
        
- Skipping such blocks may lead to errors in later code sections.

---

### 5. **Submitting Assignments**

- **Button**: Click the **"Submit Assignment"** button (top right) to submit your work for grading.
- Ensure:
    - All required code blocks are completed.
    - Code is tested and produces the expected results.

---

## Benefits of Jupyter Notebooks

- **Interactive Learning**:
    - Implement code, observe results, and iterate quickly.
- **Visualization**:
    - Easily include visual outputs like charts or tables alongside the code.
- **Streamlined Workflow**:
    - Integrates instructions, coding, and execution in one interface.

---
# Justification for Logistic Regression Cost Function

## Overview
- **Objective**: Explain the rationale for using the logistic regression cost function.
- **Key Concept**: The cost function is derived from the principle of **maximum likelihood estimation (MLE)** under the assumption that the training examples are **IID (independently and identically distributed)**.

---

## Logistic Regression Recap
- **Prediction**:

$$ \hat{y} = \sigma(W^T X + b) $$

  - $\sigma$: Sigmoid function.
  - $\hat{y}$: Probability that $y = 1$ for input $x$.

- **Probabilities**:
  - If $y = 1$: $p(y|x) = \hat{y}$.
  - If $y = 0$: $p(y|x) = 1 - \hat{y}$.

---

## Unified Probability Expression
- **Combined Equation**:
$$ p(y|x) = \hat{y}^y (1 - \hat{y})^{1-y} $$

- **Explanation**:
  - When $y = 1$: 

$$ p(y|x) = \hat{y}^1 \cdot (1 - \hat{y})^0 = \hat{y} $$

  - When $y = 0$:

$$ p(y|x) = \hat{y}^0 \cdot (1 - \hat{y})^1 = 1 - \hat{y} $$

---

## Log Likelihood and Loss Function
1. **Logarithm of $p(y|x)$**:

$$ \log p(y|x) = y \log \hat{y} + (1 - y) \log (1 - \hat{y}) $$

2. **Loss Function for a Single Example**:

$$ L(\hat{y}, y) = - \big[ y \log \hat{y} + (1 - y) \log (1 - \hat{y}) \big] $$

4. **Interpretation**:
   - The **negative sign** ensures that minimizing the loss is equivalent to maximizing the log probability.

---

## Cost Function for the Training Set
1. **Likelihood of the Training Set**:
   - Assuming examples are IID:

$$ P(\text{Labels in Training Set}) = \prod_{i=1}^m p(y^{(i)}|x^{(i)}) $$

2. **Log Likelihood**:
   - Taking the log:

$$ \log P = \sum_{i=1}^m \log p(y^{(i)}|x^{(i)}) $$

3. **Cost Function**:
   - Substitute the loss function:

$$ J(W, b) = \frac{1}{m} \sum_{i=1}^m L(\hat{y}^{(i)}, y^{(i)}) $$

   - Expanding $L$:

$$ J(W, b) = - \frac{1}{m} \sum_{i=1}^m \big[ y^{(i)} \log \hat{y}^{(i)} + (1 - y^{(i)}) \log (1 - \hat{y}^{(i)}) \big] $$

4. **Maximizing Likelihood vs Minimizing Cost**:
   - Maximizing the likelihood is equivalent to minimizing $J(W, b)$.
   - The factor $\frac{1}{m}$ scales the cost for better numerical stability.

---

## Key Takeaways
- **Why This Cost Function?**:
  - It arises naturally from the principle of maximum likelihood estimation.
  - It ensures the model fits the training data by maximizing the probability of observed labels.

- **Connection to Probability**:
  - The cost function represents the negative log likelihood of the data under the logistic regression model.

- **Assumptions**:
  - Training examples are IID.

---

## Summary
- The logistic regression cost function is derived from probabilistic principles, ensuring that the model’s parameters maximize the likelihood of the observed data.
- **Formula**:

$$ J(W, b) = - \frac{1}{m} \sum_{i=1}^m \big[ y^{(i)} \log \hat{y}^{(i)} + (1 - y^{(i)}) \log (1 - \hat{y}^{(i)}) \big] $$

With this understanding, you are now better equipped to implement logistic regression and work on related programming exercises.
