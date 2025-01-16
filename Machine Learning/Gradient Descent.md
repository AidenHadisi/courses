# Gradient Descent

## Overview
Gradient descent is a fundamental algorithm used in machine learning for minimizing functions, particularly cost functions in models like linear regression and neural networks.

### Key Concepts
- **Goal**: Minimize the cost function $J(w, b)$ by iteratively updating parameters $w$ and $b$.
- **Applications**:
  - Linear regression
  - Neural networks (deep learning)
  - General optimization problems

---

## Gradient Descent Algorithm

### General Idea
1. **Initialize Parameters**: Start with initial guesses for $w$ and $b$ (commonly $w = 0$ and $b = 0$).
2. **Iterative Update**:
   - Take small steps in the direction of steepest descent (negative gradient).
   - Repeat until $J(w, b)$ reaches a minimum.

---

### Gradient Descent Process
- **Cost Function Visualization**: The function $J(w, b)$ can be imagined as a 3D surface where:
  - $w$ and $b$ are horizontal axes.
  - $J(w, b)$ is the vertical axis (height of the surface).

#### Analogy: Walking Downhill
- Imagine $J(w, b)$ as a hilly terrain.
- At each step:
  1. Spin around 360° to find the steepest downhill direction.
  2. Take a small step in that direction.
  3. Repeat until you reach the valley (minimum).

---

### Mathematical Properties
- **Direction of Steepest Descent**: 
  - Gradient: $\nabla J = \left( \frac{\partial J}{\partial w}, \frac{\partial J}{\partial b} \right)$
  - Update rule: 

$$ w := w - \alpha \frac{\partial J}{\partial w} $$

$$ b := b - \alpha \frac{\partial J}{\partial b} $$

where $\alpha$ is the learning rate.

- **Local Minima**:
  - Gradient descent can converge to different local minima depending on the starting point.
  - Multiple valleys (local minima) exist in complex functions, e.g., neural network cost functions.

---

## Example: Gradient Descent in Action
| **Step**   | **Action**                                             | **Result**                                        |
|------------|--------------------------------------------------------|--------------------------------------------------|
| **1**      | Start at an initial point on $J(w, b)$                 | Initial cost $J(w, b)$                          |
| **2**      | Compute $\nabla J$ (gradient)                          | Direction of steepest descent                   |
| **3**      | Update $w, b$ using the gradient and learning rate     | Move to a new point on the surface              |
| **4**      | Repeat until convergence                               | Reach the bottom of a valley (local minimum)    |

---

## Notes on Local Minima
- Starting at different points may lead to different valleys (local minima).
- **Global vs. Local Minimum**:
  - Global minimum: Lowest point on the entire cost surface.
  - Local minimum: Lowest point within a region.
  
> **Callout**: In linear regression, the cost function is convex (bowl-shaped), so gradient descent always finds the global minimum. For non-convex functions (e.g., neural networks), gradient descent may converge to a local minimum.

---

# Implementing Gradient Descent

## Overview
This lecture covers the implementation of the gradient descent algorithm, with a focus on:
- Updating parameters $w$ and $b$.
- Understanding the role of the learning rate $\alpha$.
- Correctly implementing simultaneous updates.

---

## Gradient Descent Algorithm

### Update Rules
For each iteration:
1. **Parameter $w$ Update**:

$$ w := w - \alpha \frac{\partial J}{\partial w} $$

2. **Parameter $b$ Update**:

$$ b := b - \alpha \frac{\partial J}{\partial b} $$

- $\alpha$: Learning rate, controls step size (typically between $0$ and $1$).
- $\frac{\partial J}{\partial w}, \frac{\partial J}{\partial b}$: Derivatives of the cost function with respect to $w$ and $b$.

---

### Simultaneous Updates
- **Key Idea**: Update $w$ and $b$ **simultaneously** using their current values.
- **Correct Implementation**:
  1. Compute intermediate values:
  
$$ \text{temp}_w := w - \alpha \frac{\partial J}{\partial w} $$

$$ \text{temp}_b := b - \alpha \frac{\partial J}{\partial b} $$

  2. Assign the new values:
  
$$ w := \text{temp}_w $$

$$ b := \text{temp}_b $$
     
- **Incorrect Implementation**:
  - Update $w$ before computing $\frac{\partial J}{\partial b}$.
  - Results in inconsistent values, as the updated $w$ affects the computation for $b$.

> **Callout**: Always perform **simultaneous updates** for correct gradient descent behavior.

---

## Understanding Learning Rate ($\alpha$)
- Controls step size during updates:
  - **Large $\alpha$**: Fast progress but risk of overshooting the minimum.
  - **Small $\alpha$**: Stable progress but slow convergence.
- Example: A typical value might be $\alpha = 0.01$.

---

## Implementation Notes

### Assignment Operator
- In programming, `$=$` is an **assignment operator** (e.g., `$a = c$` sets $a$ to $c$).
- **Truth assertions** (e.g., $a == c$) differ from assignments.

### Iterative Updates
- Gradient descent proceeds until convergence:
  - Parameters stabilize (change becomes negligible).
  - Local minimum is reached.

---

## Summary Table

| **Aspect**         | **Correct Implementation**                              | **Incorrect Implementation**                          |
|---------------------|--------------------------------------------------------|------------------------------------------------------|
| **Update Order**    | Simultaneous: compute $\text{temp}_w$, $\text{temp}_b$ | Sequential: update $w$ before computing $b$.         |
| **Resulting Values**| Consistent parameter updates.                          | Inconsistent updates due to dependency issues.       |
| **Behavior**        | Correct gradient descent behavior.                     | Similar, but technically implements a different algorithm. |

---

# Gradient Descent Intuition

## Overview
This lecture dives deeper into gradient descent, focusing on:
- The roles of the learning rate $\alpha$ and the derivative term $\frac{\partial J}{\partial w}$.
- Intuition behind how gradient descent minimizes the cost function $J(w)$.

---

## Gradient Descent Algorithm Recap
For a single parameter $w$:

$$ w := w - \alpha \frac{\partial J}{\partial w} $$

- $\alpha$: Learning rate (controls step size).
- $\frac{\partial J}{\partial w}$: Derivative of $J$ with respect to $w$.

---

## Intuition Behind Gradient Descent

### Cost Function
- $J(w)$: Represents the cost as a function of parameter $w$.
- Horizontal axis: Parameter $w$.
- Vertical axis: Cost $J(w)$.

---

### Derivative and Slope
- **Tangent Line**: A line that touches $J(w)$ at a specific point.
- **Derivative**: The slope of the tangent line:
  - Positive slope: $\frac{\partial J}{\partial w} > 0$.
  - Negative slope: $\frac{\partial J}{\partial w} < 0$.

#### Slope Calculation
- Draw a triangle under the tangent line.
- Slope = $\text{Height} \div \text{Width}$.

---

### Examples

#### Example 1: Positive Slope
1. Starting point: $w$ on the right side of the minimum.
2. **Derivative**: $\frac{\partial J}{\partial w} > 0$.
3. **Update**:

$$ w := w - \alpha \cdot (\text{positive number}) $$

   - $w$ decreases.
4. **Result**:
   - Move left on the graph.
   - Cost $J(w)$ decreases.

#### Example 2: Negative Slope
1. Starting point: $w$ on the left side of the minimum.
2. **Derivative**: $\frac{\partial J}{\partial w} < 0$.
3. **Update**:

$$ w := w - \alpha \cdot (\text{negative number}) $$

   - Subtracting a negative number increases $w$.
5. **Result**:
   - Move right on the graph.
   - Cost $J(w)$ decreases.

---

## Insights
- **Derivative**: Guides the direction of movement.
  - Positive $\frac{\partial J}{\partial w}$: Move left.
  - Negative $\frac{\partial J}{\partial w}$: Move right.
- Gradient descent iteratively reduces $J(w)$, bringing $w$ closer to the minimum.

---

# Choosing the Learning Rate in Gradient Descent

## Overview
The learning rate $\alpha$ plays a critical role in the efficiency and success of gradient descent. This lecture explores:
- The effects of choosing $\alpha$ too small or too large.
- How gradient descent behaves at a local minimum.
- How gradient descent automatically adjusts step sizes as it approaches a minimum.

---

## Gradient Descent Rule Recap

$$ w := w - \alpha \frac{\partial J}{\partial w} $$

- $\alpha$: Learning rate, determines step size.
- $\frac{\partial J}{\partial w}$: Derivative of $J$ with respect to $w$, guides direction and magnitude of updates.

---

## Effects of Learning Rate

### Case 1: Learning Rate $\alpha$ Too Small
- **Behavior**:
  - Tiny steps are taken because $\alpha$ is a very small number (e.g., $0.0000001$).
  - Gradient descent converges to the minimum, but **very slowly**.
- **Example**:
  - Start at a point on $J(w)$.
  - Take multiple tiny steps, reducing $J(w)$ incrementally.
  - Requires many iterations to approach the minimum.
- **Summary**: Gradient descent works but is inefficient.

---

### Case 2: Learning Rate $\alpha$ Too Large
- **Behavior**:
  - Steps are too large, overshooting the minimum.
  - The cost $J(w)$ may increase or oscillate, causing divergence.
- **Example**:
  - Start near the minimum.
  - Take a large step past the minimum, increasing $J(w)$.
  - Repeated large steps cause the algorithm to fail to converge.
- **Summary**: Gradient descent fails to converge and may diverge.

---

## Gradient Descent at a Local Minimum
- **Scenario**: $w$ is at a local minimum of $J(w)$.
- **Behavior**:
  - At a local minimum, $\frac{\partial J}{\partial w} = 0$.
  - Update rule becomes:

$$ w := w - \alpha \cdot 0 $$

$$ w := w $$

  - $w$ remains unchanged, as desired.
- **Conclusion**: Gradient descent correctly identifies and stays at the local minimum.

---

## Automatic Step Size Adjustment
- **Observation**: Gradient descent naturally takes smaller steps as it approaches a minimum.
- **Reason**:
  - The derivative $\frac{\partial J}{\partial w}$ decreases as $w$ approaches a minimum.
  - Smaller $\frac{\partial J}{\partial w}$ leads to smaller updates:

$$ \Delta w = -\alpha \frac{\partial J}{\partial w} $$

- **Example**:
  - Start at a high slope $\rightarrow$ large step.
  - As slope decreases $\rightarrow$ smaller steps.
  - Steps become negligible at the minimum.
- **Benefit**: This property allows gradient descent to converge even with a fixed $\alpha$.

---

## Summary Table

| **Learning Rate ($\alpha$)** | **Behavior**                                                                                  | **Outcome**                          |
|------------------------------|----------------------------------------------------------------------------------------------|--------------------------------------|
| Too Small                    | Tiny steps, slow convergence.                                                               | Works but inefficient.               |
| Too Large                    | Overshoots or oscillates around the minimum, may diverge.                                    | Fails to converge.                   |
| Just Right                   | Steps decrease as the slope decreases, allowing efficient convergence to the local minimum. | Works efficiently and correctly.     |

---

## Key Takeaways
1. **Choosing $\alpha$**:
   - Too small: Inefficient.
   - Too large: May fail to converge.
   - A moderate value ensures efficient convergence.
2. **Gradient Descent at Local Minima**:
   - Automatically stops when $\frac{\partial J}{\partial w} = 0$.
3. **Automatic Adjustment**:
   - Step sizes decrease as $J(w)$ approaches a minimum due to smaller derivatives.

---

# Gradient Descent for Linear Regression

## Overview
This lecture combines the linear regression model, the squared error cost function, and the gradient descent algorithm to train a model that fits a straight line to the training data.

---

## Components of Linear Regression with Gradient Descent

### Linear Regression Model
- Hypothesis function:

$$ f_{w,b}(x) = w \cdot x + b $$

### Squared Error Cost Function
- Measures the model's error:

$$ J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right)^2 $$

  - $m$: Number of training examples.
  - $x^{(i)}$, $y^{(i)}$: Input features and actual output for the $i$-th example.

---

## Gradient Descent Algorithm
- Update rules for $w$ and $b$:

$$ w := w - \alpha \cdot \frac{\partial J}{\partial w} $$

$$ b := b - \alpha \cdot \frac{\partial J}{\partial b} $$

- **Partial Derivatives**:
  - For $w$:

$$ \frac{\partial J}{\partial w} = \frac{1}{m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right) \cdot x^{(i)} $$

  - For $b$:

$$ \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right) $$

- **Simultaneous Updates**: Both $w$ and $b$ are updated at each step using their current values.

---

## Key Concepts and Insights

### Derivation of Partial Derivatives (Optional)
- **For $w$**:
  - Start with the cost function:

$$ J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right)^2 $$

  - Plug in $f_{w,b}(x^{(i)}) = w \cdot x^{(i)} + b$.
  - Differentiate w.r.t. $w$, simplify, and cancel constants:

$$ \frac{\partial J}{\partial w} = \frac{1}{m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right) \cdot x^{(i)} $$

- **For $b$**:
  - Similar process, without the $x^{(i)}$ term:

$$ \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right) $$

### Convex Cost Function
- **Bowl-Shaped Function**:
  - The squared error cost function is **convex**.
  - Convex functions have a **single global minimum**.
- **Property**:
  - Gradient descent on a convex function will always converge to the global minimum (if $\alpha$ is appropriately chosen).

---

## Example of Gradient Descent
1. Initialize $w$ and $b$.
2. Use the update rules:
   - Compute $\frac{\partial J}{\partial w}$ and $\frac{\partial J}{\partial b}$.
   - Update $w$ and $b$ simultaneously.
3. Repeat until convergence.

---

## Summary Table

| **Component**                  | **Formula/Definition**                                                                                       |
|--------------------------------|-------------------------------------------------------------------------------------------------------------|
| Hypothesis                     | $f_{w,b}(x) = w \cdot x + b$                                                                               |
| Cost Function                  | $J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right)^2$                        |
| $\frac{\partial J}{\partial w}$| $\frac{1}{m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right) \cdot x^{(i)}$                       |
| $\frac{\partial J}{\partial b}$| $\frac{1}{m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right)$                                      |
| Gradient Descent Rule          | $w := w - \alpha \cdot \frac{\partial J}{\partial w}, \, b := b - \alpha \cdot \frac{\partial J}{\partial b}$ |

---

## Key Takeaways
1. The gradient descent algorithm iteratively adjusts $w$ and $b$ to minimize $J(w, b)$.
2. Squared error cost function for linear regression is convex, ensuring convergence to the global minimum.
3. Properly compute and update $w$ and $b$ simultaneously for correct implementation.

---

# Gradient Descent in Action for Linear Regression

## Overview
This lecture demonstrates the application of **batch gradient descent** to train a linear regression model. We observe how the algorithm iteratively adjusts the parameters $w$ and $b$, reducing the cost function $J(w, b)$, and eventually fits a straight line to the data.

---

## Gradient Descent Workflow

### Initial Setup
- **Model**: $f(x) = w \cdot x + b$
- **Initialization**:
  - $w = -0.1$
  - $b = 900$
  - Corresponds to $f(x) = -0.1x + 900$

### Visualization Components
1. **Model Plot** (Top Left): Shows the line $f(x)$ fitting the data.
2. **Contour Plot** (Top Right): Displays the cost function $J(w, b)$ as contour lines.
3. **Surface Plot** (Bottom): A 3D visualization of $J(w, b)$.

---

## Steps of Gradient Descent
1. **Step 1**: Gradient descent starts at the initialized point $(w, b)$ on the cost function.
   - **Effect**: The cost decreases as $w$ and $b$ are updated.
   - **Model Update**: The line $f(x)$ changes slightly to fit the data better.
2. **Step 2**: Another step is taken.
   - **Effect**: The parameters $(w, b)$ move closer to the global minimum on the cost function.
   - **Model Update**: The fit improves further.
3. **Subsequent Steps**: Continue iteratively.
   - The cost function decreases steadily.
   - The straight line $f(x)$ better approximates the data.
4. **Final Step**: Convergence at the global minimum.
   - **Result**: The model achieves an optimal fit to the training data.

---

## Key Observations
- **Global Minimum**: 
  - The cost function $J(w, b)$ for linear regression is convex.
  - Gradient descent always converges to the global minimum if the learning rate $\alpha$ is appropriately chosen.
- **Model Utility**:
  - Once trained, $f(x)$ can predict outcomes. 
  - Example: For a house of size $1250 \, \text{sq ft}$, $f(x)$ predicts a price of approximately $250,000$.

---

## Batch Gradient Descent

### Definition
- Uses **all training examples** to compute the cost and its derivatives at each step:

$$ \frac{\partial J}{\partial w} = \frac{1}{m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right) \cdot x^{(i)} $$

$$ \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right) $$

- **Characteristics**:
  - Processes the entire batch of training data at every update step.
  - Ensures stability and consistent convergence for small datasets.

### Other Gradient Descent Variants
- Use smaller subsets of the data (e.g., stochastic or mini-batch gradient descent).
- Focus of this lecture: Batch gradient descent.

## Summary Table

| **Component**            | **Description**                                                                                     |
|--------------------------|-----------------------------------------------------------------------------------------------------|
| **Gradient Descent**      | Iteratively updates $w$ and $b$ to minimize $J(w, b)$ and fit the data.                           |
| **Batch Gradient Descent**| Processes all training examples to compute derivatives and update parameters.                     |
| **Visualization**         | Shows model updates, cost reduction, and convergence to the global minimum.                      |
| **Convexity**             | The squared error cost function is convex, ensuring convergence to a global minimum.             |

---

## Key Takeaways
1. Gradient descent optimizes the model by iteratively reducing the cost function $J(w, b)$.
2. The convexity of $J(w, b)$ ensures that batch gradient descent converges to the global minimum.
3. Batch gradient descent uses the entire training set for stable parameter updates.


**Congratulations!** You’ve successfully implemented your first machine learning model. 🎉


