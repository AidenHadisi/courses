Here’s a **comprehensive cheat sheet for derivatives**, covering the basics, rules, examples, and more advanced topics. It’s designed to help you learn and understand derivatives step by step.

---

## **What is a Derivative?**

The **derivative** of a function measures how the function changes as its input changes. It represents the **slope** or **rate of change** of the function at any given point.

### Notation:

- $f′(x)$: Derivative of $f(x)$
- $\frac{dy}{dx}$: Derivative of $y$ with respect to $x$
- $\frac{d}{dx}[f(x)]$: Another way to express the derivative of $f(x)$

---

## **Basic Rules of Differentiation**

### 1. **Constant Rule**

$$\frac{d}{dx}[c] = 0$$

- The derivative of a constant is always $0$.
- Example: $\frac{d}{dx}[5] = 0$

---

### 2. **Power Rule**

$$\frac{d}{dx}[x^n] = n \cdot x^{n-1}$$

- The derivative of $x^n$ is $n$ times $x$ raised to the power of $n-1$.
- Example: $\frac{d}{dx}[x^3] = 3x^2$

---

### 3. **Constant Multiple Rule**

$$\frac{d}{dx}[c \cdot f(x)] = c \cdot \frac{d}{dx}[f(x)]$$

- A constant multiplier can be factored out.
- Example: $\frac{d}{dx}[3x^2] = 3 \cdot \frac{d}{dx}[x^2] = 6x$

---

### 4. **Sum Rule**

$$\frac{d}{dx}[f(x) + g(x)] = \frac{d}{dx}[f(x)] + \frac{d}{dx}[g(x)]$$

- The derivative of a sum is the sum of the derivatives.
- Example: $\frac{d}{dx}[x^2 + 2x] = \frac{d}{dx}[x^2] + \frac{d}{dx}[2x] = 2x + 2$

---

### 5. **Difference Rule**

$$\frac{d}{dx}[f(x) - g(x)] = \frac{d}{dx}[f(x)] - \frac{d}{dx}[g(x)]$$

- The derivative of a difference is the difference of the derivatives.
- Example: $\frac{d}{dx}[x^3 - 4x] = 3x^2 - 4$

---

## **Product and Quotient Rules**

### 6. **Product Rule**

$$\frac{d}{dx}[f(x) \cdot g(x)] = f'(x) \cdot g(x) + f(x) \cdot g'(x)$$

- Use this when two functions are multiplied together.
- Example: $\frac{d}{dx}[x \cdot e^x] = 1 \cdot e^x + x \cdot e^x = e^x(1 + x)$

---

### 7. **Quotient Rule**

$$\frac{d}{dx}\left[\frac{f(x)}{g(x)}\right] = \frac{f'(x) \cdot g(x) - f(x) \cdot g'(x)}{[g(x)]^2}$$

- Use this when one function is divided by another.
- Example: $\frac{d}{dx}\left[\frac{x^2}{\sin x}\right] = \frac{2x \cdot \sin x - x^2 \cdot \cos x}{\sin^2 x}$

---

## **Chain Rule**

### 8. Chain Rule (Composite Functions)

$$\frac{d}{dx}[f(g(x))] = f'(g(x)) \cdot g'(x)$$

- Use this for nested functions.
- Example: $\frac{d}{dx}[(2x + 1)^3]$
    - Let $u=2x+1$, so $\frac{d}{dx}[u^3] = 3u^2 \cdot \frac{du}{dx} = 3(2x + 1)^2 \cdot 2 = 6(2x + 1)^2$

---

## **Common Derivatives**

### 1. Polynomials:

$$\frac{d}{dx}[x^n] = n \cdot x^{n-1}$$

### 2. Exponentials:

$$\frac{d}{dx}[e^x] = e^x, \quad \frac{d}{dx}[a^x] = a^x \cdot \ln(a)$$

### 3. Logarithms:

$$\frac{d}{dx}[\ln x] = \frac{1}{x}, \quad \frac{d}{dx}[\log_a x] = \frac{1}{x \cdot \ln(a)}$$

### 4. Trigonometric Functions:

$$\frac{d}{dx}[\sin x] = \cos x, \quad \frac{d}{dx}[\cos x] = -\sin x, \quad \frac{d}{dx}[\tan x] = \sec^2 x$$

### 5. Inverse Trigonometric Functions:

$$\frac{d}{dx}[\arcsin x] = \frac{1}{\sqrt{1 - x^2}}, \quad \frac{d}{dx}[\arccos x] = -\frac{1}{\sqrt{1 - x^2}}$$

---

## **Higher-Order Derivatives**

1. **First Derivative** $(f'(x))$: Slope or rate of change.
2. **Second Derivative** $(f''(x))$: Concavity (curvature of the graph).
    - $f''(x) > 0$: Concave up (smile shape).
    - $f''(x) < 0$: Concave down (frown shape).

---

## **Practical Examples**

### Example 1: Simple Derivative

Find $\frac{d}{dx}[3x^3 + 2x - 5]$:

$$\frac{d}{dx}[3x^3] + \frac{d}{dx}[2x] - \frac{d}{dx}[5] = 9x^2 + 2 - 0$$

### Example 2: Product Rule

Find $\frac{d}{dx}[x \cdot \sin x]$:

$$f(x)=x$$

$$g(x)=sinx$$

$$f′(x)=1$$

$$g′(x)=cosx$$

$$dxd​[x⋅sinx]=f′(x)⋅g(x)+f(x)⋅g′(x)=1⋅sinx+x⋅cosx=sinx+xcosx$$

### Example 3: Chain Rule

Find $\frac{d}{dx}[(3x^2 + 1)^4]$:

$$f(g(x))=(3x2+1)4$$

$$g(x)=3x2+1$$

$$f'(g(x)) = 4(3x^2 + 1)^3$$

$$g'(x) = 6x$$

$$\frac{d}{dx}[(3x^2 + 1)^4] = 4(3x^2 + 1)^3 \cdot 6x = 24x(3x^2 + 1)^3$$

---

## **Tips for Derivatives**

1. **Memorize common derivatives** (e.g., powers, exponentials, and trig functions).
2. **Break complex problems into smaller parts** (use product, quotient, or chain rules).
3. **Practice interpreting graphs**:
    - Positive derivative: Function is increasing.
    - Negative derivative: Function is decreasing.

---

This cheat sheet covers the essentials and should give you a strong foundation. Let me know if you'd like more examples or advanced topics!