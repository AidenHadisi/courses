# Python 3.13 f-strings Cheatsheet 🚀

Python 3.13 introduces **new features** for f-strings, making them even more powerful and flexible. Below is a **comprehensive cheatsheet** covering all aspects of f-strings in Python 3.13.

---

## 🔹 1. Basic f-String Syntax

```python
name = "Alice"
age = 30
print(f"My name is {name} and I am {age} years old.")
# Output: My name is Alice and I am 30 years old.
```

---

## 🔹 2. Expressions Inside f-Strings

You can evaluate expressions inside `{}`.

```python
a = 10
b = 5
print(f"Sum: {a + b}, Product: {a * b}")
# Output: Sum: 15, Product: 50
```

---

## 🔹 3. String Formatting

### 📌 Width & Alignment

```python
text = "Hello"
print(f"[{text:>10}]")  # Right align
print(f"[{text:<10}]")  # Left align
print(f"[{text:^10}]")  # Center align
# Output:
# [     Hello]
# [Hello     ]
# [  Hello   ]
```

### 📌 Padding with Characters

```python
print(f"[{text:_>10}]")  # Fill empty space with "_"
# Output: [_____Hello]
```

### 📌 Truncating Strings

```python
print(f"{text:.3}")  # Only first 3 characters
# Output: Hel
```

---

## 🔹 4. Number Formatting

### 📌 Decimal Places

```python
pi = 3.1415926535
print(f"{pi:.2f}")  # 2 decimal places
# Output: 3.14
```

### 📌 Thousands Separator

```python
num = 1000000
print(f"{num:,}")  # Add commas
# Output: 1,000,000
```

### 📌 Percentage Format

```python
percent = 0.8543
print(f"{percent:.1%}")  # Convert to percentage
# Output: 85.4%
```

### 📌 Binary, Hex, and Octal

```python
num = 255
print(f"Binary: {num:b}, Hex: {num:x}, Octal: {num:o}")
# Output: Binary: 11111111, Hex: ff, Octal: 377
```

---

## 🔹 5. Nested f-Strings (New in Python 3.13)

You can now **nest** f-strings inside f-strings!

```python
name = "Alice"
style = "upper"
print(f"My name is {f'{name}'.upper() if style == 'upper' else name}")
# Output: My name is ALICE
```

---

## 🔹 6. Debugging with `=`

Using `=` inside an f-string shows both the variable name and its value.

```python
x = 42
print(f"{x=}")
# Output: x=42
```

You can combine it with expressions:

```python
print(f"{x + 5=}")
# Output: x + 5=47
```

---

## 🔹 7. Using `!r`, `!s`, and `!a`

### 📌 `!r` (repr) → Shows raw representation

```python
text = "Hello\nWorld"
print(f"{text!r}")
# Output: 'Hello\nWorld'
```

### 📌 `!s` (str) → Default string conversion (same as `str()`)

```python
print(f"{text!s}")
# Output: Hello
#         World
```

### 📌 `!a` (ascii) → ASCII representation

```python
print(f"{text!a}")
# Output: 'Hello\nWorld'
```

---

## 🔹 8. Multiline f-Strings

```python
name = "Alice"
age = 30
message = (
    f"My name is {name}.\n"
    f"I am {age} years old."
)
print(message)
# Output:
# My name is Alice.
# I am 30 years old.
```

---

## 🔹 9. Lambda Functions Inside f-Strings (New in Python 3.13)

Python 3.13 allows **inline lambdas** inside f-strings.

```python
print(f"{(lambda x: x * 2)(10)}")
# Output: 20
```

---

## 🔹 10. Assigning f-Strings to Variables (New in Python 3.13)

Python 3.13 lets you **assign** f-strings directly without using `str.format()`.

```python
x = 10
y = 20
formatted_string = f"{x} + {y} = {x + y}"
print(formatted_string)
# Output: 10 + 20 = 30
```

---

## 🔹 11. Lazy Evaluation with `=` Debugging (New in Python 3.13)

```python
import random
print(f"{random.randint(1, 100)=}")
# Output: random.randint(1, 100)=42 (for example)
```

---

## 🔹 12. Using `locals()` and `globals()`

```python
name = "Alice"
age = 30
print(f"Locals: {locals()}")
```

Outputs all local variables as a dictionary.

---

## 🔹 13. Performance Considerations

f-strings are **faster** than `.format()` and `%` formatting.

```python
import timeit
print(timeit.timeit("""f'Hello {42}'""", number=1000000))
```

---

## 🔹 14. Escaping `{` and `}`

If you need literal `{}` inside an f-string:

```python
print(f"{{This is inside curly braces}}")
# Output: {This is inside curly braces}
```

---

## 🔹 15. Combining f-Strings with JSON (New in Python 3.13)

```python
import json
data = {"name": "Alice", "age": 30}
print(f"{json.dumps(data, indent=2)}")
```

---

### 🎯 Summary of New Features in Python 3.13:

✅ **Nested f-strings** (`f"{f'{x}'}"`)  
✅ **Lambda functions in f-strings**  
✅ **Lazy evaluation in `=` debugging**  
✅ **Better variable assignments with f-strings**  
✅ **More consistent and readable string interpolation**

---

🔥 Now you have a **complete cheatsheet** for **Python 3.13 f-strings**! 🚀 Let me know if you need further clarification! 😊
