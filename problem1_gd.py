# Problem 1 - Question 5 - Extra Credit

# Dataset
data = [
    (-4, 0, -1),
    (-1, 1, 1),
    (0, -1, -1),
    (2, 1, 1),
    (3, 0, 1),
    (6, -1, -1)
]

# Initialize weights
w0 = 0
w1 = 0
w2 = 0

# Learning rate
eta = 0.1

# Convergence threshold
threshold = 1e-5

# Maximum iterations safeguard
max_iters = 10000

for iteration in range(max_iters):

    grad_w0 = 0
    grad_w1 = 0
    grad_w2 = 0

    # Compute gradients using ALL examples
    for x1, x2, y in data:

        prediction = w0 + w1 * x1 + w2 * x2
        margin = y * prediction

        # Hinge loss active only if margin < 1
        if margin < 1:
            grad_w0 += -y
            grad_w1 += -y * x1
            grad_w2 += -y * x2

    # Save old weights
    old_w0 = w0
    old_w1 = w1
    old_w2 = w2

    # Gradient descent update
    w0 = w0 - eta * grad_w0
    w1 = w1 - eta * grad_w1
    w2 = w2 - eta * grad_w2

    # Check convergence
    change = (
        abs(w0 - old_w0) +
        abs(w1 - old_w1) +
        abs(w2 - old_w2)
    )

    if change < threshold:
        break

print("Final weights:")
print("w0 =", w0)
print("w1 =", w1)
print("w2 =", w2)

# Compute training error
incorrect = 0

for x1, x2, y in data:

    prediction = w0 + w1 * x1 + w2 * x2

    if prediction >= 0:
        predicted_label = 1
    else:
        predicted_label = -1

    if predicted_label != y:
        incorrect += 1

misclassification_rate = incorrect / len(data)

print("Misclassification rate:",
      misclassification_rate)