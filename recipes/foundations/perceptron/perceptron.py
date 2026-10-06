"""The perceptron learning rule, by hand. Rosenblatt, 1958.

Note what is absent: there is no loss function and no gradient anywhere in this
file. The perceptron predates backpropagation by nearly three decades. It uses a
hard step function, which has zero derivative everywhere it is defined and no
derivative at all at the threshold -- so there is nothing to differentiate. The
update rule is driven directly by whether each prediction was wrong:

    w <- w - lr * (y_pred - y) . X
    b <- b - lr * sum(y_pred - y)

NAND is linearly separable, so a single perceptron can represent it and the rule
is guaranteed to find a separating line. XOR is not, which is what perceptron_nn.py
and the wider history of the field are about.
"""

import mlx.core as mx

# The entire input space: every combination of two bits.
X = mx.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=mx.float32)
y = mx.array([1, 1, 1, 0], dtype=mx.float32)  # NAND

lr = 0.1
max_epochs = 20

w = mx.array([0.0, 0.0], dtype=mx.float32)
b = mx.array(0.0, dtype=mx.float32)


def step_function(x):
    "Heaviside step function that returns 1 for x >= 0 and 0 for x < 0."
    return mx.where(x >= 0, 1.0, 0.0)


converged_at = None
for epoch in range(1, max_epochs + 1):
    # forward pass
    z = mx.matmul(X, w) + b
    y_pred = step_function(z)

    # how many predictions were wrong, and in which direction
    errors = y_pred - y
    mistakes = int(mx.sum(mx.abs(errors)).item())

    print(
        f"epoch {epoch:2d}: mistakes {mistakes}/{X.shape[0]}  "
        f"w = [{w[0].item():+.2f}, {w[1].item():+.2f}]  b = {b.item():+.2f}"
    )

    if mistakes == 0:
        converged_at = epoch
        break

    # update weights and bias.
    # errors is (4,) and X is (4, 2), so the matmul sums each feature's
    # contribution across the batch and gives a (2,) update for w.
    w -= lr * mx.matmul(errors, X)
    # b must stay a scalar. Writing `b -= lr * errors` here would broadcast
    # (1,) against (4,) and silently give the model one bias PER TRAINING ROW,
    # which memorises the truth table while w never leaves zero.
    b -= lr * mx.sum(errors)
    mx.eval(w, b)

if converged_at:
    print(f"\nconverged after {converged_at} epochs")
else:
    print(f"\nno separating line found in {max_epochs} epochs")

# inference
z = mx.matmul(X, w) + b
y_pred = step_function(z)
mx.eval(y_pred)
print(f"  w = [{w[0].item():+.2f}, {w[1].item():+.2f}]   b = {b.item():+.2f}")
if abs(w[1].item()) > 1e-6:
    print(f"  boundary: x2 = {-w[0].item() / w[1].item():+.2f} * x1 {-b.item() / w[1].item():+.2f}")
print("\n  x1  x2 | target  pred")
for i in range(X.shape[0]):
    x1, x2 = (int(v) for v in X[i].tolist())
    want, got = int(y[i].item()), int(y_pred[i].item())
    print(f"   {x1}   {x2} |   {want}      {got}   {'ok' if want == got else 'XX'}")
