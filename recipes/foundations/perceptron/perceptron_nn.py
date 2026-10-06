"""The same NAND gate, learned by gradient descent instead of the perceptron rule.

perceptron.py uses Rosenblatt's 1958 update rule: a hard step function and a
correction applied whenever a prediction is wrong. It works, but it is a dead end,
and the reason is the first thing this file prints -- the step function's
derivative is zero everywhere:

    d/dx step(x)    = [0.0, 0.0, 0.0, 0.0, 0.0]
    d/dx sigmoid(x) = [0.105, 0.235, 0.25, 0.235, 0.105]

With a zero gradient there is nothing to backpropagate, so a step-function network
cannot be trained by gradient descent and cannot usefully be stacked into layers.
Replacing the step with a sigmoid gives a smooth version of the same decision, with
a gradient that says how wrong the prediction was and in which direction. That
substitution is what made deep networks trainable.

The step function survives here only at prediction time, where the model has to
commit to a 0 or a 1. Training works on the logits.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

X = mx.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=mx.float32)
y = mx.array([1, 1, 1, 0], dtype=mx.float32)  # NAND

LEARNING_RATE = 0.1
NUM_ITERS = 2000
SEED = 0


class Perceptron(nn.Module):
    """One linear unit returning a raw logit.

    No activation in __call__ on purpose: the loss is computed from logits, and
    the sigmoid is applied only when predicting.
    """

    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, 1)

    def __call__(self, x):
        # nn.Linear gives (batch, 1); squeeze to (batch,) so it matches the
        # targets. Without this the loss silently broadcasts to (batch, batch).
        return self.linear(x).squeeze(-1)


def bce_with_logits(z, y):
    """Binary cross-entropy from logits -- the same function as the
    logistic-regression recipe, for the same numerical-stability reason."""
    zero = mx.zeros_like(z)
    return mx.mean(y * mx.logaddexp(zero, -z) + (1 - y) * mx.logaddexp(zero, z))


def show_why_the_step_function_fails():
    probe = mx.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    step_grad = mx.grad(lambda x: mx.sum(mx.where(x >= 0, 1.0, 0.0)))(probe)
    sigmoid_grad = mx.grad(lambda x: mx.sum(mx.sigmoid(x)))(probe)
    mx.eval(step_grad, sigmoid_grad)
    print("why not just differentiate the step function?")
    print(f"  x               {[round(v, 2) for v in probe.tolist()]}")
    print(f"  d/dx step(x)    {[round(v, 3) for v in step_grad.tolist()]}   <- no signal")
    print(f"  d/dx sigmoid(x) {[round(v, 3) for v in sigmoid_grad.tolist()]}")
    print()


def train():
    mx.random.seed(SEED)
    model = Perceptron(X.shape[1])
    optimizer = optim.SGD(learning_rate=LEARNING_RATE)
    loss_and_grad = nn.value_and_grad(model, lambda m, a, b: bce_with_logits(m(a), b))
    mx.eval(model.parameters())

    for _ in range(NUM_ITERS):
        loss, grads = loss_and_grad(model, X, y)
        optimizer.update(model, grads)
        # Force the computation without copying values back to the host.
        mx.eval(model.parameters(), optimizer.state, loss)

    return model, loss


def main():
    show_why_the_step_function_fails()

    model, loss = train()
    probs = mx.sigmoid(model(X))
    predictions = (probs >= 0.5).astype(mx.int32)
    w = model.linear.weight.squeeze(0)
    b = model.linear.bias
    mx.eval(probs, predictions, w, b, loss)

    print(f"gradient descent, SGD(lr={LEARNING_RATE}), {NUM_ITERS} iterations")
    print(f"  loss {loss.item():.6f}")
    print(f"  w = [{w[0].item():+.2f}, {w[1].item():+.2f}]   b = {b.item():+.2f}")
    if abs(w[1].item()) > 1e-6:
        print(
            f"  boundary: x2 = {-w[0].item() / w[1].item():+.2f} * x1 "
            f"{-b.item() / w[1].item():+.2f}"
        )

    print("\n  x1  x2 | target   P(1)   pred")
    for i in range(X.shape[0]):
        x1, x2 = (int(v) for v in X[i].tolist())
        want, got = int(y[i].item()), int(predictions[i].item())
        print(
            f"   {x1}   {x2} |   {want}    {probs[i].item():6.4f}    {got}   "
            f"{'ok' if want == got else 'XX'}"
        )

    print(
        "\nSame boundary as perceptron.py, found a different way. The perceptron rule\n"
        "stops the moment it classifies everything correctly; gradient descent keeps\n"
        "pushing the line toward higher confidence, which is why |w| here is larger."
    )


if __name__ == "__main__":
    main()
