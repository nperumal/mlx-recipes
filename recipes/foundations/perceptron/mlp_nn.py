"""The same XOR network, with mx.grad deriving the gradients instead of you.

mlp.py writes backpropagation out longhand -- two deltas, two transposes, four
parameter updates. This file is the same shape of network, with the whole
backward() method replaced by one line:

    loss_and_grad = nn.value_and_grad(model, loss_fn)

Everything mlp.py computes by hand, autograd computes from the forward pass.

It also shows why one neuron cannot do this, by training one alongside: it stops
at a loss of ln 2 = 0.6931 and predicts 0.5 on all four inputs.
"""

import math

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

SEED = 0
LEARNING_RATE = 0.1
NUM_ITERS = 3000
HIDDEN_DIM = 4

X = mx.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=mx.float32)
y = mx.array([0, 1, 1, 0], dtype=mx.float32)  # XOR


class Neuron(nn.Module):
    """One linear unit -- the model perceptron.py trains. Cannot represent XOR."""

    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, 1)

    def __call__(self, x):
        return self.linear(x).squeeze(-1)


class MLP(nn.Module):
    """Two layers with a nonlinearity between them."""

    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)

    def __call__(self, x):
        # Nonlinearity between the layers, nothing after the last one: the
        # output is a logit and the sigmoid is applied when predicting. Two
        # linear layers with nothing between them would collapse into one.
        #
        # tanh rather than sigmoid, and four hidden units rather than two, for
        # measured reasons rather than conventional ones -- see the README's
        # Gotchas. At width 2 this problem has a local minimum that every
        # activation falls into on most seeds.
        return self.linear2(mx.tanh(self.linear1(x))).squeeze(-1)


def bce_with_logits(z, y):
    """Binary cross-entropy from logits, as in the logistic-regression recipe."""
    zero = mx.zeros_like(z)
    return mx.mean(y * mx.logaddexp(zero, -z) + (1 - y) * mx.logaddexp(zero, z))


def train(model, iters=NUM_ITERS, lr=LEARNING_RATE):
    optimizer = optim.Adam(learning_rate=lr)
    # This single line replaces mlp.py's entire backward() method.
    loss_and_grad = nn.value_and_grad(model, lambda m, a, b: bce_with_logits(m(a), b))
    mx.eval(model.parameters())

    for _ in range(iters):
        loss, grads = loss_and_grad(model, X, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)

    return loss


def report(name, model, loss):
    probs = mx.sigmoid(model(X))
    predictions = (probs >= 0.5).astype(mx.int32)
    mx.eval(probs, predictions, loss)
    correct = sum(int(predictions[i].item()) == int(y[i].item()) for i in range(X.shape[0]))

    print(f"\n{name}")
    print(f"  loss {loss.item():.4f}   accuracy {correct}/{X.shape[0]}")
    print("  x1  x2 | target   P(1)   pred")
    for i in range(X.shape[0]):
        x1, x2 = (int(v) for v in X[i].tolist())
        want, got = int(y[i].item()), int(predictions[i].item())
        print(
            f"   {x1}   {x2} |   {want}    {probs[i].item():6.4f}    {got}   "
            f"{'ok' if want == got else 'XX'}"
        )


def main():
    print(f"XOR, Adam(lr={LEARNING_RATE}), {NUM_ITERS} iterations, seed {SEED}")

    mx.random.seed(SEED)
    neuron = Neuron(2)
    report("single neuron (the model perceptron.py trains)", neuron, train(neuron))
    print(
        f"  ln 2 = {math.log(2):.4f}. A loss of exactly that, with P(1) = 0.5 on every\n"
        "  input, is the model predicting the base rate because it has nothing better.\n"
        "  No straight line separates XOR, so there is nothing to find."
    )

    mx.random.seed(SEED)
    mlp = MLP(2, HIDDEN_DIM)
    report(f"two layers, {HIDDEN_DIM} hidden units", mlp, train(mlp))
    print(
        "\n  Same data, same optimizer, same number of steps. The only change is one\n"
        "  hidden layer with a nonlinearity -- and mx.grad derived every gradient\n"
        "  that mlp.py spells out by hand."
    )


if __name__ == "__main__":
    main()
