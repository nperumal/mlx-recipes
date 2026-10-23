"""XOR with a two-layer network, backpropagating by hand.

perceptron.py showed that a single neuron solves NAND. XOR is the problem it
cannot touch: no straight line puts {(0,1), (1,0)} on one side and
{(0,0), (1,1)} on the other. A single neuron trained on XOR settles at a loss of
exactly ln 2 and predicts 0.5 everywhere -- not struggling, but out of capacity.
Minsky and Papert published that limitation in 1969 and the field largely stopped
for a decade.

The fix is a hidden layer, and the hidden layer needs backpropagation. This file
derives it by hand so the chain rule is visible:

    output layer    delta2 = (y - a2) * sigma'(a2)
    hidden layer    delta1 = (delta2 . W2^T) * sigma'(a1)

mlp_nn.py is the same idea with mx.grad doing that derivation for you.

Four rows is the entire input space, so there is no held-out set -- this is
function fitting, not learning in the statistical sense.
"""

import mlx.core as mx

SEED = 0
LEARNING_RATE = 1.0
EPOCHS = 10000
HIDDEN_DIM = 2
LOG_EVERY = 2000


class MLP:
    """Two layers, sigmoid throughout, gradients written out longhand."""

    def __init__(self, input_dim, hidden_dim, output_dim, seed=SEED):
        mx.random.seed(seed)
        self.W1 = mx.random.normal((input_dim, hidden_dim))
        self.b1 = mx.zeros((1, hidden_dim))
        self.W2 = mx.random.normal((hidden_dim, output_dim))
        self.b2 = mx.zeros((1, output_dim))

    def sigmoid(self, x):
        return 1 / (1 + mx.exp(-x))

    def sigmoid_derivative(self, activation):
        """sigma'(z) expressed in terms of sigma(z), not z.

        The argument must be the ACTIVATION, never the pre-activation. Pass z
        here by mistake and the network still trains, just toward the wrong
        thing -- which is why the parameter is not named x.
        """
        return activation * (1 - activation)

    def forward(self, X):
        self.a1 = self.sigmoid(mx.matmul(X, self.W1) + self.b1)
        self.a2 = self.sigmoid(mx.matmul(self.a1, self.W2) + self.b2)
        return self.a2

    def backward(self, X, y, output, lr):
        # Output layer: how wrong, scaled by how much the sigmoid was moving.
        output_delta = (y - output) * self.sigmoid_derivative(output)

        # Hidden layer: push the output error back through W2, then scale by
        # the hidden sigmoid's slope. This line is the chain rule.
        hidden_error = mx.matmul(output_delta, self.W2.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.a1)

        # Ascend, because the deltas carry (y - output) rather than (output - y).
        self.W2 += lr * mx.matmul(self.a1.T, output_delta)
        self.b2 += lr * mx.sum(output_delta, axis=0, keepdims=True)
        self.W1 += lr * mx.matmul(X.T, hidden_delta)
        self.b1 += lr * mx.sum(hidden_delta, axis=0, keepdims=True)

    def train(self, X, y, epochs=EPOCHS, lr=LEARNING_RATE, log_every=0):
        for epoch in range(1, epochs + 1):
            output = self.forward(X)
            self.backward(X, y, output, lr)
            # Force the update without copying values back to the host.
            mx.eval(self.W1, self.b1, self.W2, self.b2)
            if log_every and (epoch == 1 or epoch % log_every == 0):
                mse = mx.mean(mx.square(self.forward(X) - y))
                mx.eval(mse)
                print(f"  epoch {epoch:5d}: MSE {mse.item():.6f}")


X = mx.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=mx.float32)
y = mx.array([[0], [1], [1], [0]], dtype=mx.float32)  # XOR


def main():
    print(
        f"hand-written backprop, {HIDDEN_DIM} hidden units, "
        f"lr={LEARNING_RATE}, {EPOCHS} epochs, seed {SEED}"
    )
    mlp = MLP(input_dim=2, hidden_dim=HIDDEN_DIM, output_dim=1)
    mlp.train(X, y, log_every=LOG_EVERY)

    predictions = mlp.forward(X)
    mse = mx.mean(mx.square(predictions - y))
    mx.eval(predictions, mse)

    print(f"\n  final MSE {mse.item():.6f}")
    print("\n  x1  x2 | target   out    pred")
    for i in range(X.shape[0]):
        x1, x2 = (int(v) for v in X[i].tolist())
        want = int(y[i].item())
        out = predictions[i].item()
        got = int(out >= 0.5)
        print(f"   {x1}   {x2} |   {want}    {out:6.4f}    {got}   {'ok' if got == want else 'XX'}")

    print(
        "\nA single neuron cannot do this at any learning rate or for any number of\n"
        "epochs. Two layers can, and nothing else changed."
    )


if __name__ == "__main__":
    main()
