import random
from micrograd.engine import Value

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        act = sum(wi*xi for wi, xi in zip(self.w, x)) + self.b
        out = act.tanh()
        # out = act.relu()
        return out

    def parameters(self):
        return self.w + [self.b]

class Layer:

    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        # return outs
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, xs):
        return [self(x) for x in xs]

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def backprop(self, xs, ys):
        # Forward pass
        ypred = self.forward(xs)
        # Loss
        loss = sum([(yout - ygt)**2 for ygt, yout in zip(ys, ypred)])
        # Backward pass
        loss.backward()
        # Update
        for p in self.parameters():
            p.data -= 0.2 * p.grad
            p.grad = 0.0

        return loss

    def train(self, xs, ys):
        for ix in range(25):
            loss = self.backprop(xs, ys)
            print(ix, loss.data)
