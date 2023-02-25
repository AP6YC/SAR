import random
from micrograd.engine import Value

class Neuron:
    def __init__(self, nin, nonlin=True, tanh=False):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
        self.nonlin = nonlin
        self.tanh = tanh

    def __call__(self, x):
        act = sum(wi*xi for wi, xi in zip(self.w, x)) + self.b
        if self.tanh:
            out = act.tanh()
        else:
            out = act.relu() if self.nonlin else act
        return out

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

    def parameters(self):
        return self.w + [self.b]

class Layer:

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        # return outs
        return outs[0] if len(outs) == 1 else outs

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:

    def __init__(self, nin, nouts, tanh=False):
        sz = [nin] + nouts
        # self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
        self.layers = [Layer(sz[i], sz[i+1], nonlin=(i!=len(nouts)-1), tanh=tanh) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"

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

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

