# Neural networks in C++

This library provides components used for building advanced neural networks. It is built on Eigen (for speed and ease of development) and tries to be as flexible as possible. Some of its main features are :

* Very flexible architecture: neural networks consist of a graph of nodes (cycles allowed), which allows an easy implementation of feed-forward deep networks, recurrent networks, autoencoders, complex networks that do several things in parallel and then merge results, etc.
* Built on Eigen, known for its speed
* Time-series can be trained in an efficient and incremental fashion (you can predict t(0)...t(n), then train t(n), then predict t(n+1) without having to re-predict t(0)...t(n), then train t(n+1), etc)
* Fully unit-tested

# Simple perceptron

The library consists of a `Network` class, that manages training and prediction, and several `AbstractNode` subclasses. Examples of usages of these classes can be found in the `tests/` directory, but here are some more detailed examples.

Creating a neural network starts by instantiating a `Network` object, that will keep track of nodes :

```cpp
Network *net = new Network(num_inputs);
```

Now that the `Network` object has been created, `AbstractNode` subclasses can be instantiated. The most important subclasses are `Dense` (a fully-connected weighted dense layer, that learns its weights) and the different kinds of `Activation` nodes (`SigmoidActivation`, `TanhActivation`).

```cpp
Dense *layer1 = new Dense(num_hidden, 0.005);
SigmoidActivation *layer1_act = new SigmoidActivation;
Dense *layer2 = new Dense(num_outputs, 0.005);
SigmoidActivation *layer2_act = new SigmoidActivation;
```

The above code snippet creates the nodes required for the implementation of a single-hidden-layer feed-forward perceptron. The first dense node connects the `num_inputs` inputs of the network to the `num_hidden` neurons of the hidden layer. The second dense node connects the `num_hidden` neurons of the hidden layer to the `num_outputs` output neurons. The constructor of `Dense` takes as parameter the number of output neurons of the dense node. The number of input neurons will be inferred when the nodes are connected to each other (when `Dense` knows from what to take its input).

Those two dense layers don't have any activation function by themselves (they use a linear activation), so two `SigmoidActivation` nodes will be used in order to add activations to them.

Now that the nodes are created, they can be wired together. Each `AbstractNode` subclass exposes an *output port* (producing values and consuming error signals), and can have one or several input ports. In this simple example, all the nodes used have only one input port.

```cpp
layer1->setInput(net->inputPort());
layer1_act->setInput(layer1->output());
layer2->setInput(layer1_act->output());
layer2_act->setInput(layer2->output());
```

The last step, that is simple when building feed-forward neural networks but can become tricky when building recurrent networks, consists of adding the layers to the network. The layers must be added in the order in which they have to forward their values. In this simple example, we will add `layer1`, then `layer1_act`, then `layer2` and finally `layer2_act`. If there are loops in the network graph, then an order has to be decided. It is usually the breadth-first order that is used.

```cpp
net->addNode(layer1);
net->addNode(layer1_act);
net->addNode(layer2);
net->addNode(layer2_act);
```

The first node added to `Network` is the input (it receives input vectors). The last one is the output and produces the value returned by `Network::predict()`.

This neural network is now complete and can be trained using `Network::train` (performing a single gradient update on a single example). The `tests/utils.h` file contains some utility functions that can be used in order to train a network on a batch of input/output samples.

# Merge nodes

Merge nodes are special types of nodes that take as many inputs as one wants, and merge them. Merging can be done either by adding the inputs (component-wise adding, so the first output neuron is the sum of the first neuron of all the inputs), or by multiplying them. Those nodes can be used to implement network that have gates: the output of a `Dense` node is multiplied by another one, that serves as a gate, which allows to design gated recurrent networks that can learn to forget, copy, or anything else. Those merge nodes are used internally by the GRU node.

The following snippet shows to to create a network that learns a function like `(ax + b)*(cx + d)`, with the a, b, c and d parameters learned by the `Dense` nodes.

```cpp
Network *net = new Network(1);
Dense *dense1 = new Dense(1, 0.05);
Dense *dense2 = new Dense(1, 0.05);
MergeSum *product = new MergeProduct;

dense1->setInput(net->inputPort());
dense2->setInput(net->inputPort());
product->addInput(dense1->output());
product->addInput(dense2->output());

net->addNode(dense1);
net->addNode(dense2);
net->addNode(product);
```

# Recurrent networks

Using recurrend nodes (`GRU` for instance, which works a bit like the famous `LSTM` networks but are sometimes a bit more efficient and stable) is easy. They are added to a network exactly like other nodes.

Building recurrent networks piece by piece, so by assembling `Dense`, `Activation` and merge nodes, is a bit more complicated. The topology of the network first has to be designed (which can be done by looking at formulas or graphs explaining how the recurrent network works), then all the nodes have to be instantiated and wired in the correct way.

Once the wiring is correct, they have to be added to a `Network` in the right order. The main principle is that, during the forward pass, no node can be forwarded until all its dependencies have been forwarded. If all your nodes depend on `h(t-1)`, then ensure that you set the output of the node reprenting `h` to zero before starting to train a sequence, and add the networks so that the node (usually a `Dense`) connecting `h` to the input is forwarded first. The other nodes are usually easier to connect. Here is how the GRU unit has been connected :

```cpp
// Wire h(t-1) to what depends on it (but not on other things)
_nodes.push_back(loop_output_to_updates);
_nodes.push_back(loop_output_to_resets);

// resets merges the h(t-1) (wired in the above step) and user-specified reset signals
_nodes.push_back(resets);
_nodes.push_back(reset_activation);
_nodes.push_back(reset_times_output);  // depends on h(t-1) and reset_activation, okay to add here

// Now that information can flow from resets to its activation to reset_times_output, reset_times_output can be used
_nodes.push_back(loop_reset_times_output_to_inputs);

// inputs depend on loop_reset_times_output_to_inputs, which has now been added to the network, and user-specific inputs.
_nodes.push_back(inputs);
_nodes.push_back(input_activation);

// updates depends on loop_output_to_updates (h(t-1)) and user-specific updates.
_nodes.push_back(updates);
_nodes.push_back(update_activation);
_nodes.push_back(oneminus_update_activation);
_nodes.push_back(update_times_output);
_nodes.push_back(oneminus_update_times_input);  // This node also depends on input, hence its addition near the end of this code snippet.

// Now that everything has been computed using h(t-1), the new values can be forwarded to the output, h(t)
_nodes.push_back(output);
```
