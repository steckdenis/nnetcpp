#include "test_gru.h"
#include "utils.h"

#include <network.h>
#include <dense.h>
#include <gru.h>

#include <iostream>

static std::vector<Vector> makeSequence(const std::vector<Float> &entries)
{
    std::vector<Vector> rs;

    for (Float entry : entries) {
        rs.push_back(makeVector({entry}));
    }

    return rs;
}

void TestGRU::test()
{
    // Training vectors for the "compute parity" task
    std::vector<std::vector<Vector>> inputs {
        makeSequence({0.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f}),
        makeSequence({1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f}),
        makeSequence({0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f}),
        makeSequence({1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f}),
        makeSequence({1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f}),
        makeSequence({1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f}),
        makeSequence({0.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f}),
        makeSequence({1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f}),
        makeSequence({1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f}),
        makeSequence({0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f}),
    };
    std::vector<std::vector<Vector>> outputs {
        makeSequence({0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f}),
        makeSequence({1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f}),
        makeSequence({0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f}),
        makeSequence({1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f}),
        makeSequence({1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f}),
        makeSequence({1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f}),
        makeSequence({0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f}),
        makeSequence({1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f}),
        makeSequence({1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f}),
        makeSequence({0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f}),
    };

    // Network with N GRU cells
    static const unsigned int N = 100;

    Network *net = new Network(1);
    Dense *dense_in = new Dense(N, 0.005);
    Dense *dense_z = new Dense(N, 0.005);
    Dense *dense_r = new Dense(N, 0.005);
    GRU *gru = new GRU(N, 0.005);
    Dense *out = new Dense(1, 0.005);

    dense_in->setInput(net->inputPort());
    dense_z->setInput(net->inputPort());
    dense_r->setInput(net->inputPort());
    gru->addInput(dense_in->output());
    gru->addZ(dense_z->output());
    gru->addR(dense_r->output());
    out->setInput(gru->output());

    net->addNode(dense_in);
    net->addNode(dense_z);
    net->addNode(dense_r);
    net->addNode(gru);
    net->addNode(out);

    // Train the network on the input sequences
    for (int iteration=0; iteration<100; ++iteration) {
        std::cout << "Sequence iteration " << iteration << std::endl;

        for (std::size_t i=0; i<inputs.size(); ++i) {
            checkLearning(net, inputs[i], outputs[i], 0.0, 10, false);
        }
    }

    delete net;
}
