/*
 * Copyright (c) 2015 Vrije Universiteit Brussel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "test_recurrent.h"
#include "utils.h"

#include <network.h>
#include <dense.h>
#include <gru.h>
#include <lstm.h>

#include <iostream>
#include <stdlib.h>

static const float learning_rate = 1e-2;

static std::vector<Vector> makeSequence(const std::vector<Float> &entries)
{
    std::vector<Vector> rs;

    for (Float entry : entries) {
        rs.push_back(makeVector({entry}));
    }

    return rs;
}

void TestRecurrent::testGRU()
{
    // Network with N GRU cells
    static const unsigned int N = 10;

    Network *net = new Network(1);
    Dense *dense_in = new Dense(N, learning_rate);
    Dense *dense_z = new Dense(N, learning_rate);
    Dense *dense_r = new Dense(N, learning_rate);
    GRU *gru = new GRU(N, learning_rate);
    Dense *out = new Dense(1, learning_rate);

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

    // Test this network
    testNetwork(net, 0.10);
}

void TestRecurrent::testLSTM()
{
    // Network with N LSTM cells
    static const unsigned int N = 100;

    Network *net = new Network(1);
    Dense *dense_in = new Dense(N, learning_rate);
    Dense *dense_ingate = new Dense(N, learning_rate);
    Dense *dense_outgate = new Dense(N, learning_rate);
    Dense *dense_forgetgate = new Dense(N, learning_rate);
    LSTM *lstm = new LSTM(N, learning_rate);
    Dense *out = new Dense(1, learning_rate);

    dense_in->setInput(net->inputPort());
    dense_ingate->setInput(net->inputPort());
    dense_outgate->setInput(net->inputPort());
    dense_forgetgate->setInput(net->inputPort());
    lstm->addInput(dense_in->output());
    lstm->addInGate(dense_ingate->output());
    lstm->addOutGate(dense_outgate->output());
    lstm->addForgetGate(dense_forgetgate->output());
    out->setInput(lstm->output());

    net->addNode(dense_in);
    net->addNode(dense_ingate);
    net->addNode(dense_outgate);
    net->addNode(dense_forgetgate);
    net->addNode(lstm);
    net->addNode(out);

    // Test this network
    testNetwork(net, 0.20);
}

void TestRecurrent::testNetwork(Network *net, float target)
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

    // Train the network on the input sequences (all but the last one)
    for (int iteration=0; iteration<50000; ++iteration) {
        int i = rand() % (int)inputs.size();

        checkLearning(net, inputs[i], outputs[i], 0.0, 1, true, true);
    }

    // Test the network on all the input sequences (last one included)
    std::cout << "Validation" << std::endl;

    for (std::size_t i=0; i<inputs.size(); ++i) {
        // NOTE: The MSE allowed is quite big, but this is required as correct
        //       parity nevertheless has sometimes a big error, for instance if
        //       the network overshoots (predicts 1.30 instead of 1, and -0.2
        //       instead of 0).
        CPPUNIT_ASSERT_MESSAGE(
            "A test vector failed the parity test",
            checkLearning(net, inputs[i], outputs[i], target, 1, true, true)
        );
    }

    delete net;
}
