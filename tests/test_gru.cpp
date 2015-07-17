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

#include "test_gru.h"
#include "utils.h"

#include <network.h>
#include <dense.h>
#include <gru.h>

#include <iostream>
#include <stdlib.h>

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

    // Train the network on the input sequences (all but the last one)
    for (int iteration=0; iteration<2000; ++iteration) {
        int i = rand() % (int)inputs.size();

        checkLearning(net, inputs[i], outputs[i], 0.0, 1, false);
    }

    // Test the network on all the input sequences (last one included)
    std::cout << "Validation" << std::endl;

    for (std::size_t i=0; i<inputs.size()-1; ++i) {
        // NOTE: The MSE allowed is quite big, but this is required as correct
        //       parity nevertheless has sometimes a big error, for instance if
        //       the network overshoots (predicts 1.30 instead of 1, and -0.2
        //       instead of 0). A bit of training is performed here so that the
        //       network can recover from this overshoot (but has no time to learn
        //       each sequence on the fly).
        CPPUNIT_ASSERT_MESSAGE(
            "A test vector failed the parity test",
            checkLearning(net, inputs[i], outputs[i], 0.50, 3, true)
        );
    }

    delete net;
}
