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

#ifndef __GRU_H__
#define __GRU_H__

#include "abstractrecurrentnetworknode.h"
#include "activation.h"

class MergeSum;

/**
 * @brief Gated Recurrent Units layer
 *
 * This layer contains memory and can be used to model sequences of input. At
 * each time step, the value it produces depends on all the previous time steps.
 * An episode is finished by calling reset(), which Network::reset() does.
 */
class GRU : public AbstractRecurrentNetworkNode
{
    public:
        /**
         * @brief Layer of GRU units. All the input and output ports of this
         *        layer have the same shape.
         *
         * @note This constructor wires some recurrent connexions (output to Z
         *       and R). Adding more connexions is possible by calling addInput,
         *       addZ and addR. For instance, the input is often connected to Z
         *       and R (using a different Dense than for the input itself, so we
         *       have X -> dense1 -> input, X -> dense2 -> Z and X -> dense3 -> R).
         */
        GRU(unsigned int size, Float learning_rate, Float decay = 0.9f);

        /**
         * @brief Add an X input to this network
         */
        void addInput(Port *input);

        /**
         * @brief Add a Z (update gate) input to this network
         */
        void addZ(Port *z);

        /**
         * @brief Add a R (reset gate) input to this network
         */
        void addR(Port *r);

        virtual Port* output();
        virtual void setCurrentTimestep(unsigned int timestep);

    private:
        MergeSum *_inputs;
        MergeSum *_updates;
        MergeSum *_resets;
        LinearActivation *_real_output;
        LinearActivation *_recurrent_output;
};

#endif
