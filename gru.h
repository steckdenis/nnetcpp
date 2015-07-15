#ifndef __GRU_H__
#define __GRU_H__

#include "abstractnode.h"

class MergeSum;

/**
 * @brief Gated Recurrent Units layer
 *
 * This layer contains memory and can be used to model sequences of input. At
 * each time step, the value it produces depends on all the previous time steps.
 * An episode is finished by calling reset(), which Network::reset() does.
 */
class GRU : public AbstractNode
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
        virtual ~GRU();

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
        virtual void forward();
        virtual void backward();
        virtual void update();
        virtual void clearError();

        virtual void reset();

    private:
        MergeSum *_inputs;
        MergeSum *_updates;
        MergeSum *_resets;
        MergeSum *_output;

        std::vector<AbstractNode *> _nodes;

};

#endif