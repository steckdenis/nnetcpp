#ifndef __ABSTRACTMERGENODE_H__
#define __ABSTRACTMERGENODE_H__

#include "abstractnode.h"

/**
 * @brief Merge input ports to an output port by applying an element-wise operation on them
 */
class AbstractMergeNode : public AbstractNode
{
    public:
        AbstractMergeNode();

        /**
         * @brief Add an input to the list of inputs to be summed
         */
        void addInput(Port *input);

        virtual Port *output();
        virtual void update();
        virtual void clearError();

    protected:
        Port _output;

        std::vector<Port *> _inputs;
};

#endif