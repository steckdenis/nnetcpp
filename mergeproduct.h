#ifndef __MERGEPRODUCT_H__
#define __MERGEPRODUCT_H__

#include "abstractmergenode.h"

/**
 * @brief Merge input ports to an output port by element-wise multiplying them
 */
class MergeProduct : public AbstractMergeNode
{
    public:
        MergeProduct();

        virtual void forward();
        virtual void backward();
};

#endif