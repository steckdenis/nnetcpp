#ifndef __MERGESUM_H__
#define __MERGESUM_H__

#include "abstractmergenode.h"

/**
 * @brief Merge input ports to an output port by element-wise summing them
 */
class MergeSum : public AbstractMergeNode
{
    public:
        MergeSum();

        virtual void forward();
        virtual void backward();
};

#endif