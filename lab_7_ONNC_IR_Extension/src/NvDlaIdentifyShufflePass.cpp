//===- NvDlaIdentifyShufflePass.cpp ---------------------------------------===//
//
//                             The ONNC Project
//
// See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "NvDlaIdentifyShufflePass.h"

#include "Compute/NvDlaShuffle.h"
#include "NvDlaDefine.h"

#include <onnc/Core/PassSupport.h>
#include <onnc/IR/Compute/Reshape.h>
#include <onnc/IR/Compute/Transpose.h>

using namespace onnc;
using namespace foonvdla;

//===----------------------------------------------------------------------===//
// NvDlaIdentifyShufflePass
//===----------------------------------------------------------------------===//
Pass::ReturnType NvDlaIdentifyShufflePass::runOnModule(Module& pModule)
{
  Pass::ReturnType ret = kModuleNoChanged;

  ret = BaseType::runOnModule(pModule);

  if (ret != kModuleNoChanged) {
    pModule.eraseUnusedValues();
  }

  return ret;
}

Pass::ReturnType NvDlaIdentifyShufflePass::runOnComputeGraph(ComputeGraph& pCG)
{
  Pass::ReturnType ret   = Pass::kModuleNoChanged;

  //---------------------------------------------------------------------
  // Find out all Reshape-Transpose-Reshape patterns in the model graph.
  //---------------------------------------------------------------------
  
  std::vector<Reshape*> reshapes;
  for (auto& op : pCG) {
    if (Reshape* reshape1 = dyn_cast<Reshape>(&op)) {
      if (is_shuffle(reshape1)) { // A channel-shuffle pattern is detected.
        // Save the first node of this pattern into a queue.
        // We will replace this pattern by a single Shuffle IR later on.
        reshapes.push_back(reshape1);

        // Since a node replacement will happen in the model, the model graph
        // will be changed and thus this function should return kModuleChanged.
        ret |= Pass::kModuleChanged;
      }
    }
  }

  //---------------------------------------------------------------------------
  // Replace every Reshape-Transpose-Reshape pattern with a single Shuffle IR.
  //---------------------------------------------------------------------------
  
  for (Reshape* reshape1 : reshapes) {

    // Derive the Tranpose and the second Reshape.
    auto* transpose  = dyn_cast<Transpose>(reshape1->getOutput(0)->getUses()[0].getUser());
    auto* reshape2 = dyn_cast<Reshape>(transpose->getOutput(0)->getUses()[0].getUser());

    Tensor* input_tensor = reshape1->getInput(0);
    Tensor* shape1_tensor = reshape1->getInput(1);
    auto shape1_initializer = static_cast<ComputeOperator*>(shape1_tensor->getDefine());
    Tensor* reshape1_out_tensor = reshape1->getOutput(0);
    Tensor* transpose_out = transpose->getOutput(0);
    Tensor* shape2_tensor = reshape2->getInput(1);
    auto shape2_initializer = static_cast<ComputeOperator*>(shape2_tensor->getDefine());
    Tensor* output_tensor = reshape2->getOutput(0);

    // The current ONNC IR graph status
    // ================================
    //
    //               (shape1_initializer)
    //       |              |
    //  input_tensor  shape1_tensor
    //           \      /
    //          (reshape1)
    //              |
    //     reshape1_out_tensor
    //              |      
    //         (transpose)  (shape2_initializer)
    //              |             |
    //       transpose_out  shape2_tensor
    //                   \     /
    //                 (reshape2)
    //                     |
    //               output_tensor
    //                     |

    // Create a new Shuffle.
    const auto& reshape_shape = static_cast<Int64Tensor*>(reshape1->getInput(1))->getValues();
    auto*       shuffle       = pCG.addOperator<NvDlaShuffle>(reshape_shape[1]);

    // The current ONNC IR graph status
    // ================================
    //
    //               (shape1_initializer)
    //       |              |
    //  input_tensor  shape1_tensor
    //           \      /
    //          (reshape1)
    //              |
    //     reshape1_out_tensor
    //              |      
    //         (transpose)  (shape2_initializer)
    //              |             |
    //       transpose_out  shape2_tensor
    //                   \     /
    //                 (reshape2)                (shuffle)
    //                     |
    //               output_tensor
    //                     |

    // Remove the edges between some operators and their input/output tensors.
    // Remove an edge means to erase the records within an operator's data structure about its input tensors.
    reshape1->removeAllInputs();
    reshape1->removeAllOutputs();
    transpose->removeAllInputs();
    transpose->removeAllOutputs();
    reshape2->removeAllInputs();
    reshape2->removeAllOutputs();
    shape1_initializer->removeAllOutputs();
    shape2_initializer->removeAllOutputs();

    // The current ONNC IR graph status
    // ================================
    //
    //               (shape1_initializer)
    //       |              |
    //  input_tensor  shape1_tensor
    //                 
    //          (reshape1)
    //              
    //     reshape1_out_tensor
    //                    
    //         (transpose)  (shape2_initializer)
    //                           |
    //       transpose_out  shape2_tensor
    //                        
    //                 (reshape2)                (shuffle)
    //                     
    //               output_tensor
    //                     |

    // Remove some un-used nodes in the ONNC IR graph.
    pCG.erase(*reshape1);
    pCG.erase(*transpose);
    pCG.erase(*reshape2);
    pCG.erase(*shape1_initializer);
    pCG.erase(*shape2_initializer);
    pCG.erase(*shape1_tensor);
    pCG.erase(*reshape1_out_tensor);
    pCG.erase(*transpose_out);
    pCG.erase(*shape2_tensor);

    // The current ONNC IR graph status
    // ================================
    //
    //       |
    //  input_tensor
    //
    //                                           (shuffle)
    //
    //               output_tensor
    //                     |
    
    shuffle->addInput(*input_tensor);
    shuffle->addOutput(*output_tensor);

    // The current ONNC IR graph status
    // ================================
    //
    //       |
    //  input_tensor
    //       |
    //   (shuffle)
    //       |              
    // output_tensor
    //       |
    
  }

  pCG.topologicalSort();

  return ret;
}

bool NvDlaIdentifyShufflePass::is_shuffle(Reshape* reshape1)
{
  // We are going to detect the following pattern.
  //
  //       |
  //  input_tensor
  //           \ 
  //          (reshape1)
  //              |
  //     reshape1_out_tensor
  //              |      // This tensor must have only one user.
  //         (transpose)
  //              |
  //       transpose_out
  //               \     // This tensor must have only one user.
  //             (reshape2)
  //                  |
  //            output_tensor
  //                  |
  //
  
#define SHUFFLE_ASSERT(cond) if (! (cond)) return false;

  //--------------------------
  // Check the first Reshape.
  //--------------------------

  SHUFFLE_ASSERT( reshape1->getNumOfOutputs() == 1 );

  // the output tensor of the Reshape has only one user.
  SHUFFLE_ASSERT( reshape1->getOutput(0)->getUses().size() == 1 );

  // The Reshape attribute must satisfy certain constraints.
  // The input dimension must be 4, and this Reshape splits the second dimension into two,
  // thus causing the output dimension to be 5.
  // e.g. input:  1x12x5x6, shape: [1,3,4,5,6]
  //      output: 1x3x4x5x6
  SHUFFLE_ASSERT( reshape1->getInput(0)->getNumOfDimensions() == 4 );
  SHUFFLE_ASSERT( reshape1->getInput(1)->getNumOfDimensions() == 1 ); // shape tensor must be array

  const auto& reshape1_shape = static_cast<Int64Tensor*>(reshape1->getInput(1))->getValues();
  SHUFFLE_ASSERT( reshape1_shape.size() == 5 );
  SHUFFLE_ASSERT( reshape1->getInput(0)->dimension(1) == reshape1_shape[1] * reshape1_shape[2] );
  SHUFFLE_ASSERT( reshape1->getInput(0)->dimension(2) == reshape1_shape[3] &&
                  reshape1->getInput(0)->dimension(3) == reshape1_shape[4]);
  
  //-----------------------------
  // Check the middle Transpose.
  //-----------------------------

  // the output tensor of the first Reshape has the user to be a Transpose.
  Transpose* transpose = dyn_cast<Transpose>(reshape1->getOutput(0)->getUses()[0].getUser());
  SHUFFLE_ASSERT( transpose );

  // the output tensor of the Transpose has only one user.
  SHUFFLE_ASSERT( transpose->getNumOfOutputs() == 1 );
  SHUFFLE_ASSERT( transpose->getOutput(0)->getUses().size() == 1 );

  // the attribute of Tranpose, perm, must be [0, 2, 1, 3, 4], ie. swap the 1st and 2nd dimensions.
  // e.g. input:  1x3x4x5x6
  //      output: 1x4x3x5x6
  SHUFFLE_ASSERT( transpose->getInput(0)->getNumOfDimensions() == 5 );
  SHUFFLE_ASSERT( transpose->getPerm().at(0) == 0 &&
                  transpose->getPerm().at(1) == 2 &&
                  transpose->getPerm().at(2) == 1 &&
                  transpose->getPerm().at(3) == 3 &&
                  transpose->getPerm().at(4) == 4);

  //-----------------------------
  // Check the last Reshape.
  //-----------------------------

  // the output tensor of the middle Transpose has the user to be a Reshape.
  Reshape* reshape2 = dyn_cast<Reshape>(transpose->getOutput(0)->getUses()[0].getUser());
  SHUFFLE_ASSERT( reshape2 );

  // The Reshape attribute must satisfy certain constraints.
  // The input dimension must be 5, and this Reshape merges the 2nd and 3rd dimension into one,
  // thus causing the output dimension to be 4.
  // e.g. input: 1x4x3x5x6, shape: [1,12,5,6]
  // output: 1x12x5x6
  SHUFFLE_ASSERT( reshape2->getInput(0)->getNumOfDimensions() == 5 );
  SHUFFLE_ASSERT( reshape2->getInput(1)->getNumOfDimensions() == 1 ); // shape tensor must be array

  const auto& reshape2_shape = static_cast<Int64Tensor*>(reshape2->getInput(1))->getValues();
  SHUFFLE_ASSERT( reshape2_shape.size() == 4 );
  SHUFFLE_ASSERT( reshape2->getInput(0)->dimension(1) * reshape2->getInput(0)->dimension(2) ==
                  reshape2_shape[1] );
  SHUFFLE_ASSERT( reshape2->getInput(0)->dimension(3) == reshape2_shape[2] &&
                  reshape2->getInput(0)->dimension(4) == reshape2_shape[3]);

#undef SHUFFLE_ASSERT

  return true;
}
