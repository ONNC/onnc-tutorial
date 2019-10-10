//===- NvDlaFuseAddMulReluPass.cpp ----------------------------------------===//
//
//                             The ONNC Project
//
// See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "NvDlaFuseAddMulReluPass.h"
#include "Compute/NvDlaAddMulRelu.h"

#include <onnc/Core/PassSupport.h>
#include <onnc/IR/Compute/Attributes.h>
#include <onnc/IR/Compute/Initializer.h>
#include <onnc/IR/Compute/Add.h>
#include <onnc/IR/Compute/Mul.h>
#include <onnc/IR/Compute/Relu.h>
#include <onnc/IR/ComputeOperator.h>
#include <onnc/Transforms/Optimizations/OptimizationsUtils.h>

namespace onnc {
namespace foonvdla {

//===----------------------------------------------------------------------===//
// NvDlaFuseAddMulReluPass
//===----------------------------------------------------------------------===//

Pass::ReturnType NvDlaFuseAddMulReluPass::runOnModule(Module& pModule)
{
  const Pass::ReturnType ret = BaseType::runOnModule(pModule);

  if (ret != kModuleNoChanged) {
    pModule.eraseUnusedValues();
  }

  return ret;
}

Pass::ReturnType NvDlaFuseAddMulReluPass::runOnComputeGraph(ComputeGraph& pCG)
{
  Pass::ReturnType ret = Pass::kModuleNoChanged;

  // Search for the Add-Mul-Relu patterns that can be replaced by a single AddMulRelu IR.
  std::vector<ComputeOperator*> patternList;
  for (ComputeOperator& node : pCG) {
    if (isAddMulRelu(&node)) {
      patternList.emplace_back(&node);
      ret |= Pass::kModuleChanged;
    }
  }
  
  for (ComputeOperator* node : patternList) {
    // Derive original IRs.
    Add* add = dyn_cast<Add>(node);
    Mul* mul = dyn_cast<Mul>(add->getOutput(0)->getUses()[0].getUser());
    Relu* relu = dyn_cast<Relu>(mul->getOutput(0)->getUses()[0].getUser());

    Tensor* addA = add->getInput(0);
    Tensor* addB = add->getInput(1);
    Tensor* addC = add->getOutput(0);
    Tensor* mulB;
    if (addC == mul->getInput(0)) {
      mulB = mul->getInput(1);
    } else {
      mulB = mul->getInput(0);
    }
    Tensor* mulC = mul->getOutput(0);
    Tensor* reluY = relu->getOutput(0);

    // The current ONNC IR graph status
    // ================================
    //        
    //    |      |
    //  addA   addB
    //      \   /
    //      (add)  
    //        |      |
    //       addC   mulB
    //         \   /
    //         (mul)
    //           |
    //         mulC
    //           |
    //         (relu)
    //           |
    //         reluY
    //           |

    // Create a new AddMulRelu IR.
    NvDlaAddMulRelu* compound = pCG.addOperator<NvDlaAddMulRelu>();

    // The current ONNC IR graph status
    // ================================
    //        
    //    |      |
    //  addA   addB
    //      \   /
    //      (add)  
    //        |      |
    //       addC   mulB
    //         \   /
    //         (mul)
    //           |
    //         mulC
    //           |
    //         (relu)    (compound)
    //           |
    //         reluY
    //           |
    
    add->removeAllInputs();
    add->removeAllOutputs();
    mul->removeAllInputs();
    mul->removeAllOutputs();
    relu->removeAllInputs();
    relu->removeAllOutputs();

    // The current ONNC IR graph status
    // ================================
    //        
    //    |      |
    //  addA   addB
    //        
    //      (add)  
    //               |
    //       addC   mulB
    //            
    //         (mul)
    //           
    //         mulC
    //           
    //         (relu)    (compound)
    //           
    //         reluY
    //           |

    pCG.erase(*add);
    pCG.erase(*mul);
    pCG.erase(*relu);
    pCG.erase(*addC);
    pCG.erase(*mulC);
    
    // The current ONNC IR graph status
    // ================================
    //        
    //    |      |
    //  addA   addB
    //        
    //               |
    //              mulB
    //            
    //                   (compound)
    //           
    //         reluY
    //           |

    compound->addInput(*addA);
    compound->addInput(*addB);
    compound->addInput(*mulB);
    compound->addOutput(*reluY);

    // The current ONNC IR graph status
    // ================================
    //        
    //     |     |     |
    //    addA  addB  mulB
    //       \   |    /
    //       (compound)
    //           |
    //         reluY
    //           |

  }

  pCG.topologicalSort();
  
  return ret;
}

bool NvDlaFuseAddMulReluPass::isAddMulRelu(ComputeOperator* pNode)
{
  // Check the first node.
  // It must be
  //   1) an Add and,
  //   2) has only one operator to use its result.
  if ( ! isa<Add>(pNode)) return false;
  if (pNode->getOutput(0)->getUses().size() > 1) return false;

  // Check the second node.
  // It must be
  //   1) a Mul and,
  //   2) has only one operator to use its result.
  ComputeOperator* secondNode = pNode->getOutput(0)->getUses()[0].getUser();
  if ( ! isa<Mul>(secondNode)) return false;
  if (secondNode->getOutput(0)->getUses().size() > 1) return false;

  // Check the third node.
  // It must be a Relu.
  // However, it does not need the limitation of only one operator to use its result, because
  // its result is saved in system memory which can be loaded by multiple operators for use
  // at any time.
  ComputeOperator* thirdNode = secondNode->getOutput(0)->getUses()[0].getUser();
  if ( ! isa<Relu>(thirdNode)) return false;

  return true;
}
  
} // namespace foonvdla
} // namespace onnc
