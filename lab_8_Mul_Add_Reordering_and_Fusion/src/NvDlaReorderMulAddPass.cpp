//===- NvDlaReorderMulAddPass.cpp -----------------------------------------===//
//
//                             The ONNC Project
//
// See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "NvDlaReorderMulAddPass.h"

#include <onnc/Core/PassSupport.h>
#include <onnc/IR/Compute/Attributes.h>
#include <onnc/IR/Compute/Initializer.h>
#include <onnc/IR/Compute/Mul.h>
#include <onnc/IR/Compute/Add.h>
#include <onnc/IR/ComputeOperator.h>
#include <onnc/Transforms/Optimizations/OptimizationsUtils.h>

using namespace onnc;
using namespace onnc::foonvdla;

//===----------------------------------------------------------------------===//
// NvDlaReorderMulAddPass
//===----------------------------------------------------------------------===//

unsigned NvDlaReorderMulAddPass::tensorIdx = 0;

Pass::ReturnType NvDlaReorderMulAddPass::runOnModule(Module& pModule)
{
  std::cout << "NvDlaReorderMulAddPass is called...\n";
  
  const Pass::ReturnType ret = BaseType::runOnModule(pModule);

  if (ret != kModuleNoChanged) {
    pModule.eraseUnusedValues();
  }

  return ret;
}

Pass::ReturnType NvDlaReorderMulAddPass::runOnComputeGraph(ComputeGraph& pCG)
{
  Pass::ReturnType ret = Pass::kModuleNoChanged;

  //--------------------------------------------------------
  // Search for the Mul-Add patterns that can be reordered.
  //--------------------------------------------------------
  std::vector<ComputeOperator*> mulList;
  for (ComputeOperator& node : pCG) {
    if (canBeReordered(&node)) {
      mulList.emplace_back(&node);
      ret |= Pass::kModuleChanged;
    }
  }

  //--------------------------------------------
  // Perform re-ordering on the found patterns.
  //--------------------------------------------
  // The original pattern is:
  //   outputY = (inputX * alpha) + beta
  //
  // We will re-arrange the above pattern by:
  //   outputY = (inputX + gamma) * alpha, where
  //     gamma = beta / alpha

  for (ComputeOperator* node : mulList) {
    Mul* mul = dyn_cast<Mul>(node);
    Add* add = dyn_cast<Add>(node->getOutput(0)->getUses()[0].getUser());

    Tensor* inputX;
    FloatTensor* alpha; // This kind of tensor contains constant values.
    FloatTensor* beta;
    Tensor* outputY;
    Tensor* tmp;

    // Find alpha and inputX. alpha must be a constant tensor.
    // In this example, we assume that Mul must have one constant input.
    if (isConstant(mul->getInput(0))) {
      alpha = dynamic_cast<FloatTensor*>(mul->getInput(0));
      inputX = mul->getInput(1);
    } else {
      inputX = mul->getInput(0);
      alpha = dynamic_cast<FloatTensor*>(mul->getInput(1));
    }

    // Find beta. beta must be a constant tensor.
    // In this example, we assume that Add must have one constant input.
    if (isConstant(add->getInput(0))) {
      beta = dynamic_cast<FloatTensor*>(add->getInput(0));
      tmp = add->getInput(1);
    } else {
      tmp = add->getInput(0);
      beta = dynamic_cast<FloatTensor*>(add->getInput(1));
    }

    // Find outputY
    outputY = add->getOutput(0);

    std::string addOutputTensorName = add->getOutput(0)->getName();
    std::string mulOutputTensorName = mul->getOutput(0)->getName();

    // The current ONNC IR graph status
    // ================================
    //
    //        (alphaInitializer)
    //    |      |
    // inputX  alpha
    //      \   /
    //      (mul)  (betaInitializer)
    //        |      |
    //       tmp   beta
    //         \   /
    //         (add)
    //           |
    //        outputY
    //           |
    //
    
    // Remove the edges between Mul/Add and their input/output tensors.
    // We will re-build their edges later on.
    // Remove an edge means to erase the records within an operator's data structure about its input tensors.
    mul->removeAllInputs();
    mul->removeAllOutputs();
    add->removeAllInputs();
    add->removeAllOutputs();

    // The current ONNC IR graph status
    // ================================
    //
    //        (alphaInitializer)
    //    |      |
    // inputX  alpha
    //         
    //      (mul)  (betaInitializer)
    //              |
    //       tmp   beta
    //            
    //         (add)
    //           
    //        outputY
    //           |
    //
    
    // Create a new tensor gamma.
    FloatTensor* gamma = dynamic_cast<FloatTensor*>(beta->create());

    // Give gamma tensor a unique name.
    gamma->setName(beta->getName() + "__gamma_" + std::to_string(tensorIdx++) + ")");

    // Initialize gamma.
    gamma->setDimensions(beta->getDimensions());

    // Add gamma into the ONNC IR graph.
    gamma = pCG.addValue<FloatTensor>(gamma);
    assert((gamma != nullptr) && "The name must be unique");
    
    // Create a new Initializer operator for gamma tensor. This is a must in ONNC IR graph.
    // Every tensor must have a "defining" operator. For a constant tensor, its defining
    // operator is an Initializer.
    Initializer* gammaInitializer = pCG.addOperator<Initializer>();
    gammaInitializer->setTensor(*gamma);

    // The current ONNC IR graph status
    // ================================
    //
    //        (alphaInitializer)
    //    |      |
    // inputX  alpha
    //         
    //      (mul)  (betaInitializer)
    //              |
    //       tmp   beta
    //            
    //         (add)     (gammaInitializer)
    //                      |
    //        outputY     gamma
    //           |
    //

    // Get the constant data of beta.
    const float* betaData = reinterpret_cast<const float*>(beta->getValues().data());

    // Get the constant data of alpha.
    const float* alphaData = reinterpret_cast<const float*>(alpha->getValues().data());

    // Calculate the constant data of gamma.
    int tensorSize = beta->getValues().size();
    for (int i = 0; i < tensorSize; i++) {
      gamma->getValues().push_back( betaData[i] / alphaData[i] );
    }

    // Remove beta from the ONNC IR graph. We don't need it anymore.
    Initializer* betaInitializer = static_cast<Initializer*>(beta->getDefine());
    pCG.erase(*betaInitializer);
    pCG.erase(*beta);
    
    // The current ONNC IR graph status
    // ================================
    //
    //        (alphaInitializer)
    //    |      |
    // inputX  alpha
    //         
    //      (mul)
    //         
    //       tmp
    //            
    //         (add)     (gammaInitializer)
    //                      |
    //        outputY     gamma
    //           |
    //

    // Re-connect the operators.
    add->addInput(*inputX);
    add->addInput(*gamma);
    add->addOutput(*tmp);
    mul->addInput(*tmp);
    mul->addInput(*alpha);
    mul->addOutput(*outputY);

    // The current ONNC IR graph status
    // ================================
    //
    //        (gammaInitializer)
    //    |      |
    // inputX  gamma
    //      \   /
    //      (add)  (alphaInitializer)
    //        |      |
    //       tmp   alpha
    //         \   /
    //         (mul)
    //           |
    //        outputY
    //           |
    //

    // Rename tensor tmp to become the original output tensor's name of add.
    add->getOutput(0)->setName(addOutputTensorName);
    // Rename tensor outputY to become the original output tensor's name of mul.
    mul->getOutput(0)->setName(mulOutputTensorName);
  }

  pCG.topologicalSort();
  
  return ret;
}

bool NvDlaReorderMulAddPass::canBeReordered(ComputeOperator* pNode)
{
  if (!isa<Mul>(pNode)) {
    return false;
  }

  if (!isConstant(pNode->getInput(0)) && !isConstant(pNode->getInput(1))) {
    return false;
  }
  
  Value* outv = pNode->getOutput(0);

  // if Mul's result has more than one users, we can't fuse it.
  if (outv->getUses().size() > 1) {
    return false;
  }

  ComputeOperator* userNode = outv->getUses()[0].getUser();
  if (!isa<Add>(userNode)) {
    return false;
  }

  return true;
}

bool NvDlaReorderMulAddPass::isConstant(Value* pValue)
{
  // Only if this value's (tensor's) "defining" operator is Initializer,
  // this tensor is a constant tensor.  
  ComputeOperator* op = static_cast<ComputeOperator*>(pValue->getDefine());
  if (isa<Initializer>(op)) {
    return true;
  } else {
    return false;
  }
}
