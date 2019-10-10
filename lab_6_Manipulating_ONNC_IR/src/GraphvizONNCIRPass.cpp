//===- GraphvizONNCIRPass.cpp ---------------------------------------------===//
//
//                             The ONNC Project
//
// See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "GraphvizONNCIRPass.h"

#include <onnc/Core/PassSupport.h>
#include <onnc/IR/Compute/Attributes.h>
#include <onnc/IR/Compute/Initializer.h>
#include <onnc/IR/ComputeOperator.h>
#include <onnc/Transforms/Optimizations/OptimizationsUtils.h>

namespace onnc {
namespace foonvdla {

//===----------------------------------------------------------------------===//
// GraphvizONNCIRPass
//===----------------------------------------------------------------------===//

Pass::ReturnType GraphvizONNCIRPass::runOnModule(Module& pModule)
{
  Pass::ReturnType ret = kModuleNoChanged;

  ret = BaseType::runOnModule(pModule);

  if (ret != kModuleNoChanged) {
    pModule.eraseUnusedValues();
  }

  return ret;
}

Pass::ReturnType GraphvizONNCIRPass::runOnComputeGraph(ComputeGraph& pCG)
{
  std::cout << "=== GraphvizONNCIRPass ======\n";
  std::cout << "digraph {\n";

  // Loop over every operator in this ComputeGraph.
  for (ComputeOperator& op : pCG) {

    //------------------------------------------------------------------------------------
    // Print the decleration of this operator's name according to Graphviz's requirement.
    //------------------------------------------------------------------------------------

    std::string opName = op.name().str() + "_" + std::to_string((long)&op);
    std::cout << "  " << opName << " [label=" << op.name() << "]\n";

    //-----------------------------------------------------------------
    // Print the edges between this operator and all its input tensors. 
    //-----------------------------------------------------------------
    int numInputs = op.getNumOfInputs();
    for (int i = 0; i < numInputs; ++i) {
      Value* input = op.getInput(i);

      std::cout << "  " << input->getName() << " -> " << opName << "\n";
    }

    //-------------------------------------------------------------------
    // Print the edges between this operator and all its output tensors.
    //-------------------------------------------------------------------
    int numOutputs = op.getNumOfOutputs();
    for (int i = 0; i < numOutputs; ++i) {
      Value* output = op.getOutput(i);

      std::cout << "  " << opName << " -> " << output->getName() << "\n";
      std::cout << "  " << output->getName() << " [shape=rect]\n";
    }
  }

  std::cout << "}\n";
  std::cout << "==========================\n";
  
  return Pass::kModuleNoChanged;
}

} // namespace foonvdla
} // namespace onnc
