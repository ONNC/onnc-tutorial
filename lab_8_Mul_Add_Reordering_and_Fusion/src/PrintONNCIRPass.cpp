//===- PrintONNCIRPass.cpp ------------------------------------------------===//
//
//                             The ONNC Project
//
// See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "PrintONNCIRPass.h"

#include <onnc/Core/PassSupport.h>
#include <onnc/IR/Compute/Attributes.h>
#include <onnc/IR/Compute/Initializer.h>
#include <onnc/IR/ComputeOperator.h>
#include <onnc/Transforms/Optimizations/OptimizationsUtils.h>

namespace onnc {
namespace foonvdla {

//===----------------------------------------------------------------------===//
// PrintONNCIRPass
//===----------------------------------------------------------------------===//

Pass::ReturnType PrintONNCIRPass::runOnModule(Module& pModule)
{
  const Pass::ReturnType ret = BaseType::runOnModule(pModule);

  if (ret != kModuleNoChanged) {
    pModule.eraseUnusedValues();
  }

  return ret;
}

Pass::ReturnType PrintONNCIRPass::runOnComputeGraph(ComputeGraph& pCG)
{
  Pass::ReturnType ret = Pass::kModuleNoChanged;

  std::cout << "=== PrintONNCIRPass ======\n";
  for (ComputeOperator& node : pCG) {
    node.print(std::cout);
    std::cout << "\n";
  }
  std::cout << "==========================\n";
  
  return ret;
}

} // namespace foonvdla
} // namespace onnc
