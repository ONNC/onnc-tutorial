//===- PrintONNCIRPass.h --------------------------------------------------===//
//
//                             The ONNC Project
//
// See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef ONNC_FOONVDLA_PRINT_ONNC_IR_PASS_H
#define ONNC_FOONVDLA_PRINT_ONNC_IR_PASS_H
#include <onnc/Core/CustomPass.h>
#include <utility>

namespace onnc {
namespace foonvdla {

class PrintONNCIRPass : public CustomPass<PrintONNCIRPass>
{
public:
  PrintONNCIRPass() = default;

  ReturnType runOnModule(Module& pModule) override;

  ReturnType runOnComputeGraph(ComputeGraph& pCG) override;
};

} // namespace foonvdla
} // namespace onnc

#endif
