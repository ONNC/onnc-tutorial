//===- GraphvizONNCIRPass.h -----------------------------------------------===//
//
//                             The ONNC Project
//
// See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef ONNC_FOONVDLA_GRAPHVIZ_ONNC_IR_PASS_H
#define ONNC_FOONVDLA_GRAPHVIZ_ONNC_IR_PASS_H
#include <onnc/Core/CustomPass.h>
//#include <utility>

namespace onnc {
namespace foonvdla {

class GraphvizONNCIRPass : public CustomPass<GraphvizONNCIRPass>
{
public:
  GraphvizONNCIRPass() = default;

  ReturnType runOnModule(Module& pModule) override;

  ReturnType runOnComputeGraph(ComputeGraph& pCG) override;
};

} // namespace foonvdla
} // namespace onnc

#endif
