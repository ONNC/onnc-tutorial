//===- NvDlaReorderMulAddPass.h -------------------------------------------===//
//
//                             The ONNC Project
//
// See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef ONNC_FOONVDLA_REORDER_MUL_ADD_PASS_H
#define ONNC_FOONVDLA_REORDER_MUL_ADD_PASS_H
#include <onnc/Core/CustomPass.h>

namespace onnc {
namespace foonvdla {

class NvDlaReorderMulAddPass : public CustomPass<NvDlaReorderMulAddPass>
{
public:
  NvDlaReorderMulAddPass() = default;

  ReturnType runOnModule(Module& pModule) override;

  ReturnType runOnComputeGraph(ComputeGraph& pCG) override;

private:
  bool canBeReordered(ComputeOperator* pNode);
  bool isConstant(Value* value);

  static unsigned tensorIdx;
};

} // namespace foonvdla
} // namespace onnc

#endif
