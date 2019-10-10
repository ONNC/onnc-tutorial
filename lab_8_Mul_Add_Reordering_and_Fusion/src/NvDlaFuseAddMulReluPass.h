//===- NvDlaFuseAddMulReluPass.h ------------------------------------------===//
//
//                             The ONNC Project
//
// See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef ONNC_FOONVDLA_FUSE_ADD_MUL_RELU_PASS_H
#define ONNC_FOONVDLA_FUSE_ADD_MUL_RELU_PASS_H
#include <onnc/Core/CustomPass.h>


namespace onnc {
namespace foonvdla {

class NvDlaFuseAddMulReluPass : public CustomPass<NvDlaFuseAddMulReluPass>
{
public:
  NvDlaFuseAddMulReluPass() = default;

  ReturnType runOnModule(Module& pModule) override;

  ReturnType runOnComputeGraph(ComputeGraph& pCG) override;

private:
  bool isAddMulRelu(ComputeOperator* pNode);
};

} // namespace foonvdla
} // namespace onnc

#endif
