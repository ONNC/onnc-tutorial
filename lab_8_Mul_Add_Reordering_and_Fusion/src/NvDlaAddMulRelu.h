//===- NvDlaAddMulRelu.h ------------------------------------------------------===//
//
//                             The ONNC Project
//
// See LICENSE.TXT for details.
//
//===--------------------------------------------------------------------------===//
#ifndef TARGET_NVDLA_NVDLA_ADD_MUL_RELU_H
#define TARGET_NVDLA_NVDLA_ADD_MUL_RELU_H

#include <onnc/IR/Compute/Reshape.h>
#include <onnc/IR/Compute/Transpose.h>
#include <onnc/IR/ComputeOperator.h>

namespace onnc {
namespace foonvdla {

class NvDlaAddMulRelu : public ComputeOperator
{
public:
  static char ID;

public:
  NvDlaAddMulRelu()
    : ComputeOperator("AddMulRelu", ID)
  {}

  virtual ~NvDlaAddMulRelu() {}

  // Paramater

  // Input & Ouput Tensor
  Tensor* getInput(unsigned int pIdx) override { return static_cast<Tensor*>(m_Inputs[pIdx]); }

  const Tensor* getInput(unsigned int pIdx) const override { return static_cast<Tensor*>(m_Inputs[pIdx]); }

  Tensor* getOutput(unsigned int pIdx) override { return static_cast<Tensor*>(m_Outputs[pIdx]); }

  const Tensor* getOutput(unsigned int pIdx) const override { return static_cast<Tensor*>(m_Outputs[pIdx]); }

  void printAttributes(std::ostream& pOS) const override;

  void accept(ComputeVisitor& pV) override;

  void accept(ComputeVisitor& pV) const override;

  static bool classof(const ComputeOperator* pOp);

};

} // namespace foonvdla
} // namespace onnc

#endif
