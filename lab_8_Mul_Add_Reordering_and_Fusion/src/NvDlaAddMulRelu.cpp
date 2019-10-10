//===- NvDlaAddMulRelu.cpp ------------------------------------------------===//
//
//                             The ONNC Project
//
// See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "NvDlaAddMulRelu.h"

#include "../CodeEmitVisitor.h"
#include "../NvDlaDefine.h"

using namespace onnc;
using namespace onnc::foonvdla;

char NvDlaAddMulRelu::ID = 0;

//===----------------------------------------------------------------------===//
// NvDlaAddMulRelu
//===----------------------------------------------------------------------===//
void NvDlaAddMulRelu::printAttributes(std::ostream& pOS) const
{
  pOS << "<>";
}

void NvDlaAddMulRelu::accept(ComputeVisitor& pV)
{
  CodeEmitVisitor* visitor = dyn_cast<CodeEmitVisitor>(&pV);
  if (nullptr != visitor)
    visitor->visit(*this);
}

void NvDlaAddMulRelu::accept(ComputeVisitor& pV) const
{
  CodeEmitVisitor* visitor = dyn_cast<CodeEmitVisitor>(&pV);
  if (nullptr != visitor)
    visitor->visit(*this);
}

bool NvDlaAddMulRelu::classof(const ComputeOperator* pOp)
{
  if (nullptr == pOp)
    return false;
  return (pOp->getID() == &ID);
}
