//===- FooNvdlaBackend.h -------------------------------------------------------===//
//
//                             The ONNC Project
//
// See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef TARGET_FOONVDLA_FOONVDLA_BACKEND_H
#define TARGET_FOONVDLA_FOONVDLA_BACKEND_H
#include <string>
#include <onnc/Target/TargetBackend.h>
#include "NvDlaDefine.h"
#include "NvDlaMeta.h"
#include "Version.h"

namespace onnc {
using namespace onnc::foonvdla;
  
class FooNvdlaBackend : public TargetBackend, private NvDlaConstants
{
private:
  static const Version LOADABLE_VERSION;
  static const Version BLOB_DLA_VERSION;
  static const Version BLOB_EMU_VERSION;
  
public:
  FooNvdlaBackend(const TargetOptions& pOptions);

  virtual ~FooNvdlaBackend() = default;

  void addTensorSel(PassManager& pPM) override;

  void addOnncIrOptimization(PassManager& pPM, OptimizationOptions& options) override;

  void addTensorSched(PassManager& pPM) override;
  
  void addMemAlloc(PassManager& pPM) override;

  void addCodeEmit(PassManager& pPM, const Path& pOutput) override;

  void RegisterLowers(LowerRegistry& pRegistry) const override;

private:
  NvDlaBackendMeta       m_pMeta;
};

}  // namespace onnc

#endif
