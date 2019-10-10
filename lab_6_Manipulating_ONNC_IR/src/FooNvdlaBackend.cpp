//===- FooNvdlaBackend.cpp -----------------------------------------------------===//
//
//                             The ONNC Project
//
// See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include <memory>

#include "FooNvdlaBackend.h"
#include "TargetInfo/FooNvdlaTargetInfo.h"
#include "TargetInfo/FooNvdlaTargetMemInfo.h"
#include "CodeEmitVisitor.h"
#include "NvDlaMemInfoPass.h"
#include "NvDlaTaskSubmitPass.h"
#include "NvDlaFileGenPass.h"
#include "GraphvizONNCIRPass.h"

#include <onnc/Analysis/UpdateGraphOutputSize.h>
#include <onnc/Analysis/NodeIRScheduler.h>
#include <onnc/CodeGen/BuildMemOperand.h>
#include <onnc/CodeGen/LinearScanMemAlloc.h>
#include <onnc/CodeGen/LiveIntervals.h>
#include <onnc/CodeGen/LiveValueMatrix.h>
#include <onnc/CodeGen/SetMemOperand.h>
#include <onnc/CodeGen/SlotIndexes.h>
#include <onnc/IR/CodeEmit.h>
#include <onnc/Support/Memory.h>
#include <onnc/Target/TargetRegistry.h>
#include <onnc/Target/TargetStandardPasses.h>
#include <onnc/Transforms/BookONNXGraphs.h>
#include <onnc/Transforms/BuildInitializers.h>
#include <onnc/Transforms/BuildInputOperators.h>
#include <onnc/Transforms/BuildOutputOperators.h>
#include <onnc/Transforms/DeadNodeElimination.h>
#include <onnc/Transforms/RemoveTrainingNodes.h>
#include <onnc/Transforms/TensorSel.h>
#include <onnc/Transforms/TensorSel/Standards/AddLower.h>
#include <onnc/Transforms/TensorSel/Standards/AveragePoolLower.h>
#include <onnc/Transforms/TensorSel/Standards/BatchNormalizationLower.h>
#include <onnc/Transforms/TensorSel/Standards/CastLower.h>
#include <onnc/Transforms/TensorSel/Standards/ConcatLower.h>
#include <onnc/Transforms/TensorSel/Standards/ConvLower.h>
#include <onnc/Transforms/TensorSel/Standards/FlattenLower.h>
#include <onnc/Transforms/TensorSel/Standards/GemmLower.h>
#include <onnc/Transforms/TensorSel/Standards/GlobalAveragePoolLower.h>
#include <onnc/Transforms/TensorSel/Standards/LRNLower.h>
#include <onnc/Transforms/TensorSel/Standards/LeakyReluLower.h>
#include <onnc/Transforms/TensorSel/Standards/MaxPoolLower.h>
#include <onnc/Transforms/TensorSel/Standards/MulLower.h>
#include <onnc/Transforms/TensorSel/Standards/PReluLower.h>
#include <onnc/Transforms/TensorSel/Standards/ReluLower.h>
#include <onnc/Transforms/TensorSel/Standards/ReshapeLower.h>
#include <onnc/Transforms/TensorSel/Standards/SoftmaxLower.h>
#include <onnc/Transforms/TensorSel/Standards/SplitLower.h>
#include <onnc/Transforms/TensorSel/Standards/SqueezeLower.h>
#include <onnc/Transforms/TensorSel/Standards/SumLower.h>
#include <onnc/Transforms/TensorSel/Standards/TransposeLower.h>
#include <onnc/Transforms/TensorSel/Standards/UnsqueezeLower.h>
#include <onnc/Transforms/TensorSel/Standards/UpsampleLower.h>

#include <memory>

using namespace onnc;

//===----------------------------------------------------------------------===//
// FooNvdlaBackend
//===----------------------------------------------------------------------===//
const Version FooNvdlaBackend::LOADABLE_VERSION = Version(1, 1, 255);
const Version FooNvdlaBackend::BLOB_DLA_VERSION = Version(1, 3, 0);
const Version FooNvdlaBackend::BLOB_EMU_VERSION = Version(1, 3, 0);

FooNvdlaBackend::FooNvdlaBackend(const TargetOptions& pOptions)
  : TargetBackend(pOptions)
  , NvDlaConstants(getConfig(::nvdla::ConfigSet::nv_full, ::nvdla::ExecutionMode::direct, false))
  , m_pMeta(*this) { 
  m_pMemInfo = std::make_unique<FooNvdlaTargetMemInfo>();
}

void FooNvdlaBackend::addTensorSel(PassManager& pPM)
{
  errs() << "FooNvdla is invoked\n";

  // Do ONNX graph IR optimization here.

  // Translate from ONNX graph IR into ONNC IR
  addStandardTensorSel(pPM, *this);
  
  // Now ONNC IR is ready.
  // If you need to extend ONNC IR, here is the place to add your pass that
  // adds your ONNC IR operators.
}

void FooNvdlaBackend::addOnncIrOptimization(PassManager& pPM, OptimizationOptions& options)
{
  TargetBackend::addOnncIrOptimization(pPM, options);

  pPM.add<GraphvizONNCIRPass>();
}

void FooNvdlaBackend::addTensorSched(PassManager& pPM)
{
  // After method AddTensorSel, operators have been scheduled in an
  // topological order, which totally respects the data dependency.
  // However, that might not be an optimized order for certain objective.
  // Add a scheduling optimization pass here.
}

void FooNvdlaBackend::addMemAlloc(PassManager& pPM)
{
  // Input: Module
  // Output: LiveIntervals
  addStandardCreateLiveIntervals(pPM);

  // Input: LiveIntervals
  // Output: MemAllocs
  addStandardMemoryAllocation(pPM, *this);

  // Input: MemAllocs
  // Output: Virtual memory address for each memory operands.
  addStandardSetMemOperands(pPM);

  const NvDlaConstants& constants = *this;
  pPM.add<NvDlaMemInfoPass>(constants, &m_pMeta);
}

void FooNvdlaBackend::addCodeEmit(PassManager& pPM, const Path& pOutput)
{
  static foonvdla::CodeEmitVisitor ceVisitor(*this, m_pMeta);
  pPM.add<CodeEmit>(ceVisitor)
     .add<NvDlaTaskSubmitPass>(&m_pMeta, BLOB_DLA_VERSION, BLOB_EMU_VERSION)
     .add<NvDlaFileGenPass>(&m_pMeta, LOADABLE_VERSION)
    ;
}

void FooNvdlaBackend::RegisterLowers(LowerRegistry& pRegistry) const
{
  pRegistry.emplace<AddLower>();
  pRegistry.emplace<AveragePoolLower>();
  pRegistry.emplace<BatchNormalizationLower>();
  pRegistry.emplace<ConcatLower>();
  pRegistry.emplace<ConvLower>();
  pRegistry.emplace<FlattenLower>();
  pRegistry.emplace<GemmLower>();
  pRegistry.emplace<GlobalAveragePoolLower>();
  pRegistry.emplace<LRNLower>();
  pRegistry.emplace<MaxPoolLower>();
  pRegistry.emplace<MulLower>();
  pRegistry.emplace<ReluLower>();
  pRegistry.emplace<ReshapeLower>();
  pRegistry.emplace<SoftmaxLower>();
  pRegistry.emplace<SqueezeLower>();
  pRegistry.emplace<SumLower>();
  pRegistry.emplace<TransposeLower>();
  pRegistry.emplace<UnsqueezeLower>();
}


//===----------------------------------------------------------------------===//
// Non member functions
//===----------------------------------------------------------------------===//
TargetBackend* CreateFooNvdlaBackend(const TargetOptions& pOptions)
{
  return new FooNvdlaBackend(pOptions);
}

extern "C" void InitializeFooNvdlaONNCBackend()
{
  onnc::TargetRegistry::RegisterTargetBackend(getTheFooNvdlaTarget(),
      CreateFooNvdlaBackend);
}

