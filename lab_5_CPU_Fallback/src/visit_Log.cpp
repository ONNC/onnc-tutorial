#include <onnc/IR/Compute/Log.h>

void CodeEmitVisitor::visit(const Log& pOp)
{
  printf("visit(Log) is called\n");
  
  // Get tensor attributes.
  const Tensor& input = *(pOp.getInput(0));
  const Tensor& output = *(pOp.getOutput(0));

  //--------------------------------
  // Configure emulator engine
  //--------------------------------

  // Use the class NvDlaEmuOperation rather than class NvDlaDlaOperation used in the DLA case.
  NvDlaEmuOperation* operation = new NvDlaEmuOperation();

  struct emu_log_op_desc& desc = (struct emu_log_op_desc&)(operation->op_desc);
  desc.common.op_type = NVDLA_EMU_OP_LOG;

  //----------------------------------------
  // Setup dataflow sources and destination
  //----------------------------------------

  struct emu_log_buffer_descs& surface = (struct emu_log_buffer_descs&)(operation->op_buf);

  // Setup input tensor source.
  const NvDlaCubeInfo inputCubeInfo       = makeCubeInfo(*this, NVDLA_CUBE_FEATURE, input);
  int input_mid                           = m_pMeta.getMemoryListEntryId(input);
  surface.src_data.addressIndex           = issueEmuAddr(input_mid);
  surface.src_data.size                   = m_pMeta.getMemoryListEntrySize(input_mid);
  surface.src_data.format                 = PRECISION_FP16;
  surface.src_data.width                  = inputCubeInfo.dim_w;
  surface.src_data.height                 = inputCubeInfo.dim_h;
  surface.src_data.channel                = inputCubeInfo.dim_c;
  surface.src_data.line_stride            = inputCubeInfo.stride_line;
  surface.src_data.surf_stride            = inputCubeInfo.stride_surface;

  // Setup output tensor destination.
  const NvDlaCubeInfo outputCubeInfo = makeCubeInfo(*this, NVDLA_CUBE_FEATURE, output);
  int output_mid                = m_pMeta.getMemoryListEntryId(output);
  surface.dst_data.addressIndex = issueEmuAddr(output_mid);
  surface.dst_data.size         = m_pMeta.getMemoryListEntrySize(output_mid);
  surface.dst_data.format       = PRECISION_FP16;
  surface.dst_data.width        = outputCubeInfo.dim_w;
  surface.dst_data.height       = outputCubeInfo.dim_h;
  surface.dst_data.channel      = outputCubeInfo.dim_c;
  surface.dst_data.line_stride  = outputCubeInfo.stride_line;
  surface.dst_data.surf_stride  = outputCubeInfo.stride_surface;

  //----------------------------------------
  //  enlist the operation 
  //----------------------------------------
  issueEmuOp(operation);
}
