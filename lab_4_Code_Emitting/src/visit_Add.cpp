#include <onnc/IR/Compute/Add.h>


void CodeEmitVisitor::visit(const Add& pOp)
{
  printf("visit(Add) is called\n");
  
  // Get tensor attributes.
  const Tensor& first = *(pOp.getInput(0));
  const Tensor& second = *(pOp.getInput(1));
  const Tensor& output = *(pOp.getOutput(0));

  // For this example, we only support a special case where the first tensor is activation data
  // stored in memory and the 2nd tensor is a constant
  assert( (!isConstant(first) && isConstant(second)) &&
          "support only the case that the first tensor is activation data and the second constant");

  //--------------------------------
  // Configure hardware block
  //--------------------------------

  NvDlaDlaOperation* operation = new NvDlaDlaOperation();
  // Set hardware block type.
  operation->op_dep.op_type = DLA_OP_SDP;

  struct dla_sdp_op_desc& desc = (struct dla_sdp_op_desc&)(operation->op_desc);
  desc.src_precision     = PRECISION_FP16;
  desc.dst_precision     = PRECISION_FP16;
  // No look up table is required.
  desc.lut_index         = -1;

  // For this example, we only support batch == 1.
  desc.batch_num         = 1;
  desc.batch_stride      = 0;

  // Enable X1 block.
  desc.x1_op.enable      = 1;

  // X1 operation Options: Disable (SDP_OP_NONE) / ALU only (SDP_OP_ADD) /
  //                       Multiplier only (SDP_OP_MUL) / ALU+MUL (SDP_OP_BOTH)
  desc.x1_op.type        = SDP_OP_ADD;

  // ALU type options: SUM/MIN/MAX 
  desc.x1_op.alu_type    = SDP_ALU_OP_SUM;

  // Disable ReLU
  desc.x1_op.act         = ACTIVATION_NONE;   

  // Set per_layer/per_channel/per_point mode based on the broadcasting type.
  // For this example we only support per_point mode.
  desc.x1_op.mode        = SDP_OP_PER_POINT;

  // Set the datapath precision to be fp16.
  desc.x1_op.precision   = PRECISION_FP16;

  //----------------------------------------
  // Setup dataflow sources and destination
  //----------------------------------------

  struct dla_sdp_surface_desc& surface = (struct dla_sdp_surface_desc&)(operation->op_surf);

  // Setup 1st tensor source.
  const NvDlaCubeInfo firstCubeInfo   = makeCubeInfo(*this, NVDLA_CUBE_FEATURE, first);
  // The 1st input tensor can be read from:
  //   external DRAM via the interface of MCIF: DLA_MEM_MC
  //   SRAM via the interface of CVIF: DLA_MEM_CV
  //   the output of CONV hardware block: DLA_MEM_HW
  // In this example, we only support the 1st input tensor is stored at external DRAM.
  surface.src_data.type               = DLA_MEM_MC;
  // Setup memory allocation and DMA configuration for 1st input tensor.
  surface.src_data.address            = issueDlaAddr(first, firstCubeInfo);
  surface.src_data.size               = m_pMeta.getMemoryListEntrySize(first);
  surface.src_data.width              = firstCubeInfo.dim_w;
  surface.src_data.height             = firstCubeInfo.dim_h;
  surface.src_data.channel            = firstCubeInfo.dim_c;
  surface.src_data.line_stride        = firstCubeInfo.stride_line;
  surface.src_data.surf_stride        = firstCubeInfo.stride_surface;

  // Setup 2nd tensor source.
  MemoryListEntryId   memoryId;
  const NvDlaCubeInfo secondCubeInfo = makeCubeInfo(*this, getSdpXSingleCubeType(second, DLA_PRECISION), second);
  // The 2nd input tensor is stored at DRAM and accessed through the interface of MCIF.
  surface.x1_data.type               = DLA_MEM_MC;
  // Setup memory allocation and DMA configuration for 2nd input tensor.
  // In addition, the 2nd tensor is constant so need be packed into a blob and becomes a part of loadable.
  surface.x1_data.address            = issueSDPOperand(second, secondCubeInfo, memoryId);
  surface.x1_data.size               = m_pMeta.getMemoryListEntrySize(memoryId);
  surface.x1_data.width              = secondCubeInfo.dim_w;
  surface.x1_data.height             = secondCubeInfo.dim_h;
  surface.x1_data.channel            = secondCubeInfo.dim_c;
  surface.x1_data.line_stride        = secondCubeInfo.stride_line;
  surface.x1_data.surf_stride        = secondCubeInfo.stride_surface;

  // Setup output tensor destination.
  const NvDlaCubeInfo outputCubeInfo = makeCubeInfo(*this, NVDLA_CUBE_FEATURE, output);
  // The output tensor is stored at DRAM.
  surface.dst_data.type         = DLA_MEM_MC;
  surface.dst_data.address      = issueDlaAddr(output, outputCubeInfo);
  surface.dst_data.size         = m_pMeta.getMemoryListEntrySize(output);
  surface.dst_data.width        = outputCubeInfo.dim_w;
  surface.dst_data.height       = outputCubeInfo.dim_h;
  surface.dst_data.channel      = outputCubeInfo.dim_c;
  surface.dst_data.line_stride  = outputCubeInfo.stride_line;
  surface.dst_data.surf_stride  = outputCubeInfo.stride_surface;

  //----------------------------------------
  //  enlist the operation 
  //----------------------------------------
  issueDlaOp(operation, NULL, m_pMeta.m_pPrevOp);
}

