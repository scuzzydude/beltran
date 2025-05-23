#ifndef __EMU_AGGREGATION_H
#define __EMU_AGGREGATION_H

#include "emu_ctrl.h"

//*******************************************************************************************************
//** Model Specific Data Structures
//*******************************************************************************************************
typedef struct _emu_aggregation_model
{
	int num_controllers;

	void **ppvHostCtrls;
	void **ppvDevCtrls;

    BufferPtr d_ctrls_buff;



} emu_aggregation_model;



//*******************************************************************************************************
//** Host Functions  
//*******************************************************************************************************
extern uint32_t emu_model_aggregation_private_init(bam_host_emulator *pEmu, bam_emu_target_model *pModel);



//*******************************************************************************************************
//** Device Functions  
//*******************************************************************************************************
__device__ int emu_model_agg_sq_enqueue(emu_aggregation_model *pAggModel, uint16_t cidx, uint16_t qidx, storage_next_emuluator_context *pContext);


__device__ inline void emu_dump_sn_context(storage_next_emuluator_context *pContext)
{
	int	verbose = bam_get_verbosity(BAM_EMU_DBGLVL_INFO, BAM_DBG_CODE_PATH_D_AGG);

	uint32_t *dw = &pContext->pCmd->nvme_cmd.dword[0];

	BAM_EMU_DEV_DBG_PRINT4(verbose, "NVME[00:03] 0x%08x %08x %08x %08x\n", dw[0], dw[1], dw[2], dw[3]);
	BAM_EMU_DEV_DBG_PRINT4(verbose, "NVME[04:07] 0x%08x %08x %08x %08x\n", dw[4], dw[5], dw[6], dw[7]);
	BAM_EMU_DEV_DBG_PRINT4(verbose, "NVME[08:11] 0x%08x %08x %08x %08x\n", dw[8], dw[9], dw[10], dw[11]);
	BAM_EMU_DEV_DBG_PRINT4(verbose, "NVME[12:15] 0x%08x %08x %08x %08x\n", dw[12], dw[13], dw[14], dw[15]);


}


__device__ inline int emu_model_aggregation_submit(bam_emu_target_model *pModel, storage_next_emuluator_context *pContext, void ** ppvThreadContext)
{
	emu_aggregation_model *pAggModel;
	
	int	verbose = bam_get_verbosity(BAM_EMU_DBGLVL_INFO, BAM_DBG_CODE_PATH_D_AGG);
	BAM_EMU_DEV_DBG_PRINT3(verbose, "emu_model_aggregation_submit(%p, %p, %p)\n", pModel, pContext, ppvThreadContext);
	
	emu_dump_sn_context(pContext);

	//TODO: Map it to drive and LBA

	pAggModel = (emu_aggregation_model *)pModel->pvDevPrivate;

	emu_model_agg_sq_enqueue(pAggModel, 0, 0, pContext);
	
	return 0;
}



#endif

