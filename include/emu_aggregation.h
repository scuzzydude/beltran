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

uint32_t emu_model_aggregation_private_init(bam_host_emulator *pEmu, bam_emu_target_model *pModel) 
{
	int	verbose = bam_get_verbosity(BAM_EMU_DBGLVL_INFO, BAM_DBG_CODE_PATH_H_INIT_AGG);
	int num_controllers = (pEmu->emulationTargetFlags & BAM_EMU_TARGET_AGG_CONT_BITMASK) >> BAM_EMU_TARGET_AGG_CONT_BITSHIFT;
	size_t size = sizeof(emu_aggregation_model);
	emu_aggregation_model *pAggModel;
	char devPath[32];
	EmuController *pCntl;
	uint32_t queueDepth = 16;
	uint32_t numberOfQueues = 1;
	
		
	
	BAM_EMU_HOST_DBG_PRINT(verbose, "emu_model_aggregation_private_init(pEmu = %p, pModel = %p) num_controllers = %d size = %ld\n", pEmu, pModel, num_controllers, size);

	pModel->pvHostPrivate = malloc(size);

	BAM_EMU_HOST_ASSERT(pModel->pvHostPrivate);

	pAggModel = (emu_aggregation_model *)pModel->pvHostPrivate;

	pAggModel->num_controllers = num_controllers;

	pAggModel->ppvHostCtrls = (void **)malloc(sizeof(EmuController *) * num_controllers);

	BAM_EMU_HOST_ASSERT(pAggModel->ppvHostCtrls);
	
	pAggModel->d_ctrls_buff = createBuffer(num_controllers * sizeof(EmuController*), pEmu->cudaDevice);

	pAggModel->ppvDevCtrls = (void **)pAggModel->d_ctrls_buff.get();


	pModel->d_model_private = createBuffer(size, pEmu->cudaDevice);

	pModel->pvDevPrivate = pModel->d_model_private.get();

	void *pvtemp;
	            
	for (size_t i = 0 ; i < num_controllers; i++)
	{
		sprintf(devPath, "/dev/libnvm%d", i);


		
		pCntl = new EmuController(devPath, 1, pEmu->cudaDevice, queueDepth, numberOfQueues);

		pAggModel->ppvHostCtrls[i] = (void *)pCntl;


		BAM_EMU_HOST_DBG_PRINT(verbose, "emu_model_aggregation_private_init() %s d_ctrl_ptr = %p ppDevCtrls = %p sizeof(EmuController*) = %ld\n", devPath, pCntl->d_ctrl_ptr, pAggModel->ppvDevCtrls, sizeof(EmuController*));

		
		BAM_EMU_HOST_DBG_PRINT(verbose, "emu_model_aggregation_private_init() %p %p %ld\n", pAggModel->ppvDevCtrls+i, pCntl->d_ctrl_ptr, sizeof(EmuController*));
		
		cuda_err_chk(cudaMemcpy(pAggModel->ppvDevCtrls+i, &pCntl->d_ctrl_ptr, sizeof(EmuController*), cudaMemcpyHostToDevice));

		cuda_err_chk(cudaMemcpy(&pvtemp, pAggModel->ppvDevCtrls+0, sizeof(EmuController*), cudaMemcpyDeviceToHost));
		
		BAM_EMU_HOST_DBG_PRINT(verbose, "emu_model_aggregation_private_init() pvTemp(0) = %p\n", pvtemp);

	}
	


	cuda_err_chk(cudaMemcpy(pModel->pvDevPrivate, pModel->pvHostPrivate, size, cudaMemcpyHostToDevice));


	

	return 0;
}


//*******************************************************************************************************
//** Device Functions  
//*******************************************************************************************************


__device__ inline void emu_dump_sn_context(storage_next_emuluator_context *pContext)
{
	int	verbose = bam_get_verbosity(BAM_EMU_DBGLVL_INFO, BAM_DBG_CODE_PATH_D_AGG);

	uint32_t *dw = &pContext->pCmd->nvme_cmd.dword[0];

	BAM_EMU_DEV_DBG_PRINT4(verbose, "NVME[00:03] 0x%08x %08x %08x %08x\n", dw[0], dw[1], dw[2], dw[3]);
	BAM_EMU_DEV_DBG_PRINT4(verbose, "NVME[04:07] 0x%08x %08x %08x %08x\n", dw[4], dw[5], dw[6], dw[7]);
	BAM_EMU_DEV_DBG_PRINT4(verbose, "NVME[08:11] 0x%08x %08x %08x %08x\n", dw[8], dw[9], dw[10], dw[11]);
	BAM_EMU_DEV_DBG_PRINT4(verbose, "NVME[12:15] 0x%08x %08x %08x %08x\n", dw[12], dw[13], dw[14], dw[15]);


}



typedef ulonglong4 agg_copy_type;


__device__ inline int emu_model_agg_sq_enqueue(emu_aggregation_model *pAggModel, uint16_t cidx, uint16_t qidx, storage_next_emuluator_context *pContext)
{
	int	verbose = bam_get_verbosity(BAM_EMU_DBGLVL_INFO, BAM_DBG_CODE_PATH_D_AGG);
	EmuController *pCtrl;
	EmuQueuePair *pQP;
	nvm_cmd_t *pCmd;
	bool    bRingDoorbell = true;
	
	pCtrl = (EmuController *)pAggModel->ppvDevCtrls[cidx];

	
	BAM_EMU_DEV_DBG_PRINT3(verbose, "emu_model_agg_sq_enqueue(pCtrl=%p, ppDevCtrl = %p,%p)\n", pCtrl, pAggModel->ppvDevCtrls, pContext);
	
	pQP = pCtrl->d_qps + qidx;
	
	BAM_EMU_DEV_DBG_PRINT2(verbose, "emu_model_agg_sq_enqueue(pQP=%p, sq_mem_size = %d )\n", pQP, pQP->sq_mem_size);


	pCmd = emu_ctrl_sq_enqueue(&pQP->sq);

	if(pCmd)
	{
		agg_copy_type *queue_loc = (agg_copy_type *)pCmd;
		agg_copy_type *cmd_loc = (agg_copy_type *)pContext->pCmd;
		
		BAM_EMU_DEV_DBG_PRINT1(verbose, "emu_model_agg_sq_enqueue(pCmd=%p)\n", pCmd);

#pragma unroll
		for (uint32_t i = 0; i < 64/sizeof(agg_copy_type); i++) 
		{
			queue_loc[i] = cmd_loc[i];
		}

		if(bRingDoorbell)
		{

		}

	}
	else
	{
		return 1;
	}

	
	
	BAM_EMU_DEVICE_ASSERT(0);

	return 0;
}




__device__ inline int emu_model_aggregation_submit(bam_emu_target_model *pModel, storage_next_emuluator_context *pContext, void ** ppvThreadContext)
{
	emu_aggregation_model *pAggModel;
	
	int	verbose = bam_get_verbosity(BAM_EMU_DBGLVL_INFO, BAM_DBG_CODE_PATH_D_AGG);
	BAM_EMU_DEV_DBG_PRINT3(verbose, "emu_model_aggregation_submit(%p, %p, %p)\n", pModel, pContext, ppvThreadContext);
	
	emu_dump_sn_context(pContext);

	//TODO: Map it to drive and LBA

	pAggModel = (emu_aggregation_model *)pModel->pvDevPrivate;

	return emu_model_agg_sq_enqueue(pAggModel, 0, 0, pContext);
	

}



#endif

