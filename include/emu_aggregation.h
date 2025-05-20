#ifndef __EMU_AGGREGATION_H
#define __EMU_AGGREGATION_H

#include <ctrl.h>
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
__device__ inline int emu_model_aggregation_submit(bam_emu_target_model *pModel, storage_next_emuluator_context *pContext, void ** ppvThreadContext)
{
	int	verbose = bam_get_verbosity(BAM_EMU_DBGLVL_INFO, BAM_DBG_CODE_PATH_D_AGG);
	BAM_EMU_DEV_DBG_PRINT3(verbose, "emu_model_aggregation_submit(%p, %p, %p)\n", pModel, pContext, ppvThreadContext);
	

	return 0;
}




#endif

