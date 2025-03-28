#ifndef __EMU_AGGREGATION_H
#define __EMU_AGGREGATION_H

//*******************************************************************************************************
//** Host Functions  
//*******************************************************************************************************


uint32_t emu_model_aggregation_private_init(bam_host_emulator *pEmu, bam_emu_target_model *pModel)
{
	return 0;
}



//*******************************************************************************************************
//** Device Functions  
//*******************************************************************************************************
__device__ inline int emu_model_aggregation_submit(bam_emu_target_model *pModel, storage_next_emuluator_context *pContext, void *pvThreadContext)
{

	return 0;
}




#endif

