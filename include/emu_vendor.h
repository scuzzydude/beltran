#ifndef __EMU_VENDOR_H
#define __EMU_VENDOR_H

//*******************************************************************************************************
//** This is just a template, 3rd parties can design their own models
//** Implementing these functions
//*******************************************************************************************************





//*******************************************************************************************************
//** Host Functions  
//*******************************************************************************************************
static inline const char *get_vendor_model_name(void)
{
	return "Generice Vendor Model (non-functional)";
}

uint32_t emu_model_vendor_private_init(bam_host_emulator *pEmu, bam_emu_target_model *pModel)
{

	int	verbose = bam_get_verbosity(BAM_EMU_DBGLVL_INFO, BAM_DBG_CODE_PATH_H_INIT_VENDOR);

	BAM_EMU_HOST_DBG_PRINT(verbose, "emu_model_vendor_private_init(pEmu = %p, pModel = %p)\n", pEmu, pModel);

	return 0;
}




//*******************************************************************************************************
//** Device Functions  
//*******************************************************************************************************

__device__ inline int emu_model_vendor_submit(bam_emu_target_model *pModel, storage_next_emuluator_context *pContext, void **ppvThreadContext)
{
	
	int	verbose = bam_get_verbosity(BAM_EMU_DBGLVL_INFO, BAM_DBG_CODE_PATH_D_VENDOR);

	BAM_EMU_DEV_DBG_PRINT3(verbose, "emu_model_vendor_submit(%p, %p, %p)\n", pModel, pContext, ppvThreadContext);
	
	return 0;
}




#endif
