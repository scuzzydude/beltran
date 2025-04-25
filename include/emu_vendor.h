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
	return 0;
}




//*******************************************************************************************************
//** Device Functions  
//*******************************************************************************************************

__device__ inline int emu_model_vendor_submit(bam_emu_target_model *pModel, storage_next_emuluator_context *pContext, void **ppvThreadContext)
{

	return 0;
}




#endif
