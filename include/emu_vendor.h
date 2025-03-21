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




//*******************************************************************************************************
//** Device Functions  
//*******************************************************************************************************

__device__ inline int emu_model_vendor_submit(bam_emu_target_model *pModel, storage_next_emuluator_context *pContext)
{

	return 0;
}




#endif
