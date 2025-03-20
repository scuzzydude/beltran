#ifndef __EMU_MAPPER_H
#define __EMU_MAPPER_H




//*******************************************************************************************************
//** Host Functions  
//*******************************************************************************************************

static inline void emulator_init_mapper(bam_host_emulator *pEmu, uint32_t mapType, uint32_t modelType)
{
	
	int	verbose = bam_get_verbosity(BAM_EMU_DBGLVL_INFO, BAM_DBG_CODE_PATH_H_INIT_MAPPER);

	BAM_EMU_HOST_DBG_PRINT(verbose, "emulator_init_mapper(%p, mapType=%d, modelType=%d)\n", pEmu, mapType, modelType);
	

}



//*******************************************************************************************************
//** Device Functions  
//*******************************************************************************************************



#endif /* __EMU_MAPPER_H */
