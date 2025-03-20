#ifndef __EMU_MAPPER_H
#define __EMU_MAPPER_H




//*******************************************************************************************************
//** Host Functions  
//*******************************************************************************************************

static inline void emulator_init_mapper(bam_host_emulator *pEmu, uint32_t mapType, uint32_t modelType)
{
	bam_emu_target_model *pModel;
	bam_emu_mapper *pMap;

	bam_target_emulator *pTgt;
	
	int	verbose = bam_get_verbosity(BAM_EMU_DBGLVL_INFO, BAM_DBG_CODE_PATH_H_INIT_MAPPER);

	BAM_EMU_HOST_DBG_PRINT(verbose, "emulator_init_mapper(%p, mapType=%d, modelType=%d)\n", pEmu, mapType, modelType);

	BAM_EMU_HOST_ASSERT(pEmu);

	pTgt = &pEmu->tgt;

	
	pTgt->mapper.uMapType = mapType;
	
	switch(mapType)
	{
		case EMU_MAP_TYPE_DIRECT:
			strcpy(pTgt->mapper.szMapName, "Direct Mapper");
			break;



		default:
			BAM_EMU_HOST_DBG_PRINT(BAM_EMU_DBGLVL_ERROR, "Invalid Map Type %d\n", mapType);
			BAM_EMU_HOST_ASSERT(0);
			break;

	}

	pTgt->mapper.model.uModelType = modelType; 

	switch(modelType)
	{
		case EMU_MODEL_TYPE_LATENCY:
			strcpy(pTgt->mapper.model.szModelName, "Latency Model");
			break;
			
		case EMU_MODEL_TYPE_AGGREGATION:
			strcpy(pTgt->mapper.model.szModelName, "NVMe Aggregation Model");
			break;

		case EMU_MODEL_TYPE_VENDOR:
			strcpy(pTgt->mapper.model.szModelName, get_vendor_model_name());
			break;
		

		default:
			BAM_EMU_HOST_DBG_PRINT(BAM_EMU_DBGLVL_ERROR, "Invalid Model Type %d\n", modelType);
			BAM_EMU_HOST_ASSERT(0);
			break;
			
	}


	BAM_EMU_HOST_DBG_PRINT(verbose, "Initialized Mapper with type = %d (%s)\n", mapType, pTgt->mapper.szMapName);
	BAM_EMU_HOST_DBG_PRINT(verbose, "Initialized Model  with type = %d (%s)\n", modelType, pTgt->mapper.model.szModelName);
	

	

}

//*******************************************************************************************************
//** Device Functions  
//*******************************************************************************************************



#endif /* __EMU_MAPPER_H */
