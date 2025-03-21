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

	
	pTgt->pTgt_control->d_mapper = createBuffer(sizeof(bam_emu_mapper), pEmu->cudaDevice);
	pTgt->pTgt_control->pDevMapper = (bam_emu_mapper *)pTgt->pTgt_control->d_mapper.get();

	BAM_EMU_HOST_DBG_PRINT(verbose, "pDevMapper = %p size = %ld\n", mapType, sizeof(bam_emu_mapper));

    cuda_err_chk(cudaMemcpy(pTgt->pTgt_control->pDevMapper, &pTgt->mapper, sizeof(bam_emu_mapper), cudaMemcpyHostToDevice));




	BAM_EMU_HOST_DBG_PRINT(verbose, "Initialized Mapper with type = %d (%s)\n", mapType, pTgt->mapper.szMapName);
	BAM_EMU_HOST_DBG_PRINT(verbose, "Initialized Model  with type = %d (%s)\n", modelType, pTgt->mapper.model.szModelName);
	

	

}

//*******************************************************************************************************
//** Device Functions  
//*******************************************************************************************************

__device__ inline int emu_tgt_map_model_submit(bam_emu_mapper *pDevMapper, storage_next_emuluator_context *pContext)
{

	/* Device function pointers complicated in CUDA and using all inline functions */
	/* If burdensome later on, we can make the Models compile time selectable */
	switch(pDevMapper->model.uModelType)
	{

		case EMU_MODEL_TYPE_LATENCY:
			return emu_model_latency_submit(&pDevMapper->model, pContext);
			
				
		case EMU_MODEL_TYPE_AGGREGATION:
			return emu_model_aggregation_submit(&pDevMapper->model, pContext);
			
		
		case EMU_MODEL_TYPE_VENDOR:
			return emu_model_vendor_submit(&pDevMapper->model, pContext);

		default:
		BAM_EMU_DEV_DBG_PRINT1(BAM_EMU_DBGLVL_ERROR, "emu_tgt_map_model_submit() : Invalid Model Type %d\n", pDevMapper->model.uModelType);
		break;
		
	}

	return 1;

}


/* The mapper will be used later on to map to targets or map to other peer kernel threads based on data location (could be multi-path) or different kernel engines */ 
__device__ inline int emu_tgt_map_Submit(bam_emu_mapper *pDevMapper, storage_next_emuluator_context *pContext)
{

	/* Device function pointers complicated in CUDA and using all inline functions */
	/* If burdensome later on, we can make the Models compile time selectable */
	switch(pDevMapper->uMapType)
	{
		case EMU_MAP_TYPE_DIRECT:
			return emu_tgt_map_model_submit(pDevMapper, pContext);
		
		
		
		default:
			BAM_EMU_DEV_DBG_PRINT1(BAM_EMU_DBGLVL_ERROR, "emu_tgt_map_Submit() : Invalid Map Type %d\n", pDevMapper->uMapType);
			break;


	}

	return 1;
}


#endif /* __EMU_MAPPER_H */
