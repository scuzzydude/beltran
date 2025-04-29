#ifndef __EMU_MAPPER_H
#define __EMU_MAPPER_H




//*******************************************************************************************************
//** Host Functions  
//*******************************************************************************************************

static inline int emulator_init_mapper(bam_host_emulator *pEmu, uint32_t mapType, uint32_t modelType)
{
	int model_mem_size = 0;
	int private_size = 0;
	bam_emu_target_model *pModel;
	bam_emu_mapper *pMap;
	bam_target_emulator *pTgt;
	fnModelPrivateInit fnMdlPrvInit = NULL;
	
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
	pTgt->mapper.model.block_size = pEmu->sectorSize;

	BAM_EMU_HOST_DBG_PRINT(verbose, "emulator_init_mapper(block_size = %d)  ** \n", pTgt->mapper.model.block_size);

	switch(modelType)
	{
		case EMU_MODEL_TYPE_LATENCY:
			strcpy(pTgt->mapper.model.szModelName, "Latency Model");
		//	private_size = emu_model_latency_private_size();
			fnMdlPrvInit = emu_model_latency_private_init;
			break;
			
		case EMU_MODEL_TYPE_AGGREGATION:
			strcpy(pTgt->mapper.model.szModelName, "NVMe Aggregation Model");
			fnMdlPrvInit = emu_model_aggregation_private_init;
			break;

		case EMU_MODEL_TYPE_VENDOR:
			strcpy(pTgt->mapper.model.szModelName, get_vendor_model_name());
			fnMdlPrvInit = emu_model_vendor_private_init;
			break;
		

		default:
			BAM_EMU_HOST_DBG_PRINT(BAM_EMU_DBGLVL_ERROR, "Invalid Model Type %d\n", modelType);
			BAM_EMU_HOST_ASSERT(0);
			break;
			
	}

	
	pTgt->pTgt_control->d_mapper = createBuffer(sizeof(bam_emu_mapper), pEmu->cudaDevice);
	pTgt->pTgt_control->pDevMapper = (bam_emu_mapper *)pTgt->pTgt_control->d_mapper.get();

	BAM_EMU_HOST_DBG_PRINT(verbose, "pDevMapper = %p size = %ld\n", mapType, sizeof(bam_emu_mapper));

	model_mem_size += sizeof(bam_emu_mapper);


	if(fnMdlPrvInit)
	{
		model_mem_size += fnMdlPrvInit(pEmu, &pTgt->mapper.model);
	}

    cuda_err_chk(cudaMemcpy(pTgt->pTgt_control->pDevMapper, &pTgt->mapper, sizeof(bam_emu_mapper), cudaMemcpyHostToDevice));

	BAM_EMU_HOST_DBG_PRINT(verbose, "Initialized Mapper with type = %d (%s)\n", mapType, pTgt->mapper.szMapName);
	BAM_EMU_HOST_DBG_PRINT(verbose, "Initialized Model  with type = %d (%s)\n", modelType, pTgt->mapper.model.szModelName);
	BAM_EMU_HOST_DBG_PRINT(verbose, "Model pvDevPrivate           = %p     \n", pTgt->mapper.model.pvDevPrivate);
	BAM_EMU_HOST_DBG_PRINT(verbose, "Model pvHostPrivate          = %p     \n", pTgt->mapper.model.pvHostPrivate);
	

	return model_mem_size;

}

//*******************************************************************************************************
//** Device Functions  
//*******************************************************************************************************

__device__ inline int emu_tgt_map_model_submit(bam_emu_mapper *pDevMapper, storage_next_emuluator_context *pContext, void **ppvThreadContext)
{
	
	uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

	int verbose = bam_get_verbosity(BAM_EMU_DBGLVL_NONE, BAM_DBG_CODE_PATH_D_MAPPER);

	BAM_EMU_DEV_DBG_PRINT2(verbose, "TGT: emu_tgt_map_model_submit(%ld) uModelType = %d\n", tid, pDevMapper->model.uModelType);

	/* Device function pointers complicated in CUDA and using all inline functions */
	/* If burdensome later on, we can make the Models compile time selectable */
	switch(pDevMapper->model.uModelType)
	{

		case EMU_MODEL_TYPE_LATENCY:
			return emu_model_latency_submit(&pDevMapper->model, pContext, ppvThreadContext);
			
				
		case EMU_MODEL_TYPE_AGGREGATION:
			return emu_model_aggregation_submit(&pDevMapper->model, pContext, ppvThreadContext);
			
		
		case EMU_MODEL_TYPE_VENDOR:
			return emu_model_vendor_submit(&pDevMapper->model, pContext, ppvThreadContext);

		default:
		BAM_EMU_DEV_DBG_PRINT1(BAM_EMU_DBGLVL_ERROR, "emu_tgt_map_model_submit() : Invalid Model Type %d\n", pDevMapper->model.uModelType);
		break;
		
	}

	return 1;

}


/* The mapper will be used later on to map to targets or map to other peer kernel threads based on data location (could be multi-path) or different kernel engines */ 
__device__ inline int emu_tgt_map_Submit(bam_emu_mapper *pDevMapper, storage_next_emuluator_context *pContext, void ** ppvThreadContext)
{
	uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

	int verbose = bam_get_verbosity(BAM_EMU_DBGLVL_NONE, BAM_DBG_CODE_PATH_D_MAPPER);

	BAM_EMU_DEV_DBG_PRINT2(verbose, "TGT: emu_tgt_map_Submit(%ld) uMapType = %d\n", tid, pDevMapper->uMapType);

	/* Device function pointers complicated in CUDA and using all inline functions */
	/* If burdensome later on, we can make the Models compile time selectable */
	switch(pDevMapper->uMapType)
	{
		case EMU_MAP_TYPE_DIRECT:
			return emu_tgt_map_model_submit(pDevMapper, pContext, ppvThreadContext);
			
		
		default:
			BAM_EMU_DEV_DBG_PRINT1(BAM_EMU_DBGLVL_ERROR, "emu_tgt_map_Submit() : Invalid Map Type %d\n", pDevMapper->uMapType);
			break;


	}

	return 1;
}

__device__ inline storage_next_emuluator_context * emu_tgt_model_Cull(bam_emu_mapper *pDevMapper, void ** ppvThreadContext)
{
	switch(pDevMapper->model.uModelType)
	{

		case EMU_MODEL_TYPE_LATENCY:
			return emu_model_latency_cull(&pDevMapper->model, ppvThreadContext);
			
				
		case EMU_MODEL_TYPE_AGGREGATION:
			return NULL;//emu_model_aggregation_submit(&pDevMapper->model, pContext, pvThreadContext);
			
		
		case EMU_MODEL_TYPE_VENDOR:
			return NULL; //emu_model_vendor_submit(&pDevMapper->model, pContext, pvThreadContext);

		default:
			BAM_EMU_DEV_DBG_PRINT1(BAM_EMU_DBGLVL_ERROR, "emu_tgt_model_Cull() : Invalid Model Type %d\n", pDevMapper->model.uModelType);
		break;
	}

	return NULL;
	
}

__device__ inline storage_next_emuluator_context * emu_tgt_map_Cull(bam_emu_mapper *pDevMapper, void ** ppvThreadContext)
{

	switch(pDevMapper->uMapType)
	{
		case EMU_MAP_TYPE_DIRECT:
			return emu_tgt_model_Cull(pDevMapper, ppvThreadContext);
				
			
		default:
			BAM_EMU_DEV_DBG_PRINT1(BAM_EMU_DBGLVL_ERROR, "emu_tgt_map_Cull() : Invalid Map Type %d\n", pDevMapper->uMapType);
			break;
	
	
	}
	return NULL;
}



#endif /* __EMU_MAPPER_H */
