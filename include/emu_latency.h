#ifndef __EMU_LATENCY_H
#define __EMU_LATENCY_H

//*******************************************************************************************************
//** Model Specific Data Structures
//*******************************************************************************************************

/* This is arbitrary, we can probably extend latency chain longer, but just in case our implementation needs 
   per level context */
#define EMU_LATENCY_MAX_CHAINS STORAGE_NEXT_CONTEXT_LEVELS

typedef struct
{
	/* temp, this needs to be atomic */
	uint64_t time_free_ns;

} emu_latency_channel;

#define EMU_LATENCY_MNEMONIC_LEN 32

typedef struct
{
	int      bLoopack;
	uint32_t channels;
	uint32_t latency_ns;
	uint32_t jitter;
	uint32_t per_k_transfer_multiplier;
	
	char     mnemonic[EMU_LATENCY_MNEMONIC_LEN];
	
	emu_latency_channel *pChannels;



} emu_latency_chain;


typedef struct
{
	uint32_t nLoobackLevel; //null zero, so index to level is -1 
	uint32_t chain_count;   
	
	emu_latency_chain latency_chain[EMU_LATENCY_MAX_CHAINS];

} emu_latency_model;


//*******************************************************************************************************
//** Models
//*******************************************************************************************************

/* Just have one model for reads, later we will have model for reads and writes seperately */
emu_latency_model simple_generic =
{
	0, //loopback
	5, //chain_count
	{
		{0, 1,  100,   0, 0, "L2P_DDR",       NULL}, //0
		{0, 16, 30000, 0, 0, "TREAD_NAND",    NULL}, //1
		{0, 1,  500,   0, 1, "TRANSFER_NAND", NULL}, //2
		{0, 1,  200,   0, 1, "DECODE_NAND",   NULL}, //3
		{0, 1,  200,   1, 1, "FLUSH",         NULL}, //4
			

	}

};




//*******************************************************************************************************
//** Host Functions  
//*******************************************************************************************************
emu_latency_model *pWorkingModel = &simple_generic;


uint32_t emu_model_latency_private_size(void)
{
	int verbose = bam_get_verbosity(BAM_EMU_DBGLVL_INFO, BAM_DBG_CODE_PATH_H_LAT_PSIZE);
	int i;
	
	uint32_t size = sizeof(emu_latency_model);

	for(i = 0; i < pWorkingModel->chain_count; i++)
	{
		size += pWorkingModel->latency_chain[i].channels * sizeof(emu_latency_channel);	
	}

	BAM_EMU_HOST_DBG_PRINT(verbose, "emu_model_latency_private_size() size = %d channels = %d\n", size, pWorkingModel->chain_count);

	return size;

}


uint32_t emu_model_latency_private_init(bam_host_emulator *pEmu, bam_emu_target_model *pModel)
{
	
	int verbose = bam_get_verbosity(BAM_EMU_DBGLVL_INFO, BAM_DBG_CODE_PATH_H_LAT_PSIZE);
	uint32_t size = emu_model_latency_private_size();
	emu_latency_model *pLatModel;
	emu_latency_channel *pChan;
	int i;

	pModel->pvHostPrivate = malloc(size);
	memcpy(pModel->pvHostPrivate, pWorkingModel, sizeof(emu_latency_model));
	
	

	pModel->d_model_private = createBuffer(size, pEmu->cudaDevice);
	pModel->pvDevPrivate = pModel->d_model_private.get();

	pLatModel = (emu_latency_model *)pModel->pvHostPrivate;

	pChan = (emu_latency_channel *)((char *)pModel->pvDevPrivate + sizeof(emu_latency_model));
	
	for(i = 0; i < pWorkingModel->chain_count; i++)
	{
		/* These are device pointers, host should never touch */
		pLatModel->latency_chain[i].pChannels = pChan;
	
		BAM_EMU_HOST_DBG_PRINT(verbose, "emu_model_latency_private_init() pChannels = %p channels = %d max_chains = %d\n", pChan, pWorkingModel->latency_chain[i].channels);

		pChan += pWorkingModel->latency_chain[i].channels;

		
		
	}

	cuda_err_chk(cudaMemcpy(pModel->pvDevPrivate, pModel->pvHostPrivate, size, cudaMemcpyHostToDevice));

	
	

	return size;

}



//*******************************************************************************************************
//** Device Functions  
//*******************************************************************************************************
__device__ inline int emu_model_private_data_check(bam_emu_target_model *pModel, storage_next_emuluator_context *pContext)
{
	emu_latency_model *pLatModel = (emu_latency_model *)pModel->pvDevPrivate;
	int i;

	BAM_EMU_DEV_DBG_PRINT4(BAM_EMU_DBGLVL_INFO, "private_data_check(%p) nLoobackLevel=%d chain_count=%d max_chains=%d \n", pLatModel, pLatModel->nLoobackLevel, pLatModel->chain_count, EMU_LATENCY_MAX_CHAINS);

	for(i = 0; i <  pLatModel->chain_count; i++)
	{


	}
	

	return 0;
}

__device__ inline int emu_model_latency_submit(bam_emu_target_model *pModel, storage_next_emuluator_context *pContext)
{
	//temp
	emu_model_private_data_check(pModel, pContext);

	return 0;
}




#endif

