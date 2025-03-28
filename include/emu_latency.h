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

#define EMU_LATENCY_MNEMONIC_LEN 16

typedef union _latency_context
{
	storage_next_emuluator_context context;
	struct
	{
		void *pvCmd;
		uint64_t start_ns;
		uint64_t done_ns;
		union _latency_context *pNext;
		
	} lat_context;
		
} latency_context;


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

#define LAT_LOOPBACK_LEVEL_TOP 0xFFFFFFFF
typedef struct
{
	uint32_t nLoobackLevel; //null zero, so index to level is -1 
	uint32_t chain_count;   
	uint32_t channel_offset;
	uint32_t total_channel_size;
	
	emu_latency_chain latency_chain[EMU_LATENCY_MAX_CHAINS];

} emu_latency_model;


//*******************************************************************************************************
//** Models
//*******************************************************************************************************

/* Just have one model for reads, later we will have model for reads and writes seperately */
emu_latency_model simple_generic =
{
	LAT_LOOPBACK_LEVEL_TOP, //0, //loopback
	5, //chain_count
	0, //channel_offset - dynamically filled in
	0, //total_channel_size - dyanmically filled in
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


uint32_t emu_model_latency_private_size(uint32_t *pTotal_channel_size)
{
	int verbose = bam_get_verbosity(BAM_EMU_DBGLVL_INFO, BAM_DBG_CODE_PATH_H_LAT_PSIZE);
	int i;
	
	uint32_t size = sizeof(emu_latency_model);
	*pTotal_channel_size = 0;

	for(i = 0; i < pWorkingModel->chain_count; i++)
	{
		uint32_t ch_size = pWorkingModel->latency_chain[i].channels * sizeof(emu_latency_channel);
		size += ch_size;
		*pTotal_channel_size += ch_size;
	}

	BAM_EMU_HOST_DBG_PRINT(verbose, "emu_model_latency_private_size() size = %d channels = %d\n", size, pWorkingModel->chain_count);

	return size;

}


uint32_t emu_model_latency_private_init(bam_host_emulator *pEmu, bam_emu_target_model *pModel)
{
	
	int verbose = bam_get_verbosity(BAM_EMU_DBGLVL_INFO, BAM_DBG_CODE_PATH_H_LAT_PSIZE);
	uint32_t total_channel_size;
	uint32_t size = emu_model_latency_private_size(&total_channel_size);
	emu_latency_model *pLatModel;
	emu_latency_channel *pChan;
	int i;

	BAM_EMU_HOST_DBG_PRINT(verbose, "emu_model_latency_private_init() size = %d total_channel_size = %d\n", size, total_channel_size);


	pModel->pvHostPrivate = malloc(size);
	memcpy(pModel->pvHostPrivate, pWorkingModel, sizeof(emu_latency_model));
	
	

	pModel->d_model_private = createBuffer(size, pEmu->cudaDevice);
	pModel->pvDevPrivate = pModel->d_model_private.get();

	pLatModel = (emu_latency_model *)pModel->pvHostPrivate;

	pLatModel->channel_offset = sizeof(emu_latency_model);
	pLatModel->total_channel_size = total_channel_size;


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
	
	int verbose = bam_get_verbosity(BAM_EMU_DBGLVL_INFO, BAM_DBG_CODE_PATH_D_LATENCY);
	emu_latency_model *pLatModel = (emu_latency_model *)pModel->pvDevPrivate;
	int i;
	

	BAM_EMU_DEV_DBG_PRINT1(verbose, "LAT: emu_model_private_data_check() pLatModel=%p\n", pLatModel);
	BAM_EMU_DEV_DBG_PRINT3(verbose, "LAT: emu_model_private_data_check() nLoobackLevel=%d chain_count = %d MAX_CHAINS =%d\n", pLatModel->nLoobackLevel, pLatModel->chain_count, EMU_LATENCY_MAX_CHAINS);
	
	for(i = 0; i <  pLatModel->chain_count; i++)
	{
		BAM_EMU_DEV_DBG_PRINT4(verbose,"LAT:(%d) bLoopback = %d channels = %d latency_ns = %d\n", i, pLatModel->latency_chain[i].bLoopack,pLatModel->latency_chain[i].channels, pLatModel->latency_chain[i].latency_ns); 
		BAM_EMU_DEV_DBG_PRINT4(verbose,"LAT:(%d) jitter    = %d per_k_transfer_multiplier = %d mnemonic = %s\n", i, pLatModel->latency_chain[i].bLoopack,pLatModel->latency_chain[i].per_k_transfer_multiplier, pLatModel->latency_chain[i].mnemonic); 
		
	}
	
	assert(0);
	
	return 0;
}

__device__ inline int emu_model_latency_submit(bam_emu_target_model *pModel, storage_next_emuluator_context *pContext)
{

	int verbose = bam_get_verbosity(BAM_EMU_DBGLVL_INFO, BAM_DBG_CODE_PATH_D_LATENCY);
	emu_latency_model *pLatModel = (emu_latency_model *)pModel->pvDevPrivate;
	
	BAM_EMU_DEV_DBG_PRINT2(verbose, "LAT:emu_model_latency_submit() pModel = %p pContext = %p\n", pModel, pContext);

	switch(SN_CONTEXT_OP(pContext))
	{
		case SN_OP_READ:
			break;


		default:
			assert(0);
			break;					

	}

	//temp
	emu_model_private_data_check(pModel, pContext);

	return 0;
}




#endif

