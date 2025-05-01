#ifndef __EMU_LATENCY_H
#define __EMU_LATENCY_H

//*******************************************************************************************************
//** Model Specific Data Structures
//*******************************************************************************************************

/* This is arbitrary, we can probably extend latency chain longer, but just in case our implementation needs 
   per level context */

#define EMU_LAT_MODEL_UPNEXT      (1)
#define EMU_LAT_MODEL_CHECK_AVAIL (2)
#define EMU_LAT_MODEL_POP_ALGO    EMU_LAT_MODEL_UPNEXT 

//arbitrary - can make dynmaic later.  the emu_latency_model is static, but the channels are dynamic allocated at the end of the 
//emu_latency_model structure when allocated, requiring fix up of pointers.  Making the emu_latency_model dynamic requires extra 
//calclulations, and results in a deeper recursive (inline) call stack, so should put in some stack safety checks
#define EMU_LATENCY_MAX_CHAINS (8) 

typedef struct
{
	simt::atomic<uint64_t, simt::thread_scope_device> time_free_ns;
	uint8_t pad0[24];

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
		uint32_t xfer_bytes;
		uint32_t xfer_kbytes;
		uint32_t lat_error;
	} lat_context;
		
} latency_context;

#define LAT_STAT_GOOD 0
#define LAT_STAT_ERR  1 

typedef struct
{
	int      bLoopack;
	uint32_t channels;
	uint32_t latency_ns;
	uint32_t jitter;
	uint32_t per_k_transfer_multiplier;
	
	char     mnemonic[EMU_LATENCY_MNEMONIC_LEN];



	
	emu_latency_channel *pChannels;



#if(EMU_LAT_MODEL_POP_ALGO  == EMU_LAT_MODEL_UPNEXT) 
			simt::atomic<uint32_t, simt::thread_scope_device> up_next;
			uint8_t pad0[28];
#endif
	


} emu_latency_chain;

#define LAT_LOOPBACK_LEVEL_TOP 0xFFFFFFFF
typedef struct
{
	uint32_t nLoobackLevel; //null zero, so index to level is -1 
	uint32_t chain_count;   
	uint32_t channel_offset;
	uint32_t total_channel_size;
	uint32_t total_alloc_size;
	uint32_t model_op_type;   //TODO: First pass model for homogenus flows, all read, all write, figure out combo and shared latency stages later
	
	emu_latency_chain latency_chain[EMU_LATENCY_MAX_CHAINS];

} emu_latency_model;


//*******************************************************************************************************
//** Models
//*******************************************************************************************************

/* Just have one model for reads, later we will have model for reads and writes seperately */
emu_latency_model simple_generic =
{
	0,// LAT_LOOPBACK_LEVEL_TOP, //0, //loopback
	5, //chain_count
	0, //channel_offset - dynamically filled in
	0, //total_channel_size - dyanmically filled in
	0, //total_alloc_size - dynmaically_filled in
	SN_OP_READ, 
	{
		{0, 1,  100,   0, 0, "L2P_DDR",       NULL}, //0
		{0, 16, 30000, 0, 0, "TREAD_NAND",    NULL}, //1
		{0, 1,  500,   0, 1, "TRANSFER_NAND", NULL}, //2
		{0, 1,  200,   0, 1, "DECODE_NAND",   NULL}, //3
		{0, 1,  200,   10, 1, "FLUSH",         NULL}, //4
			

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
	pLatModel->total_alloc_size = size;

	

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

__device__ inline void emu_model_latency_enqueue(latency_context **ppLatListHead, latency_context *pLatContext)
{
	int verbose = bam_get_verbosity(BAM_EMU_DBGLVL_NONE, BAM_DBG_CODE_PATH_D_LATENCY);

	latency_context *pTemp = *ppLatListHead;
	
	BAM_EMU_DEV_DBG_PRINT4(verbose, "LAT:emu_model_latency_enqueue() pTemp = %p ppLatListHead = %p pLatContext = %p  done_ns = %ld\n", pTemp, ppLatListHead, pLatContext, pLatContext->lat_context.done_ns);
	
	if(NULL == pTemp)
	{
		*ppLatListHead = pLatContext;
		pLatContext->lat_context.pNext = NULL;

	}
	else
	{
		uint64_t ns = pLatContext->lat_context.done_ns;
		while(pTemp)
		{

			BAM_EMU_DEV_DBG_PRINT2(verbose, "LAT:emu_model_latency_enqueue() pTemp = %p pTemp.done_ns  %ld\n", pTemp, pTemp->lat_context.done_ns);

			if(ns >= pTemp->lat_context.done_ns)
			{
				pLatContext->lat_context.pNext = pTemp->lat_context.pNext;
				pTemp->lat_context.pNext = pLatContext;
				break;
			}
			else
			{
				//this shouldn't happen with loopback, but could wiht multi channels latency + jitter when 
				//one IO lucks into a shorter path;
				//deal with later
				assert(0);
			}

		}

	}


}

#if(EMU_LAT_MODEL_POP_ALGO == EMU_LAT_MODEL_UPNEXT)
__device__ inline int emu_mode_find_channel(emu_latency_chain *pChain, uint64_t *pChDoneNs)
{
	int verbose = bam_get_verbosity(BAM_EMU_DBGLVL_NONE, BAM_DBG_CODE_PATH_D_LAT_RECURSE);
	int channel = -1;
	uint32_t cht;

	if(1 == pChain->channels)
	{
		channel = 0;
	}
	else 
	{
		cht = pChain->up_next.fetch_add(1, simt::memory_order_relaxed);

		channel = cht % pChain->channels;

		BAM_EMU_DEV_DBG_PRINT4(verbose, "emu_mode_find_channel(%p) cht = %d total_channels = %d channel = %d \n", pChain, cht, pChain->channels, channel);

	}
	
	*pChDoneNs = pChain->pChannels[channel].time_free_ns.load(simt::memory_order_acquire);
		
	BAM_EMU_DEV_DBG_PRINT1(verbose, "emu_mode_find_channel() *pChDoneNs = %ld\n", *pChDoneNs);




	
	return channel;
}

#elif (EMU_LAT_MODEL_POP_ALGO == EMU_LAT_MODEL_CHECK_AVAIL)


#else
#error INVALID EMU_LAT_MODEL_POP_ALGO
#endif


__device__ inline void emu_model_update_channel_done_ns(emu_latency_chain *pChain, int channel, uint64_t channel_done_ns)
{
	pChain->pChannels[channel].time_free_ns.store(channel_done_ns, simt::memory_order_release);
}



#define NO_RECURSION
#ifdef NO_RECURSION
#define STOP_RECURSION 1
#define CONTINUE_RECURSION 0
#else
#define STOP_RECURSION 0
#endif

__device__ inline int emu_model_latency_recurse(emu_latency_model *pLatModel, latency_context *pLatContext, uint32_t level)
{
	int error = 0;
	int verbose = bam_get_verbosity(BAM_EMU_DBGLVL_NONE, BAM_DBG_CODE_PATH_D_LAT_RECURSE);
	int channel;
	emu_latency_chain *pChain;
	uint64_t channel_done_ns;
	uint64_t latency_adder_ns;
	const int stop_recursion = STOP_RECURSION;

	
	static int counter = 0;

	BAM_EMU_DEV_DBG_PRINT2(verbose, "emu_model_latency_recurse(%d) counter = %d\n", level, counter++);

	
	
	assert(pLatModel);
	assert(pLatContext);
	assert(level < EMU_LATENCY_MAX_CHAINS);
	
	if(level == pLatModel->chain_count)
	{
		return stop_recursion;
	}

	assert(level < pLatModel->chain_count);

	if(pLatModel->nLoobackLevel)
	{
		if((1 << level) & pLatModel->nLoobackLevel)
		{
			return stop_recursion;
		}

	}
	

	assert(level < pLatModel->chain_count);
	assert(level < EMU_LATENCY_MAX_CHAINS);

	BAM_EMU_DEV_DBG_PRINT2(verbose, "emu_model_latency_recurse(%d) pChain = %p\n", level, &pLatModel->latency_chain[level]);
		
	pChain = &pLatModel->latency_chain[level];

	BAM_EMU_DEV_DBG_PRINT2(verbose, "emu_model_latency_recurse(%d) CALL emu_mode_find_channel pChain = %p\n", level, pChain);

	channel = emu_mode_find_channel(pChain, &channel_done_ns);

	
	if((0 == channel_done_ns) || (channel_done_ns < pLatContext->lat_context.start_ns))
	{
		//First time it's run or the model chain-link on this channel has gone idle ( less than case) 
		channel_done_ns = pLatContext->lat_context.done_ns;
	}

	BAM_EMU_DEV_DBG_PRINT4(verbose, "LAT:emu_model_latency_recurse(%d) chain_count = %d channel = %d channel_done_ns = %ld\n", level, pLatModel->chain_count, channel, channel_done_ns);
	
	latency_adder_ns = pChain->latency_ns;

	if(pChain->jitter)
	{
		//TODO: BA - come up with a more semi-random (ns timestamps are always even....)
		uint32_t rand_factor = ((uint32_t)pLatContext->lat_context.start_ns & 0xFFFF);
		
		uint32_t cur_jitter = rand_factor % pChain->jitter;

		//BAM_EMU_DEV_DBG_PRINT4(verbose, "LAT:emu_model_latency_recurse(%d) JITTER rand_factor = %d jitter = %d cur_jitter = %d\n", level, rand_factor, pChain->jitter, rand_factor % pChain->jitter);
		
		latency_adder_ns += (cur_jitter);
	}

	if(pChain->per_k_transfer_multiplier)
	{
		uint32_t xfer_factor = pLatContext->lat_context.xfer_kbytes * pChain->per_k_transfer_multiplier;

		BAM_EMU_DEV_DBG_PRINT3(verbose, "LAT:emu_model_latency_recurse(%d) PER_K_XFER_MULT = %d xfer_factor = %d\n", level, pChain->per_k_transfer_multiplier, xfer_factor);

		latency_adder_ns += xfer_factor;
	}

	

	pLatContext->lat_context.done_ns += latency_adder_ns;

	emu_model_update_channel_done_ns(pChain, channel, pLatContext->lat_context.done_ns);

	BAM_EMU_DEV_DBG_PRINT4(verbose, "LAT:emu_model_latency_recurse(%d) latency_adder_ns = %ld channel = %d context.done_ns = %ld\n", level, latency_adder_ns, channel, pLatContext->lat_context.done_ns);
#ifdef NO_RECURSION
	return CONTINUE_RECURSION;
#else
	return emu_model_latency_recurse(pLatModel, pLatContext, level + 1);
#endif
}

__device__ inline int emu_model_latency_submit(bam_emu_target_model *pModel, storage_next_emuluator_context *pContext, void **ppvThreadContext)
{
	latency_context **pLatListHead = (latency_context **) ppvThreadContext;
	latency_context *pLatContext = (latency_context *)pContext;
	int verbose = bam_get_verbosity(BAM_EMU_DBGLVL_NONE, BAM_DBG_CODE_PATH_D_LATENCY);
	emu_latency_model *pLatModel = (emu_latency_model *)pModel->pvDevPrivate;
	
	BAM_EMU_DEV_DBG_PRINT4(verbose, "LAT:emu_model_latency_submit() pModel = %p pContext = %p pLatListHead = %p op = %x\n", pModel, pContext, pLatListHead, SN_CONTEXT_OP(pContext));

	pLatContext->lat_context.start_ns = NS_Clock();

	
	pLatContext->lat_context.done_ns = pLatContext->lat_context.start_ns;
	
	if(LAT_LOOPBACK_LEVEL_TOP == pLatModel->nLoobackLevel)
	{
		//short circuit, it will be ready for completion immediately, because we set done_ns to start_ns above
	
		BAM_EMU_DEV_DBG_PRINT2(verbose, "LAT:emu_model_latency_submit() LAT_LOOPBACK_LEVEL_TOP = 0x%08x start_ns = %ld\n", pLatModel->nLoobackLevel, pLatContext->lat_context.start_ns);

		emu_model_latency_enqueue(pLatListHead, pLatContext);

		return 0;
	}

	//calculate the xfer parameters
	pLatContext->lat_context.xfer_bytes = SN_CONTEXT_NUM_BLOCKS(pContext) * pModel->block_size;

	if(pLatContext->lat_context.xfer_bytes < 1024)
	{
		pLatContext->lat_context.xfer_kbytes = 1;	
	}
	else
	{
		pLatContext->lat_context.xfer_kbytes = pLatContext->lat_context.xfer_bytes / 1024;
		if(pLatContext->lat_context.xfer_bytes % 1024)
		{
			pLatContext->lat_context.xfer_kbytes++;
		}

	}
		
	pLatContext->lat_context.lat_error = LAT_STAT_GOOD;
	
	
	BAM_EMU_DEV_DBG_PRINT4(verbose, "LAT:emu_model_latency_submit() xfer_bytes = %d block_size = %d dword[12] = 0x%08x kbytes = %d\n", pLatContext->lat_context.xfer_bytes, pModel->block_size, pContext->pCmd->nvme_cmd.dword[12], pLatContext->lat_context.xfer_kbytes);

	
	switch(SN_CONTEXT_OP(pContext))
	{
		case SN_OP_READ:
			if(SN_OP_READ == pLatModel->model_op_type)
			{
	
#ifdef NO_RECURSION
				uint32_t i;

				
				for(i = 0; i < EMU_LATENCY_MAX_CHAINS; i++)
				{
					if(STOP_RECURSION == emu_model_latency_recurse(pLatModel, pLatContext, i))
					{
						break;
					}

				}
				if(LAT_STAT_GOOD == pLatContext->lat_context.lat_error)
				{
					emu_model_latency_enqueue(pLatListHead, pLatContext);
				}
				else
				{
					BAM_EMU_DEV_DBG_PRINT1(BAM_EMU_DBGLVL_ERROR, "LAT:CALL emu_model_latency_recurse() CALL(SN_OP_READ) ERROR %d\n", pLatContext->lat_context.lat_error);
				}
#else
			
				if(emu_model_latency_recurse(pLatModel, pLatContext, 0))
				{
					BAM_EMU_DEV_DBG_PRINT1(BAM_EMU_DBGLVL_ERROR, "LAT:CALL emu_model_latency_recurse() CALL(SN_OP_READ) ERROR %d\n", 0);
				}
				else
				{
					BAM_EMU_DEV_DBG_PRINT1(verbose, "LAT:emu_model_latency_recurse() DONE(SN_OP_READ) total_latency = %ld (ns)\n", pLatContext->lat_context.done_ns - pLatContext->lat_context.start_ns);
					emu_model_latency_enqueue(pLatListHead, pLatContext);
				}

#endif

			}
			else
			{
				BAM_EMU_DEV_DBG_PRINT2(BAM_EMU_DBGLVL_ERROR, "LAT:emu_model_latency_submit() (SN_OP_READ)op = %d model_op_type = %d ERROR!!!\n", SN_OP_READ, pLatModel->model_op_type);
				assert(0);
			}
			break;


		default:
			assert(0);
			break;					

	}


	return 0;
}


__device__ inline storage_next_emuluator_context * emu_model_latency_cull(bam_emu_target_model *pModel, void **ppvThreadContext)
{
	latency_context **ppLatListHead = (latency_context **) ppvThreadContext;
	storage_next_emuluator_context *pTemp;
	int verbose = bam_get_verbosity(BAM_EMU_DBGLVL_NONE, BAM_DBG_CODE_PATH_D_LATENCY);
	BAM_EMU_DEV_DBG_PRINT2(verbose, "emu_model_latency_cull(%p, %p)\n", pModel, *ppLatListHead);
	

	if(*ppLatListHead)
	{
		uint64_t now_ns = NS_Clock();

		BAM_EMU_DEV_DBG_PRINT2(verbose, "emu_model_latency_cull(%ld, %ld)\n", now_ns, (*ppLatListHead)->lat_context.done_ns);

		if(now_ns >= (*ppLatListHead)->lat_context.done_ns)	
		{
			pTemp = (storage_next_emuluator_context *) *ppLatListHead;
			*ppLatListHead = (*ppLatListHead)->lat_context.pNext;
			return pTemp;
		}
	}


	return NULL;
}

#endif

