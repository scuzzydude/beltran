#ifndef _EMU_CTRL_H
#define _EMU_CTRL_H

#define EMU_MAX_QUEUES 32


struct EmuQueuePair
{
    uint32_t            pageSize;
    uint32_t            block_size;
    uint32_t            block_size_log;
    uint32_t            block_size_minus_1;
    uint32_t            nvmNamespace;
    //void*               prpList;
    //uint64_t*           prpListIoAddrs;
    nvm_queue_t         sq;
    nvm_queue_t         cq;
    uint16_t            qp_id;
    DmaPtr              sq_mem;
    DmaPtr              cq_mem;
    DmaPtr              prp_mem;
    BufferPtr           sq_tickets;
    //BufferPtr           sq_head_mark;
    BufferPtr           sq_tail_mark;
    BufferPtr           sq_cid;
    //BufferPtr           cq_tickets;
    BufferPtr           cq_head_mark;
    //BufferPtr           cq_tail_mark;
    BufferPtr           cq_pos_locks;
    //BufferPtr           cq_clean_cid;

	bool sq_need_prp;  
	bool cq_need_prp;  
			
	size_t sq_mem_size; 
	size_t cq_mem_size;
	uint64_t cq_size;
	uint64_t sq_size;

};



struct EmuController
{
    simt::atomic<uint64_t, simt::thread_scope_device> access_counter;
    nvm_ctrl_t*             ctrl;
    nvm_aq_ref              aq_ref;
    DmaPtr                  aq_mem;
    struct nvm_ctrl_info    info;
    struct nvm_ns_info      ns;
    uint16_t                n_sqs;
    uint16_t                n_cqs;
    uint16_t                n_qps;
    uint32_t                deviceId;
    EmuQueuePair**             h_qps;
    EmuQueuePair*              d_qps;

    simt::atomic<uint64_t, simt::thread_scope_device> queue_counter;

    uint32_t page_size;
    uint32_t blk_size;
    uint32_t blk_size_log;


    void* d_ctrl_ptr;
    BufferPtr d_ctrl_buff;


    EmuController(const char* path, uint32_t nvmNamespace, uint32_t cudaDevice, uint64_t queueDepth, uint64_t numQueues);
		
    void reserveQueues();

    void reserveQueues(uint16_t numSubmissionQueues);

    void reserveQueues(uint16_t numSubmissionQueues, uint16_t numCompletionQueues);

    void print_reset_stats(void);

    ~EmuController();
};

inline void EmuController::reserveQueues()
{
    reserveQueues(n_sqs, n_cqs);
}



inline void EmuController::reserveQueues(uint16_t numSubmissionQueues)
{
    reserveQueues(numSubmissionQueues, n_cqs);
}



inline void EmuController::reserveQueues(uint16_t numSubs, uint16_t numCpls)
{

    int status = nvm_admin_request_num_queues(aq_ref, &numSubs, &numCpls);
    if (!nvm_ok(status))
    {
        throw error(string("Failed to reserve queues: ") + nvm_strerror(status));
    }

    n_sqs = numSubs;
    n_cqs = numCpls;

}

static void initializeEmuController(struct EmuController& ctrl, uint32_t ns_id)
{
    // Create admin queue reference
    int status = nvm_aq_create(&ctrl.aq_ref, ctrl.ctrl, ctrl.aq_mem.get());
    if (!nvm_ok(status))
    {
        throw error(string("Failed to reset controller: ") + nvm_strerror(status));
    }

    // Identify controller
    status = nvm_admin_ctrl_info(ctrl.aq_ref, &ctrl.info, NVM_DMA_OFFSET(ctrl.aq_mem, 2), ctrl.aq_mem->ioaddrs[2]);
    if (!nvm_ok(status))
    {
        throw error(nvm_strerror(status));
    }

    // Identify namespace
    status = nvm_admin_ns_info(ctrl.aq_ref, &ctrl.ns, ns_id, NVM_DMA_OFFSET(ctrl.aq_mem, 2), ctrl.aq_mem->ioaddrs[2]);
    if (!nvm_ok(status))
    {
        throw error(nvm_strerror(status));
    }

    // Get number of queues
    status = nvm_admin_get_num_queues(ctrl.aq_ref, &ctrl.n_cqs, &ctrl.n_sqs);
    if (!nvm_ok(status))
    {
        throw error(nvm_strerror(status));
    }
}

inline EmuController::EmuController(const char* path, uint32_t ns_id, uint32_t cudaDevice, uint64_t queueDepth, uint64_t numQueues)
	: ctrl(nullptr)
    , aq_ref(nullptr)
    , deviceId(cudaDevice)
{
	unsigned int mmFlag = cudaHostRegisterIoMemory;

	{
    	int fd = open(path, O_RDWR);
    	if (fd < 0)
    	{
        	throw error(string("Failed to open descriptor: ") + strerror(errno));
    	}

    	// Get controller reference
    	int status = nvm_ctrl_init(&ctrl, fd);
    	if (!nvm_ok(status))
    	{
        	throw error(string("Failed to get controller reference: ") + nvm_strerror(status));
    	}
		close(fd);

	}

	// Create admin queue memory
	aq_mem = createDma(ctrl, ctrl->page_size * 3);
	
	initializeEmuController(*this, ns_id);

	cudaError_t err = cudaHostRegister((void*) ctrl->mm_ptr, ctrl->mm_size, mmFlag);
	
	if (err != cudaSuccess)
	{
		throw error(string("Unexpected error while mapping IO memory (cudaHostRegister): ") + cudaGetErrorString(err));
	}



    queue_counter = 0;
    page_size = ctrl->page_size;
    blk_size = this->ns.lba_data_size;
    blk_size_log = std::log2(blk_size);
    reserveQueues(EMU_MAX_QUEUES,EMU_MAX_QUEUES);
    n_qps = std::min(n_sqs, n_cqs);
    n_qps = std::min(n_qps, (uint16_t)numQueues);
		
	size_t h_qp_size = sizeof(EmuQueuePair) * n_qps;
	
    printf("EmuController:SQs: %d\tCQs: %d\tn_qps: %d queueDepth = %ld\n", n_sqs, n_cqs, n_qps, queueDepth );
    h_qps = (EmuQueuePair**) malloc(h_qp_size);

	printf("h_qps = %p h_qp_size = %ld \n", h_qps, h_qp_size);
	
	
    cuda_err_chk(cudaMalloc((void**)&d_qps, sizeof(EmuQueuePair)*n_qps));
    for (size_t i = 0; i < n_qps; i++) 
	{
     //   printf("started creating qp %ld\n", i);
     //   h_qps[i] = new QueuePair(ctrl, cudaDevice, ns, info, aq_ref, i+1, queueDepth, pEmu);
    //    printf("finished creating qp %ld\n", i);
        cuda_err_chk(cudaMemcpy(d_qps+i, h_qps[i], sizeof(EmuQueuePair), cudaMemcpyHostToDevice));
   //     printf("finished copy QP Memory to device %ld\n", i);

		

    }
    printf("finished creating all qps\n");


    
    d_ctrl_buff = createBuffer(sizeof(EmuController), cudaDevice);
    d_ctrl_ptr = d_ctrl_buff.get();
    cuda_err_chk(cudaMemcpy(d_ctrl_ptr, this, sizeof(EmuController), cudaMemcpyHostToDevice));





}



inline EmuController::~EmuController()
{

    cudaFree(d_qps);
    for (size_t i = 0; i < n_qps; i++) {
        delete h_qps[i];
    }
    free(h_qps);
    nvm_aq_destroy(aq_ref);
    nvm_ctrl_free(ctrl);

}



#endif
