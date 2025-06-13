#ifndef _EMU_CTRL_H
#define _EMU_CTRL_H

#define EMU_MAX_QUEUES 32

#define BAM_USE_BAM_CTRL_QUEUES


//We don't need much of the logic used by the BAM defined nvm_queue_t
//resulting in bigger memory footprint than required.
//However, the libnvm libarary functions are using the same structure definition
//so we'd need to redefine or link to different instance.
//review later, for standalone emulator, probably worth doing
//for now BAM_USE_BAM_CTRL_QUEUES is the default
typedef struct __align__(64) 
{
    uint16_t                no;             // Queue number (must be unique per SQ/CQ pair)
    uint16_t                max_entries;    // Maximum number of queue entries supported
    uint16_t                entry_size;     // Queue entry size
    uint32_t                head;           // Queue's head pointer
    uint32_t                tail;           // Queue's tail pointer
    // TODO: Create bitfield for phase, add a remote field indicating
    //       if queue is far memory nor not, in which case we whould NOT do
    //       cache operations
    int16_t                 phase;          // Current phase bit
    uint32_t                last;           // Used internally to check db writes
    volatile uint32_t*      db;             // Pointer to doorbell register (NB! write only)
    volatile void*          vaddr;          // Virtual address to start of queue memory
    uint64_t                ioaddr;         // Physical/IO address of the memory page
} __attribute__((aligned (64))) emu_nvm_queue_t;



struct EmuQueuePair
{
    uint32_t            pageSize;
    uint32_t            block_size;
    uint32_t            block_size_log;
    uint32_t            block_size_minus_1;
    uint32_t            nvmNamespace;
    //void*               prpList;
    //uint64_t*           prpListIoAddrs;
#ifdef BAM_USE_BAM_CTRL_QUEUES
    nvm_queue_t         sq;
    nvm_queue_t         cq;
#else
	emu_nvm_queue_t     sq;
	emu_nvm_queue_t     cq;
#endif
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

#define EMU_MAX_SQ_ENTRIES_64K  (64*1024/64)
#define EMU_MAX_CQ_ENTRIES_64K  (64*1024/16)

	
	inline void emu_init_gpu_specific_struct( const uint32_t cudaDevice) {
#ifdef BAM_USE_BAM_CTRL_QUEUES
		  //this->sq_tickets = createBuffer(this->sq.qs * sizeof(padded_struct), cudaDevice);
		  //this->sq_head_mark = createBuffer(this->sq.qs * sizeof(padded_struct), cudaDevice);
		  //this->sq_tail_mark = createBuffer(this->sq.qs * sizeof(padded_struct), cudaDevice);
		  this->sq_cid = createBuffer(65536 * sizeof(padded_struct), cudaDevice);
		  //this->sq.tickets = (padded_struct*) this->sq_tickets.get();
		  //this->sq.head_mark = (padded_struct*) this->sq_head_mark.get();
		  //this->sq.tail_mark = (padded_struct*) this->sq_tail_mark.get();
		  this->sq.cid = (padded_struct*) this->sq_cid.get();
	  //	std::cout << "init_gpu_specific: " << std::hex << this->sq.cid <<  std::endl;
		  this->sq.qs_minus_1 = this->sq.qs - 1;
		  this->sq.qs_log2 = (uint32_t) std::log2(this->sq.qs);
	
	
		  //this->cq_tickets = createBuffer(this->cq.qs * sizeof(padded_struct), cudaDevice);
		  this->cq_head_mark = createBuffer(this->cq.qs * sizeof(padded_struct), cudaDevice);
		  //this->cq_tail_mark = createBuffer(this->cq.qs * sizeof(padded_struct), cudaDevice);
		  //this->cq.tickets = (padded_struct*) this->cq_tickets.get();
		  this->cq.head_mark = (padded_struct*) this->cq_head_mark.get();
		  //this->cq.tail_mark = (padded_struct*) this->cq_tail_mark.get();
		  this->cq.qs_minus_1 = this->cq.qs - 1;
		  this->cq.qs_log2 = (uint32_t) std::log2(this->cq.qs);
		  this->cq_pos_locks = createBuffer(this->cq.qs * sizeof(padded_struct), cudaDevice);
		  this->cq.pos_locks = (padded_struct*) this->cq_pos_locks.get();
	
		  //this->cq_clean_cid = createBuffer(this->cq.qs * sizeof(uint16_t), cudaDevice);
		 // this->cq.clean_cid = (uint16_t*) this->cq_clean_cid.get();
#endif
	}


	inline void emu_queue_pair_prepare(const nvm_ctrl_t* ctrl, const uint32_t cudaDevice, const struct nvm_ns_info ns, const struct nvm_ctrl_info info, const uint16_t qp_id, const uint64_t queueDepth)
		{
			
			int verbose = bam_get_verbosity(BAM_EMU_DBGLVL_INFO, BAM_DBG_CODE_PATH_H_EMU_CTRL);
	//		 printf("queue_pair_prepare 0 mm_ptr = %p\n", ctrl->mm_ptr);
			
			uint64_t cap = ((volatile uint64_t*) ctrl->mm_ptr)[0];
			  bool cqr = (cap & 0x0000000000010000) == 0x0000000000010000;
			  //uint64_t sq_size = 16;
			  //uint64_t cq_size = 16;
	
			 
			  
			  
			  uint64_t sq_size = (cqr) ?
				  ((EMU_MAX_SQ_ENTRIES_64K <= ((((volatile uint16_t*) ctrl->mm_ptr)[0] + 1) )) ? EMU_MAX_SQ_ENTRIES_64K :  ((((volatile uint16_t*) ctrl->mm_ptr)[0] + 1) ) ) :
				  ((((volatile uint16_t*) ctrl->mm_ptr)[0] + 1) );
			  uint64_t cq_size = (cqr) ?
				  ((EMU_MAX_CQ_ENTRIES_64K <= ((((volatile uint16_t*) ctrl->mm_ptr)[0] + 1) )) ? EMU_MAX_CQ_ENTRIES_64K :  ((((volatile uint16_t*) ctrl->mm_ptr)[0] + 1) ) ) :
				  ((((volatile uint16_t*) ctrl->mm_ptr)[0] + 1) );

				BAM_EMU_HOST_DBG_PRINT(verbose, "TARGET Q_SIZE (0 index) = %d\n",  ((volatile uint16_t*) ctrl->mm_ptr)[0] + 1);

				  
			  printf("EMU ** queue_pair_prepare sq_size = %ld cq_size = %ld, mm_ptr[0] = %04x\n", sq_size, cq_size, ((volatile uint16_t*) ctrl->mm_ptr)[0]);
			  
			  sq_size = std::min(queueDepth, sq_size);
			  cq_size = std::min(queueDepth, cq_size);
	
			  this->cq_size = cq_size;
			  this->sq_size = sq_size;
			  
			  sq_need_prp = false;//(!cqr) || (sq_size > MAX_SQ_ENTRIES_64K);
			  cq_need_prp = false;// (!cqr) || (cq_size > MAX_CQ_ENTRIES_64K);
			
			  sq_mem_size = sq_size * sizeof(nvm_cmd_t) + sq_need_prp*(64*1024) ;
			  cq_mem_size = cq_size * sizeof(nvm_cpl_t) + cq_need_prp*(64*1024) ;

	//		  sq_mem_size = sq_mem_size * 2;
	//		  cq_mem_size = cq_mem_size * 2;
			  
	
	//		  std::cout << sq_size << "\t" << sq_mem_size << std::endl;
			//size_t queueMemSize = ctrl.info.page_size * 2;
			//size_t prpListSize = ctrl.info.page_size * numThreads * (doubleBuffered + 1);
			//size_t prp_mem_size = sq_size * (4096) * 2;
	//		  std::cout << "Started creating DMA\n";
			// qmem->vaddr will be already a device pointer after the following call
			this->sq_mem = createDma(ctrl, NVM_PAGE_ALIGN(sq_mem_size, 1UL << 16), cudaDevice);
	 // 	  std::cout << "Finished creating sq dma vaddr: " << this->sq_mem.get()->vaddr << "\tioaddr: " << std::hex<< this->sq_mem.get()->ioaddrs[0] << std::dec << std::endl;
			this->cq_mem = createDma(ctrl, NVM_PAGE_ALIGN(cq_mem_size, 1UL << 16), cudaDevice);
			//this->prp_mem = createDma(ctrl, NVM_PAGE_ALIGN(prp_mem_size, 1UL << 16), cudaDevice, adapter, segmentId);
	 // 	  std::cout << "Finished creating cq dma vaddr: " << this->cq_mem.get()->vaddr << "\tioaddr: " << std::hex << this->cq_mem.get()->ioaddrs[0] << std::dec << std::endl;
	
			// Set members
			this->pageSize = info.page_size;
			this->block_size = ns.lba_data_size;
	
			this->block_size_minus_1 = ns.lba_data_size-1;
			this->block_size_log = std::log2(ns.lba_data_size);
	  //	  std::cout << "block size: " << this->block_size << "\tblock_size_log: " << this->block_size_log << std::endl ;
			this->nvmNamespace = ns.ns_id;
	
			//this->prpList = NVM_DMA_OFFSET(this->prp_mem, 0);
			//this->prpListIoAddrs = this->prp_mem->ioaddrs;
			this->qp_id = qp_id;
	
			
		  //  std::cout << "before nvm_admin_cq_create\n";
		}
	
		
		inline EmuQueuePair( const nvm_ctrl_t* ctrl, const uint32_t cudaDevice, const struct nvm_ns_info ns, const struct nvm_ctrl_info info, nvm_aq_ref& aq_ref, const uint16_t qp_id, const uint64_t queueDepth, bam_host_emulator *pEmu = NULL)
		{
			int need_device_ptr = 1;
			cudaError_t err;
			int verbose = bam_get_verbosity(BAM_EMU_DBGLVL_INFO, BAM_DBG_CODE_PATH_H_EMU_CTRL);

			emu_queue_pair_prepare(ctrl,cudaDevice, ns, info, qp_id, queueDepth);
	
			int status = nvm_admin_cq_create(aq_ref, &this->cq, qp_id, this->cq_mem.get(), 0, cq_size, cq_need_prp);
			if (!nvm_ok(status))
			{
				throw error(string("Failed to create completion queue: ") + nvm_strerror(status));
			}
			// std::cout << "after nvm_admin_cq_create\n";
	
			// Get a valid device pointer for CQ doorbell
			void* devicePtr = nullptr;
	
			
			BAM_EMU_HOST_DBG_PRINT(verbose,"QP(%d) CQ post mm_ptr = %p db = %p DIF  = %p pEmu = %p\n", qp_id, ctrl->mm_ptr, this->cq.db, ((uint64_t)this->cq.db - (uint64_t)ctrl->mm_ptr), pEmu);
	
			//this->cq.db = emu_host_get_db_pointer((qp_id - 1), 1, pEmu, &this->cq, &need_device_ptr);
	
			if(need_device_ptr)
			{
				err = cudaHostGetDevicePointer(&devicePtr, (void*) this->cq.db, 0);
				if (err != cudaSuccess)
				{
					throw error(string("Failed to get device pointer") + cudaGetErrorString(err));
				}
				this->cq.db = (volatile uint32_t*) devicePtr;
			}
	
	
			//printf("DEVICE cq.db = %p\n", this->cq.db);
			
				
	
			// Create submission queue
			//	nvm_admin_sq_create(nvm_aq_ref ref, nvm_queue_t* sq, const nvm_queue_t* cq, uint16_t id, const nvm_dma_t* dma, size_t offset, size_t qs, bool need_prp = false)
			//printf("CALL nvm_admin_sq_create() sq_size =%ld\n", sq_size);
			
			status = nvm_admin_sq_create(aq_ref, &this->sq, &this->cq, qp_id, this->sq_mem.get(), 0, sq_size, sq_need_prp);
			if (!nvm_ok(status))
			{
				throw error(string("Failed to create submission queue: ") + nvm_strerror(status));
			}
	
	
			// Get a valid device pointer for SQ doorbell
	
			//this->sq.db = emu_host_get_db_pointer((qp_id - 1), 0, pEmu, &this->sq, &need_device_ptr);
	
			if( need_device_ptr)
			{
				err = cudaHostGetDevicePointer(&devicePtr, (void*) this->sq.db, 0);
				if (err != cudaSuccess)
				{
					throw error(string("Failed to get device pointer") + cudaGetErrorString(err));
				}
				this->sq.db = (volatile uint32_t*) devicePtr;
			}
	
			//std::cout << "Finish Making Queue\n";
	
			emu_init_gpu_specific_struct(cudaDevice);
	
			//printf("init_gpu_specific_struct() RETURN \n");
					
	
			return;
	
	
	
		}
	
	


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
	
	int verbose = bam_get_verbosity(BAM_EMU_DBGLVL_INFO, BAM_DBG_CODE_PATH_H_EMU_CTRL);
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


	BAM_EMU_HOST_DBG_PRINT(verbose,"sizeof(EmuController) = %ld sizeof(EmuQueuePair) = %ld\n", sizeof(EmuController), sizeof(EmuQueuePair));

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
        h_qps[i] = new EmuQueuePair(ctrl, cudaDevice, ns, info, aq_ref, i+1, queueDepth, NULL);
    //    printf("finished creating qp %ld\n", i);
        cuda_err_chk(cudaMemcpy(d_qps+i, h_qps[i], sizeof(EmuQueuePair), cudaMemcpyHostToDevice));
   //     printf("finished copy QP Memory to device %ld\n", i);

		

    }
    printf("finished creating all qps %d\n", n_qps);
    
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

//************************************************************************************************************
//************************************************************************************************************
//**  Device Code 
//************************************************************************************************************
//************************************************************************************************************
#ifdef BAM_USE_BAM_CTRL_QUEUES
#define EMUQ_GET_HEAD(_q) _q->head.load(simt::memory_order_relaxed)
#define EMUQ_GET_TAIL(_q) _q->tail.load(simt::memory_order_relaxed)
#define EMUQ_GET_QS(_q) _q->qs
#define EMUQ_TAIL_INC(_q) _q->tail.fetch_add(1, simt::memory_order_relaxed)
#define EMUQ_HEAD_INC(_q) _q->head.fetch_add(1, simt::memory_order_relaxed)
#define EMUQ_TAIL_STORE(_q, _v) _q->tail.store(_v, simt::memory_order_release)
#define EMUQ_HEAD_STORE(_q, _v) _q->head.store(_v, simt::memory_order_release)
#define EMUQ_GET_LAST(_q) _q->last
#define EMUQ_RING_DB(_q, _v) asm volatile ("st.mmio.relaxed.sys.global.u32 [%0], %1;" :: "l"(_q->db),"r"((uint32_t)_v) : "memory")
#else
#define EMUQ_GET_HEAD(_q) _q->head
#define EMUQ_GET_TAIL(_q) _q->tail
#define EMUQ_GET_QS(_q) _q->qs
#define EMUQ_TAIL_INC(_q) _q->tail++
#define EMUQ_HEAD_INC(_q) _q->head++
#define EMUQ_TAIL_STORE(_q, _v) _q->tail = _v
#define EMUQ_HEAD_STORE(_q, _v) _q->head = _v
#define EMUQ_GET_LAST(_q) _q->last
#define EMUQ_RING_DB(_q, _v) *((volatile uint32_t*) _q->db) = _v
#endif



__device__ static inline void emu_ctrl_nvm_sq_update(nvm_queue_t* pSq)
{
	int verbose = bam_get_verbosity(BAM_EMU_DBGLVL_NONE, BAM_DBG_CODE_PATH_H_EMU_CTRL);

	EMUQ_HEAD_INC(pSq);

	BAM_EMU_DEV_DBG_PRINT2(verbose, "emu_ctrl_nvm_sq_update(qs = %d head = %d)\n", EMUQ_GET_HEAD(pSq), EMUQ_GET_QS(pSq));

	if(EMUQ_GET_HEAD(pSq) == EMUQ_GET_QS(pSq))
	{
		EMUQ_HEAD_STORE(pSq, 0);

		BAM_EMU_DEV_DBG_PRINT1(verbose, "emu_ctrl_nvm_sq_update(qs = %d head = 0) FLIP\n", EMUQ_GET_QS(pSq));

	}
	

}
		

__device__ static inline nvm_cmd_t* emu_ctrl_sq_enqueue(nvm_queue_t* pSq)

{
	nvm_cmd_t* pCmd;
	int verbose = bam_get_verbosity(BAM_EMU_DBGLVL_NONE, BAM_DBG_CODE_PATH_H_EMU_CTRL);
	uint16_t tail = EMUQ_GET_TAIL(pSq);
	
	BAM_EMU_DEV_DBG_PRINT3(verbose, "emu_ctrl_sq_enqueue(qs = %d head = %d tail = %d)\n", EMUQ_GET_QS(pSq), EMUQ_GET_HEAD(pSq), EMUQ_GET_TAIL(pSq));
	
	if ((uint16_t) ((tail - EMUQ_GET_HEAD(pSq)) % EMUQ_GET_QS(pSq)) == (EMUQ_GET_QS(pSq) - 1))
	{
		BAM_EMU_DEV_DBG_PRINT3(verbose, "emu_ctrl_sq_enqueue(qs = %d head = %d tail = %d) QFULL!!!!\n", EMUQ_GET_QS(pSq), EMUQ_GET_HEAD(pSq), EMUQ_GET_TAIL(pSq));

		return NULL;
	}

	pCmd = (((nvm_cmd_t*)(pSq->vaddr)) + tail);
	
	EMUQ_TAIL_INC(pSq);

	if(EMUQ_GET_TAIL(pSq) == EMUQ_GET_QS(pSq))
	{
		pSq->phase = !pSq->phase;
		
		BAM_EMU_DEV_DBG_PRINT3(verbose, "emu_ctrl_sq_enqueue(qs = %d head = %d tail = %d) FILP!!!!\n", EMUQ_GET_QS(pSq), EMUQ_GET_HEAD(pSq), EMUQ_GET_TAIL(pSq));

		EMUQ_TAIL_STORE(pSq, 0);
	}


	return pCmd;
	
}

__device__ static inline void emu_ctrl_nvm_sq_submit(nvm_queue_t* pSq)
{

	int verbose = bam_get_verbosity(BAM_EMU_DBGLVL_NONE, BAM_DBG_CODE_PATH_H_EMU_CTRL);
	uint16_t tail = EMUQ_GET_TAIL(pSq);

	
	BAM_EMU_DEV_DBG_PRINT3(verbose, "emu_ctrl_nvm_sq_submit(qs = %d last = %d tail = %d) \n", EMUQ_GET_QS(pSq), EMUQ_GET_LAST(pSq), tail);
	
    if (EMUQ_GET_LAST(pSq) != tail)
    {
        //nvm_cache_flush((void*) sq->vaddr, sizeof(nvm_cmd_t) * sq->max_entries);
        //nvm_wcb_flush(); 
		BAM_EMU_DEV_DBG_PRINT1(verbose, "emu_ctrl_nvm_sq_submit() RING SQ DOORBELL = %d\n", tail);
		EMUQ_RING_DB(pSq, tail);

        pSq->last = tail;
    }
}

__device__ static inline nvm_cpl_t* emu_ctrl_nvm_walk_and_find_cq_cid(nvm_queue_t* pCq, uint16_t cid)
{
	int verbose = bam_get_verbosity(BAM_EMU_DBGLVL_INFO, BAM_DBG_CODE_PATH_H_EMU_CTRL);
	nvm_cpl_t* cpl;

	for(uint32_t i = 0; i < EMUQ_GET_QS(pCq); i++)
	{
		cpl = (nvm_cpl_t*) (((unsigned char*) pCq->vaddr) + sizeof(nvm_cpl_t) * i);
		uint32_t cpl_entry = cpl->dword[3];

		BAM_EMU_DEV_DBG_PRINT4(verbose, "emu_ctrl_nvm_walk_and_find_cq_cid[%d] cpl = %p cpl_entry = 0x%08x search_cid = %04x\n", i,cpl, cpl_entry, cid);

		if((cpl_entry & 0xFFFF) == cid)
		{
			BAM_EMU_DEV_DBG_PRINT3(verbose, "Completion cid = %x found at slot = %d dword[2] = %08x\n", cid, i, cpl->dword[2]);
			break;
		}
			

	}

	return cpl;
}

__device__ static inline nvm_cpl_t* emu_ctrl_nvm_cq_poll(nvm_queue_t* pCq)
{
	int verbose = bam_get_verbosity(BAM_EMU_DBGLVL_NONE, BAM_DBG_CODE_PATH_H_EMU_CTRL);
	
	uint16_t head = EMUQ_GET_HEAD(pCq);
		
    nvm_cpl_t* cpl = (nvm_cpl_t*) (((unsigned char*) pCq->vaddr) + (sizeof(nvm_cpl_t) * head));

//	nvm_cpl_t* cpl = &((nvm_cpl_t*)pCq->vaddr)[head];

	
	uint32_t cpl_entry = cpl->dword[3];
	uint8_t phase = (uint8_t)((cpl_entry & 0x00010000) >> 16);
	
	BAM_EMU_DEV_DBG_PRINT4(verbose, "emu_ctrl_nvm_cq_cull(%p) head = %d phase = %d cpl_entry = %08x\n", pCq,EMUQ_GET_HEAD(pCq), pCq->phase, cpl_entry);
	

    // Check if new completion is ready by checking the phase tag
    if (phase != pCq->phase)
    {
		BAM_EMU_DEV_DBG_PRINT2(verbose, "emu_ctrl_nvm_cq_cull(NULL) !!_RB(*NVM_CPL_STATUS(cpl) = %d != phase = %d\n", !!_RB(*NVM_CPL_STATUS(cpl), 0, 0), pCq->phase);

		return NULL;
    }

	BAM_EMU_DEV_DBG_PRINT2(verbose, "emu_ctrl_nvm_cq_cull(GOOD) cpl = %p cid = 0x%04x\n", cpl, (cpl_entry & 0x0000ffff));
#if 0
	EMUQ_HEAD_INC(pCq);
	head = EMUQ_GET_HEAD(pCq);
	
	if(head == EMUQ_GET_QS(pCq))
	{
		head = 0;
		EMUQ_HEAD_STORE(pCq, head);

		pCq->phase = (pCq->phase ? 0 : 1);
	}

	if(EMUQ_GET_LAST(pCq) != head)
	{
		
		BAM_EMU_DEV_DBG_PRINT1(verbose, "emu_ctrl_nvm_cq_cull() RING CQ DOORBELL = %d\n", head);
		EMUQ_RING_DB(pCq, head);
		pCq->last = head;

	}
#endif

    return cpl;
}

__device__ static inline void emu_ctrl_nvm_cq_dequeue(nvm_queue_t* pCq)
{
	int verbose = bam_get_verbosity(BAM_EMU_DBGLVL_NONE, BAM_DBG_CODE_PATH_H_EMU_CTRL);
	uint16_t head;

	EMUQ_HEAD_INC(pCq);
	head = EMUQ_GET_HEAD(pCq);
	
	if(head == EMUQ_GET_QS(pCq))
	{
		head = 0;
		EMUQ_HEAD_STORE(pCq, head);

		pCq->phase = (pCq->phase ? 0 : 1);
	}

	if(EMUQ_GET_LAST(pCq) != head)
	{
		
		BAM_EMU_DEV_DBG_PRINT1(verbose, "emu_ctrl_nvm_cq_cull() RING CQ DOORBELL = %d\n", head);
		EMUQ_RING_DB(pCq, head);
		pCq->last = head;

	}

}



#endif
