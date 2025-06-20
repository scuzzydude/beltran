#ifndef __BENCHMARK_QUEUEPAIR_H__
#define __BENCHMARK_QUEUEPAIR_H__
// #ifndef __CUDACC__
// #define __device__
// #define __host__
// #endif

#include <algorithm>
#include <cstdint>
#include "buffer.h"
#include "ctrl.h"
#include "cuda.h"
#include "nvm_types.h"
#include "nvm_util.h"
#include "nvm_error.h"
#include "nvm_admin.h"
#include <stdexcept>
#include <string>
#include <iostream>
#include <cmath>
#include "util.h"
#include "emu.h"

using error = std::runtime_error;
using std::string;


struct QueuePair
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



#define MAX_SQ_ENTRIES_64K  (64*1024/64)
#define MAX_CQ_ENTRIES_64K  (64*1024/16)
#define MAX_CID 65536
//#define MAX_CID 4096



    inline void init_gpu_specific_struct( const uint32_t cudaDevice) {
        this->sq_tickets = createBuffer(this->sq.qs * sizeof(padded_struct), cudaDevice);
        //this->sq_head_mark = createBuffer(this->sq.qs * sizeof(padded_struct), cudaDevice);
        this->sq_tail_mark = createBuffer(this->sq.qs * sizeof(padded_struct), cudaDevice);
        this->sq_cid = createBuffer(MAX_CID * sizeof(padded_struct), cudaDevice);
        this->sq.tickets = (padded_struct*) this->sq_tickets.get();
        //this->sq.head_mark = (padded_struct*) this->sq_head_mark.get();
        this->sq.tail_mark = (padded_struct*) this->sq_tail_mark.get();
        this->sq.cid = (padded_struct*) this->sq_cid.get();
    //    std::cout << "init_gpu_specific: " << std::hex << this->sq.cid <<  std::endl;
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
    }



	inline void queue_pair_prepare(const nvm_ctrl_t* ctrl, const uint32_t cudaDevice, const struct nvm_ns_info ns, const struct nvm_ctrl_info info, const uint16_t qp_id, const uint64_t queueDepth)
    {
//		 printf("queue_pair_prepare 0 mm_ptr = %p\n", ctrl->mm_ptr);
		
		uint64_t cap = ((volatile uint64_t*) ctrl->mm_ptr)[0];
		  bool cqr = (cap & 0x0000000000010000) == 0x0000000000010000;
		  //uint64_t sq_size = 16;
		  //uint64_t cq_size = 16;

		 
		  
		  
		  uint64_t sq_size = (cqr) ?
			  ((MAX_SQ_ENTRIES_64K <= ((((volatile uint16_t*) ctrl->mm_ptr)[0] + 1) )) ? MAX_SQ_ENTRIES_64K :  ((((volatile uint16_t*) ctrl->mm_ptr)[0] + 1) ) ) :
			  ((((volatile uint16_t*) ctrl->mm_ptr)[0] + 1) );
		  uint64_t cq_size = (cqr) ?
			  ((MAX_CQ_ENTRIES_64K <= ((((volatile uint16_t*) ctrl->mm_ptr)[0] + 1) )) ? MAX_CQ_ENTRIES_64K :  ((((volatile uint16_t*) ctrl->mm_ptr)[0] + 1) ) ) :
			  ((((volatile uint16_t*) ctrl->mm_ptr)[0] + 1) );

//		  printf("queue_pair_prepare sq_size = %ld cq_size = %ld, mm_ptr[0] = %04x\n", sq_size, cq_size, ((volatile uint16_t*) ctrl->mm_ptr)[0]);
		  
		  sq_size = std::min(queueDepth, sq_size);
		  cq_size = std::min(queueDepth, cq_size);

		  this->cq_size = cq_size;
		  this->sq_size = sq_size;
		  
		  sq_need_prp = false;//(!cqr) || (sq_size > MAX_SQ_ENTRIES_64K);
		  cq_need_prp = false;// (!cqr) || (cq_size > MAX_CQ_ENTRIES_64K);
		
		  sq_mem_size =	sq_size * sizeof(nvm_cmd_t) + sq_need_prp*(64*1024);
		  cq_mem_size =	cq_size * sizeof(nvm_cpl_t) + cq_need_prp*(64*1024);

//        std::cout << sq_size << "\t" << sq_mem_size << std::endl;
        //size_t queueMemSize = ctrl.info.page_size * 2;
        //size_t prpListSize = ctrl.info.page_size * numThreads * (doubleBuffered + 1);
        //size_t prp_mem_size = sq_size * (4096) * 2;
//        std::cout << "Started creating DMA\n";
        // qmem->vaddr will be already a device pointer after the following call
        this->sq_mem = createDma(ctrl, NVM_PAGE_ALIGN(sq_mem_size, 1UL << 16), cudaDevice);
 //       std::cout << "Finished creating sq dma vaddr: " << this->sq_mem.get()->vaddr << "\tioaddr: " << std::hex<< this->sq_mem.get()->ioaddrs[0] << std::dec << std::endl;
        this->cq_mem = createDma(ctrl, NVM_PAGE_ALIGN(cq_mem_size, 1UL << 16), cudaDevice);
        //this->prp_mem = createDma(ctrl, NVM_PAGE_ALIGN(prp_mem_size, 1UL << 16), cudaDevice, adapter, segmentId);
 //       std::cout << "Finished creating cq dma vaddr: " << this->cq_mem.get()->vaddr << "\tioaddr: " << std::hex << this->cq_mem.get()->ioaddrs[0] << std::dec << std::endl;

        // Set members
        this->pageSize = info.page_size;
        this->block_size = ns.lba_data_size;

        this->block_size_minus_1 = ns.lba_data_size-1;
        this->block_size_log = std::log2(ns.lba_data_size);
  //      std::cout << "block size: " << this->block_size << "\tblock_size_log: " << this->block_size_log << std::endl ;
        this->nvmNamespace = ns.ns_id;

        //this->prpList = NVM_DMA_OFFSET(this->prp_mem, 0);
        //this->prpListIoAddrs = this->prp_mem->ioaddrs;
        this->qp_id = qp_id;

        if (cq_need_prp) {
            size_t iters = (size_t)ceil(((float)cq_size*sizeof(nvm_cpl_t))/((float)ctrl->page_size));
            uint64_t* cpu_vaddrs = (uint64_t*) malloc(64*1024);
            memset((void*)cpu_vaddrs, 0, 64*1024);
            for (size_t i = 0; i < iters; i++) {
                size_t page_64  = i/(64*1024);
                size_t page_4 = i%(64*1024/ctrl->page_size);
                cpu_vaddrs[i] = this->cq_mem.get()->ioaddrs[1 + page_64] + (page_4 * ctrl->page_size);
            }

            if (this->cq_mem.get()->vaddr) {
                cuda_err_chk(cudaMemcpy(this->cq_mem.get()->vaddr, cpu_vaddrs, 64*1024, cudaMemcpyHostToDevice));
            }

            this->cq_mem.get()->vaddr = (void*)((uint64_t)this->cq_mem.get()->vaddr + 64*1024);

            free(cpu_vaddrs);
        }

        if (sq_need_prp) {
            size_t iters = (size_t)ceil(((float)sq_size*sizeof(nvm_cpl_t))/((float)ctrl->page_size));
            uint64_t* cpu_vaddrs = (uint64_t*) malloc(64*1024);
            memset((void*)cpu_vaddrs, 0, 64*1024);
            for (size_t i = 0; i < iters; i++) {
                size_t page_64  = i/(64*1024);
                size_t page_4 = i%(64*1024/ctrl->page_size);
                cpu_vaddrs[i] = this->sq_mem.get()->ioaddrs[1 + page_64] + (page_4 * ctrl->page_size);
            }

            if (this->sq_mem.get()->vaddr) {
                cuda_err_chk(cudaMemcpy(this->sq_mem.get()->vaddr, cpu_vaddrs, 64*1024, cudaMemcpyHostToDevice));
            }

            this->sq_mem.get()->vaddr = (void*)((uint64_t)this->sq_mem.get()->vaddr + 64*1024);

            free(cpu_vaddrs);
        }
      //  std::cout << "before nvm_admin_cq_create\n";
	}

	
    inline QueuePair( const nvm_ctrl_t* ctrl, const uint32_t cudaDevice, const struct nvm_ns_info ns, const struct nvm_ctrl_info info, nvm_aq_ref& aq_ref, const uint16_t qp_id, const uint64_t queueDepth, bam_host_emulator *pEmu = NULL)
    {
		int need_device_ptr;
		cudaError_t err;
		
		queue_pair_prepare(ctrl,cudaDevice, ns, info, qp_id, queueDepth);

        int status = nvm_admin_cq_create(aq_ref, &this->cq, qp_id, this->cq_mem.get(), 0, cq_size, cq_need_prp);
        if (!nvm_ok(status))
        {
            throw error(string("Failed to create completion queue: ") + nvm_strerror(status));
        }
        // std::cout << "after nvm_admin_cq_create\n";

        // Get a valid device pointer for CQ doorbell
        void* devicePtr = nullptr;

		
		//printf("QP(%d) CQ post mm_ptr = %p db = %p DIF  = %p pEmu = %p\n", qp_id, ctrl->mm_ptr, this->cq.db, ((uint64_t)this->cq.db - (uint64_t)ctrl->mm_ptr), pEmu);

		this->cq.db = emu_host_get_db_pointer((qp_id - 1), 1, pEmu, &this->cq, &need_device_ptr);

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
        //  nvm_admin_sq_create(nvm_aq_ref ref, nvm_queue_t* sq, const nvm_queue_t* cq, uint16_t id, const nvm_dma_t* dma, size_t offset, size_t qs, bool need_prp = false)
        //printf("CALL nvm_admin_sq_create() sq_size =%ld\n", sq_size);
		
        status = nvm_admin_sq_create(aq_ref, &this->sq, &this->cq, qp_id, this->sq_mem.get(), 0, sq_size, sq_need_prp);
        if (!nvm_ok(status))
        {
            throw error(string("Failed to create submission queue: ") + nvm_strerror(status));
        }


        // Get a valid device pointer for SQ doorbell

		this->sq.db = emu_host_get_db_pointer((qp_id - 1), 0, pEmu, &this->sq, &need_device_ptr);

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

        init_gpu_specific_struct(cudaDevice);

		//printf("init_gpu_specific_struct() RETURN \n");
				

        return;



    }

};
#endif
