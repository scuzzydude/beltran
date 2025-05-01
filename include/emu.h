#ifndef __BAM_EMU_H 
#define __BAM_EMU_H 

	
#include <cstdint>
#include "buffer.h"
#include "nvm_types.h"
#include "nvm_ctrl.h"
#include "nvm_aq.h"
#include "nvm_admin.h"
#include "nvm_util.h"
#include "nvm_error.h"
#include <string>
#include <memory>
#include <vector>
#include <stdexcept>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cstdio>
#include <stdarg.h>
#include <algorithm>
#include <unistd.h>
#include <fcntl.h>
#include <cstdlib>
#include <algorithm>
#include <simt/atomic>
#include <ctrl.h>

#include "queue.h"

#include <cuda/std/chrono>

/* Emulation Headers */

#include "emu_common.h"
#include "emu_latency.h"
#include "emu_aggregation.h"
#include "emu_vendor.h"
#include "emu_mapper.h"



volatile uint32_t * emu_host_get_db_pointer(int qidx, int cq, bam_host_emulator *pEmu, nvm_queue_t *pQueue, int *pNeedDevicePtr)
{
		int verbose = bam_get_verbosity(BAM_EMU_DBGLVL_NONE, BAM_DBG_CODE_PATH_H_CREATE_Q );

		BAM_EMU_HOST_DBG_PRINT(verbose, "emu_host_get_db_pointer(%d, %d, %p, %p, %p)\n", qidx, cq, pEmu, pQueue, pNeedDevicePtr);

		*pNeedDevicePtr = 1;

#if(BAM_EMU_DOORBELL_TYPE == EMU_DB_MEM_ATOMIC_MANAGED) 
		*pNeedDevicePtr = 0;
		return ((0 != cq) ? (uint32_t *)&pEmu->tgt.pTgt_control->atomic_doorbells[qidx].cq_db : (uint32_t *)&pEmu->tgt.pTgt_control->atomic_doorbells[qidx].sq_db);
#elif(BAM_EMU_DOORBELL_TYPE == EMU_DB_MEM_ATOMIC_DEVICE) 
		*pNeedDevicePtr = 0;
		return pQueue->db;
#else
		//pEmu will be NULL if normal BaM compile or if File mapped, no redirection neccessary
		return pQueue->db;
#endif	



}

__device__ inline uint32_t emu_tgt_read_doorbell(bam_emulated_queue *pEmuQ)
{
#if(BAM_EMU_DOORBELL_TYPE == EMU_DB_MEM_MAPPED_FILE) 
	//In this case, the pointer is mapped host (memory mapped file), and thus sequuenced and forced to be coherent with the applications BaM doorbell writes
	//That's the theory, anyway.....
	return *pEmuQ->db;

#else
	auto atomic_casted = reinterpret_cast<simt::atomic<uint32_t, simt::thread_scope_device>*>(const_cast<uint32_t*>(pEmuQ->db));
	return atomic_casted->load(simt::memory_order_relaxed);
#endif
}


//#define EMU_TIME_SIM_DMA 

__device__ void emu_tgt_DMA(void *dst_addr, void *src_addr, int copy_size, int direction)
{
	int verbose = bam_get_verbosity(BAM_EMU_DBGLVL_NONE, BAM_DBG_CODE_PATH_D_SQ_PROCESS);
	int i;
#ifdef EMU_TIME_SIM_DMA
	uint64_t start_ticks;
	uint64_t end_ticks;
#endif
	BAM_EMU_DEV_DBG_PRINT2(verbose, "TGT: emu_tgt_DMA(%d) %d\n", copy_size, direction);

	emu_copy_type *pSrc = (emu_copy_type *)src_addr;
	emu_copy_type *pDst = (emu_copy_type *)dst_addr;
	int limit = copy_size / sizeof(emu_copy_type);
	int remainder = copy_size % sizeof(emu_copy_type);
	if(remainder & 0x3)
	{
		BAM_EMU_DEV_DBG_PRINT2(BAM_EMU_DBGLVL_ERROR, "TGT: emu_tgt_DMA() limit = %d remainder =	%d NOT DWORD ALIGNED!!!!\n", limit, remainder);
	}
	
	BAM_EMU_DEV_DBG_PRINT2(verbose, "TGT: emu_tgt_DMA() limit = %d sizeof(copy_type) =  %ld\n", limit, sizeof(emu_copy_type));
	BAM_EMU_DEV_DBG_PRINT3(verbose, "TGT: emu_tgt_DMA() pSrc = %p pDst =  %p remainder = %d\n", pSrc, pDst, remainder);
	
#ifdef EMU_TIME_SIM_DMA
	start_ticks = NS_Clock();
#endif
	for(i = 0; i < limit; i++)
	{
		pDst[i] = pSrc[i];
	}

	BAM_EMU_DEV_DBG_PRINT3(verbose, "TGT: emu_tgt_DMA() LIMIT copy done pSrc = %p pDst =  %p remainder = %d\n", pSrc, pDst, remainder);
	
	if(remainder)
	{
		BAM_EMU_DEV_DBG_PRINT2(verbose, "TGT: copy i = %d remainder = %d\n", i, remainder);
		uint32_t *pDwSrc = (uint32_t *)&pSrc[i];
		uint32_t *pDwDst = (uint32_t *)&pDst[i];
	
		for(i = 0; i < (remainder / 4); i++ )
		{
			pDwDst[i] = pDwSrc[i];
		}
	}


#ifdef EMU_TIME_SIM_DMA
	end_ticks = NS_Clock();
#endif
	

#ifdef EMU_TIME_SIM_DMA
	BAM_EMU_DEV_DBG_PRINT2(verbose, "TGT: emu_tgt_DMA() copied %d bytes in  =  %ld ns\n", copy_size, (end_ticks - start_ticks));
	BAM_EMU_DEV_DBG_PRINT2(verbose, "TGT: emu_tgt_DMA() %d bytes copied @  =  %f GB/sec \n", copy_size, get_GBs_per_sec((end_ticks - start_ticks), copy_size));
#endif
	//BAM_EMU_DEV_DBG_PRINT2(verbose, "TGT emu_tgt_DMA(): hexdump(%p, %d)\n", pDst, 32);
	//hexdump(pDst, 32);


}


__device__ inline int emu_tgt_SQ_Process(bam_emulated_target_control    *pMgtTgtControl, bam_emulated_queue_pair     *pQP, uint32_t db_tail)
{
	int verbose = bam_get_verbosity(BAM_EMU_DBGLVL_NONE, BAM_DBG_CODE_PATH_D_SQ_PROCESS);
	int slot_count = 0;
	void *src_addr;
	void *dst_addr;
	int copy_size;
	
	BAM_EMU_DEV_DBG_PRINT3(verbose, "TGT: emu_tgt_SQ_Process() db_tail = %d sq_head = %d sq_tail = %d\n",  db_tail, pQP->sQ.head, pQP->sQ.tail);

	if(db_tail > pQP->sQ.tail)
	{
		//single contiguous DMA
		slot_count = (db_tail - pQP->sQ.tail);
		//TODO: use inline/macro to get the address, will be DMA in disagregated model
		src_addr = (void *)&(((nvm_cmd_t *)(pQP->sQ.ioaddr))[pQP->sQ.tail]);
		dst_addr = (void *)&(((nvm_cmd_t *)(pQP->sQ.pEmuQ))[pQP->sQ.tail]);
		copy_size  = slot_count * sizeof(nvm_cmd_t);	
		
		BAM_EMU_DEV_DBG_PRINT2(verbose, "TGT: emu_tgt_SQ_Process(%d) ioaddr = %p\n", pQP->q_number, (void *)pQP->sQ.ioaddr);
		BAM_EMU_DEV_DBG_PRINT2(verbose, "TGT: emu_tgt_SQ_Process() slot_count = %d src_addr = %p\n", slot_count, src_addr);
		BAM_EMU_DEV_DBG_PRINT2(verbose, "TGT: emu_tgt_SQ_Process() copy_size = %d dst_addr = %p\n", copy_size, dst_addr);

		emu_tgt_DMA(dst_addr, src_addr, copy_size, BAM_EMU_DATA_IN);

	
	}
	else if(db_tail < pQP->sQ.tail)
	{
		int slot_count2 = db_tail;
		slot_count = (pQP->sQ.q_size_minus_1 - db_tail);
		
		BAM_EMU_DEV_DBG_PRINT3(verbose, "TGT: emu_tgt_SQ_Process() rollover detected, Split DMA: slot_count = %d slot_count = %d total_entries = %d\n", slot_count, slot_count2, slot_count + slot_count2);

		src_addr = (void *)&(((nvm_cmd_t *)(pQP->sQ.ioaddr))[pQP->sQ.tail]);
		dst_addr = (void *)&(((nvm_cmd_t *)(pQP->sQ.pEmuQ))[pQP->sQ.tail]);
		copy_size  = slot_count * sizeof(nvm_cmd_t);	
		
		BAM_EMU_DEV_DBG_PRINT3(verbose, "TGT: emu_tgt_SQ_Process() Split-DMA1 copy_size = %d src_addr = %p dst_addr = %p\n", copy_size, src_addr, dst_addr );
		emu_tgt_DMA(dst_addr, src_addr, copy_size, BAM_EMU_DATA_IN);

		copy_size  = slot_count2 * sizeof(nvm_cmd_t);	
		src_addr = (void *)&(((nvm_cmd_t *)(pQP->sQ.ioaddr))[0]);
		dst_addr = (void *)&(((nvm_cmd_t *)(pQP->sQ.pEmuQ))[0]);

		BAM_EMU_DEV_DBG_PRINT3(verbose, "TGT: emu_tgt_SQ_Process() Split-DMA2 copy_size = %d src_addr = %p dst_addr = %p\n", copy_size, src_addr, dst_addr );
		emu_tgt_DMA(dst_addr, src_addr, copy_size, BAM_EMU_DATA_IN);


	}

	if(0)
	{
		verbose = BAM_EMU_DBGLVL_ERROR;
		uint32_t *pDW = (uint32_t *)pQP->sQ.ioaddr;
		BAM_EMU_DEV_DBG_PRINT4(verbose, "TGT: emu_tgt_SQ_Process() SRC  = 0x%08x 0x%08x 0x%08x 0x%08x\n", pDW[0], pDW[16], pDW[32], pDW[48] );
		pDW = (uint32_t *)pQP->sQ.pEmuQ;
		BAM_EMU_DEV_DBG_PRINT4(verbose, "TGT: emu_tgt_SQ_Process() DST  = 0x%08x 0x%08x 0x%08x 0x%08x\n", pDW[0], pDW[16], pDW[32], pDW[48] );
	}

	pQP->sQ.head = db_tail;

	return slot_count;

}
__device__ inline void emu_tgt_CQ_Drain(bam_emulated_target_control *pMgtTgtControl, bam_emulated_queue_pair *pQP, uint32_t cq_db_head)
{
	int verbose = bam_get_verbosity(BAM_EMU_DBGLVL_NONE, BAM_DBG_CODE_PATH_D_CQ_DRAIN);
	int slot_count;
	void *dst_addr;
	void *src_addr;
	int copy_size;

	BAM_EMU_DEV_DBG_PRINT2(verbose, "TGT: emu_tgt_CQ_Drain() CALL cq_db_head = %d cq_tail = %d\n", cq_db_head, pQP->cQ.tail);

	if(cq_db_head == pQP->cQ.tail)
	{
		BAM_EMU_DEV_DBG_PRINT2(verbose, "TGT: emu_tgt_CQ_Drain() EMPTY QUEUE cq_db_head = %d cq_tail = %d\n", cq_db_head, pQP->cQ.tail);
	}
	else if(cq_db_head < pQP->cQ.tail)
	{
		BAM_EMU_DEV_DBG_PRINT3(verbose, "TGT: emu_tgt_CQ_Drain(%d) cq_db_head = %d cq_tail = %d\n", pQP->q_number, cq_db_head, pQP->cQ.tail);
		slot_count =  (pQP->cQ.tail - cq_db_head);
		dst_addr = (void *)&(((nvm_cpl_t *)(pQP->cQ.ioaddr))[cq_db_head]);
		src_addr = (void *)&(((nvm_cpl_t *)(pQP->cQ.pEmuQ))[cq_db_head]);
		copy_size  = slot_count * sizeof(nvm_cpl_t);	


		BAM_EMU_DEV_DBG_PRINT2(verbose, "TGT: emu_tgt_CQ_Drain() slot_count = %d src_addr = %p\n", slot_count, src_addr);
		BAM_EMU_DEV_DBG_PRINT2(verbose, "TGT: emu_tgt_CQ_Drain() copy_size = %d dst_addr = %p\n", copy_size, dst_addr);

		emu_tgt_DMA(dst_addr, src_addr, copy_size, BAM_EMU_DATA_OUT);

	}
	else 
	{
		BAM_EMU_DEV_DBG_PRINT2(verbose, "TGT: emu_tgt_CQ_Drain() Split-DMA cq_db_head = %d cq_tail = %d\n", cq_db_head, pQP->cQ.tail);

		slot_count =  (pQP->cQ.q_size - cq_db_head);
		dst_addr = (void *)&(((nvm_cpl_t *)(pQP->cQ.ioaddr))[cq_db_head]);
		src_addr = (void *)&(((nvm_cpl_t *)(pQP->cQ.pEmuQ))[cq_db_head]);
		copy_size  = slot_count * sizeof(nvm_cpl_t);	

		BAM_EMU_DEV_DBG_PRINT2(verbose, "TGT: emu_tgt_CQ_Drain() DMA1 slot_count = %d src_addr = %p\n", slot_count, src_addr);
		emu_tgt_DMA(dst_addr, src_addr, copy_size, BAM_EMU_DATA_OUT);

		slot_count =  (pQP->cQ.tail);
		dst_addr = (void *)&(((nvm_cpl_t *)(pQP->cQ.ioaddr))[0]);
		src_addr = (void *)&(((nvm_cpl_t *)(pQP->cQ.pEmuQ))[0]);
		copy_size  = slot_count * sizeof(nvm_cpl_t);	

		BAM_EMU_DEV_DBG_PRINT2(verbose, "TGT: emu_tgt_CQ_Drain() DMA2 slot_count = %d src_addr = %p\n", slot_count, src_addr);
		emu_tgt_DMA(dst_addr, src_addr, copy_size, BAM_EMU_DATA_OUT);


	}

	
	//debug 
	if(0)
	{
		
		uint32_t *pDw = (uint32_t *)pQP->cQ.ioaddr;			

		BAM_EMU_DEV_DBG_PRINT1(verbose, "TGT: emu_tgt_CQ_Drain() ptr = %p\n", pDw);

		BAM_EMU_DEV_DBG_PRINT4(verbose, "TGT: emu_tgt_CQ_Drain(dst) dw3(0..4) 0x%08x 0x%08x 0x%08x 0x%08x\n", pDw[3], pDw[7], pDw[11], pDw[15]);

		pDw = (uint32_t *)pQP->cQ.pEmuQ; 		

		BAM_EMU_DEV_DBG_PRINT4(verbose, "TGT: emu_tgt_CQ_Drain(src) dw3(0..4) 0x%08x 0x%08x 0x%08x 0x%08x\n", pDw[3], pDw[7], pDw[11], pDw[15]);

	}
	
}






__device__ inline int emu_tgt_NVMe_loopback(bam_emulated_target_control * pMgtTgtControl,
	 bam_emulated_queue_pair * pQP, uint16_t cid, uint32_t cq_db_head)
{
	int 			verbose = bam_get_verbosity(BAM_EMU_DBGLVL_NONE, BAM_DBG_CODE_PATH_D_NVME_LOOP);
	uint32_t		phase = 0x10000;
	uint32_t		retries = 0;
	const uint32_t	retry_limit = 10;
	const uint32_t	retry_ns = 64;

	//	uint32_t phase = 0;
	BAM_EMU_DEV_DBG_PRINT4(verbose, "emu_tgt_NVMe_loopback() db_head = %d cq_tail = %d next_tail = %d cid = 0x%x\n",
		 cq_db_head, pQP->cQ.tail, ((pQP->cQ.tail + 1) &pQP->cQ.q_size_minus_1), cid);

	while (retries < retry_limit)
	{
		if (cq_db_head != (pQP->cQ.tail + 1))
		{
			nvm_cpl_t * 	pCmp = & (((nvm_cpl_t *) (pQP->cQ.pEmuQ))[pQP->cQ.tail]);

			verbose 			= 0;

			if (pQP->cQ.rollover & 0x1)
			{
				phase				= 0;
			}

			BAM_EMU_DEV_DBG_PRINT3(verbose, "emu_tgt_NVMe_loopback() cid = 0x%04x cq_tail = %d rollover = %d\n", cid,
				 pQP->cQ.tail, pQP->cQ.rollover);


			pCmp->dword[0]		= 0;
			pCmp->dword[1]		= 0;
			pCmp->dword[2]		= ((uint32_t) pQP->q_number << 16) | (pQP->sQ.head);
			pCmp->dword[3]		= phase | cid;


			BAM_EMU_DEV_DBG_PRINT3(verbose, "emu_tgt_NVMe_loopback() %p val[2] = %x val[3] = %x\n", &pCmp->dword[2],
				 pCmp->dword[2], pCmp->dword[3]);

			pQP->cQ.tail++;

			pQP->cQ.tail		&= pQP->cQ.q_size_minus_1;

			if (0 == pQP->cQ.tail)
			{
				pQP->cQ.rollover++;
			}

			return 0;
		}
		else 
		{
			BAM_EMU_DEV_DBG_PRINT3(BAM_EMU_DBGLVL_ERROR, "emu_tgt_NVMe_loopback() !!!QFULL db_head = %d cq_tail = %d retries = %d\n",
				 cq_db_head, pQP->cQ.tail, retries);
			retries++;
			
		    __nanosleep(retries * retry_ns);
			
		}
	}

	return 1;
}





#ifdef BAM_EMU_TGT_SIMPLE_MODE_NVME_LOOPBACK
__device__ inline int emu_tgt_NVMe_execute(bam_emulated_target_control    *pMgtTgtControl, bam_emulated_queue_pair     *pQP, storage_next_emuluator_context *pContext, uint32_t cq_db_head)
{

	int verbose = bam_get_verbosity(BAM_EMU_DBGLVL_NONE, BAM_DBG_CODE_PATH_D_NVME_EXE);
	uint16_t cid;
	uint8_t opcode;
	uint64_t   lba;
	int err;
	nvm_cmd_t *pCmd = (nvm_cmd_t *)pContext->pCmd;

	
	cid = pCmd->dword[0] >> 16;
	opcode = pCmd->dword[0] & 0x7f;
	lba = ((uint64_t)pCmd->dword[11] << 32) | pCmd->dword[10];
	BAM_EMU_DEV_DBG_PRINT3(verbose, "TGT: emu_tgt_NVMe_execute() cid = 0x%04x opcode = 0x%02x lba = %lx\n", cid, opcode, lba);

	err = emu_tgt_NVMe_loopback(pMgtTgtControl, pQP, cid, cq_db_head);
	return err;
	
}
#else /* Normal Emulator Implementation */
__device__ inline int emu_tgt_NVMe_execute(bam_emulated_target_control    *pMgtTgtControl, bam_emulated_queue_pair     *pQP, storage_next_emuluator_context *pContext, uint32_t cq_db_head)
{
	uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	int verbose = bam_get_verbosity(BAM_EMU_DBGLVL_NONE, BAM_DBG_CODE_PATH_D_NVME_EXE);

	
	BAM_EMU_DEV_DBG_PRINT3(verbose, "TGT: emu_tgt_NVMe_execute(%ld) call emu_tgt_map_Submit(%p, %p)\n", tid, pMgtTgtControl->pDevMapper, pContext);

	return emu_tgt_map_Submit(pMgtTgtControl->pDevMapper, pContext, &pQP->pvThreadContext);
}
#endif


__device__ inline int emu_tgt_NVMe_Complete(storage_next_emuluator_context *pContext, bam_emulated_target_control    *pMgtTgtControl, bam_emulated_queue_pair      *pQP, uint32_t cq_db_head)
{
	int verbose = bam_get_verbosity(BAM_EMU_DBGLVL_NONE, BAM_DBG_CODE_PATH_D_COMP);
	uint32_t		retries = 0;
	const uint32_t	retry_limit = 10;
	const uint32_t	retry_ns = 64;
	uint16_t cid = SN_CONTEXT_TAG(pContext);
	uint32_t		phase = 0x10000;
		
		
		
	//	uint32_t phase = 0;
	BAM_EMU_DEV_DBG_PRINT4(verbose, "emu_tgt_NVMe_Complete() db_head = %d cq_tail = %d next_tail = %d cid = 0x%x\n",
		 cq_db_head, pQP->cQ.tail, ((pQP->cQ.tail + 1) &pQP->cQ.q_size_minus_1), cid);

	while (retries < retry_limit)
	{
		if (cq_db_head != (pQP->cQ.tail + 1))
		{
			nvm_cpl_t * 	pCmp = & (((nvm_cpl_t *) (pQP->cQ.pEmuQ))[pQP->cQ.tail]);


			if (pQP->cQ.rollover & 0x1)
			{
				phase				= 0;
			}

			BAM_EMU_DEV_DBG_PRINT3(verbose, "emu_tgt_NVMe_Complete() cid = 0x%04x cq_tail = %d rollover = %d\n", cid,
				 pQP->cQ.tail, pQP->cQ.rollover);


			pCmp->dword[0]		= 0;
			pCmp->dword[1]		= 0;
			pCmp->dword[2]		= ((uint32_t) pQP->q_number << 16) | (pQP->sQ.head);
			pCmp->dword[3]		= phase | cid;


			BAM_EMU_DEV_DBG_PRINT3(verbose, "emu_tgt_NVMe_Complete() %p val[2] = %x val[3] = %x\n", &pCmp->dword[2],
				 pCmp->dword[2], pCmp->dword[3]);

			pQP->cQ.tail++;

			pQP->cQ.tail		&= pQP->cQ.q_size_minus_1;

			if (0 == pQP->cQ.tail)
			{
				pQP->cQ.rollover++;
			}

			return 0;
		}
		else 
		{
			BAM_EMU_DEV_DBG_PRINT3(BAM_EMU_DBGLVL_ERROR, "emu_tgt_NVMe_Complete() !!!QFULL db_head = %d cq_tail = %d retries = %d\n",
				 cq_db_head, pQP->cQ.tail, retries);
			retries++;
			
		    __nanosleep(retries * retry_ns);
			
		}
	}

	return 1;




}

__device__ inline uint32_t emu_tgt_Cull(bam_emulated_target_control    *pMgtTgtControl, bam_emulated_queue_pair      *pQP, uint32_t cq_db_head)
{
	int verbose = bam_get_verbosity(BAM_EMU_DBGLVL_NONE, BAM_DBG_CODE_PATH_D_CULL);
	uint32_t count = 0;
	storage_next_emuluator_context *pContext;
	
	BAM_EMU_DEV_DBG_PRINT2(verbose, "TGT: emu_tgt_Cull() pMgtTgtControl = %p pvThreadContext = %p\n", pMgtTgtControl, pQP->pvThreadContext);

	while(NULL != (pContext = emu_tgt_map_Cull(pMgtTgtControl->pDevMapper, &pQP->pvThreadContext)))
	{
		BAM_EMU_DEV_DBG_PRINT2(verbose, "TGT: emu_tgt_Cull(%d) pContext = %p \n", count, pContext);
		
		emu_tgt_NVMe_Complete(pContext, pMgtTgtControl, pQP, cq_db_head);
		

		count++;
		
	}
	return count;

	
}


__device__ inline uint32_t emu_tgt_NVMe_Submit(bam_emulated_target_control    *pMgtTgtControl, bam_emulated_queue_pair     *pQP, uint32_t *pSubmit_count)
{
	int verbose = bam_get_verbosity(BAM_EMU_DBGLVL_NONE, BAM_DBG_CODE_PATH_D_NVME_SUB);
	nvm_cmd_t *pCmd;
	nvm_cmd_t *pQ = &(((nvm_cmd_t *)(pQP->sQ.pEmuQ))[0]);
	int count = 0;
	uint32_t cq_db_head;
	storage_next_emuluator_context *pContext;

//	cq_db_head = *pQP->cQ.db;

	cq_db_head = emu_tgt_read_doorbell(&pQP->cQ);


	while(pQP->sQ.head != pQP->sQ.tail)
	{
			
		BAM_EMU_DEV_DBG_PRINT4(verbose, "TGT: emu_tgt_NVMe_Submit(%d) head = %d tail = %d pMgtTgtControl = %p\n", count, pQP->sQ.head, pQP->sQ.tail, pMgtTgtControl);

		pCmd = &pQ[pQP->sQ.tail];

		BAM_EMU_DEV_DBG_PRINT1(verbose, "TGT: emu_tgt_NVMe_Submit(pCmd = %p)\n", pCmd);
		
		pContext = &pQP->pContext[pQP->sQ.tail];

		BAM_EMU_DEV_DBG_PRINT1(verbose, "TGT: emu_tgt_NVMe_Submit(pContext = %p)\n", pContext);

		pContext->pCmd = (storage_next_command *)pCmd;

		BAM_EMU_DEV_DBG_PRINT3(verbose, "TGT: emu_tgt_NVMe_Submit() pCmd = %p pContext = %p pContext->pCmd = %p\n", pCmd, pContext, pContext->pCmd);

		BAM_EMU_DEV_DBG_PRINT2(verbose, "TGT: emu_tgt_NVMe_Submit() *** = %p  %p\n", pMgtTgtControl, pMgtTgtControl->pDevMapper);

		if(emu_tgt_NVMe_execute(pMgtTgtControl, pQP, pContext, cq_db_head))
		{
			BAM_EMU_DEV_DBG_PRINT1(BAM_EMU_DBGLVL_ERROR, "TGT: emu_tgt_NVMe_execute(%d) ERROR!!!\n", pQP->q_number);
		}
		else
		{
			pQP->sQ.tail++;

			pQP->sQ.tail &= pQP->sQ.q_size_minus_1;

			if(0 == pQP->sQ.tail)
			{
				pQP->sQ.rollover++;
			}
			count++;
		}
	

	}

	BAM_EMU_DEV_DBG_PRINT3(verbose, "TGT: emu_tgt_NVMe_Submit(%d) EXIT = cq_db_head %d tail = %d\n", count, cq_db_head, pQP->sQ.tail);

	*pSubmit_count = count;

	return cq_db_head;
	

}

__device__ inline int emu_tgt_SQ_Check(bam_emulated_target_control    *pMgtTgtControl, bam_emulated_queue_pair     *pQP)
{
	int verbose = bam_get_verbosity(BAM_EMU_DBGLVL_NONE, BAM_DBG_CODE_PATH_D_SQ_CHECK);
	uint32_t db_tail;
	nvm_cmd_t* pCmd;
	int q_number = pQP->q_number;
	
	BAM_EMU_DEV_DBG_PRINT1(verbose, "TGT: emu_tgt_SQ_Check(%p) \n", pQP);
	BAM_EMU_DEV_DBG_PRINT1(verbose, "TGT: emu_tgt_SQ_Check(%s) \n", pMgtTgtControl->szName);
	BAM_EMU_DEV_DBG_PRINT1(verbose, "TGT: emu_tgt_SQ_Check() q_number = %d \n", q_number);


	
	BAM_EMU_DEV_DBG_PRINT2(verbose, "TGT: emu_tgt_SQ_Check(%d) %d CALL\n", pMgtTgtControl->bRun, q_number);

	if(pMgtTgtControl->bRun)
	{
		if(pQP->qp_enabled)		
		{
			BAM_EMU_DEV_DBG_PRINT2(verbose, "TGT: emu_tgt_SQ_Check(%d) ENABLED size = %d\n", q_number, pQP->sQ.q_size);
			BAM_EMU_DEV_DBG_PRINT2(verbose, "TGT: emu_tgt_SQ_Check(%p) %d \n", (void *)pQP->sQ.ioaddr, q_number);

			pCmd = (nvm_cmd_t *)pQP->sQ.ioaddr;

			BAM_EMU_DEV_DBG_PRINT2(verbose, "TGT: emu_tgt_SQ_Check() cmd[0] = 0x%08x cmd[1] = 0x%08x \n", pCmd->dword[0], pCmd->dword[1]);
			BAM_EMU_DEV_DBG_PRINT2(verbose, "TGT: emu_tgt_SQ_Check() db = %p  db = %p \n", pQP->cQ.db, pQP->sQ.db);

			db_tail = emu_tgt_read_doorbell(&pQP->sQ);
			
				
			BAM_EMU_DEV_DBG_PRINT2(verbose, "TGT: emu_tgt_SQ_Check() db = %d	head = %d \n",  db_tail, pQP->sQ.head);

			if(db_tail != pQP->sQ.head)
			{
				emu_tgt_SQ_Process(pMgtTgtControl, pQP, db_tail);
			}
				
		}


		
	}
	
	BAM_EMU_DEV_DBG_PRINT2(verbose, "TGT: emu_tgt_SQ_Check(%d) exit qp_enabled = %d\n", q_number, pQP->qp_enabled);

	return pQP->qp_enabled;
	
}


__device__ inline void memCpy32(uint32_t *pDest, uint32_t *pSrc, uint32_t len)
{
	uint32_t i;
	uint32_t limit = (len / 4);
		
	if(len & 0x3)
	{
		assert(0);
	}

	for(i = 0; i < limit; i++)
	{
		pDest[i] = pSrc[i];
	}


}
__device__ inline bam_emulated_queue_pair * emu_tgt_init_QueuePair(bam_emulated_target_control  *pMgtTgtControl, bam_emulated_queue_pair         *pDevQPairs, bam_emulated_queue_pair *pQProxy, uint32_t *pQueues_per_thread, uint32_t *pBase_q_idx)
{
	uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t base_q_idx;
	int verbose = bam_get_verbosity(BAM_EMU_DBGLVL_NONE, BAM_DBG_CODE_PATH_D_INIT_Q_PAIR);
	
	*pQueues_per_thread = 1;

	if(pMgtTgtControl->numEmuThreads)
	{
		*pQueues_per_thread = pMgtTgtControl->numQueues / (uint32_t)pMgtTgtControl->numEmuThreads;
	
	}

	base_q_idx = tid * (*pQueues_per_thread);
	
	*pBase_q_idx = base_q_idx;
	
	BAM_EMU_DEV_DBG_PRINT4(verbose, "TGT:(%ld) numQueues = %d numEmuThreads = %ld Queues_per_thread =%d\n", tid, pMgtTgtControl->numQueues, pMgtTgtControl->numEmuThreads, *pQueues_per_thread);
	BAM_EMU_DEV_DBG_PRINT3(verbose, "TGT:(%ld) base_q_idx = %d pQProxy = %p\n", tid, base_q_idx, pQProxy);

	if(NULL != pQProxy)
	{
		memCpy32((uint32_t *)pQProxy,(uint32_t *)&pDevQPairs[base_q_idx],sizeof(bam_emulated_queue_pair) * (*pQueues_per_thread));
		return pQProxy;
	}
	else
	{

		
		return &pDevQPairs[base_q_idx];
	}
}




//max performance case
#if (BAM_EMU_USE_KCONTEXT_Q_CTRL == 1)
__device__ inline bam_emulated_queue_pair * emu_tgt_get_QueuePair(bam_emulated_queue_pair         *pDevQPairs, bam_emulated_queue_pair *pQProxy, uint32_t queues_per_thread, uint32_t count, uint32_t base_q_idx)
{
	return &pQProxy[0];
}
#else

__device__ inline bam_emulated_queue_pair * emu_tgt_get_QueuePair(bam_emulated_queue_pair         *pDevQPairs, bam_emulated_queue_pair *pQProxy, uint32_t queues_per_thread, uint32_t count, uint32_t base_q_idx)
{
	uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	int verbose = bam_get_verbosity(BAM_EMU_DBGLVL_NONE, BAM_DBG_CODE_PATH_D_GET_Q_PAIR);

	uint32_t q_idx = (count % queues_per_thread); 

	BAM_EMU_DEV_DBG_PRINT3(verbose, "TGT:(%ld) count =%d q_idx = %d\n", tid, count, q_idx);
	
	if(NULL != pQProxy)
	{
		return &pQProxy[q_idx];
	}
	else
	{
		q_idx += base_q_idx;
		
		return &pDevQPairs[q_idx];
	}
	

}
#endif

#ifdef BAM_RUN_EMU_IN_BAM_KERNEL
#define EMU_KERNEL_ENTRY_TYPE __device__
#else 
#define EMU_KERNEL_ENTRY_TYPE __global__
#endif

EMU_KERNEL_ENTRY_TYPE void kernel_Emulator(bam_emulated_target_control    *pMgtTgtControl, bam_emulated_queue_pair     *pDevQPairs)

{
	bam_emulated_queue_pair 	   *pQP;
	uint32_t cq_db_head;
	int verbose = bam_get_verbosity(BAM_EMU_DBGLVL_NONE, BAM_DBG_CODE_PATH_D_KER_QSTRM);
	uint32_t count = 0;
	uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	bam_emulated_queue_pair *pRegQp;
	uint32_t queues_per_thread;
	uint32_t base_q_idx;
#ifdef BAM_EMU_USE_KCONTEXT_Q_CTRL
	bam_emulated_queue_pair sharedQP[BAM_EMU_USE_KCONTEXT_Q_CTRL];
	pRegQp = &sharedQP[0];
#else
	pRegQp = NULL;
#endif

	pQP = emu_tgt_init_QueuePair(pMgtTgtControl, pDevQPairs, pRegQp, &queues_per_thread, &base_q_idx);

	BAM_EMU_DEV_DBG_PRINT3(verbose, "TGT: kernel_Emulator ENTER pMgtTgtControl = %p pQP = %p tid=%ld\n", pMgtTgtControl, pQP, tid);

	BA_DBG_SET(pMgtTgtControl, 1, 0xBABA0001);
	


	while(pMgtTgtControl->bRun)
	{
		uint32_t submit_count;

		pQP = emu_tgt_get_QueuePair(pDevQPairs, pRegQp, queues_per_thread, count, base_q_idx);

#ifdef BAM_EMU_TGT_SIMPLE_MODE_NVME_LOOPBACK
	 	BA_DBG_SET(pMgtTgtControl, 2, 0xBABA0002);
		if(emu_tgt_SQ_Check(pMgtTgtControl, pQP))
		{
			BA_DBG_SET(pMgtTgtControl, 3, 0xBABA0003);
			cq_db_head = emu_tgt_NVMe_Submit(pMgtTgtControl, pQP, &submit_count);

			if(submit_count)
			{
				emu_tgt_CQ_Drain(pMgtTgtControl, pQP, cq_db_head);
			}
		}
#else
	
	 	BA_DBG_SET(pMgtTgtControl, 2, 0xBABA0002);
		if(emu_tgt_SQ_Check(pMgtTgtControl, pQP))
		{
			BA_DBG_SET(pMgtTgtControl, 3, 0xBABA0003);
			cq_db_head = emu_tgt_NVMe_Submit(pMgtTgtControl, pQP, &submit_count);

	
			if(emu_tgt_Cull(pMgtTgtControl, pQP, cq_db_head))
			{
				emu_tgt_CQ_Drain(pMgtTgtControl, pQP, cq_db_head);
			}


		}
#endif



#ifdef BAM_EMU_QTHREAD_ONE_SHOT
	    if(count >= pMgtTgtControl->nOneShot)
		{			
			break;
		}
#endif
		count++;

	}	


	if(count)
	{
		BA_DBG_SET(pMgtTgtControl, BA_DBG_IDX_MARK_RUN, BA_DBG_VAL_MARK_RUN);
	}
	BA_DBG_SET(pMgtTgtControl, BA_DBG_IDX_RUN_COUNT, count);

	BAM_EMU_DEV_DBG_PRINT3(verbose, "TGT: kernel_Emulator oneshot = %d count = %d x = %d EXIT\n",0, count, 0);	
}



//#define BAM_EMU_DOUBLE_CHECK_DEVICE_Q_COPY


static void emulator_update_d_queue(bam_host_emulator *pEmu,  uint16_t q_number, int bEnable = 1)
{
	int	verbose = bam_get_verbosity(BAM_EMU_DBGLVL_NONE, BAM_DBG_CODE_PATH_H_UPDATEDQ);
	uint16_t q_idx = q_number - 1;
	
	
	BAM_EMU_HOST_DBG_PRINT(verbose, "emulator_update_d_queue() q_number = %d bEnable = %d\n", q_number, bEnable);

	if(pEmu->tgt.queuePairs[q_idx].sQ.enabled)
	{
		if(pEmu->tgt.queuePairs[q_idx].cQ.enabled)
		{
			BAM_EMU_HOST_DBG_PRINT(verbose, "emulator_update_d_queue() q_idx = %d BOTH Qs ENABLED, UPDATING\n", q_idx);

			pEmu->tgt.queuePairs[q_idx].qp_enabled = bEnable;
			pEmu->tgt.queuePairs[q_idx].q_number = q_number;
			
			
			cuda_err_chk(cudaMemcpy(&pEmu->tgt.pDevQPairs[q_idx], &pEmu->tgt.queuePairs[q_idx], sizeof(bam_emulated_queue_pair), cudaMemcpyHostToDevice));

			BAM_EMU_HOST_DBG_PRINT(verbose, "*** Copy Good to q_idx=%d dev_ptr = %p dev_base = %p sq.db = %p cq.db = %p\n", q_idx, &pEmu->tgt.pDevQPairs[q_idx], pEmu->tgt.pDevQPairs, pEmu->tgt.queuePairs[q_idx].sQ.db, pEmu->tgt.queuePairs[q_idx].cQ.db);
			
#ifdef BAM_EMU_DOUBLE_CHECK_DEVICE_Q_COPY
			cuda_err_chk(cudaMemcpyAsync(&aQP, &pEmu->tgt.pDevQPairs[q_idx], sizeof(bam_emulated_queue_pair), cudaMemcpyDeviceToHost,  pEmu->tgt.queueStream));
			
			BAM_EMU_HOST_DBG_PRINT(verbose, "*** Async %d = %p  device_base = %p\n", 3, &pEmu->tgt.pDevQPairs[q_idx], pEmu->tgt.pDevQPairs );
			
			BAM_EMU_HOST_DBG_PRINT(verbose, "*** pQ qp_enabled = %d target pointer = %p\n", (uint32_t)aQP.qp_enabled, &pEmu->tgt.pDevQPairs[q_idx]);
			BAM_EMU_HOST_DBG_PRINT(verbose, "*** SQ %d size = %d\n", (uint32_t)aQP.sQ.q_number, (uint32_t)aQP.sQ.q_size);
			BAM_EMU_HOST_DBG_PRINT(verbose, "*** CQ %d size = %d\n", (uint32_t)aQP.cQ.q_number, (uint32_t)aQP.cQ.q_size);

#endif
		}
		else
		{
			BAM_EMU_HOST_DBG_PRINT(verbose, "emulator_update_d_queue() q_idx = %d CQ NOT ENABLED YET, SKIPPING UPDATE\n", q_idx);
		}

	}
	else
	{	
		//This is a hack, because we call emulator_update_d_queue() twice to update doorbells.  Let's count only first time when only SQ enabled
		if(bEnable)
		{
			pEmu->tgt.pTgt_control->numQueues++;
		}

		BAM_EMU_HOST_DBG_PRINT(verbose, "emulator_update_d_queue() q_idx = %d SQ NOT ENABLED YET, SKIPPING UPDATE\n", q_idx);
	}

	BAM_EMU_HOST_DBG_PRINT(verbose, "emulator_update_d_queue() EXIT q_idx = %d bEnable = %d\n", q_idx, bEnable);

		

}

static void emulator_create_queue(bam_host_emulator *pEmu, uint16_t q_number, uint16_t q_size, uint64_t ioaddr, uint16_t cq)
{
	int	verbose = bam_get_verbosity(BAM_EMU_DBGLVL_NONE, BAM_DBG_CODE_PATH_H_CREATE_Q);
	uint16_t q_idx = q_number - 1;
	uint32_t context_size = 0;
	uint32_t mem_size;
	
	

	
	bam_emulated_queue *pQ;
	
	BAM_EMU_HOST_DBG_PRINT(verbose, "emulator_create_queue() pEmu = %p q_number = %d q_size = %d ioaddr = %p cq = %d\n", pEmu, q_number, q_size, ioaddr, cq);

	if(cq)
	{
		pQ = &pEmu->tgt.queuePairs[q_idx].cQ;

		mem_size = q_size * sizeof(nvm_cpl_t);
			


	}
	else
	{
		pQ = &pEmu->tgt.queuePairs[q_idx].sQ;

		mem_size = q_size * sizeof(nvm_cmd_t);
		
		context_size = mem_size;
		
	}

	pQ->cq = cq;
	pQ->ioaddr = ioaddr;
	pQ->q_size = q_size;
	pQ->q_size_minus_1 = (q_size - 1);
	pQ->q_number = q_number;
	pQ->enabled = 1;


	

	/* One to One mapping of Command to context, so we can index directly and not have to manage a queue of contexts */
	BAM_EMU_HOST_ASSERT(sizeof(storage_next_command) == sizeof(nvm_cmd_t));
	BAM_EMU_HOST_ASSERT(sizeof(storage_next_command) == sizeof(storage_next_emuluator_context));
	
	BAM_EMU_HOST_DBG_PRINT(verbose, "emulator_create_queue() q_mem_size = %d context_size = %d total = %d\n", mem_size, context_size, mem_size + context_size);


	pEmu->tgt.devQMem[q_idx].target_q_mem[cq] = createDma(pEmu->pCtrl, NVM_PAGE_ALIGN(mem_size + context_size, 1UL << 16), pEmu->cudaDevice);
	

	pQ->pEmuQ = pEmu->tgt.devQMem[q_idx].target_q_mem[cq]->vaddr;


	if(context_size)
	{
		pEmu->tgt.queuePairs[q_idx].pContext = (storage_next_emuluator_context *)((char *)pQ->pEmuQ + mem_size);

	}
	

	
	
	emulator_update_d_queue(pEmu, q_number);
	

}

static void emulator_rpc_callout(void *pvCtrl, void* pvCmd, void *pvCpl)
{
	int	verbose = bam_get_verbosity(BAM_EMU_DBGLVL_NONE, BAM_DBG_CODE_PATH_H_EMU_RPC);
	nvm_cpl_t* pCpl = (nvm_cpl_t*)pvCpl;

	nvm_cmd_t *pCmd = (nvm_cmd_t *)pvCmd;
	nvm_ctrl_t *pCtrl = (nvm_ctrl_t *)pvCtrl;
	bam_host_emulator *pEmu = (bam_host_emulator *)pCtrl->pvEmu;
	unsigned char opcode = pCmd->dword[0] & 0x7F;
	uint16_t q_number;
	uint16_t q_size;
	uint16_t cq = 0;
	uint64_t ioaddr;
	
	BAM_EMU_HOST_DBG_PRINT(verbose,"emulator_rpc_callout(%p, %p) pEmu = %p opcode = %x\n", pvCtrl, pvCmd, pEmu, opcode);

	switch(opcode)
	{
		case NVM_ADMIN_CREATE_CQ:
			cq = 1;
			 [[fallthrough]];
		case NVM_ADMIN_CREATE_SQ:
		{
			q_number = pCmd->dword[10] & 0xFFFF;
			q_size = ((pCmd->dword[10] >> 16) & 0xFFFF) + 1;
			ioaddr =  ((uint64_t)pCmd->dword[7] << 32UL) | (uint64_t)pCmd->dword[6];
		
			
			BAM_EMU_HOST_DBG_PRINT(verbose,"emulator_rpc_callout() CREATE_Q  q_number = %d q_size = %d ioaddr =%p\n", q_number, q_size, ioaddr);
			
			emulator_create_queue(pEmu, q_number, q_size, ioaddr, cq);

			
		
		}
		break;

		case NVM_ADMIN_GET_FEATURES:
			BAM_EMU_HOST_DBG_PRINT(verbose,"emulator_rpc_callout(NVM_ADMIN_GET_FEATURES) pEmu = %p opcode = %x\n", pEmu, opcode);

			uint32_t numqueues = 1024;
			numqueues--;

			pCpl->dword[0] = (numqueues << 16) | numqueues;
			
			
			

			break;
			

		

	}

}




#ifdef BAM_EMU_TARGET_HOST_THREAD


__global__ void kernel_oneshotStream(bam_emulated_target_control    *pMgtTgtControl, bam_emulated_queue_pair     *pDevQPairs)
{
	int verbose = bam_get_verbosity(BAM_EMU_DBGLVL_INFO, BAM_DBG_CODE_PATH_D_KER_QSTRM);
	static int count = 0;

	BAM_EMU_DEV_DBG_PRINT1(verbose, "kernel_oneshotStream(%d)\n", count++);
	
}


void * launch_emu_target(void *pvEmu)
{
#ifndef	BAM_RUN_EMU_IN_BAM_KERNEL

	bam_host_emulator * pEmu = (bam_host_emulator *)pvEmu;
	int             heartbeat_usec = 1000 * 1000;
	int             count = 0;
	int 			verbose = bam_get_verbosity(BAM_EMU_DBGLVL_INFO, BAM_DBG_CODE_PATH_H_EMU_THREAD);


#ifdef BAM_EMU_QTHREAD_ONE_SHOT
	pEmu->tgt.pTgt_control->nOneShot = 1;
#else

#ifdef RESIDENT_STREAM_DEBUG
	kernel_Emulator<<< pEmu->g_size, pEmu->b_size, 0, pEmu->tgt.queueStream >>> (5);
#else
	kernel_Emulator<<< pEmu->g_size, pEmu->b_size, 0, pEmu->tgt.queueStream >>> (pEmu->tgt.pTgt_control, pEmu->tgt.pDevQPairs);
#endif


	//kernel_Emulator<<<pEmu->g_size, pEmu->b_size>>> (pEmu->tgt.pMgtTgtControl, pEmu->tgt.pDevQPairs);
	BAM_EMU_HOST_DBG_PRINT(verbose, "kernel_Emulator RETURN(%d)\n", 0);

#endif

	BAM_EMU_HOST_DBG_PRINT(verbose, "launch_emu_target(%p)\n", pvEmu);

	while(pEmu->bRun)
	{
		if(0 == (count % heartbeat_usec))
		{
			BAM_EMU_HOST_DBG_PRINT(verbose, "emu_thread heartbeat(%d)\n", count);

		}	
#ifdef BAM_EMU_QTHREAD_ONE_SHOT
		kernel_Emulator<<< pEmu->g_size, pEmu->b_size, 0, pEmu->tgt.queueStream >>> (pEmu->tgt.pTgt_control, pEmu->tgt.pDevQPairs);

		cudaStreamSynchronize(pEmu->tgt.queueStream);
#else	
		sleep(1);
#endif

		count++;
	}
	
	BAM_EMU_HOST_DBG_PRINT(verbose, "launch_emu_target(%d) RETURN\n", count);
#endif
	return NULL;
}





#endif
static void start_emulation_target(bam_host_emulator *pEmu)
{
	int	verbose = bam_get_verbosity(BAM_EMU_DBGLVL_INFO, BAM_DBG_CODE_PATH_H_START_EMU);


	BAM_EMU_HOST_DBG_PRINT(verbose,"start_emulation_target(%p) g_size = %d b_size = %d\n", pEmu, pEmu->g_size, pEmu->b_size);

	cuda_err_chk(cudaStreamCreateWithFlags (&pEmu->tgt.bamStream, (cudaStreamNonBlocking)));

	cuda_err_chk(cudaMemcpy(&pEmu->tgt.pDevQPairs[0], &pEmu->tgt.queuePairs[0], sizeof(bam_emulated_queue_pair) * BAM_EMU_MAX_QUEUES, cudaMemcpyHostToDevice));


#ifdef BAM_EMU_TARGET_HOST_THREAD
	pEmu->bRun = 1;

	pthread_create(&pEmu->emu_threads[0], NULL, launch_emu_target, pEmu); 

#else
	try
	{
		cuda_err_chk(cudaStreamCreate (&pEmu->tgt.queueStream));

		kernel_Emulator<<< pEmu->g_size, pEmu->b_size, 0, pEmu->tgt.queueStream >>> (pEmu->tgt.pTgt_control, pEmu->tgt.pDevQPairs);

		BAM_EMU_HOST_DBG_PRINT(verbose,"HOST: kernel_Emulator RETURN(%d)\n",0);

	}
	catch (const error& e) 
	{
		BAM_EMU_HOST_DBG_PRINT(BAM_EMU_DBGLVL_ERROR,"start_emulation_target(CATCH): %s\n", e.what());
		BAM_EMU_HOST_ASSERT(0);
	}
#endif
	//BAM_EMU_HOST_ASSERT(0);

}
	
static inline void cleanup_emulator_target(bam_host_emulator *pEmu)
{
	int	verbose = bam_get_verbosity(BAM_EMU_DBGLVL_NONE, BAM_DBG_CODE_PATH_H_CLEANUP_EMU);
	int i;
	

#ifdef KERNEL_DBG_ARRAY
	for(int i = 0; i < 32; i++)
	{
		if(pEmu->tgt.pTgt_control->debugA[i])
		{
			printf("DEBUG[%d] = 0x%08x\n", i, pEmu->tgt.pTgt_control->debugA[i]);
		}
	}

	if(pEmu->tgt.pTgt_control->debugA[BA_DBG_IDX_MARK_RUN] != BA_DBG_VAL_MARK_RUN)
	{

		printf("*************************************************************************************************\n");
		printf("*************************************************************************************************\n");
		printf("!!!! EMULATOR THREAD TID=0 DID NOT RUN CYCLES!!!!\n");
		printf("*************************************************************************************************\n");
		printf("*************************************************************************************************\n");

	}
	else
	{
		printf("EMULATOR THREAD TID=0 Ran %d Cycles\n", pEmu->tgt.pTgt_control->debugA[BA_DBG_IDX_RUN_COUNT]);
	}
#endif


	
	BAM_EMU_HOST_DBG_PRINT(verbose,"cleanup_emulator_target(%p) CALL %d queues configured\n", pEmu, pEmu->tgt.pTgt_control->numQueues);

	pEmu->bRun = 0;
	pEmu->tgt.pTgt_control->bRun = 0;

	for(i = 0; i < BAM_EMU_MAX_QUEUES; i++)
	{
		if(pEmu->tgt.queuePairs[i].qp_enabled)
		{
			BAM_EMU_HOST_DBG_PRINT(verbose,"cleanup_emulator_target() qp = %d ENABLED\n", i); 
			//emulator_update_d_queue(pEmu, i, 0);
			
		}
		else
		{
		//	BAM_EMU_HOST_DBG_PRINT(verbose,"cleanup_emulator_target() qp = %d \n", i + 1); 
			
		}
	}

	
	

	//TODO: Cleanup 		
#if 0
	cudaError_t err;

	BAM_EMU_HOST_DBG_PRINT(verbose,"cleanup_emulator_target() cudaStreamSynchronize(tgtStream) CALL = %d\n", 0); 
	err = cudaStreamSynchronize(pEmu->tgt.queueStream);	
	BAM_EMU_HOST_DBG_PRINT(verbose,"cleanup_emulator_target() cudaStreamSynchronize(tgtStream) RETURN = %d\n", err); 

	

	if(pEmu->tgt.pTgt_control)
	{
		cudaFree(pEmu->tgt.pTgt_control);
	}

	
	if(pEmu->tgt.queuePairs)
	{
		cudaFreeHost(pEmu->tgt.pTgt_control);
	}

	
	err = cudaStreamQuery(pEmu->tgt.tgtStream);
	BAM_EMU_HOST_DBG_PRINT(verbose,"cleanup_emulator_target() tgtStream Query = %d \n", err);
	if(cudaSuccess == err)
	{
		BAM_EMU_HOST_DBG_PRINT(verbose,"cleanup_emulator_target() tgtStream Query == cudaSuccess \n", err);
	}
	else
	{
	
		BAM_EMU_HOST_DBG_PRINT(verbose,"cleanup_emulator_target() tgtStream Query != cudaSuccess calling destroy\n", err);
		err = cudaStreamDestroy(pEmu->tgt.tgtStream);
	}
	

	err = cudaStreamQuery(pEmu->tgt.queueStream);
	BAM_EMU_HOST_DBG_PRINT(verbose,"cleanup_emulator_target() queueStream Query = %d \n", err);
	if(cudaSuccess == err)
	{
		BAM_EMU_HOST_DBG_PRINT(verbose,"cleanup_emulator_target() queueStream Query == cudaSuccess \n", err);
	}
	else
	{
	
		BAM_EMU_HOST_DBG_PRINT(verbose,"cleanup_emulator_target() queueStream Query != cudaSuccess calling destroy\n", err);
		err = cudaStreamDestroy(pEmu->tgt.queueStream);
	}
	


	err = cudaStreamQuery(pEmu->tgt.bamStream);
	BAM_EMU_HOST_DBG_PRINT(verbose,"cleanup_emulator_target() bamStream Query = %d \n", err);
	if(cudaSuccess == err)
	{
		BAM_EMU_HOST_DBG_PRINT(verbose,"cleanup_emulator_target() bamStream Query == cudaSuccess \n", err);
	}
	else
	{
	
		BAM_EMU_HOST_DBG_PRINT(verbose,"cleanup_emulator_target() bamStream Query != cudaSuccess calling destroy\n", err);
		err = cudaStreamDestroy(pEmu->tgt.bamStream);
	}
	
#endif

	free(pEmu);

	pEmu = NULL;

	BAM_EMU_HOST_DBG_PRINT(verbose,"cleanup_emulator_target(%p) RETURN\n", pEmu);

}

static inline nvm_ctrl_t* initializeEmulator(uint32_t ns_id, uint32_t cudaDevice, uint64_t queueDepth, uint64_t numQueues, bam_host_emulator **pEmulator, uint64_t emulationTargetFlags, uint32_t blkSize, uint32_t sectorSize)
{

	int	verbose = bam_get_verbosity(BAM_EMU_DBGLVL_INFO, BAM_DBG_CODE_PATH_H_INIT_EMU);
	nvm_ctrl_t        *pCtrl;
	int err;
	bam_host_emulator *pEmu;
	const char  *temppath = "working_device_mem.bin";
	unsigned char *pFileMem;
	int fd;
    DmaPtr                  aq_mem;
	int qall_size = sizeof(bam_emulated_queue_pair) * BAM_EMU_MAX_QUEUES;
	int mem_size = NVM_CTRL_MEM_MINSIZE * 2;
	int map_model_size = 0;

	

	BAM_EMU_HOST_DBG_PRINT(verbose,"initializeEmulator() sizeof(bam_emulated_queue_pair) = %ld\n", sizeof(bam_emulated_queue_pair));
	
	if(posix_memalign((void **)&pFileMem, 4096, mem_size))
	{
		BAM_EMU_HOST_ASSERT(0);
	}
				
	memset(pFileMem, 0, mem_size);

	BAM_EMU_HOST_DBG_PRINT(verbose,"initializeEmulator() sizeof nvm_ctrl_t = %d pFileMem = %p ns_id = %d queueDepth = %ld\n", sizeof(nvm_ctrl_t), pFileMem, ns_id, queueDepth);


	/* Fix up the Emulators Controller Capabilities here */
	((volatile uint64_t*) pFileMem)[0] = 0x1FFFF;
	
				
	fd = open(temppath, O_CREAT | O_RDWR, S_IRWXU);
			
	int res = write(fd, pFileMem, mem_size);

	BAM_EMU_HOST_DBG_PRINT(verbose,"initializeEmulator() tempath = %s fd = %d res = %d\n", temppath, fd,res);

	close(fd);

	//fd = open(temppath, O_RDWR);
	fd = open(temppath, O_RDWR, S_IRWXU);
	

	BAM_EMU_HOST_DBG_PRINT(verbose,"initializeEmulator(FD2) tempath = %s fd = %d \n", temppath, fd);

	int status = nvm_ctrl_init(&pCtrl, fd, 1);
	
	BAM_EMU_HOST_DBG_PRINT(verbose,"initializeEmulator() pCtrl = %p status = %d sizeof(bam_host_emulator) %ld\n", pCtrl, status, sizeof(bam_host_emulator));


	err = posix_memalign((void **)&pEmu, 4096, sizeof(bam_host_emulator));
	
	if (err) 
	{
		   throw error(string("Failed to allocate host memory: ") + std::to_string(err));
	}

	BAM_EMU_HOST_ASSERT(pEmu);

	//TODO: causes warning: ‘void* memset(void*, int, size_t)’ clearing an object of type ‘struct bam_host_emulator’ with no trivial copy-assignment; use assignment or value-initialization instead [-Wclass-memaccess]
	//Removing it causes segmentation fault below @pEmu->tgt.d_queue_q_mem = createDma(...)
	//Despite explict initialization of the member of the pEmu and pEmu->tgt members
	//Mystery, come back to it.
	memset(pEmu, 0, sizeof(bam_host_emulator));
	
	BAM_EMU_HOST_DBG_PRINT(verbose, "initializeEmulator(%p) pCtrl\n", pCtrl);

	strcpy(pEmu->name, "BAM_EMULATOR");

	pEmu->sectorSize = sectorSize;

	*pEmulator = pEmu;
	
	pCtrl->page_size = 4096;
	pCtrl->emulated = 1;
	pCtrl->pvEmu = pEmu;
	pCtrl->fnEmuCallout = emulator_rpc_callout;

	
	pEmu->emulationTargetFlags = emulationTargetFlags;


#define EMU_GRID_SIZE_DEFAULT 32

	if(numQueues <= blkSize)
	{
		pEmu->g_size = 1;
		pEmu->b_size = numQueues;

	}
	else
	{
		pEmu->g_size = ((numQueues - 1) / blkSize) + 1;
		pEmu->b_size = blkSize;
		
	}
	printf("numQueues = %ld g_size = %d b_size = %d EMU_GRID_SIZE_DEFAULT=%d\n", numQueues, pEmu->g_size, pEmu->b_size, (uint32_t)EMU_GRID_SIZE_DEFAULT); 

	pEmu->pCtrl = pCtrl;
	pEmu->cudaDevice = cudaDevice;

	pEmu->tgt.d_queue_q_mem = NULL;
	pEmu->tgt.d_target_control_mem = NULL;
	pEmu->tgt.tgtStream = 0;
	pEmu->tgt.queueStream = 0;
	pEmu->tgt.bamStream = 0;
	pEmu->tgt.queuePairs = NULL;
	pEmu->tgt.pTgt_control = NULL; 
	pEmu->tgt.pDevQPairs = NULL;

	
	cuda_err_chk(cudaMallocHost(&pEmu->tgt.queuePairs, qall_size, 0));

	printf("host queuePairs = %p size = %d\n", pEmu->tgt.queuePairs, qall_size);
	
	cuda_err_chk(cudaMallocManaged(&pEmu->tgt.pTgt_control, sizeof(bam_emulated_target_control)));

	
	pEmu->tgt.pTgt_control->numQueues = 0;
	pEmu->tgt.pTgt_control->bRun = 1;
	pEmu->tgt.pTgt_control->bDone = 0;
		

	strcpy(pEmu->tgt.pTgt_control->szName, "george");

	
	 
	
	pEmu->tgt.d_queue_q_mem = createDma(pEmu->pCtrl, NVM_PAGE_ALIGN(qall_size, 1UL << 16), pEmu->cudaDevice);
	
	pEmu->tgt.pDevQPairs = (bam_emulated_queue_pair *)pEmu->tgt.d_queue_q_mem.get()->vaddr;

	BAM_EMU_HOST_DBG_PRINT(verbose, "initializeEmulator() pEmu->tgt.pDevQPairs = %p &tgt.queuePairs = %p, qall_size = %ld bam_emulated_target_control = %d \n", pEmu->tgt.pDevQPairs, &pEmu->tgt.queuePairs, qall_size, sizeof(bam_emulated_target_control));


#ifndef BAM_EMU_TGT_SIMPLE_MODE_NVME_LOOPBACK
	map_model_size = emulator_init_mapper(pEmu, EMU_MAP_TYPE_DIRECT, EMU_MODEL_TYPE_LATENCY);
#endif


	pEmu->shared_size = 0;

	BAM_EMU_HOST_DBG_PRINT(verbose, "initializeEmulator() Total Shared Size = %d map_model_size = %d \n", pEmu->shared_size, map_model_size);
	cuda_err_chk(cudaMemcpy(pEmu->tgt.pDevQPairs, &pEmu->tgt.queuePairs, qall_size, cudaMemcpyHostToDevice));

#ifdef	BAM_EMU_START_EMU_POST_Q_CONFIG
	BAM_EMU_HOST_DBG_PRINT(verbose, "initializeEmulator() POSTPONING start_emulation_target() until after QConfig = %d (BAM_EMU_START_EMU_POST_Q_CONFIG)\n", 0);
#else
	start_emulation_target(pEmu);
#endif

	return pCtrl;
}

#endif

