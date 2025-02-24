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


__device__	  inline  int64_t  NS_Clock() 
{
	  auto							TimeSinceEpoch_ns  =  cuda::std::chrono::duration_cast<cuda::std::chrono::nanoseconds>( cuda::std::chrono::system_clock::now().time_since_epoch() );
	  return  static_cast<int64_t>( TimeSinceEpoch_ns.count() );
}
__device__ __host__ inline float get_GBs_per_sec(uint64_t elap_ns, int bytes)
{

	float gbs = (float)bytes / (float)elap_ns;

	return gbs;
}

#define BAM_EMU_MAX_QUEUES 1024

//controls emulation compile and benchmark enablement
//#define BAM_EMU_COMPILE 

#define BAM_EMU_TARGET_DISABLE    0
#define BAM_EMU_TARGET_ENABLE     0x00000001
#define BAM_EMU_HOST_ASSERT(__assert) if(!(__assert))do { \
	printf("BAM_EMU_HOST_ASSERT @LINE=%d in %s\n", __LINE__, __FILE__); 	\
	printf("\n\n***** EXIT(0) ***"); \
	exit(0); } while(0)


#define BAM_EMU_DBGLVL_NONE    0
#define BAM_EMU_DBGLVL_IOPATH  1
#define BAM_EMU_DBGLVL_ERROR   2
#define BAM_EMU_DBGLVL_INFO    3
#define BAM_EMU_DBGLVL_DETAIL  4
#define BAM_EMU_DBGLVL_VERBOSE 5

#define BAM_EMU_DBGLVL_COMPILE BAM_EMU_DBGLVL_ERROR

#define BAM_DBG_CODE_PATH_ALL             0xFFFFFFFFFFFFFFFFUL
#define BAM_DBG_CODE_PATH_H_CREATE_Q      0x1
#define BAM_DBG_CODE_PATH_H_EMU_RPC       0x2
#define BAM_DBG_CODE_PATH_H_INIT_EMU      0x4
#define BAM_DBG_CODE_PATH_H_START_EMU     0x8
#define BAM_DBG_CODE_PATH_H_UPDATEDQ      0x10
#define BAM_DBG_CODE_PATH_H_EMU_THREAD    0x20
#define BAM_DBG_CODE_PATH_H_CLEANUP_EMU   0x40

#define BAM_DBG_CODE_DEVICE_OFFSET        32UL
#define BAM_DBG_CODE_MACRO_DEVICE(_cd) ((uint64_t)_cd << BAM_DBG_CODE_DEVICE_OFFSET)
#define BAM_DBG_CODE_PATH_D_KER_QSTRM     BAM_DBG_CODE_MACRO_DEVICE(0x1) 
#define BAM_DBG_CODE_PATH_D_SQ_CHECK      BAM_DBG_CODE_MACRO_DEVICE(0x2)
#define BAM_DBG_CODE_PATH_D_SQ_PROCESS    BAM_DBG_CODE_MACRO_DEVICE(0x4)
#define BAM_DBG_CODE_PATH_D_NVME_SUB      BAM_DBG_CODE_MACRO_DEVICE(0x8)
#define BAM_DBG_CODE_PATH_D_NVME_EXE      BAM_DBG_CODE_MACRO_DEVICE(0x10)
#define BAM_DBG_CODE_PATH_D_NVME_LOOP     BAM_DBG_CODE_MACRO_DEVICE(0x20)
#define BAM_DBG_CODE_PATH_D_CQ_DRAIN      BAM_DBG_CODE_MACRO_DEVICE(0x40)

#define BAM_EMU_DEFAULT_CODE_PATH_VERBOSITY 0//BAM_DBG_CODE_PATH_ALL
static uint64_t gCodePathVerbosity = BAM_EMU_DEFAULT_CODE_PATH_VERBOSITY;
__device__ static uint64_t gDeviceCodePathVerbosity = BAM_EMU_DEFAULT_CODE_PATH_VERBOSITY;


__host__  static inline void bam_emu_dbg_printf(int verbose, const char *fmt, ...)
{
    va_list args;
   
    if(verbose >=  BAM_EMU_DBGLVL_COMPILE )
    {  

    	printf("BAM_HOST_EMU:") ;       
        va_start(args, fmt);
        vprintf(fmt, args);
        va_end(args);

    }
}

__host__ __device__ static inline int bam_get_verbosity(int local, uint64_t code_path)
{
#ifdef __CUDA_ARCH__
	if(code_path & gDeviceCodePathVerbosity)
#else
	if(code_path & gCodePathVerbosity)
#endif

	{
		return BAM_EMU_DBGLVL_ERROR;
	}
	else
	{
		//TO TURN OFF ALL debug print return 0;
    	return local;
	}

}

#define BAM_EMU_HOST_DBG_PRINT(__verbose, __format, ...) do { if( __verbose >= BAM_EMU_DBGLVL_COMPILE) bam_emu_dbg_printf(__verbose, __format, __VA_ARGS__); } while (0)
//Kludge because of vprintf in device code
//Not big deal these should only be used for debug anyway
#define BAM_EMU_DEV_DBG_PRINT1(__verbose, __format, _v1) do { if( __verbose >= BAM_EMU_DBGLVL_COMPILE) printf(__format, _v1); } while (0)
#define BAM_EMU_DEV_DBG_PRINT2(__verbose, __format, _v1, _v2) do { if( __verbose >= BAM_EMU_DBGLVL_COMPILE) printf(__format, _v1, _v2); } while (0)
#define BAM_EMU_DEV_DBG_PRINT3(__verbose, __format, _v1, _v2, _v3) do { if( __verbose >= BAM_EMU_DBGLVL_COMPILE) printf(__format, _v1, _v2, _v3); } while (0)
#define BAM_EMU_DEV_DBG_PRINT4(__verbose, __format, _v1, _v2, _v3, _v4) do { if( __verbose >= BAM_EMU_DBGLVL_COMPILE) printf(__format, _v1, _v2, _v3, _v4); } while (0)




typedef struct
{	
	uint16_t 			q_number;  
	uint16_t 			q_size;
	uint16_t            q_size_minus_1;
	uint16_t 			rsvd;

	uint8_t 			cq;
	uint8_t 			enabled;
	uint32_t            rollover;

	
	uint64_t 			ioaddr;
	volatile uint32_t            *db;

	uint32_t            head;
	uint32_t            tail;

	DmaPtr              target_q_mem;
	void *              pEmuQ;
	

} bam_emulated_queue;

typedef struct
{
	bam_emulated_queue    sQ;
	bam_emulated_queue    cQ;

	int                   qp_enabled;
	int                   q_number;
	
} bam_emulated_queue_pair;

typedef struct 
{
		
		int 					numQueues;
		volatile int            bRun;
		int                     bDone;
		int                     nOneShot;
		int                     rsvd;
		char                    szName[32];
		
} bam_emulated_target_control;
	


typedef struct
{
	cudaStream_t            tgtStream;
	cudaStream_t            queueStream;
	cudaStream_t            bamStream;
	
//	bam_emulated_queue_pair queuePairs[BAM_EMU_MAX_QUEUES];

	bam_emulated_queue_pair *queuePairs;

	

	DmaPtr d_queue_q_mem;
	DmaPtr d_target_control_mem;

	bam_emulated_target_control *pTgt_control; //managed, shared with device
	
	bam_emulated_queue_pair        *pDevQPairs;
	
} bam_target_emulator;


typedef struct 
{
	char name[64];
	
	uint64_t emulationTargetFlags;
	int id;
	int g_size;
	int b_size;
	
	bam_target_emulator   tgt;
	nvm_ctrl_t            *pCtrl;
	uint32_t 	          cudaDevice;
	pthread_t          	  emu_threads[3];
	int                   bRun;
	
	
} bam_host_emulator;

//TODO: Async Memcopy when Queues are configured seemed to either stall the EMU threads or never complete
//Want to eventually have the emulator running so that the ADMIN queues can be created by the emulator and 
//don't need to be faked out in libnvm.  For now, to move forward, bring up the emulator after the queues are configured
#define BAM_EMU_START_EMU_POST_Q_CONFIG
#define BAM_EMU_START_EMU_IN_APP_LAYER

#define BAM_EMU_TARGET_HOST_THREAD
//#define BAM_EMU_LAUNCH_DUMMY_THREAD
//#define BAM_EMU_LAUNCH_BAM_DUMMY_IN_START_EMU
//#define BMA_EMU_DISABLE_LAUNCH_EMU_THREAD
//#define BAM_EMU_APP_LAYER_EMU_DUMMY
//#define BAM_EMU_APP_LAYER_BAM_DUMMY
#define BAM_EMU_QTHREAD_ONE_SHOT

//using SCSI terminology, data-in from Host -> target
#define BAM_EMU_DATA_IN  0
#define BAM_EMU_DATA_OUT 1
typedef ulonglong4 emu_copy_type;

__device__ void emu_tgt_DMA(void *dst_addr, void *src_addr, int copy_size, int direction)
{
	int verbose = bam_get_verbosity(BAM_EMU_DBGLVL_NONE, BAM_DBG_CODE_PATH_D_SQ_PROCESS);
	int i;
	uint64_t start_ticks;
	uint64_t end_ticks;
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
	
	start_ticks = NS_Clock();
	for(i = 0; i < limit; i++)
	{
		pDst[i] = pSrc[i];
	}
	
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


	end_ticks = NS_Clock();

	

	BAM_EMU_DEV_DBG_PRINT2(verbose, "TGT: emu_tgt_DMA() copied %d bytes in  =  %ld ns\n", copy_size, (end_ticks - start_ticks));
	BAM_EMU_DEV_DBG_PRINT2(verbose, "TGT: emu_tgt_DMA() %d bytes copied @  =  %f GB/sec \n", copy_size, get_GBs_per_sec((end_ticks - start_ticks), copy_size));

	//BAM_EMU_DEV_DBG_PRINT2(verbose, "TGT emu_tgt_DMA(): hexdump(%p, %d)\n", pDst, 32);
	//hexdump(pDst, 32);


}


__device__ inline void emu_tgt_SQ_Process(bam_emulated_target_control    *pMgtTgtControl, bam_emulated_queue_pair     *pQP, uint32_t db_tail)
{
	int verbose = bam_get_verbosity(BAM_EMU_DBGLVL_NONE, BAM_DBG_CODE_PATH_D_SQ_PROCESS);
	int slot_count;
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

#define BAM_EMU_TGT_NVME_LOOPBACK
#define BAM_EMU_TGT_NVME_LOOPBACK_WITH_XFER



__device__ inline int emu_tgt_NVMe_loopback(bam_emulated_target_control    *pMgtTgtControl, bam_emulated_queue_pair     *pQP, uint16_t cid, uint32_t cq_db_head)
{
	int verbose = bam_get_verbosity(BAM_EMU_DBGLVL_NONE, BAM_DBG_CODE_PATH_D_NVME_LOOP);
	uint32_t phase = 0x10000;
//	uint32_t phase = 0;
	


	BAM_EMU_DEV_DBG_PRINT4(verbose, "emu_tgt_NVMe_loopback() db_head = %d cq_tail = %d next_tail = %d cid = 0x%x\n", cq_db_head, pQP->cQ.tail, ((pQP->cQ.tail + 1) & pQP->cQ.q_size_minus_1), cid);

	//if((db_head == pQP->cQ.tail) || (((pQP->cQ.tail + 1) & pQP->cQ.q_size_minus_1) != db_head))


	if(cq_db_head != (pQP->cQ.tail + 1))
	{
		nvm_cpl_t *pCmp = &(((nvm_cpl_t *)(pQP->cQ.pEmuQ))[pQP->cQ.tail]);

		verbose = 0;	

		if(pQP->cQ.rollover & 0x1)
		{
			phase = 0;
		}

		BAM_EMU_DEV_DBG_PRINT3(verbose, "emu_tgt_NVMe_loopback() cid = 0x%04x cq_tail = %d rollover = %d\n", cid, pQP->cQ.tail, pQP->cQ.rollover);

	
		pCmp->dword[0] = 0;
		pCmp->dword[1] = 0;
		pCmp->dword[2] = ((uint32_t)pQP->q_number << 16) | (pQP->sQ.head);
		pCmp->dword[3] = phase | cid;
	
	
		BAM_EMU_DEV_DBG_PRINT3(verbose, "emu_tgt_NVMe_loopback() %p val[2] = %x val[3] = %x\n", &pCmp->dword[2] , pCmp->dword[2],  pCmp->dword[3]);

		pQP->cQ.tail++;

		pQP->cQ.tail &= pQP->cQ.q_size_minus_1;

		if(0 == pQP->cQ.tail)
		{
			pQP->cQ.rollover++;
		}
	}
	else
	{
		BAM_EMU_DEV_DBG_PRINT3(BAM_EMU_DBGLVL_ERROR, "emu_tgt_NVMe_loopback() !!!QFULL db_head = %d cq_tail = %d cq_size = %d\n", cq_db_head, pQP->cQ.tail, pQP->cQ.q_size);
		return 1;
	}

	return 0;
}

__device__ inline int emu_tgt_NVMe_execute(bam_emulated_target_control    *pMgtTgtControl, bam_emulated_queue_pair     *pQP, nvm_cmd_t *pCmd, uint32_t cq_db_head)
{
	int verbose = bam_get_verbosity(BAM_EMU_DBGLVL_NONE, BAM_DBG_CODE_PATH_D_NVME_EXE);
	uint16_t cid;
	uint8_t opcode;
	uint64_t   lba;
	int err;
	
	cid = pCmd->dword[0] >> 16;
	opcode = pCmd->dword[0] & 0x7f;
	lba = ((uint64_t)pCmd->dword[11] << 32) | pCmd->dword[10];
	BAM_EMU_DEV_DBG_PRINT3(verbose, "TGT: emu_tgt_NVMe_execute() cid = 0x%04x opcode = 0x%02x lba = %lx\n", cid, opcode, lba);

#ifdef BAM_EMU_TGT_NVME_LOOPBACK
	err = emu_tgt_NVMe_loopback(pMgtTgtControl, pQP, cid, cq_db_head);
#else	
	//deal with other targets
#endif
	return err;
	
}


__device__ inline uint32_t emu_tgt_NVMe_Submit(bam_emulated_target_control    *pMgtTgtControl, bam_emulated_queue_pair     *pQP, uint32_t *pSubmit_count)
{
	int verbose = bam_get_verbosity(BAM_EMU_DBGLVL_NONE, BAM_DBG_CODE_PATH_D_NVME_SUB);
	nvm_cmd_t *pCmd;
	nvm_cmd_t *pQ = &(((nvm_cmd_t *)(pQP->sQ.pEmuQ))[0]);
	int count = 0;
	uint32_t cq_db_head;

	cq_db_head = *pQP->cQ.db;
	
	while(pQP->sQ.head != pQP->sQ.tail)
	{
			
		BAM_EMU_DEV_DBG_PRINT3(verbose, "TGT: emu_tgt_NVMe_Submit(%d) head = %d tail = %d\n", count, pQP->sQ.head, pQP->sQ.tail);

		pCmd = &pQ[pQP->sQ.tail];

		if(emu_tgt_NVMe_execute(pMgtTgtControl, pQP, pCmd, cq_db_head))
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
	int q_number = pQP->q_number;
	uint32_t db_tail;
	int verbose = bam_get_verbosity(BAM_EMU_DBGLVL_NONE, BAM_DBG_CODE_PATH_D_SQ_CHECK);
	

	 nvm_cmd_t* pCmd;

	
//	BAM_EMU_DEV_DBG_PRINT2(verbose, "TGT: emu_tgt_SQ_Check(%d) %d CALL\n", pMgtTgtControl->bRun, q_number);
//	BAM_EMU_DEV_DBG_PRINT1(verbose, "TGT: emu_tgt_SQ_Check(%s) \n", pMgtTgtControl->szName);

	if(pMgtTgtControl->bRun)
	{
		if(pQP->qp_enabled)		
		{
			BAM_EMU_DEV_DBG_PRINT2(verbose, "TGT: emu_tgt_SQ_Check(%d) ENABLED size = %d\n", q_number, pQP->sQ.q_size);
			BAM_EMU_DEV_DBG_PRINT2(verbose, "TGT: emu_tgt_SQ_Check(%p) %d \n", (void *)pQP->sQ.ioaddr, q_number);

			pCmd = (nvm_cmd_t *)pQP->sQ.ioaddr;

			BAM_EMU_DEV_DBG_PRINT2(verbose, "TGT: emu_tgt_SQ_Check() cmd[0] = 0x%08x cmd[1] = 0x%08x \n", pCmd->dword[0], pCmd->dword[1]);
			BAM_EMU_DEV_DBG_PRINT2(verbose, "TGT: emu_tgt_SQ_Check() db = %p  db = %p \n", pQP->cQ.db, pQP->sQ.db);
	
			db_tail = *pQP->sQ.db;

				
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



__global__ void kernel_queueStream(bam_emulated_target_control    *pMgtTgtControl, bam_emulated_queue_pair     *pDevQPairs)
{
	int verbose = bam_get_verbosity(BAM_EMU_DBGLVL_NONE, BAM_DBG_CODE_PATH_D_KER_QSTRM);
	uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	int numQueues;
	uint32_t count = 0;
	uint32_t submit_count = 0;
	bam_emulated_queue_pair 	   *pQP = &pDevQPairs[tid];
	uint32_t cq_db_head;
	
	
	BAM_EMU_DEV_DBG_PRINT1(verbose, "TGT: kernel_queueStream ENTER numQueues = %d\n", numQueues);
#if 0
	uint32_t laneid = lane_id();
	uint32_t bid = blockIdx.x;
	uint32_t smid = get_smid();
	BAM_EMU_DEV_DBG_PRINT1(verbose,"TGT: kernel_queueStream tid = %ld\n", tid);
	BAM_EMU_DEV_DBG_PRINT1(verbose,"TGT: kernel_queueStream laneid = %d\n", laneid);
	BAM_EMU_DEV_DBG_PRINT1(verbose,"TGT: kernel_queueStream bid = %d\n", bid);
	BAM_EMU_DEV_DBG_PRINT1(verbose,"TGT: kernel_queueStream smid = %d\n", smid);
#endif

	while(pMgtTgtControl->bRun)
	{
		if(emu_tgt_SQ_Check(pMgtTgtControl, pQP))
		{
			cq_db_head = emu_tgt_NVMe_Submit(pMgtTgtControl, pQP, &submit_count);

			if(submit_count)
			{
				emu_tgt_CQ_Drain(pMgtTgtControl, pQP, cq_db_head);
			}
		}
		
		if(count >= pMgtTgtControl->nOneShot)
		{			
			break;
		}
		count++;

	}	
	
}

__global__ void dummy_queueStream(bam_emulated_target_control    *pMgtTgtControl, bam_emulated_queue_pair     *pDevQPairs)
{
	int verbose = bam_get_verbosity(BAM_EMU_DBGLVL_INFO, BAM_DBG_CODE_PATH_D_KER_QSTRM);
	int count = 0;
	int ns_sleep_q_enabled = 1000000 * 700; 
	const int display_freq = 3000;
	
	while(pMgtTgtControl->bRun)
	{
		
		__nanosleep(ns_sleep_q_enabled);
		
		if((0 == count) || (0 == (count % display_freq)))
		{
	
				BAM_EMU_DEV_DBG_PRINT1(verbose,"*TGT*: dummy_queueStream count = %d\n", count);
		}
		count++;
	}

	BAM_EMU_DEV_DBG_PRINT1(verbose,"*TGT*: dummy_queueStream EXIT, %d\n", count);

}

//#define BAM_EMU_DOUBLE_CHECK_DEVICE_Q_COPY

static void emulator_update_d_queue(bam_host_emulator *pEmu,  uint16_t q_number, int bEnable = 1)
{
	int	verbose = bam_get_verbosity(BAM_EMU_DBGLVL_NONE, BAM_DBG_CODE_PATH_H_UPDATEDQ);
	bam_emulated_queue_pair aQP;
	uint16_t q_idx = q_number - 1;
	
	
	BAM_EMU_HOST_DBG_PRINT(verbose, "emulator_update_d_queue() q_number = %d bEnable = %d\n", q_number, bEnable);

	if(pEmu->tgt.queuePairs[q_idx].sQ.enabled)
	{
		if(pEmu->tgt.queuePairs[q_idx].cQ.enabled)
		{
			BAM_EMU_HOST_DBG_PRINT(verbose, "emulator_update_d_queue() q_idx = %d BOTH Qs ENABLED, UPDATING\n", q_idx);

			pEmu->tgt.queuePairs[q_idx].qp_enabled = bEnable;
			pEmu->tgt.queuePairs[q_idx].q_number = q_number;
			
			if(bEnable)
			{
				pEmu->tgt.pTgt_control->numQueues++;
			}
			
			cuda_err_chk(cudaMemcpy(&pEmu->tgt.pDevQPairs[q_idx], &pEmu->tgt.queuePairs[q_idx], sizeof(bam_emulated_queue_pair), cudaMemcpyHostToDevice));

			BAM_EMU_HOST_DBG_PRINT(verbose, "*** Async %d\n", 1);
			
#ifdef BAM_EMU_DOUBLE_CHECK_DEVICE_Q_COPY
			cuda_err_chk(cudaMemcpyAsync(&aQP, &pEmu->tgt.pDevQPairs[q_idx], sizeof(bam_emulated_queue_pair), cudaMemcpyDeviceToHost,  pEmu->tgt.queueStream));
			
			BAM_EMU_HOST_DBG_PRINT(verbose, "*** Async %d = %p\n", 3, &pEmu->tgt.pDevQPairs[q_idx] );
			
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
		BAM_EMU_HOST_DBG_PRINT(verbose, "emulator_update_d_queue() q_idx = %d SQ NOT ENABLED YET, SKIPPING UPDATE\n", q_idx);
	}

//	usleep(100000);
	BAM_EMU_HOST_DBG_PRINT(verbose, "emulator_update_d_queue() EXIT q_idx = %d bEnable = %d\n", q_idx, bEnable);

		

}

static void emulator_create_queue(bam_host_emulator *pEmu, uint16_t q_number, uint16_t q_size, uint64_t ioaddr, uint16_t cq)
{
	int	verbose = bam_get_verbosity(BAM_EMU_DBGLVL_NONE, BAM_DBG_CODE_PATH_H_CREATE_Q);
	uint16_t q_idx = q_number - 1;
	
	bam_emulated_queue *pQ;
	
	BAM_EMU_HOST_DBG_PRINT(verbose, "emulator_create_queue() pEmu = %p q_number = %d q_size = %d ioaddr = %p cq = %d\n", pEmu, q_number, q_size, ioaddr, cq);

	if(cq)
	{
		pQ = &pEmu->tgt.queuePairs[q_idx].cQ;
	}
	else
	{
		pQ = &pEmu->tgt.queuePairs[q_idx].sQ;
	}

	pQ->cq = cq;
	pQ->ioaddr = ioaddr;
	pQ->q_size = q_size;
	pQ->q_size_minus_1 = (q_size - 1);
	pQ->q_number = q_number;
	pQ->enabled = 1;
	pQ->target_q_mem = createDma(pEmu->pCtrl, NVM_PAGE_ALIGN(q_size, 1UL << 16), pEmu->cudaDevice);
	pQ->pEmuQ = pQ->target_q_mem->vaddr;
		
	BAM_EMU_HOST_DBG_PRINT(verbose, "emulator_create_queue() target_q_mem = %p\n", pQ->target_q_mem);

	
	
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
	bam_host_emulator * pEmu = (bam_host_emulator *)pvEmu;
	int             heartbeat_usec = 1000 * 1000;
	int             count = 0;
	int 			verbose = bam_get_verbosity(BAM_EMU_DBGLVL_INFO, BAM_DBG_CODE_PATH_H_EMU_THREAD);

	BAM_EMU_HOST_DBG_PRINT(verbose, "launch_emu_target(%d)\n", 0);

#ifdef BAM_EMU_QTHREAD_ONE_SHOT
	pEmu->tgt.pTgt_control->nOneShot = 1;
#else
	kernel_queueStream<<< pEmu->g_size, pEmu->b_size, 0, pEmu->tgt.queueStream >>> (pEmu->tgt.pTgt_control, pEmu->tgt.pDevQPairs);
	//kernel_queueStream<<<pEmu->g_size, pEmu->b_size>>> (pEmu->tgt.pMgtTgtControl, pEmu->tgt.pDevQPairs);
	BAM_EMU_HOST_DBG_PRINT(verbose, "kernel_queueStream RETURN(%d)\n", 0);

#endif


	while(pEmu->bRun)
	{
		if(0 == (count % heartbeat_usec))
		{
			BAM_EMU_HOST_DBG_PRINT(verbose, "emu_thread heartbeat(%d)\n", count);

		}	
#ifdef BAM_EMU_QTHREAD_ONE_SHOT
					kernel_queueStream<<< pEmu->g_size, pEmu->b_size, 0, pEmu->tgt.queueStream >>> (pEmu->tgt.pTgt_control, pEmu->tgt.pDevQPairs);
					//kernel_oneshotStream<<< pEmu->g_size, pEmu->b_size, 0, pEmu->tgt.queueStream >>> (pEmu->tgt.pTgt_control, pEmu->tgt.pDevQPairs);

		cudaStreamSynchronize(pEmu->tgt.queueStream);
#endif

		count++;
	}
	
	BAM_EMU_HOST_DBG_PRINT(verbose, "launch_emu_target(%d) RETURN\n", count);

	return NULL;
}

void * launch_dummy_target(void *pvEmu)
{
	bam_host_emulator * pEmu = (bam_host_emulator *)pvEmu;
	int             heartbeat_sec = 5;
	int             count = 0;
	int 			verbose = bam_get_verbosity(BAM_EMU_DBGLVL_INFO, BAM_DBG_CODE_PATH_H_EMU_THREAD);

//	cuda_err_chk(cudaStreamCreateWithFlags (&pEmu->tgt.tgtStream, (cudaStreamDefault | cudaStreamNonBlocking)));
//	cuda_err_chk(cudaStreamCreate(&pEmu->tgt.queueStream));

	BAM_EMU_HOST_DBG_PRINT(verbose, "launch_emu_target(%d)\n", 0);



	dummy_queueStream<<< pEmu->g_size, pEmu->b_size, 0, pEmu->tgt.tgtStream>>> (pEmu->tgt.pTgt_control, pEmu->tgt.pDevQPairs);

	BAM_EMU_HOST_DBG_PRINT(verbose, "dummy_queueStream RETURN(%d)\n", 0);


	while(pEmu->bRun)
	{
		if(0 == (count % heartbeat_sec))
		{
			BAM_EMU_HOST_DBG_PRINT(verbose, "dummy_thread heartbeat(%d)\n", count);
		}	
		sleep(1);
		count++;
	}

	return NULL;
}



__global__ void dummy_Stream(bam_emulated_target_control     *pMgtTgtControl)
{
	int verbose = bam_get_verbosity(BAM_EMU_DBGLVL_INFO, BAM_DBG_CODE_PATH_D_KER_QSTRM);
	int count = 0;
	int ns_sleep_q_enabled = 1000000 * 800; 
	const int display_freq = 2000;
	
	BAM_EMU_DEV_DBG_PRINT1(verbose,"BAM: dummy_Stream CALL = %d\n", count);
	
	while(pMgtTgtControl->bRun)
	{
		if((0 == count) || (0 == (count % display_freq)))
		{
			BAM_EMU_DEV_DBG_PRINT1(verbose,"BAM: dummy_Stream count = %d\n", count);
		}
		__nanosleep(ns_sleep_q_enabled);
		count++;
	}
}

#define DUMMY_IO_THREAD
void * bam_io_thread(void *pvEmu)
{
	bam_host_emulator *pEmu = (bam_host_emulator *)pvEmu;
//	launch_param *pL = (launch_param *)pvLaunchParam;	
	
	int count = 0;
	
	printf("bam_io_thread(%p)\n", pEmu);
//	cuda_err_chk(cudaStreamCreateWithFlags (&bamStream, (cudaStreamNonBlocking)));
//	cuda_err_chk(cudaStreamCreate(&pEmu->tgt.bamStream));

//CUDA REference
//https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-dynamic-parallelism
#ifdef DUMMY_IO_THREAD
	int limit = 100;	
	dummy_Stream<<< 1, 1>>> (pEmu->tgt.pTgt_control);

	while(pEmu->bRun)
	{
		if(0 == (count % 5))
		{
			//cudaError_t err = cudaStreamQuery(pEmu->tgt.bamStream);
			int err = 0;
			
			printf("bam_io_thread count = %d  StreamQuery  = %d cudaErrorNotReady = %d \n", count, (int)err, (int)cudaErrorNotReady);
		}
		sleep(1);

		count++;

		if(count > limit)
		{
//			break;
		}
	}
#endif

	return NULL;

}



#endif
static void start_emulation_target(bam_host_emulator *pEmu)
{
	int	verbose = bam_get_verbosity(BAM_EMU_DBGLVL_INFO, BAM_DBG_CODE_PATH_H_START_EMU);


	BAM_EMU_HOST_DBG_PRINT(verbose,"start_emulation_target(%p) g_size = %d b_size = %d\n", pEmu, pEmu->g_size, pEmu->b_size);

	cuda_err_chk(cudaStreamCreateWithFlags (&pEmu->tgt.bamStream, (cudaStreamNonBlocking)));
	cuda_err_chk(cudaStreamCreateWithFlags (&pEmu->tgt.queueStream, (cudaStreamNonBlocking)));
	cuda_err_chk(cudaStreamCreateWithFlags (&pEmu->tgt.tgtStream, (cudaStreamNonBlocking)));


	cuda_err_chk(cudaMemcpy(&pEmu->tgt.pDevQPairs[0], &pEmu->tgt.queuePairs[0], sizeof(bam_emulated_queue_pair) * BAM_EMU_MAX_QUEUES, cudaMemcpyHostToDevice));

	
//	printf("*** (%d) cq.db = %p sq.db = %p\n", 0, pEmu->tgt.queuePairs[0].cQ.db, pEmu->tgt.queuePairs[0].sQ.db);
//	printf("*** (%d) cq.db = %p sq.db = %p\n", 1, pEmu->tgt.queuePairs[1].cQ.db, pEmu->tgt.queuePairs[1].sQ.db);


#ifdef BAM_EMU_TARGET_HOST_THREAD
	pEmu->bRun = 1;

#ifdef	BMA_EMU_DISABLE_LAUNCH_EMU_THREAD

#else
	pthread_create(&pEmu->emu_threads[0], NULL, launch_emu_target, pEmu); 
#endif

#ifdef BAM_EMU_LAUNCH_DUMMY_THREAD
	pthread_create(&pEmu->emu_threads[1], NULL, launch_dummy_target, pEmu); 
#endif
#ifdef BAM_EMU_LAUNCH_BAM_DUMMY_IN_START_EMU
	pthread_create(&pEmu->emu_threads[2], NULL, bam_io_thread, pEmu); 
#endif
#else
	try
	{
		h_szStr = (char *)malloc(ssize);
		cuda_err_chk(cudaMalloc((void **) &d_szStr, ssize));
//		cuda_err_chk(cudaStreamCreateWithFlags (&pEmu->tgt.queueStream, (cudaStreamDefault | cudaStreamNonBlocking)));
		cuda_err_chk(cudaStreamCreate (&pEmu->tgt.queueStream));
		
		
	//sync	kernel_queueStream<<<pEmu->g_size, pEmu->b_size>>>(pEmu->tgt.numQueues, d_szStr);
		kernel_queueStream<<<pEmu->g_size, pEmu->b_size, 0, pEmu->tgt.queueStream>>>(pEmu->tgt.pMgtTgtControl, pEmu->tgt.pDevQPairs);

		BAM_EMU_HOST_DBG_PRINT(verbose,"HOST: kernel_queueStream RETURN(%d)\n",0);
		BAM_EMU_HOST_DBG_PRINT(verbose,"HOST: Sleep DONE(%s)\n", h_szStr);

		


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
	int	verbose = bam_get_verbosity(BAM_EMU_DBGLVL_INFO, BAM_DBG_CODE_PATH_H_CLEANUP_EMU);
	int i;
	
	
	BAM_EMU_HOST_DBG_PRINT(verbose,"cleanup_emulator_target(%p) CALL %d queues configured\n", pEmu, pEmu->tgt.pTgt_control->numQueues);

	pEmu->tgt.pTgt_control->bRun = 0;

	for(i = 0; i < BAM_EMU_MAX_QUEUES; i++)
	{
		if(pEmu->tgt.queuePairs[i].qp_enabled)
		{
			BAM_EMU_HOST_DBG_PRINT(verbose,"cleanup_emulator_target() qp = %d ENABLED\n", i); 
			emulator_update_d_queue(pEmu, i, 0);
			
		}
		else
		{
		//	BAM_EMU_HOST_DBG_PRINT(verbose,"cleanup_emulator_target() qp = %d \n", i + 1); 
			
		}
	}

	sleep(1);

	pEmu->bRun = 0;


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

	cudaDeviceReset();

}



static inline nvm_ctrl_t* initializeEmulator(uint32_t ns_id, uint32_t cudaDevice, uint64_t queueDepth, uint64_t numQueues, bam_host_emulator **pEmulator, uint64_t emulationTargetFlags)
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
	
	if(posix_memalign((void **)&pFileMem, 4096, NVM_CTRL_MEM_MINSIZE))
	{
		BAM_EMU_HOST_ASSERT(0);
	}
				
	memset(pFileMem, 0, NVM_CTRL_MEM_MINSIZE);

	BAM_EMU_HOST_DBG_PRINT(verbose,"initializeEmulator() sizeof nvm_ctrl_t = %d pFileMem = %p\n", sizeof(nvm_ctrl_t), pFileMem);


	/* Fix up the Emulators Controller Capabilities here */
	((volatile uint64_t*) pFileMem)[0] = 0x1FFFF;
	
				
	fd = open(temppath, O_CREAT | O_RDWR, S_IRWXU);
			
	int res = write(fd, pFileMem, NVM_CTRL_MEM_MINSIZE);

	BAM_EMU_HOST_DBG_PRINT(verbose,"initializeEmulator() tempath = %s fd = %d res = %d\n", temppath, fd,res);

	close(fd);

	fd = open(temppath, O_RDWR);

	BAM_EMU_HOST_DBG_PRINT(verbose,"initializeEmulator(FD2) tempath = %s fd = %d \n", temppath, fd);

	int status = nvm_ctrl_init(&pCtrl, fd, 1);
	
	BAM_EMU_HOST_DBG_PRINT(verbose,"initializeEmulator() pCtrl = %p status = %d sizeof(bam_host_emulator) %ld\n", pCtrl, status, sizeof(bam_host_emulator));


	err = posix_memalign((void **)&pEmu, 4096, sizeof(bam_host_emulator));
	
	if (err) 
	{
		   throw error(string("Failed to allocate host memory: ") + std::to_string(err));
	}

	BAM_EMU_HOST_ASSERT(pEmu);

	memset(pEmu, 0, sizeof(bam_host_emulator));
	
	
	BAM_EMU_HOST_DBG_PRINT(verbose, "initializeEmulator(%p) pCtrl\n", pCtrl);

	strcpy(pEmu->name, "BAM_EMULATOR");

	*pEmulator = pEmu;
	
	pCtrl->page_size = 4096;
	pCtrl->emulated = 1;
	pCtrl->pvEmu = pEmu;
	pCtrl->fnEmuCallout = emulator_rpc_callout;

	
	pEmu->emulationTargetFlags = emulationTargetFlags;


#define EMU_GRID_SIZE_DEFAULT 32

	if(numQueues <= EMU_GRID_SIZE_DEFAULT)
	{
		pEmu->g_size = 1;
		pEmu->b_size = numQueues;

	}
	else
	{
		pEmu->g_size = ((numQueues - 1) / EMU_GRID_SIZE_DEFAULT) + 1;
		pEmu->b_size = EMU_GRID_SIZE_DEFAULT;
		
	}
	printf("numQueues = %d g_size = %d b_size = %d EMU_GRID_SIZE_DEFAULT=%d\n", numQueues, pEmu->g_size, pEmu->b_size, EMU_GRID_SIZE_DEFAULT); 

	pEmu->pCtrl = pCtrl;
	pEmu->cudaDevice = cudaDevice;


	cuda_err_chk(cudaMallocHost(&pEmu->tgt.queuePairs, sizeof(bam_emulated_queue_pair) * BAM_EMU_MAX_QUEUES, 0));
	
	cuda_err_chk(cudaMallocManaged(&pEmu->tgt.pTgt_control, sizeof(bam_emulated_target_control)));
	
	pEmu->tgt.pTgt_control->numQueues = 0;
	pEmu->tgt.pTgt_control->bRun = 1;
	pEmu->tgt.pTgt_control->bDone = 0;

	strcpy(pEmu->tgt.pTgt_control->szName, "george");
		
	 
	
	pEmu->tgt.d_queue_q_mem = createDma(pEmu->pCtrl, NVM_PAGE_ALIGN(qall_size, 1UL << 16), pEmu->cudaDevice);
	pEmu->tgt.pDevQPairs = (bam_emulated_queue_pair *)pEmu->tgt.d_queue_q_mem.get()->vaddr;

	BAM_EMU_HOST_DBG_PRINT(verbose, "initializeEmulator() pEmu->tgt.pDevQPairs = %p &tgt.queuePairs = %p, qall_size = %ld\n", pEmu->tgt.pDevQPairs, &pEmu->tgt.queuePairs, qall_size);

	cuda_err_chk(cudaMemcpy(pEmu->tgt.pDevQPairs, &pEmu->tgt.queuePairs, qall_size, cudaMemcpyHostToDevice));

	pEmu->tgt.d_target_control_mem = createDma(pEmu->pCtrl, NVM_PAGE_ALIGN(sizeof(bam_emulated_target_control), 1UL << 16), pEmu->cudaDevice);


#ifdef	BAM_EMU_START_EMU_POST_Q_CONFIG
	BAM_EMU_HOST_DBG_PRINT(verbose, "initializeEmulator() POSTPONING start_emulation_target() until after QConfig = %d (BAM_EMU_START_EMU_POST_Q_CONFIG)\n", 0);
#else
	start_emulation_target(pEmu);
#endif

	return pCtrl;
}




#endif

