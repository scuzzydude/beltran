#ifndef __EMU_AGGREGATION_H
#define __EMU_AGGREGATION_H

#include "emu_ctrl.h"

//*******************************************************************************************************
//** Model Specific Data Structures
//*******************************************************************************************************
typedef struct _emu_aggregation_model
{
	int num_controllers;

	void **ppvHostCtrls;
	void **ppvDevCtrls;

    BufferPtr d_ctrls_buff;

	void *pvDevAqc;

    BufferPtr d_aqc_buff;


} emu_aggregation_model;

typedef union _agg_context
{
	storage_next_emuluator_context context;
	struct
	{
		void *pvCmd;                       //0
		uint64_t start_ns;                 //1  
		uint64_t done_ns;                  //2
		union _agg_context *pNext;         //3  
		union _agg_context *pPrev;         //4
		uint16_t cid;
		uint16_t ctrl;
		uint16_t qidx;
		uint16_t rsv16;                    //5
		uint64_t rsv64a;                   //6
		uint64_t rsv64b;                   //7
		
	} agg_context;
		
} agg_context;

typedef struct
{
	agg_context *pHead;
	agg_context *pTail;
	uint32_t     count;
	uint32_t     total_submitted;
} agg_queue_control;

#define MAX_AGG_QUEUES 16

typedef struct
{
	agg_queue_control aqc[MAX_AGG_QUEUES];

} agg_control;

//*******************************************************************************************************
//** Host Functions  
//*******************************************************************************************************

uint32_t emu_model_aggregation_private_init(bam_host_emulator *pEmu, bam_emu_target_model *pModel) 
{
	int	verbose = bam_get_verbosity(BAM_EMU_DBGLVL_INFO, BAM_DBG_CODE_PATH_H_INIT_AGG);
	int num_controllers = (pEmu->emulationTargetFlags & BAM_EMU_TARGET_AGG_CONT_BITMASK) >> BAM_EMU_TARGET_AGG_CONT_BITSHIFT;
	size_t size = sizeof(emu_aggregation_model);
	emu_aggregation_model *pAggModel;
	char devPath[32];
	EmuController *pCntl;
	uint32_t queueDepth = 16;
	uint32_t numberOfQueues = 1;
	
		
	
	BAM_EMU_HOST_DBG_PRINT(verbose, "emu_model_aggregation_private_init(pEmu = %p, pModel = %p) num_controllers = %d size = %ld\n", pEmu, pModel, num_controllers, size);

	pModel->pvHostPrivate = malloc(size);

	BAM_EMU_HOST_ASSERT(pModel->pvHostPrivate);

	pAggModel = (emu_aggregation_model *)pModel->pvHostPrivate;

	pAggModel->num_controllers = num_controllers;

	pAggModel->ppvHostCtrls = (void **)malloc(sizeof(EmuController *) * num_controllers);

	BAM_EMU_HOST_ASSERT(pAggModel->ppvHostCtrls);
	
	pAggModel->d_ctrls_buff = createBuffer(num_controllers * sizeof(EmuController*), pEmu->cudaDevice);

	pAggModel->ppvDevCtrls = (void **)pAggModel->d_ctrls_buff.get();


	pModel->d_model_private = createBuffer(size, pEmu->cudaDevice);

	pModel->pvDevPrivate = pModel->d_model_private.get();

	

	            
	for (size_t i = 0 ; i < num_controllers; i++)
	{
		sprintf(devPath, "/dev/libnvm%d", i);


		
		pCntl = new EmuController(devPath, 1, pEmu->cudaDevice, queueDepth, numberOfQueues);

		pAggModel->ppvHostCtrls[i] = (void *)pCntl;


		BAM_EMU_HOST_DBG_PRINT(verbose, "emu_model_aggregation_private_init() %s d_ctrl_ptr = %p ppDevCtrls = %p sizeof(EmuController*) = %ld\n", devPath, pCntl->d_ctrl_ptr, pAggModel->ppvDevCtrls, sizeof(EmuController*));

		
		BAM_EMU_HOST_DBG_PRINT(verbose, "emu_model_aggregation_private_init() %p %p %ld\n", pAggModel->ppvDevCtrls+i, pCntl->d_ctrl_ptr, sizeof(EmuController*));
		
		cuda_err_chk(cudaMemcpy(pAggModel->ppvDevCtrls+i, &pCntl->d_ctrl_ptr, sizeof(EmuController*), cudaMemcpyHostToDevice));


	}
	
	
	int aggqsize = pEmu->tgt.pTgt_control->numEmuThreads * sizeof(agg_control);

	BAM_EMU_HOST_DBG_PRINT(verbose, "emu_model_aggregation_private_init() numEmuThread = %d aggqsize = %d\n", pEmu->tgt.pTgt_control->numEmuThreads, aggqsize);

	pAggModel->d_aqc_buff = createBuffer(aggqsize, pEmu->cudaDevice);
	pAggModel->pvDevAqc = pAggModel->d_aqc_buff.get();


	cuda_err_chk(cudaMemcpy(pModel->pvDevPrivate, pModel->pvHostPrivate, size, cudaMemcpyHostToDevice));


	

	return 0;
}


//*******************************************************************************************************
//** Device Functions  
//*******************************************************************************************************

__device__ void emu_model_agg_set_agg_control_ptr(bam_emulated_queue_pair *pQP, void *pvModel, int queue_per_thread, uint32_t qidx)
{
	int	verbose = bam_get_verbosity(BAM_EMU_DBGLVL_INFO, BAM_DBG_CODE_PATH_D_AGG);

	if(1 == queue_per_thread)
	{
		emu_aggregation_model *pAggModel = (emu_aggregation_model *)pvModel;
		agg_control *pAggControlBase = (agg_control *)pAggModel->pvDevAqc;
		agg_control *pAggControl = (pAggControlBase + qidx);

		pQP->pvThreadContext = pAggControl;	

		BAM_EMU_DEV_DBG_PRINT4(verbose, " emu_model_agg_set_agg_control_ptr(pAggModel = %p pAggControlBase = %p, pAggControl = %p, pvThreadContext = %p)\n", pAggModel, pAggControlBase, pAggControl, pQP->pvThreadContext);

	}
	else
	{
		BAM_EMU_DEV_DBG_PRINT1(BAM_EMU_DBGLVL_ERROR, " emu_model_agg_set_agg_control_ptr() queue_per_thread != 1 == %d \n", queue_per_thread);
		BAM_EMU_DEVICE_ASSERT_DBG(0);
	}

}


// MicrochipChatbot: Add a new agg_context node to the tail of the list
// Code generated by MCHP Chatbot
// Add a new agg_context node to the tail of the queue
__device__ void add_agg_context_tail(agg_queue_control *queue, agg_context *newNode)
{
    if (!queue || !newNode) return;

    newNode->agg_context.pNext = NULL;
    newNode->agg_context.pPrev = NULL;

    if (queue->pTail == NULL) {
        // List is empty
        queue->pHead = queue->pTail = newNode;
    } else {
        // Append to tail
        newNode->agg_context.pPrev = queue->pTail;
        queue->pTail->agg_context.pNext = newNode;
        queue->pTail = newNode;
    }
    queue->count++;
}

// MicrochipChatbot: Add a new agg_context node to the tail of the list
// Code generated by MCHP Chatbot
// Find a node by cid and remove it from the queue
__device__ agg_context* find_and_remove_agg_context_by_cid(agg_queue_control *queue, uint16_t cid)
{
    if (!queue) return NULL;

    agg_context *curr = queue->pHead;
    while (curr) {
        if (curr->agg_context.cid == cid) {
            // Remove from list
            if (curr->agg_context.pPrev)
                curr->agg_context.pPrev->agg_context.pNext = curr->agg_context.pNext;
            else
                queue->pHead = curr->agg_context.pNext;

            if (curr->agg_context.pNext)
                curr->agg_context.pNext->agg_context.pPrev = curr->agg_context.pPrev;
            else
                queue->pTail = curr->agg_context.pPrev;

            curr->agg_context.pNext = NULL;
            curr->agg_context.pPrev = NULL;
            if (queue->count > 0)
            {
				queue->count--;
            }
			else
			{
				BAM_EMU_DEV_DBG_PRINT1(BAM_EMU_DBGLVL_ERROR, "find_and_remove_agg_context_by_cid() QUEUE COUNT already zero (%p)\n", queue);
				BAM_EMU_DEVICE_ASSERT_DBG(0);
			}
            return curr;
        }
        curr = curr->agg_context.pNext;
    }
    return NULL; // Not found
}



__device__ inline void emu_dump_sn_context(storage_next_emuluator_context *pContext)
{
	int	verbose = bam_get_verbosity(BAM_EMU_DBGLVL_INFO, BAM_DBG_CODE_PATH_D_AGG);

	uint32_t *dw = &pContext->pCmd->nvme_cmd.dword[0];

	BAM_EMU_DEV_DBG_PRINT4(verbose, "NVME[00:03] 0x%08x %08x %08x %08x\n", dw[0], dw[1], dw[2], dw[3]);
	BAM_EMU_DEV_DBG_PRINT4(verbose, "NVME[04:07] 0x%08x %08x %08x %08x\n", dw[4], dw[5], dw[6], dw[7]);
	BAM_EMU_DEV_DBG_PRINT4(verbose, "NVME[08:11] 0x%08x %08x %08x %08x\n", dw[8], dw[9], dw[10], dw[11]);
	BAM_EMU_DEV_DBG_PRINT4(verbose, "NVME[12:15] 0x%08x %08x %08x %08x\n", dw[12], dw[13], dw[14], dw[15]);


}

__device__  agg_queue_control* emu_model_agg_aqc(void ** ppvThreadContext, uint16_t qidx)
{
	int	verbose = bam_get_verbosity(BAM_EMU_DBGLVL_INFO, BAM_DBG_CODE_PATH_D_AGG);
	agg_control *pAgg = (agg_control *)*ppvThreadContext; 
	agg_queue_control *pAqc;

	BAM_EMU_DEV_DBG_PRINT3(verbose, "emu_model_agg_aqc(ppvThreadContext = %p pAgg = %p qidx = %d)\n", ppvThreadContext, pAgg, qidx);
	

	BAM_EMU_DEVICE_ASSERT_DBG(pAgg);
	
	BAM_EMU_DEVICE_ASSERT_DBG(qidx < MAX_AGG_QUEUES);

	pAqc =  &pAgg->aqc[qidx];

	BAM_EMU_DEV_DBG_PRINT4(verbose, "emu_model_agg_aqc(pAqc = %p, count = %ld tail = %p head = %p)\n", pAqc, pAqc->count, pAqc->pTail, pAqc->pHead);

	return pAqc;
}


typedef ulonglong4 agg_copy_type;

__device__ int gLive = 0;
__device__ inline int emu_model_agg_sq_enqueue(emu_aggregation_model *pAggModel, uint16_t cidx, uint16_t qidx, storage_next_emuluator_context *pContext)
{
	int	verbose = bam_get_verbosity(BAM_EMU_DBGLVL_INFO, BAM_DBG_CODE_PATH_D_AGG);
	EmuController *pCtrl;
	EmuQueuePair *pQP;
	nvm_cmd_t *pCmd;
	bool    bRingDoorbell = true;
	
	pCtrl = (EmuController *)pAggModel->ppvDevCtrls[cidx];

	
	BAM_EMU_DEV_DBG_PRINT3(verbose, "emu_model_agg_sq_enqueue(pCtrl=%p, ppDevCtrl = %p,%p)\n", pCtrl, pAggModel->ppvDevCtrls, pContext);
	
	pQP = pCtrl->d_qps + qidx;
	
	BAM_EMU_DEV_DBG_PRINT2(verbose, "emu_model_agg_sq_enqueue(pQP=%p, sq_mem_size = %d )\n", pQP, pQP->sq_mem_size);


	pCmd = emu_ctrl_sq_enqueue(&pQP->sq);

	if(pCmd)
	{
		agg_copy_type *queue_loc = (agg_copy_type *)pCmd;
		agg_copy_type *cmd_loc = (agg_copy_type *)pContext->pCmd;
		
		BAM_EMU_DEV_DBG_PRINT3(verbose, "emu_model_agg_sq_enqueue(pCmd=%p dword[0] = 0x%08x cid = 0x%04x)\n", pCmd, pCmd->dword[0], ((pCmd->dword[0] >> 16) & 0xFFFF));

#pragma unroll
		for (uint32_t i = 0; i < 64/sizeof(agg_copy_type); i++) 
		{
			queue_loc[i] = cmd_loc[i];
		}

		if(bRingDoorbell)
		{
			emu_ctrl_nvm_sq_submit(&pQP->sq);
		}

		__nanosleep(1000000);
		gLive = 1;

	}
	else
	{
		return EMU_SUBMIT_QFULL;
	}

	
	

	return EMU_SUBMIT_GOOD;
}




__device__ inline int emu_model_aggregation_submit(bam_emu_target_model *pModel, storage_next_emuluator_context *pContext, void ** ppvThreadContext)
{
	emu_aggregation_model *pAggModel;
	
	agg_context *pAggContext = (agg_context *)pContext;
	int	verbose = bam_get_verbosity(BAM_EMU_DBGLVL_INFO, BAM_DBG_CODE_PATH_D_AGG);
	BAM_EMU_DEV_DBG_PRINT3(verbose, "emu_model_aggregation_submit(%p, %p, %p)\n", pModel, pContext, ppvThreadContext);
	int ret;
	uint16_t cidx = 0;
	uint16_t qidx = 0;
	emu_dump_sn_context(pContext);

	//TODO: Map it to drive and LBA

	pAggModel = (emu_aggregation_model *)pModel->pvDevPrivate;

	ret	= emu_model_agg_sq_enqueue(pAggModel, cidx, qidx, pContext);

	if(EMU_SUBMIT_GOOD == ret)
	{
		add_agg_context_tail(emu_model_agg_aqc(ppvThreadContext, qidx), pAggContext);
	}


	return ret;	

}

__device__ inline storage_next_emuluator_context * emu_model_aggregation_cull(bam_emu_target_model *pModel, void **ppvThreadContext)
{
	//We only need this to keep a list for error handling	
	//**ppLatListHead = (latency_context **) ppvThreadContext;
	//storage_next_emuluator_context *pTemp;
	int verbose = bam_get_verbosity(BAM_EMU_DBGLVL_INFO, BAM_DBG_CODE_PATH_D_LATENCY);
	EmuController *pCtrl;
	EmuQueuePair *pQP;
	uint16_t cidx = 0;
	uint16_t qidx = 0;
	nvm_cpl_t *pCpl;
	agg_context *pAggContext = NULL;
	emu_aggregation_model *pAggModel;

	pAggModel = (emu_aggregation_model *)pModel->pvDevPrivate;
	

	pCtrl = (EmuController *)pAggModel->ppvDevCtrls[cidx];
	
	pQP = pCtrl->d_qps + qidx;


	BAM_EMU_DEVICE_ASSERT_DBG(pModel);
	//BAM_EMU_DEVICE_ASSERT_DBG(ppvThreadContext);


	BAM_EMU_DEV_DBG_PRINT1(verbose, "emu_model_latency_cull(%p, %p)\n", pModel);
	pCpl =  emu_ctrl_nvm_cq_cull(&pQP->cq);

	if(pCpl)
	{
		
		uint32_t cid = pCpl->dword[3] & 0xFFFF;

		BAM_EMU_DEV_DBG_PRINT2(verbose, "emu_model_latency_cull(%p) COMPLETION cid 0x%04x\n", pCpl, cid);

		pAggContext = find_and_remove_agg_context_by_cid(emu_model_agg_aqc(ppvThreadContext, qidx), cid);

		BAM_EMU_DEV_DBG_PRINT2(verbose, "emu_model_latency_cull(%p) ContextFound cid 0x%04x\n", pAggContext, cid);

		BAM_EMU_DEVICE_ASSERT(pAggContext);
		

	}
	else
	{
		BAM_EMU_DEV_DBG_PRINT1(verbose, "emu_model_latency_cull(%p) NO COMPLETION!!! \n", pCpl);

		if(gLive)
		{
			gLive++;
		}
		if(gLive > 100)
		{
			BAM_EMU_DEV_DBG_PRINT1(verbose, "emu_model_latency_cull(%p) gLive = %d!!! \n", gLive);
			BAM_EMU_DEVICE_ASSERT(0);

		}
	}

	return (storage_next_emuluator_context *)pAggContext;
}



#endif

