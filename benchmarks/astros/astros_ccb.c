#include "astros.h"

int gDqDebug = ASTROS_DBGLVL_NONE;

#ifdef ASTROS_LINUX_KERNEL_LISTS
void astros_ccb_list_init(ccb_list *pL, int count, char *szName)
{
	ASTROS_ASSERT(pL);

	pL->count = 0;
	pL->szName = szName;
	pL->verbose = 0;

	INIT_LIST_HEAD(&pL->queue_list);


}

void astros_ccb_list_free(ccb_list *pL)
{
	int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);
	ASTROS_ASSERT(pL);

	ASTROS_DBG_PRINT(verbose, "astros_ccb_list_free(%s) empty = %d\n", pL->szName, list_empty(&pL->queue_list) );
}


void astros_ccb_list_enqueue(ccb_list *pL, ccb *pCCB)
{
	ASTROS_ASSERT(pL);
	ASTROS_ASSERT(pCCB);
	

	list_add_tail(&pCCB->queue_list,&pL->queue_list);

}



ccb * astros_ccb_list_dequeue(ccb_list *pL)
{
	ccb *pCCB = NULL;
	int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);

	ASTROS_ASSERT(pL);


	if(list_empty(&pL->queue_list))
	{
		ASTROS_DBG_PRINT(verbose, "astros_ccb_list_dequeue(%s) EMPTY\n", pL->szName);
	}
	else
	{
		int cpu;


		if(pL->verbose)
		{
			verbose = gDqDebug; //astros_get_verbosity(ASTROS_DBGLVL_NONE);
			
			ASTROS_GET_CPU(&cpu);
		}	


		
		ASTROS_DBG_PRINT(verbose, "astros_ccb_list_dequeue(%s) %d list_first_entry = %px\n", pL->szName, cpu, pL);
				
		pCCB = list_entry(pL->queue_list.next, struct __command_control_block, queue_list);

		ASTROS_DBG_PRINT(verbose, "astros_ccb_list_dequeue(%s) %d pCCB = %px\n", pL->szName, cpu, pCCB);

		ASTROS_ASSERT(pCCB);

		ASTROS_DBG_PRINT(verbose, "astros_ccb_list_dequeue(%s) %d pCCB = %px idx = %d queue_list=%px\n", pL->szName, cpu, pCCB, pCCB->idx, &pCCB->queue_list);

		ASTROS_DBG_PRINT(verbose, "astros_ccb_list_dequeue(%s) %d next=%px prev=%px\n", pL->szName, cpu, pCCB->queue_list.next, pCCB->queue_list.prev);

		list_del(pL->queue_list.next);
		
		ASTROS_DBG_PRINT(verbose, "astros_ccb_list_dequeue(%s) %d pCCB = %px idx = %d LIST_DEL\n", pL->szName, cpu, pCCB, pCCB->idx);

	}

	
	return pCCB;
}





#else


#ifdef ASTROS_SIMPLE_LIST

void astros_ccb_list_init(ccb_list *pL, int count, char *szName)
{
	int size = sizeof(ccb *) * ASTROS_MAX_CCBS;
	pL->count = 0;
	pL->szName = szName;

	pL->ccbA = ASTROS_ALLOC(64, size);

	ASTROS_ASSERT(pL->ccbA);

}
void astros_ccb_list_free(ccb_list *pL)
{
	int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);

	ASTROS_DBG_PRINT(verbose, "astros_ccb_list_free count = %d\n", pL->count);

	if(pL->count != 0)
	{
		ASTROS_DBG_PRINT(ASTROS_DBGLVL_ERROR, "astros_ccb_list_free NOT ZERO count = %d NAME:%s\n", pL->count,pL->szName);
	}

	ASTROS_FREE(pL->ccbA);
	
}


void astros_ccb_list_enqueue(ccb_list *pL, ccb *pCCB)
{
	ASTROS_ASSERT(pCCB);

	ASTROS_ASSERT(pL->count < ASTROS_MAX_CCBS);
	
	pL->ccbA[pL->count] = pCCB;

	pL->count++;

}


ccb * astros_ccb_list_dequeue(ccb_list *pL)
{
	ccb *pCCB = NULL;

	if(pL->count > 0)
	{
		pL->count--;
		pCCB = pL->ccbA[pL->count];
		pL->ccbA[pL->count] = NULL;	
	}

	if(pCCB)
	{
		ASTROS_DBG_PRINT(ASTROS_DBGLVL_NONE, "_list_dequeue pCCB->idx = %d, count = %d\n", pCCB->idx, pL->count);
	}
	
	return pCCB;
}

#else
void astros_ccb_list_init(ccb_list *pL, int count, char *szName)
{
	pL->pTail = NULL;
	pL->pHead = NULL;
	pL->count = 0;
	pL->szName = szName;

}
void astros_ccb_list_free(ccb_list *pL)
{
	int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);

	ASTROS_DBG_PRINT(verbose, "astros_ccb_list_free count = %d pHead = %px pTail = %px\n", pL->count, pL->pHead, pL->pTail);

	if(pL->count != 0)
	{
		ASTROS_DBG_PRINT(ASTROS_DBGLVL_ERROR, "astros_ccb_list_free NOT ZERO count = %d pHead = %px pTail = %px NAME:%s\n", pL->count, pL->pHead, pL->pTail, pL->szName);
	}

	//ASTROS_ASSERT(pL->count == 0);
	
}


void astros_ccb_list_enqueue(ccb_list *pL, ccb *pCCB)
{
	ASTROS_ASSERT(pCCB);

	pCCB->pListNext = NULL;

	if(NULL == pL->pHead)
	{

		if(pL->count != 0)
		{
			ASTROS_DBG_PRINT(ASTROS_DBGLVL_ERROR, "astros_ccb_list_enqueue(%s) HEAD = EMPTY but pL->count = %d : %px %px\n", pL->szName, pL->count, pL->pHead, pL->pTail); 
		}
		ASTROS_ASSERT(0 == pL->count);

		pL->pHead = pCCB;
		pL->pTail = pCCB;
	
	}
	else
	{
		pL->pTail->pListNext = pCCB;
		pL->pTail = pCCB;
		
	}
	pL->count++;

	ASTROS_DBG_PRINT(ASTROS_DBGLVL_NONE, "_list_enqueue pCCB->idx = %d, count = %d\n", pCCB->idx, pL->count);

	

}


ccb * astros_ccb_list_dequeue(ccb_list *pL)
{
	ccb *pCCB;

	pCCB = pL->pHead;

	if(pCCB)
	{
		pL->pHead = pCCB->pListNext;

		if(NULL == pL->pHead)
		{
			if(pL->count != 1)
			{

				ASTROS_DBG_PRINT(ASTROS_DBGLVL_ERROR, "astros_ccb_list_dequeue(%s) goint to EMPTY but count = %d pTail = %px pCCB = %px idx =%d\n",
					pL->szName, pL->count, pL->pTail, pCCB, pCCB->idx);
				//ASTROS_ASSERT(0);
			}
					

			pL->pTail = NULL;
		}

		pCCB->pListNext = NULL;
		pL->count--;

		

	}

	if(pCCB)
	{
		ASTROS_DBG_PRINT(ASTROS_DBGLVL_NONE, "_list_dequeue pCCB->idx = %d, count = %d\n", pCCB->idx, pL->count);
	}
	
	return pCCB;
}
#endif
#endif


void astros_ccb_list_init_locked(astros_ccb_list_locked *pLockedFifo)
{

	astros_ccb_list_init(&pLockedFifo->ccb_list, 0, "ccb_list_locked");

	pLockedFifo->ccb_list.verbose = 1;
	
	ASTROS_SPINLOCK_INIT(pLockedFifo->ccb_list_lock);
}


void astros_ccb_list_free_locked(astros_ccb_list_locked *pLockedFifo)
{

	astros_ccb_list_free(&pLockedFifo->ccb_list);
	
	ASTROS_SPINLOCK_DESTROY(pLockedFifo->ccb_list_lock);
}


void astros_ccb_list_enqueue_locked (astros_ccb_list_locked *pLockedFifo, ccb *pCCB)
{
	unsigned long flags;

	ASTROS_UNUSED(flags);

	ASTROS_SPINLOCK_IRQSAVE(pLockedFifo->ccb_list_lock, flags);

	astros_ccb_list_enqueue(&pLockedFifo->ccb_list, pCCB);

	ASTROS_SPINUNLOCK_IRQSAVE(pLockedFifo->ccb_list_lock, flags);
	

}


ccb * astros_ccb_list_dequeue_locked (astros_ccb_list_locked *pLockedFifo)
{
	ccb *pCCB = NULL;
//	unsigned long flags;
#ifdef ASTROS_LINUX_KERNEL_LISTS
	ccb *paCCB;
	ccb *pbCCB;

	int cpu;
	int count = 0;
	
	ASTROS_GET_CPU(&cpu);
#endif	
	ASTROS_ASSERT(pLockedFifo);

	gDqDebug = ASTROS_DBGLVL_INFO;

	//ASTROS_SPINLOCK_IRQSAVE(pLockedFifo->ccb_list_lock, flags);

//	if(spin_trylock(&pLockedFifo->ccb_list_lock))
	ASTROS_SPINLOCK_LOCK(pLockedFifo->ccb_list_lock);
	{


#ifdef ASTROS_LINUX_KERNEL_LISTS
	list_for_each_entry_safe(paCCB, pbCCB, &pLockedFifo->ccb_list.queue_list, queue_list)
	{
		ASTROS_DBG_PRINT(ASTROS_DBGLVL_INFO, "%d]dque = pCCB(%d) cpu = %d pCCB=%px\n", count, cpu, paCCB->idx, paCCB);\
		count++;
	}
	if(count)
	{
		ASTROS_DBG_PRINT(ASTROS_DBGLVL_INFO, "dque = DONE cpu = %d count = %d\n", cpu, count);
	}
#endif


		pCCB = astros_ccb_list_dequeue(&pLockedFifo->ccb_list);

	}
	ASTROS_SPINLOCK_UNLOCK(pLockedFifo->ccb_list_lock);

	gDqDebug = ASTROS_DBGLVL_NONE;

	return pCCB;
}







#if (CCB_Q_TYPE == CCB_Q_TYPE_LOCKED)

void astros_ccb_queue_init(ccb_queue *pQ, int count, char *szName)
{
	pQ->ppCCBqueue[0] = ASTROS_ALLOC(64, count * sizeof(ccb *));
	pQ->ppCCBqueue[1] = ASTROS_ALLOC(64, count * sizeof(ccb *));

	ASTROS_ASSERT(pQ->ppCCBqueue[0]);
	ASTROS_ASSERT(pQ->ppCCBqueue[1]);

	pQ->qidx[0] = 0;
	pQ->qidx[1] = 0;
	pQ->beat = 0;

	pQ->max_count = count;
	pQ->szName = szName;

	ASTROS_SPINLOCK_INIT(pQ->ccbqlock);
	

}


void astros_ccb_queue_free(ccb_queue *pQ)
{
	ASTROS_FREE(pQ->ppCCBqueue[0]);
	ASTROS_FREE(pQ->ppCCBqueue[1]);

	ASTROS_SPINLOCK_DESTROY(pQ->ccbqlock);
	
}


void astros_ccb_queue_enqueue(ccb_queue *pQ, ccb *pCCB)
{
//    int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);
	int beat;
	unsigned long flags;
	

	ASTROS_SPINLOCK_IRQSAVE(pQ->ccbqlock, flags);

	beat = pQ->beat & 1;

	pQ->ppCCBqueue[beat][pQ->qidx[beat]] = pCCB;

	pQ->qidx[beat]++;
	
	ASTROS_SPINUNLOCK_IRQSAVE(pQ->ccbqlock, flags);

	ASTROS_DBG_PRINT(ASTROS_DBGLVL_NONE, "_ccb_enqueue pCCB->idx = %d, beat = %d qidx= %d\n", pCCB->idx, pQ->beat, pQ->qidx[beat]);


	

}

ccb ** astros_ccb_queue_multi_dequeue(ccb_queue *pQ, int *pCount)
{
	unsigned long flags;
	ccb ** ppCcb;
//    int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);
	int beat;

	ASTROS_SPINLOCK_IRQSAVE(pQ->ccbqlock, flags);
	beat = pQ->beat & 1;

	ppCcb = pQ->ppCCBqueue[beat];

	*pCount = pQ->qidx[beat];

	if(pQ->qidx[beat] > 0)
	{

		pQ->beat++;
	
		beat = pQ->beat & 1;

		pQ->qidx[beat] = 0;

	}

	ASTROS_SPINUNLOCK_IRQSAVE(pQ->ccbqlock, flags);

	if(*pCount)
	{
		ASTROS_DBG_PRINT(ASTROS_DBGLVL_NONE, "_multi_dequeue beat = %d count = %d\n", pQ->beat, *pCount);	
	}
	
	return ppCcb;
}



#else
void astros_ccb_queue_free(ccb_queue *pQ)
{
	ASTROS_FREE(pQ->ppCCBqueue);
}


void astros_ccb_queue_init(ccb_queue *pQ, int count, char *szName)
{
	pQ->ppCCBqueue = ASTROS_ALLOC(64, count * sizeof(ccb *));

	ASTROS_ASSERT(pQ->ppCCBqueue);

	pQ->max_count = count;
	pQ->tail = -1;
	pQ->head = 0;
	ASTROS_SET_ATOMIC(pQ->cur_count, 0);
	pQ->szName = szName;
	pQ->sig = 0xBABA0001;
	

}

inline int astros_ccb_queue_full(ccb_queue *pQ)
{
	if(pQ->max_count == ASTROS_GET_ATOMIC(pQ->cur_count))
	{
		return 1;
	}
	else
	{
		return 0;
	}

}

inline int astros_ccb_queue_empty(ccb_queue *pQ)
{
	if(ASTROS_GET_ATOMIC(pQ->cur_count) == 0)
	{
		return 1;
	}
	else
	{
		return 0;
	}
}

void astros_ccb_queue_enqueue(ccb_queue *pQ, ccb *pCCB)
{
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);


	if(astros_ccb_queue_full(pQ))
	{

#ifdef ALTUVE_ZOMBIE
		int cpu;
		aioengine  *pEngine;

		ASTROS_GET_CPU(&cpu);

		pEngine = pCCB->pvEngine;

		
		ASTROS_DBG_PRINT(ASTROS_DBGLVL_ERROR, "ccb_eq :: cpu = %d pCCB->idx = %d pEngine->cpu = %d\n", cpu, pCCB->idx, pEngine->cpu);

#endif


		ASTROS_DBG_PRINT(ASTROS_DBGLVL_ERROR, "ccb_eq(%s): %px idx = %d pi = %d ci = %d cur_count = %d max_count = %d\n", 
			pQ->szName, pCCB, pCCB->idx, pQ->tail, pQ->head, ASTROS_GET_ATOMIC(pQ->cur_count) , pQ->max_count);

		//TODO: wf zombie hitting this but not sure why.  test continues because all ccbs are gathered after batter is done from a different list 
		ASTROS_ASSERT(0);
		
	}

	pQ->tail = (pQ->tail + 1) % pQ->max_count;
	
	pQ->ppCCBqueue[pQ->tail] = pCCB;

	ASTROS_INC_ATOMIC(&pQ->cur_count);
	
	
	ASTROS_DBG_PRINT(verbose, "ccb_eq(%s): %px idx = %d pi = %d ci = %d\n", pQ->szName, pCCB, pCCB->idx, pQ->tail, pQ->head);
	
	

}

ccb * astros_ccb_queue_dequeue(ccb_queue *pQ)
{
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);

	ccb *pCCB = NULL;

	ASTROS_DBG_PRINT(verbose, "ccb_dq(%s): head = %d tail = %d\n", pQ->szName, pQ->head, pQ->tail);

	if(astros_ccb_queue_empty(pQ))
	{

	}
	else
	{

		pCCB = pQ->ppCCBqueue[pQ->head];

		pQ->head = (pQ->head + 1)  % pQ->max_count;
		
		ASTROS_DEC_ATOMIC(&pQ->cur_count);
		

		ASTROS_ASSERT(pCCB);
		
	
		ASTROS_DBG_PRINT(verbose, "ccb_dq(%s): %px idx = %d \n", pQ->szName, pCCB, pCCB->idx);

	}
	return pCCB;
}
#endif

#define DEFAULT_RANDOM_SEED 0x5532489


UINT64 gCCBRandSeed = DEFAULT_RANDOM_SEED;

UINT64 ccb_get_random64(void)
{


 	UINT64 m = 0x7ffffffffff;

	gCCBRandSeed = (gCCBRandSeed * 48271) % m;

	return gCCBRandSeed;





}

static inline UINT32 ccb_get_random32(void)
{
	return ASTROS_GET_RAND32();
}



void ccb_get_random_lba(ccb *pCCB)
{
	UINT64 rand;
	UINT64 blk4k; 
	
	int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);

	ASTROS_ASSERT(pCCB);
	ASTROS_ASSERT(pCCB->pTarget);
	
	blk4k  = pCCB->pTarget->capacity4kblock;

	ASTROS_DBG_PRINT(verbose, "random_lba(%d) CALL get_random64()\n", pCCB->idx);
	
	rand = (UINT64)ccb_get_random64();
		

	rand = rand % blk4k;


	pCCB->lba = rand * 8; //8 blocks == 4k

#if 0
	pCCB->lba = pCCB->idx * ((1024 * 1024 * 1024) / 4096);
	pCCB->offset = pCCB->lba * 4096;


#endif

	ASTROS_DBG_PRINT(verbose, "random_lba(%d) RETURN get_random64(%lld) lba = %lx\n", pCCB->idx, rand, pCCB->lba);


#ifdef ASTROS_KERNEL_STATS
	{
		aioengine *pEngine;
		int cpu;
		const UINT64 mask = ~(ASTROS_KERNEL_STATS_LBA_MASK);
		UINT64 marker;
		UINT64 olba = pCCB->lba;


		ASTROS_DBG_PRINT(verbose, "random_lba(%d) KERNEL_STATS %px \n", pCCB->idx, pCCB->pvEngine );

		pEngine = pCCB->pvEngine;

		ASTROS_ASSERT(pEngine);

		
		cpu = pEngine->cpu;


		ASTROS_DBG_PRINT(verbose, "random_lba(%d) KERNAL_STATS pEngine = %px \n", cpu, pEngine);
				

		pCCB->lba = olba & mask;

		marker = GET_KSTAT_LBA_MARKER(cpu);
		

		pCCB->lba |= marker;


		ASTROS_DBG_PRINT(verbose, "random_lba(%d) olba = %ld lba = %ld marker = %ld cpu = %ld lba  =%lx\n", pCCB->idx, olba, pCCB->lba, marker, cpu, pCCB->lba);


	}
#endif


	pCCB->offset = pCCB->lba * 512;
	
	ASTROS_DBG_PRINT(ASTROS_DBGLVL_NONE, "random_lba(%d) blk4k = %lld lba = %lld offset = %lld \n", pCCB->idx, blk4k, rand, pCCB->offset);



}


inline UINT64 ccb_get_next_seq_lba(target *pTarget, ccb *pCCB, int bidx)
{
	UINT32 nextlba;

	

	switch(pTarget->sequential_control.mode)
	{
		case ASTROS_SEQ_MODE_SPLIT_STREAM_RST_INNING:
		case ASTROS_SEQ_MODE_SPLIT_STREAM_RST_ATBAT:
		{

			nextlba = pTarget->sequential_control.bat[bidx].next_lba;
			
			pTarget->sequential_control.bat[bidx].next_lba += pCCB->block_count;

#if 0   
					if(next_lba > capacity_blocks)
						set nextlba to zero
#endif

		}
		break;

		default:
			ASTROS_ASSERT(0);
		break;
		



	}


	return nextlba;

}

void ccb_get_seq_lba(ccb *pCCB)
{
	int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);
	UINT64 blk4k; 
	UINT64 nextlba;
	aioengine *pEngine;
	batter *pBatter;
	int bidx;
	
	pEngine = pCCB->pvEngine;

	pBatter = pEngine->pvBatter;


	ASTROS_ASSERT(pCCB);
	ASTROS_ASSERT(pCCB->pTarget);
	
	blk4k  = pCCB->pTarget->capacity4kblock;

	bidx = pBatter->idx;

 //ASTROS_DBG_PRINT(ASTROS_DBGLVL_INFO, "seq_lba(%d)\n", pCCB->pTarget->sequential_control.mode);

	nextlba = ccb_get_next_seq_lba(pCCB->pTarget, pCCB, bidx);

	pCCB->lba = nextlba;
	

	ASTROS_DBG_PRINT(verbose, "seq_lba(%d) RETURN bat(%d) lba = %llx\n", pCCB->idx, bidx, pCCB->lba );


	pCCB->offset = pCCB->lba * 512;

	
	ASTROS_DBG_PRINT(ASTROS_DBGLVL_NONE, "seq_lba(%d) blk4k = %lld nextlba = %lld offset = %lld \n", pCCB->idx, blk4k, nextlba, pCCB->offset);



}










void astros_ccb_put(ccb **pHead, ccb *pCCB)
{

    int verbose = astros_get_verbosity(ASTROS_DBGLVL_IOPATH);
    ASTROS_DBG_PRINT(verbose, "ccb_put(%p:%p)\n", pHead, pCCB);


    if(*pHead)
    {
        pCCB->pNext = *pHead;
        *pHead = pCCB;
    }
    else
    {
        pCCB->pNext = NULL;
        *pHead = pCCB;
    }

}



int astros_ccb_count(ccb **pHead)
{
	int count = 0;

	ccb *pCCB = *pHead;

	while(pCCB)
	{
		
		
		count++;
		pCCB = pCCB->pNext;
	}

	return count;
	
}

ccb * astros_ccb_get(ccb **pHead)
{
    ccb *pCCB = NULL;

    if(*pHead)
    {
        pCCB = *pHead;

        *pHead = pCCB->pNext;
    }

	if(pCCB)
	{
		pCCB->pNext = NULL;
	}
    return pCCB;
}


bool astros_ccb_find_and_get(ccb **pHead, ccb *pCCB)
{	
	bool bFound = false;
	ccb *pTemp = *pHead;
	ccb *pPrev = NULL;
	int verbose = astros_get_verbosity(ASTROS_DBGLVL_IOPATH);

	ASTROS_DBG_PRINT(verbose, "astros_ccb_find_and_get(head = %d count = %d)\n", (*pHead ? (*pHead)->idx : -1), astros_ccb_count(pHead));
	
	while(pTemp)
	{

		ASTROS_DBG_PRINT(verbose, "astros_ccb_find_and_get(head = %d find = %d cur = %d next = %d\n)",
			(*pHead)->idx, (pCCB->idx), (pTemp->idx), (pTemp->pNext ? pTemp->pNext->idx : -1));

	
		if(pCCB == pTemp)
		{
			bFound = true;
			if(pPrev)
			{
				pPrev->pNext = pTemp->pNext;
				pTemp->pNext = NULL;
				
			}
			else
			{
				*pHead = pTemp->pNext;
				pTemp->pNext = NULL;
				
			}
			break;
		}


		pPrev = pTemp;
		pTemp = pTemp->pNext;
	}


	ASTROS_DBG_PRINT(verbose, "astros_ccb_find_and_get(bFound = %d head = %d\n)",
			bFound, (*pHead ? (*pHead)->idx : -1));

	return bFound;


}




ccb * astros_get_free_ccb(astros_lineup *pLineup)
{
    ccb *pCCB = NULL;
    
    if(pLineup->gametime.pCCBFree)
    {
        pCCB = pLineup->gametime.pCCBFree;

        pLineup->gametime.pCCBFree = pCCB->pNext;
        
        pCCB->pNext = NULL;

        
    }

    return pCCB;
    

}

void astros_put_free_ccb(astros_lineup *pLineup, ccb *pCCB)
{
	if(pLineup->gametime.pCCBFree)
	{
		pCCB->pNext = pLineup->gametime.pCCBFree;
	}
	else
	{
		pCCB->pNext = NULL;
		
	}
	
	pLineup->gametime.pCCBFree = pCCB;

}




int astros_ccb_callback_sustained(void *pvCCB)
{
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);
	ccb *pCCB = pvCCB;
	
    aioengine *pEngine = pCCB->pvEngine;

    ASTROS_DBG_PRINT(verbose, "astros_callback_sustained(%d) pEngine = %p\n", pCCB->idx, pEngine);

    pEngine->pfnSetup(pEngine,pCCB);

    return 0;
}


int astros_ccb_callback_burst(void *pvCCB)
{
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);
	ccb *pCCB = pvCCB;
	int cur_qd;
    aioengine *pEngine = pCCB->pvEngine;
	astros_inning *pInning = pEngine->pvInning;
	int trough_qd = INNING_PARAM(pInning, InParQueueDepthTrough);
	target *pTarget = pCCB->pTarget;

	cur_qd = ASTROS_GET_ATOMIC(pTarget->atomic_qd);
	

    ASTROS_DBG_PRINT(verbose, "astros_callback_sustained(%d) pEngine = %p cur_qd = %d trough_qd = %d\n", pCCB->idx, pEngine, cur_qd, trough_qd);

	
	if(astros_ccb_find_and_get(&pEngine->pPendingHead, pCCB))
	{
 		ASTROS_DBG_PRINT(verbose, "astros_callback_burst(%d) PENDING CCB FOUND qd = %d\n", pCCB->idx, cur_qd);

		astros_ccb_fifo_enqueue(&pEngine->fifo, pCCB);


	}
	else
	{
		ASTROS_ASSERT(0);
	}

	pCCB = NULL;

	if(cur_qd == trough_qd)
	{
		int count = 0;
		ccb *pStart = astros_ccb_fifo_dequeue(&pEngine->fifo);		
		pCCB  = pStart;
		
		do
		{
			if(pCCB->pTarget == pTarget)
			{
				astros_ccb_put(&pEngine->pPendingHead, pCCB);

				pEngine->pfnSetup(pEngine,pCCB);

				count++;

			}
			else
			{
				astros_ccb_fifo_enqueue(&pEngine->fifo,pCCB);
			}

		
			pCCB = astros_ccb_fifo_dequeue(&pEngine->fifo); 	

		} while ((pCCB) && (pCCB != pStart));

 		ASTROS_DBG_PRINT(verbose, "astros_callback_burst(DRAIN) cur_qd == trough_qd = %d ccb_drained = %d\n", cur_qd, count);

	}



    return 0;
}


void astros_ccb_fifo_enqueue (astros_ccb_fifo *pFifo, ccb *pCCB)
{
	pCCB->pNext = NULL;
	
	if((NULL == pFifo->pHead) && (NULL == pFifo->pTail))
	{
		pFifo->pHead = pCCB;
		pFifo->pTail = pCCB;
	}
	else
	{	
		pFifo->pTail->pNext = pCCB;
		pFifo->pTail = pCCB;
	}

}
ccb * astros_ccb_fifo_dequeue (astros_ccb_fifo *pFifo)
{
	ccb *pCCB = NULL;
	if(NULL == pFifo->pHead)
	{

	}
	else
	{
		pCCB = pFifo->pHead;
			
		pFifo->pHead = pFifo->pHead->pNext;
		if(NULL == pFifo->pHead)
		{
			pFifo->pTail = NULL;
		}

		pCCB->pNext = NULL;
		
	}

	return pCCB;

}

/*
typedef struct
{
	
	int pow2_loads[2][ASTROS_FIXLOAD_POW2_SLOTS];
	int odd_block_mipdpoint[2];
	int percent_write;
	
} astros_fixed_load;
*/

void astros_ccb_dump_fixed_load(astros_fixed_load *pLoad)
{

	int verbose = astros_get_verbosity(ASTROS_DBGLVL_INFO);
	int i;


	
	ASTROS_DBG_PRINT(verbose, "Write POW2\n", 0);

	for(i = 0; i < ASTROS_FIXLOAD_POW2_SLOTS; i++)
	{
		ASTROS_DBG_PRINT(verbose, "[%d]%d,", (1 << i), pLoad->pow2_loads[0][i]);
		
	}
	
	ASTROS_DBG_PRINT(verbose, "\nWrite POW2\n", 0);

	for(i = 0; i < ASTROS_FIXLOAD_POW2_SLOTS; i++)
		
	{
		ASTROS_DBG_PRINT(verbose, "[%d]%d,", (1 << i), pLoad->pow2_loads[0][i]);
		
	}


}


void astros_ccb_set_fixed_load(ccb *pCCB, aioengine *pEngine)
{
	int i;
	int rand32 = 0;
	int percent;
	int tl = -1;
	astros_fixed_load *pLoad = astros_get_fixed_load(pEngine);

    int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);


	ASTROS_ASSERT(pLoad);
	//astros_ccb_dump_fixed_load(pLoad);
	
	
	if(100 == pLoad->percent_write )
	{
		pCCB->op = ASTROS_CCB_OP_WRITE;
	}
	else if(0 == pLoad->percent_write )
	{
		pCCB->op = ASTROS_CCB_OP_READ;
	}
	else
	{
		rand32 = ccb_get_random32() % 100;

		if(rand32 < pLoad->percent_write)
		{
			pCCB->op = ASTROS_CCB_OP_WRITE;
		}
		else
		{
			pCCB->op = ASTROS_CCB_OP_READ;
		}
	
	}

	ASTROS_DBG_PRINT(verbose, "astros_ccb_set_fixed_load() op = %d rand32 = %d\n", pCCB->op, rand32);

	rand32 = ccb_get_random32() % 100;
	percent = 0;
	
	for(i = 0; i < ASTROS_FIXLOAD_POW2_SLOTS; i++)
	{
	
		percent +=	pLoad->pow2_loads[pCCB->op][i];

		ASTROS_DBG_PRINT(verbose, "astros_ccb_set_fixed_load(%d) percent = %d rand32 = %d pow2load = %d\n", i, percent, rand32, pLoad->pow2_loads[pCCB->op][i]);

		if(rand32 < percent)
		{
			tl = i;
			break;
		}

	}

	if(tl < 0)
	{
		tl = pLoad->odd_block_midpoint[pCCB->op];
		int mask = 0xF;
		

		if(rand32 < 50)
		{
			tl -= (rand32 & mask);
		}
		else
		{
			tl += (rand32 & mask);
		}

	    if(tl < 0)
	    {
			tl = pLoad->odd_block_midpoint[pCCB->op];
	    }

		pCCB->io_size = tl * ASTROS_FB_BLOCK_SIZE;
		
	}
	else
	{
		pCCB->io_size = (1 << tl) * ASTROS_FB_BLOCK_SIZE;
	}


	ASTROS_DBG_PRINT(verbose, "astros_ccb_set_fixed_load() tl = %d io_size = %d rand32 = %d\n", tl, pCCB->io_size, rand32);


}

void astros_reset_target_sequentials(void)
{


}

