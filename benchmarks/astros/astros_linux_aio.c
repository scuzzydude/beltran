#include "astros.h"



int astros_aio_prep_ccb(void *pvEngine, ccb *pCCB, target *pTarget, void * pvfnCallback)
{
    int error = 0;
    aioengine *pEngine = pvEngine;
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_IOPATH);
//    int verbose = astros_get_verbosity(ASTROS_DBGLVL_INFO);

    pCCB->pvEngine = pvEngine;
    pCCB->pTarget = pTarget;
    pCCB->pfnCCBCallback = pvfnCallback;
    
 
    ASTROS_DBG_PRINT(verbose, "astros_aio_prep_ccb(%p) [%d] error = %d\n", pCCB , pCCB->idx, error);
    
	astros_ccb_put(&pEngine->pStartQueueHead, pCCB);

   
    return error;
}


#ifdef ASTROS_LIBURING



int astros_iouring_init(void *pvEngine)

{
    aioengine *pEngine = pvEngine;
//    int verbose = astros_get_verbosity(ASTROS_DBGLVL_IOPATH);
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_INFO);
    int ret;
    struct io_uring *pRing;
	int flags = 0; //IORING_SETUP_IOPOLL;
	
    ASTROS_ASSERT(pvEngine);
        

    ASTROS_DBG_PRINT(verbose, "astros_iouring_init(%p) depth = %d\n", pvEngine, pEngine->depth);

    pRing = &pEngine->engineSpecific.iouring.ring;

    ret = io_uring_queue_init(pEngine->depth, pRing, flags);

    pEngine->engineSpecific.iouring.pIov = malloc(sizeof(struct iovec) * pEngine->depth);

	if(ret != 0)
	{

    	ASTROS_DBG_PRINT(ASTROS_DBGLVL_ERROR, "astros_iouring_init(%p) ret = %d : STRERROR=%s\n", pvEngine, ret, strerror(ret));
		ASTROS_ASSERT(0);
	}
    return 0;
}

int astros_iouring_free(void *pvEngine)
{
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_IOPATH);
    struct io_uring *pRing;
    int error = 0;
    aioengine *pEngine = pvEngine;

    ASTROS_ASSERT(pvEngine);

    pRing = &pEngine->engineSpecific.iouring.ring;

    io_uring_queue_exit(pRing);

    ASTROS_DBG_PRINT(verbose, "astros_iouring_free(%p) error = %d\n", pvEngine, error);

    return error;

}

int astros_iouring_setup_ccb(void *pvEngine, ccb *pCCB)
{
    aioengine *pEngine = pvEngine;
    struct io_uring_sqe *pSqe;
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);
    int error = 0;
    int iov_idx  = pCCB->engine_scratch.iouring.iov_idx;

    pSqe = io_uring_get_sqe(&pEngine->engineSpecific.iouring.ring);

    ASTROS_DBG_PRINT(verbose, "astros_iouring_setup_ccb(%p) [%d] op = %d pSqe = %p fd = %d iov_idx=%d\n", 
		pCCB , pCCB->idx, pCCB->op, pSqe, pCCB->pTarget->fd, iov_idx);




	ASTROS_ASSERT(pCCB->io_size);

	if(pCCB->io_size)
	{

		

	}
	else
	{
		astros_ccb_set_fixed_load(pCCB, pEngine);
	}




	ccb_get_random_lba(pCCB);


    if(ASTROS_CCB_OP_READ == pCCB->op)
    {

		ASTROS_DBG_PRINT(verbose, "astros_iouring_setup_ccb READ_FIXED(%p) [%d] pSqe=%p fd=%d pData=%p, io_size=%d offset=%d iov_idx=%d\n", pCCB , pCCB->idx, pSqe, pCCB->pTarget->fd, pCCB->pData, pCCB->io_size, pCCB->offset, iov_idx);

		io_uring_prep_read_fixed(pSqe, pCCB->pTarget->fd, pCCB->pData, pCCB->io_size, pCCB->offset, iov_idx);

    }
    else if(ASTROS_CCB_OP_WRITE == pCCB->op)
    {
		io_uring_prep_write_fixed(pSqe, pCCB->pTarget->fd, pCCB->pData, pCCB->io_size, pCCB->offset, iov_idx);
    }
    else
    {
        ASTROS_ASSERT(0);
    }

    io_uring_sqe_set_data(pSqe, pCCB);

	ASTROS_INC_ATOMIC(&pCCB->pTarget->atomic_qd);
	
    ASTROS_DBG_PRINT(verbose, "astros_iouring_setup_ccb(%p) [%d] pSqe flags = 0x%08x atomic_qd = %d\n", pCCB , pCCB->idx, pSqe->flags, ASTROS_GET_ATOMIC(pCCB->pTarget->atomic_qd));

	if(pEngine->bLatency)
	{
		pCCB->start_ns = ASTROS_PS_HRCLK_GET();
		astros_batter_calculate_inter_latency(pCCB);
 	}



    return error;

}



int astros_iouring_reset(void *pvEngine)
{
    aioengine *pEngine = pvEngine;

    pEngine->engineSpecific.iouring.cur_iov_idx = 0;
    

    return 0;
}

int astros_iouring_register(void *pvEngine)
{
    int error = 0;
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);
    aioengine *pEngine;
    ccb *pCCB;
    struct iovec *pIov;
    
    pEngine = pvEngine;

    pCCB = pEngine->pStartQueueHead;

    ASTROS_DBG_PRINT(verbose, "astros_iouring_regsiter() pEngine->pPendingHead = %p\n", pCCB);

    while(pCCB)
    {

        pIov = &pEngine->engineSpecific.iouring.pIov[pEngine->engineSpecific.iouring.cur_iov_idx];

        pCCB->engine_scratch.iouring.iov_idx = pEngine->engineSpecific.iouring.cur_iov_idx;

        pIov->iov_base = pCCB->pData;
        pIov->iov_len = pCCB->io_size;

        ASTROS_DBG_PRINT(verbose, "astros_iouring_regsiter CCB(%d) iov_idx = %d iov_base = %p iov_len = %d\n", pCCB->idx, pEngine->engineSpecific.iouring.cur_iov_idx, pIov->iov_base, pIov->iov_len );
        

        pEngine->engineSpecific.iouring.cur_iov_idx++;
        
        pCCB = pCCB->pNext;
    }

    if(pEngine->engineSpecific.iouring.cur_iov_idx > 0)
    {
		int retries = 0;

       	pIov = pEngine->engineSpecific.iouring.pIov;

	   	do
	   	{

       		error = io_uring_register_buffers(&pEngine->engineSpecific.iouring.ring, pIov, pEngine->engineSpecific.iouring.cur_iov_idx);

	
	   		if(error != 0)
	   		{
		   		ASTROS_DBG_PRINT(ASTROS_DBGLVL_INFO, "astros_iouring_regsiter  io_uring_register_buffers() error = %d : %s pIov = %px idx = %d\n", error, strerror(error * -1), pIov, pEngine->engineSpecific.iouring.cur_iov_idx);
	   		}

	   		ASTROS_DBG_PRINT(verbose, "astros_iouring_regsiter  io_uring_register_buffers() error = %d : %s\n", error, strerror(error * -1));

			if(error != 0)
			{
				ASTROS_BATTER_SLEEPUS(100);
			}
			retries++;
			
		} while( (error != 0) && (retries < 2));

       ASTROS_ASSERT(error == 0);
    }


    return error;

}



int astros_iouring_queue_pending(void *pvEngine)
{
    int count;
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);
    aioengine *pEngine = pvEngine;


    count = io_uring_submit(&pEngine->engineSpecific.iouring.ring);


    if(count > 0)
    {
        ASTROS_DBG_PRINT(verbose, "astros_iouring_queue_pending() count = %d\n", count);
    }

    return count;
}




int astros_iouring_complete(void *pvEngine, bool bDrain)
{
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);
    aioengine *pEngine = pvEngine;
    struct io_uring_cqe *pCqe;
    int count = 0;
    ccb *pCCB;
	int error;

    
    ASTROS_DBG_PRINT(verbose, "astros_iouring_complete CALL(%p)\n", pEngine);
    
    error = io_uring_wait_cqe(&pEngine->engineSpecific.iouring.ring, &pCqe);

    if(error < 0)
    {
        ASTROS_ASSERT(0);
    }

	if(pCqe)
	{

    	pCCB = (ccb *)io_uring_cqe_get_data(pCqe);

    	ASTROS_ASSERT(pCCB);
		
		count++;
		
		if(pEngine->bLatency)
		{
			pCCB->end_ns = ASTROS_PS_HRCLK_GET();

			astros_batter_calculate_cmd_latency(pCCB);
		}
		

		ASTROS_DEC_ATOMIC(&pCCB->pTarget->atomic_qd);
	    ASTROS_DBG_PRINT(verbose, "astros_iouring_complete CCB(%p) idx = %d user_data = %p atomic_qd = %d\n", pCCB, pCCB->idx, (void *) pCqe->user_data, ASTROS_GET_ATOMIC(pCCB->pTarget->atomic_qd));

		if(pCqe->res != pCCB->io_size)
		{
			if(pCqe->res < 0)
			{
				
				ASTROS_DBG_PRINT(ASTROS_DBGLVL_ERROR, "astros_iouring_complete CCB(%p) idx = %d res = %d strerror = %s tgtidx = %d : %s offset=%lld\n", pCCB, pCCB->idx,  pCqe->res, strerror( -1 * pCqe->res), pCCB->pTarget->idx, pCCB->pTarget->path, pCCB->offset);	
				ASTROS_ASSERT(0);
			}
		}



    	io_uring_cqe_seen(&pEngine->engineSpecific.iouring.ring, pCqe);

		if(bDrain)
		{

		}
    	else if(pCCB->pfnCCBCallback)
    	{
        	pCCB->pfnCCBCallback(pCCB);
    	}
	}

    return count;
}

#endif

/************************************************************************************************************/
/* LIBAIO */
/************************************************************************************************************/

#ifdef ASTROS_LIBAIO
//#define ASTROS_LIBAIO_VECTORED


int astros_libaio_init(void *pvEngine)

{
    aioengine *pEngine = pvEngine;
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);
    int ret;
	int size;
 	
	ASTROS_DBG_PRINT(verbose, "astros_libaio_init ALLOC depth = %d\n", pEngine->depth);

	if((ret = (io_setup(pEngine->depth, &pEngine->engineSpecific.libaio.ioctx))))
	{
		ASTROS_DBG_PRINT(ASTROS_DBGLVL_NONE, "Error io_setup(%d)\n", ret);


	}
	

	ASTROS_ASSERT(ret == 0);

	size = sizeof(struct iocb *) * pEngine->depth;

	pEngine->engineSpecific.libaio.iocbA = ASTROS_ALLOC(64, size);

	ASTROS_DBG_PRINT(verbose, "astros_libaio_init ALLOC size = %d iocbA = %px\n", size, pEngine->engineSpecific.libaio.iocbA);
	pEngine->engineSpecific.libaio.idx = 0;
	
	pEngine->engineSpecific.libaio.pEvent = ASTROS_ALLOC(64,sizeof(struct io_event) * pEngine->depth);
	



    return 0;
}

int astros_libaio_free(void *pvEngine)
{
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);
    int error = 0;
    aioengine *pEngine = pvEngine;

    ASTROS_ASSERT(pvEngine);

 
	error = io_destroy(pEngine->engineSpecific.libaio.ioctx);
	
    ASTROS_DBG_PRINT(verbose, "astros_libaio_free(%p) error = %d\n", pvEngine, error);

	ASTROS_ASSERT(error == 0);
	
	if(pEngine->engineSpecific.libaio.iocbA)
	{
		ASTROS_FREE(pEngine->engineSpecific.libaio.iocbA);
		pEngine->engineSpecific.libaio.iocbA = NULL;
	}

	

    return error;

}




int astros_libaio_setup_ccb(void *pvEngine, ccb *pCCB)
{
    aioengine *pEngine = pvEngine;
	struct iocb *pIocb;

    int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);
    int error = 0;
#ifdef ASTROS_LIBAIO_VECTORED
	int iov_flags = 0;
#endif

	
	
	pIocb = &pCCB->engine_scratch.libaio.aIocb;
	

	
   

	if(astros_is_fixed_load(pEngine))
	{
		astros_ccb_set_fixed_load(pCCB, pEngine);
	}


	pCCB->pfnCCB(pCCB);
	

    ASTROS_DBG_PRINT(verbose, "astros_libaio_setup_ccb(%p) [%d] op = %d pSqe = %p fd = %d iov_idx=%d io_size = %d\n", 
		pCCB , pCCB->idx, pCCB->op, NULL, pCCB->pTarget->fd, pEngine->engineSpecific.libaio.idx, pCCB->io_size );



#ifdef ASTROS_LIBAIO_VECTORED
	pCCB->engine_scratch.libaio.iov.iov_base = pCCB->pData;
	pCCB->engine_scratch.libaio.iov.iov_len = pCCB->io_size;

    if(ASTROS_CCB_OP_READ == pCCB->op)
    {
		ASTROS_DBG_PRINT(verbose, "astros_libaio_setup_ccb READ_FIXED(%p) [%d] pSqe=%p fd=%d pData=%p, io_size=%d offset=%d iov_idx=%d\n", pCCB , pCCB->idx, NULL, pCCB->pTarget->fd, pCCB->pData, pCCB->io_size, pCCB->offset, 0);

		io_prep_preadv(pIocb, pCCB->pTarget->fd, &pCCB->engine_scratch.libaio.iov, 1, pCCB->offset);
    }
    else if(ASTROS_CCB_OP_WRITE == pCCB->op)
    {
		io_prep_pwritev(pIocb, pCCB->pTarget->fd, &pCCB->engine_scratch.libaio.iov, 1, pCCB->offset);
    }
#else
    if(ASTROS_CCB_OP_READ == pCCB->op)
    {
		ASTROS_DBG_PRINT(verbose, "astros_libaio_setup_ccb READ_FIXED(%p) [%d] fd=%d pData=%p, io_size=%d offset=%lld lba=%lld\n", 
		                                                             pCCB , pCCB->idx, pCCB->pTarget->fd, pCCB->pData, pCCB->io_size, pCCB->offset, pCCB->lba);

		io_prep_pread(pIocb, pCCB->pTarget->fd, pCCB->pData, pCCB->io_size, pCCB->offset);

    }
    else if(ASTROS_CCB_OP_WRITE == pCCB->op)
    {

		ASTROS_DBG_PRINT(verbose, "astros_libaio_setup_ccb WRITE_FIXED(%p) [%d] fd=%d pData=%p, io_size=%d offset=%lld lba=%lld\n", 
		                                                             pCCB , pCCB->idx, pCCB->pTarget->fd, pCCB->pData, pCCB->io_size, pCCB->offset, pCCB->lba);


		io_prep_pwrite(pIocb, pCCB->pTarget->fd, pCCB->pData, pCCB->io_size, pCCB->offset);
    }
#endif
    else
    {
        ASTROS_ASSERT(0);
    }

	ASTROS_INC_ATOMIC(&pCCB->pTarget->atomic_qd);
	
    ASTROS_DBG_PRINT(verbose, "astros_iouring_setup_ccb(%p) [%d] aio_lio_opcode = 0x%08x atomic_qd = %d\n", pCCB , pCCB->idx, pIocb->aio_lio_opcode, ASTROS_GET_ATOMIC(pCCB->pTarget->atomic_qd));

	if(pEngine->bLatency)
	{
		pCCB->start_ns = ASTROS_PS_HRCLK_GET();
		astros_batter_calculate_inter_latency(pCCB);
 	}

	pIocb->data = pCCB;

	pEngine->engineSpecific.libaio.iocbA[pEngine->engineSpecific.libaio.idx] = pIocb;
	pEngine->engineSpecific.libaio.idx++;



    return error;

}



//nano #define ASTROS_LIBAIO_LOOPBACK
#ifdef ASTROS_LIBAIO_LOOPBACK
int astros_libaio_queue_pending(void *pvEngine)
{
    int count = 0;
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);
    aioengine *pEngine = pvEngine;
	int i;
	struct iocb *pIocb;
	ccb *pCCB;
	

	for(i = 0; i < pEngine->engineSpecific.libaio.idx; i++)
	{
		pIocb = pEngine->engineSpecific.libaio.iocbA[i];

		pCCB = pIocb->data;

		astros_ccb_fifo_enqueue(&pEngine->fifo,pCCB);


		count++;
	}
	pEngine->engineSpecific.libaio.idx = 0;
	

	return count;
	

}

int astros_libaio_complete(void *pvEngine, bool bDrain)
{
    int count = 0;
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);
    aioengine *pEngine = pvEngine;
	ccb *pCCB;
	

	while((pCCB = astros_ccb_fifo_dequeue(&pEngine->fifo)))
	{
		if(bDrain)
		{

		}
    	else if(pCCB->pfnCCBCallback)
    	{
        	pCCB->pfnCCBCallback(pCCB);
    	}

		count++;
	}

	return count;
	
}

#else



int astros_libaio_queue_pending(void *pvEngine)
{
    int count = 0;
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);
    aioengine *pEngine = pvEngine;


	if(pEngine->engineSpecific.libaio.idx)
	{
		

    	count = io_submit(pEngine->engineSpecific.libaio.ioctx, pEngine->engineSpecific.libaio.idx, pEngine->engineSpecific.libaio.iocbA);
		


    	if(count != pEngine->engineSpecific.libaio.idx)
    	{
			ASTROS_DBG_PRINT(ASTROS_DBGLVL_ERROR, "astros_iouring_queue_pending() count = %d idx = %d ERROR\n", count, pEngine->engineSpecific.libaio.idx);
			ASTROS_ASSERT(0);
    	}
		else
		{
			 ASTROS_DBG_PRINT(verbose, "astros_iouring_queue_pending() GOOD count = %d idx = %d\n", count, pEngine->engineSpecific.libaio.idx);
		 	pEngine->engineSpecific.libaio.idx = 0;
		}
	}
    return count;
}








#ifdef ASTROS_IO_MULTI_COMPLETE

int astros_libaio_complete(void *pvEngine, bool bDrain)
{
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);
    aioengine *pEngine = pvEngine;
    int count;
    ccb *pCCB;
	static struct timespec timeout = { 0, 10000 };
	int i;
	struct io_event *pEvent;	

	
    ASTROS_DBG_PRINT(verbose, "astros_libaio_complete CALL(%p)\n", pEngine);
    
    count = io_getevents(pEngine->engineSpecific.libaio.ioctx, 0, pEngine->min_reap, pEngine->engineSpecific.libaio.pEvent, &timeout);

	if(0 == count)	
	{
		const int tmo_ms = 100;
		const int tmo_sleep_us = 20;
		const int tmo_stall_check = 20;
		const int tmo_assert_limit = ((tmo_ms * 1000L) / tmo_sleep_us) + tmo_stall_check;
			
		if(pEngine->stalls > tmo_stall_check)
		{
			if((pEngine->stalls - tmo_stall_check) == 2)
			{
				ASTROS_DBG_PRINT(verbose, "astros_libaio_complete STALL(%px) stalls = %d\n", pEngine, pEngine->stalls);
			}
			else if(pEngine->stalls > tmo_assert_limit)
			{
				batter *pBatter;
				astros_lineup *pLineup;
				
			

				ASTROS_DBG_PRINT(ASTROS_DBGLVL_INFO, "astros_libaio_complete HARD STALL!!(%px) stalls = %d limit = %d cpu = %d \n", 
				pEngine, pEngine->stalls, tmo_assert_limit, pEngine->cpu);

				ASTROS_ASSERT(pEngine->pvBatter);

				pBatter = pEngine->pvBatter;
				pLineup = pBatter->pvLineup;

				ASTROS_ASSERT(pLineup);
				
				/* cheat code, turn this into generalizeed function that saves the error and moves on */
							astros_batter_force_out(pEngine, ASTROS_ERR_LIBAIO_IO_GET_EVENT_STALL);
	
				
			}
			ASTROS_BATTER_SLEEPUS(tmo_sleep_us);



		}
		

		pEngine->stalls++;

		return 0;
	}
	else
	{
		pEngine->stalls = 0;
	}

	for(i = 0; i < count; i++)
	{

		pEvent = &pEngine->engineSpecific.libaio.pEvent[i];


		pCCB = pEvent->data;
		

    	ASTROS_ASSERT(pCCB);
		

		if(pEngine->bLatency)
		{
			pCCB->end_ns = ASTROS_PS_HRCLK_GET();

			astros_batter_calculate_cmd_latency(pCCB);
		}
		

		ASTROS_DEC_ATOMIC(&pCCB->pTarget->atomic_qd);
	    ASTROS_DBG_PRINT(verbose, "astros_libaio_complete CCB(%p) idx = %d user_data = %p atomic_qd = %d\n", pCCB, pCCB->idx, (void *)pEvent->data, ASTROS_GET_ATOMIC(pCCB->pTarget->atomic_qd));

		if((pEvent->res != pCCB->io_size) || (pEvent->res2 != 0))
		{
			int error = pEvent->res;
			ASTROS_DBG_PRINT(ASTROS_DBGLVL_ERROR, "astros_libaio_complete CCB(%p) idx = %d res = %d strerror = %s tgtidx = %d : %s offset=%lld res2 = %d\n", pCCB, pCCB->idx,  error, strerror(error), pCCB->pTarget->idx, pCCB->pTarget->path, pCCB->offset, pEvent->res2);	
		}

		if(bDrain)
		{

		}
    	else if(pCCB->pfnCCBCallback)
    	{
        	pCCB->pfnCCBCallback(pCCB);
    	}

	}


    return count;
}



#else

int astros_libaio_complete(void *pvEngine, bool bDrain)
{
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_INFO);
    aioengine *pEngine = pvEngine;
    int error = 0;
    ccb *pCCB;
	static struct timespec timeout = { 0, 10000 };
	
	struct io_event event;
		

	
    ASTROS_DBG_PRINT(verbose, "astros_libaio_complete CALL(%p)\n", pEngine);
    
    error = io_getevents(pEngine->engineSpecific.libaio.ioctx, 0, 1, &event, &timeout);



	if(1 == error)
	{

		pCCB = event.data;
		

    	ASTROS_ASSERT(pCCB);
		

		if(pEngine->bLatency)
		{
			pCCB->end_ns = ASTROS_PS_HRCLK_GET();

			astros_batter_calculate_cmd_latency(pCCB);
		}
		

		ASTROS_DEC_ATOMIC(&pCCB->pTarget->atomic_qd);
	    ASTROS_DBG_PRINT(verbose, "astros_libaio_complete CCB(%p) idx = %d user_data = %p atomic_qd = %d\n", pCCB, pCCB->idx, (void *)event.data, ASTROS_GET_ATOMIC(pCCB->pTarget->atomic_qd));

		if((event.res != pCCB->io_size) || (event.res2 != 0))
		{
			error = event.res;
			ASTROS_DBG_PRINT(ASTROS_DBGLVL_ERROR, "astros_libaio_complete CCB(%p) idx = %d res = %d strerror = %s tgtidx = %d : %s offset=%lld res2 = %d\n", pCCB, pCCB->idx,  error, strerror(error), pCCB->pTarget->idx, pCCB->pTarget->path, pCCB->offset, event.res2);	
		}

		if(bDrain)
		{

		}
    	else if(pCCB->pfnCCBCallback)
    	{
        	pCCB->pfnCCBCallback(pCCB);
    	}

		error = 0;
	}
	else
	{
		error = 1;
	}

    return error;
}
#endif
#endif
int astros_libaio_reset(void *pvEngine)
{
    aioengine *pEngine = pvEngine;

    pEngine->engineSpecific.libaio.idx = 0;
    

    return 0;
}

int astros_libaio_register(void *pvEngine)
{
	return 0;
}
#endif



