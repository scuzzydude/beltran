#include "astros.h"



int astros_sync_engine_free(void *pvEngine)
{

	return 0;
}

int astros_sync_engine_init(void *pvEngine)
{

	return 0;
}



static inline bool astros_sync_batter_done(int io_count, int io_limit, sync_engine_spec *pEngSpec)
{
	if(io_count >= io_limit)
	{
		ASTROS_SPINLOCK_LOCK(pEngSpec->donelock);

		if(0 == pEngSpec->end_ns)
		{
			pEngSpec->end_ns = ASTROS_PS_HRCLK_GET();

			pEngSpec->total_io_count = ASTROS_GET_ATOMIC(pEngSpec->atomic_io_count);
		}

		ASTROS_SPINLOCK_UNLOCK(pEngSpec->donelock);
	
		return true;
	}

	return false;
}

static inline void astros_sync_batter_calculate_cmd_latency(ccb *pCCB, astros_sync_engine_batter *pSyncBatter)
{
	int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);
	UINT64 elap;

	
	elap = pCCB->end_ns - pCCB->start_ns;

	ASTROS_DBG_PRINT(verbose, "astros_sync_batter_calculate_cmd_latency(%d) start_ns = %ld end_ns = %ld elap = %ld pSyncBatter = %px\n", pCCB->idx, pCCB->start_ns, pCCB->end_ns, elap, pSyncBatter);

	if(pSyncBatter->cmd_lat.count == 0)
	{
		pSyncBatter->cmd_lat.hi = elap;
		pSyncBatter->cmd_lat.lo = elap;
	}
	else 
	{
		if(elap > pSyncBatter->cmd_lat.hi)
		{
			pSyncBatter->cmd_lat.hi = elap;
		}
		else if(elap < pSyncBatter->cmd_lat.lo)
		{
			pSyncBatter->cmd_lat.lo = elap;
		}

	}

	pSyncBatter->cmd_lat.total_elap_ns += elap;
	pSyncBatter->cmd_lat.count++;

}


ASTROS_THREAD_FN_RET astros_batter_sync_rotation(void *pvSyncBatter)
{
	
    astros_sync_engine_batter *pSyncBatter = pvSyncBatter;
    //astros_lineup  *pLineup = pBatter->pvLineup;
    UINT32 cpu;
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);
    aioengine *pEngine;
	ccb *pCCB;
	//FnBatterTakePitches pfnTakePitches;
	ASTROS_THREAD_FN_RET retval = 0;
	sync_engine_spec    *pEngSpec;
	astros_inning *pInning;
	size_t bytes_xfered;
	int fd;
	void *buf;
	size_t io_size;
	int cur_count;
	int io_limit;
		
	int op; 
	pEngSpec = pSyncBatter->pEngSpec;

	ASTROS_ASSERT(pEngSpec);

	pInning = pEngSpec->pvInning;

	ASTROS_ASSERT(pInning);

	io_limit = astros_inning_get_io_limit(pEngSpec->pvInning);

	io_limit--; //because cur_count will hold the value BEFORE the increment
	
	
	op = INNING_PARAM(pInning, InParOperation);


	ASTROS_DBG_PRINT(verbose, "astros_batter_sync_rotation(%d) \n", pSyncBatter->idx);

	ASTROS_GET_CPU(&cpu);
	
    ASTROS_DBG_PRINT(verbose, "astros_batter_sync_rotation(%d) Precall cpu = %ld\n",  pSyncBatter->idx, cpu);

	ASTROS_INC_ATOMIC(&pEngSpec->atomic_sync_threads_ready);

	pCCB = pSyncBatter->pCCB;
	fd = pSyncBatter->fd;


	ASTROS_ASSERT(pCCB);
	ASTROS_ASSERT(fd);

	buf = pCCB->pData;
	io_size = pCCB->io_size;

	pEngine = pEngSpec->pvEngine;

	ASTROS_ASSERT(pEngine);



	while(ASTROS_GET_ATOMIC(pEngSpec->atomic_start_sync))
	{
		ASTROS_BATTER_SLEEPUS(20);
	}

    ASTROS_DBG_PRINT(verbose, "astros_batter_sync_rotation(%d) SYNCED = %ld\n",  pSyncBatter->idx, cpu);


	pSyncBatter->start_ns = ASTROS_PS_HRCLK_GET();

	while(pEngSpec->bRun)
	{
		ccb_get_random_lba(pCCB);

		
		if(pEngine->bLatency)
		{
			pCCB->start_ns = ASTROS_PS_HRCLK_GET();
		}	

		if(ASTROS_CCB_OP_WRITE == op)
		{
			bytes_xfered = ASTROS_SYNC_WRITE(fd, buf, io_size, pCCB->offset);
		}
		else if(ASTROS_CCB_OP_READ == op)
		{
			bytes_xfered = ASTROS_SYNC_READ(fd, buf, io_size, pCCB->offset);
		}
		else
		{
			ASTROS_ASSERT(0);
		}

		
		if(pEngine->bLatency)
		{
			pCCB->end_ns = ASTROS_PS_HRCLK_GET();
			
			astros_sync_batter_calculate_cmd_latency(pCCB, pSyncBatter);
		}	


		if(bytes_xfered == io_size)
		{

			cur_count = ASTROS_INC_ATOMIC(&pEngSpec->atomic_io_count);
		

			if(astros_sync_batter_done(cur_count, io_limit, pEngSpec))
			{
				ASTROS_DBG_PRINT(verbose, "astros_batter_sync_rotation(%d) DONE\n", pSyncBatter->idx);

				pSyncBatter->end_ns = ASTROS_PS_HRCLK_GET();

				break;
			}



		}
		else
		{
			ASTROS_DBG_PRINT(verbose, "astros_batter_sync_rotation(%d) op = %d byte_xfered = %d io_size = %d ERROR\n",  pSyncBatter->idx, op, bytes_xfered, io_size);
		}
		

	}
	


	 ASTROS_INC_ATOMIC(&pEngSpec->atomic_sync_batter_done);



	return retval;
	
}


void astros_sync_eng_spec(batter *pBatter)
{
    aioengine *pEngine;
	sync_engine_spec    *pEngSpec;
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);

	ASTROS_ASSERT(pBatter);

	pEngine = pBatter->pEngine;



	ASTROS_ASSERT(pEngine);

	pEngSpec = &pEngine->engineSpecific.sync;

	
	ASTROS_ASSERT(pEngSpec);

	memset(pEngSpec, 0, sizeof(sync_engine_spec));


	if(pEngine->type == ASTROS_AIOENGINE_BAM_ARRAY)
	{
		pEngSpec->bBam = true;
	}

	ASTROS_DBG_PRINT(verbose, "astros_sync_eng_spec() pEngine = %px type = %d\n", pEngine, pEngine->type);		

	pEngSpec->pvEngine = pEngine;


	ASTROS_SPINLOCK_INIT(pEngSpec->donelock);


	
    ASTROS_SET_ATOMIC(pEngSpec->atomic_sync_threads_ready, 0);
    ASTROS_SET_ATOMIC(pEngSpec->atomic_io_count, 0);
    ASTROS_SET_ATOMIC(pEngSpec->atomic_sync_batter_done, 0);
    ASTROS_SET_ATOMIC(pEngSpec->atomic_start_sync, 1);

	pEngSpec->bRun = false;

	ASTROS_ASSERT(pBatter->pvInning);

	pEngSpec->pvInning = pBatter->pvInning;
	

	




}

sync_engine_spec * astros_sync_get_eng_spec(astros_inning * pInning)
{
    aioengine *pEngine;
	sync_engine_spec    *pEngSpec = NULL;
	batter *pBatter;
	astros_lineup *pLineup = pInning->pvLineup;

	ASTROS_ASSERT(pLineup);

	pBatter = &pLineup->gametime.batters[0];
	

	ASTROS_ASSERT(pBatter);

	pEngine = pBatter->pEngine;

	ASTROS_ASSERT(pEngine);

	pEngSpec = &pEngine->engineSpecific.sync;
	
	ASTROS_ASSERT(pEngSpec);

	return pEngSpec;
	

}

void astros_sync_wait_engine_ready(astros_inning * pInning)
{
	int count = 0;
	sync_engine_spec    *pEngSpec = astros_sync_get_eng_spec(pInning);
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);
	int threads_ready = 0;
	int thread_count = pEngSpec->total_threads;
	const UINT64 sleep_us = 20;
	const UINT64 timeout_us = 1000000; //1us
	int timeout_count = timeout_us / sleep_us;
	
    ASTROS_DBG_PRINT(verbose, "astros_sync_wait_engine_ready() waiting on thread_count = %d\n", thread_count);

	while(threads_ready < thread_count)
	{
		ASTROS_BATTER_SLEEPUS(sleep_us);

		threads_ready = ASTROS_GET_ATOMIC(pEngSpec->atomic_sync_threads_ready);

		ASTROS_DBG_PRINT(verbose, "astros_sync_wait_engine_ready(%d) threads_ready  = %d thread_count = %d\n", count, threads_ready, thread_count);

		count++;

		if(count > timeout_count)
		{
			ASTROS_DBG_PRINT(verbose, "astros_sync_wait_engine_ready(%d) threads_ready	= %d thread_count = %d count = %d timeout_count =%d sleep_us =%ld\n", 
				count, threads_ready, thread_count, thread_count, timeout_count, sleep_us);

			ASTROS_ASSERT(0);
			
		}

		
	}

	pEngSpec->bRun = true;
	
    ASTROS_SET_ATOMIC(pEngSpec->atomic_start_sync, 0);

	pEngSpec->start_ns = ASTROS_PS_HRCLK_GET(); 


}
	
#define ASTROS_SYNC_FD_PER_THREAD

#define SYNC_CPU_SPEC_BATTERS

int astros_sync_distribute_targets_to_sync_batters(astros_inning * pInning, astros_atbat *pAtbat)
{
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);
	int dcount = 0;
	int tidx, op;
	int bidx = 0;
	batter *pBatter;
    aioengine *pEngine;
    target *pTarget;
    astros_lineup *pLineup = pInning->pvLineup;
	int fd;
	ccb *pCCB;
	astros_sync_engine_batter *pSyncBatterArray;
	astros_sync_engine_batter *pSyncBatter;
	int result;

	

	sync_engine_spec    *pEngSpec = astros_sync_get_eng_spec(pInning);
	
	
    ASTROS_DBG_PRINT(verbose, "astros_sync_distribute_targets_to_sync_batters(%d)\n", pAtbat->total_ccb);

	pSyncBatterArray = pLineup->gametime.pvSyncBatterArray;

	ASTROS_ASSERT(pSyncBatterArray);

	astros_sync_eng_spec(&pLineup->gametime.batters[bidx]);


	memset(pSyncBatterArray, 0, pLineup->gametime.sync_batter_array_size);


	pEngSpec->pvSyncBatterArray = pSyncBatterArray;



	while(dcount < pAtbat->total_ccb)
	{

		ASTROS_DBG_PRINT(verbose, "astros_sync_distribute_targets_to_sync_batters(%d) START LOOP\n", dcount);

		pSyncBatter = &pSyncBatterArray[dcount];
		

		tidx = dcount % INNING_PARAM(pInning, InParTargetCount);


		op = INNING_PARAM(pInning, InParOperation);

		
		pTarget = &pLineup->gametime.targets[tidx];
			
		ASTROS_DBG_PRINT(verbose, "astros_sync_distribute_targets_to_sync_batters(%d) OPEN FILE = %s\n", dcount, pTarget->path);
			//astros_draft_blocks picks out the valid block devices

		if(false == pEngSpec->bBam)
		{

#ifdef ASTROS_SYNC_FD_PER_THREAD

			astros_open_target(pLineup, pTarget->path, tidx, op);

			fd = pTarget->fd;
#else
			fd = astros_open_target_file(pLineup, pTarget->path, tidx, op);

#endif

		

			ASTROS_DBG_PRINT(verbose, "astros_sync_distribute_targets_to_sync_batters(%d) FILE OPENED = %d\n", dcount, fd);

			if(fd < 1)
			{
				ASTROS_ASSERT(0);
			}
		

			ASTROS_ASSERT(fd);

		}
		ASTROS_ASSERT(pTarget->pReady);
		
		pBatter = &pLineup->gametime.batters[bidx];

		ASTROS_ASSERT(pBatter);

		pEngine = pBatter->pEngine;

		ASTROS_ASSERT(pEngine);
						
		pEngine->pvInning = pInning;

		pCCB =	astros_ccb_get(&pTarget->pReady);

		ASTROS_ASSERT(pCCB);

		pCCB->pTarget = pTarget;


		ASTROS_DBG_PRINT(verbose, "astros_batters_distribute()... TGTIDX = %d FN: %s BIDX = %d op = %d fd = %d pCCB->idx = %d\n", tidx, pTarget->path, bidx, op, fd, pCCB->idx);

		pCCB->pvEngine = pEngine;

		pSyncBatter->fd = fd;
		pSyncBatter->pCCB = pCCB;
		
		pSyncBatter->idx = dcount;
		pSyncBatter->pEngSpec = &pEngine->engineSpecific.sync;

		if(1) //TODO: Macro magic incase we need to port to windows, this is ptreads stuff direct to set stack size
		{
			int cpu = dcount % (ASTROS_PS_GET_NCPUS() - 2); //leave 2 CPUs free otherwise this will really stall system while running which might lead to stack up stall back to the IO.

#ifdef ASTROS_WIN
			pSyncBatter->id = CreateThread(NULL, 0, astros_batter_sync_rotation, pSyncBatter, 0, NULL);

			if(pSyncBatter->id)
			{
				DWORD dwThreadAffinityMask = 1 << cpu;
					
				SetThreadAffinityMask(pSyncBatter->id,  &dwThreadAffinityMask); 

			}

#else

			//void *pIntentionalLeak = NULL;
			pthread_attr_t attr;														 
			int rc;
			size_t stack;
			

#ifdef SYNC_CPU_SPEC_BATTERS
			int maxbats = INNING_PARAM(pInning, InParFixedBatters);

			if(maxbats > (ASTROS_PS_GET_NCPUS() - 2))
			{
					
			}
			else
			{
				cpu = dcount % maxbats;
			}
			
#endif





			rc = pthread_attr_init(&attr);												
		   	ASTROS_ASSERT (rc != -1);
																						
		    stack = 16 * 1024;																	
		    rc = pthread_attr_setstacksize(&attr, stack);   															
		    ASTROS_ASSERT (rc != -1);

			//pIntentionalLeak = ASTROS_ALLOC(64, (64 * 1024));

			//ASTROS_ASSERT(pIntentionalLeak);

			//ASTROS_DBG_PRINT(verbose, "pIntentionalLeak = %px\n", pIntentionalLeak);

		    ASTROS_DBG_PRINT(verbose, "astros_sync_distribute_targets_to_sync_batters(%d) PHTREAD CREATE CALL\n", dcount);

			if(pEngSpec->bBam)
			{
		    	result = pthread_create(&pSyncBatter->id, &attr,  astros_batter_bam_rotation,pSyncBatter);
	//	    	result = pthread_create(&pSyncBatter->id, &attr,  astros_batter_sync_rotation,pSyncBatter);
			}
			else
			{
		    	result = pthread_create(&pSyncBatter->id, &attr,  astros_batter_sync_rotation,pSyncBatter);
			}
			
			ASTROS_BATTER_SET_CPU(pSyncBatter->id, cpu);

			ASTROS_DBG_PRINT(verbose, "astros_sync_distribute_targets_to_sync_batters(%d) cpu = %d\n", dcount, cpu);


			pEngine->cpu = cpu;

			ASTROS_DBG_PRINT(verbose, "astros_sync_distribute_targets_to_sync_batters(%d) PHTREAD RETURN result = %d\n", dcount, result);

			if(dcount > 128)
			{
				ASTROS_BATTER_SLEEPUS(dcount);
			}

			

			if(0 != result)
			{

				ASTROS_DBG_PRINT(ASTROS_DBGLVL_ERROR, "astros_batters_distribute(%d) ROTATION CREATE result =  %d \n", dcount, result);
				ASTROS_DUMP_MEMORY("VmSize:", "AT ASSERT");
				// asm("int $3");
				ASTROS_ASSERT(0);

			}

#endif
		}
		dcount++;
	}

	
	pEngSpec->total_threads = dcount;


    ASTROS_DBG_PRINT(verbose, "astros_sync_distribute_targets_to_sync_batters(%d) DONE\n", dcount);

	return 0;
}


bool astros_sync_wait_sync_threads_done(astros_inning *pInning)
{
	bool bDone = false;
	int threads_done = 0;
	sync_engine_spec * pEngSpec = astros_sync_get_eng_spec(pInning);
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);
	UINT64 elap_ns;
	const UINT64 onebillion = 1000000000L;
	const UINT64 stall_ns =   (onebillion * 1L); //1 sec
	const UINT64 timeout_ns = (onebillion * 5L); //5 second

	ASTROS_DBG_PRINT(verbose, "astros_sync_wait_sync_threads_done(%p) CALL\n", pInning);


	threads_done = ASTROS_GET_ATOMIC(pEngSpec->atomic_sync_batter_done);

	if(threads_done >= pEngSpec->total_threads)
	{

		
		ASTROS_DBG_PRINT(verbose, "astros_sync_wait_sync_threads_done(%d) bDONE=true threads_done = %d total_threads = %d stall=%d\n", INNING_PARAM(pInning, InParInningNumber), threads_done,  pEngSpec->total_threads, pEngSpec->stalls);
	
		bDone = true;
	}
	else
	{
		//TODO:  Handle a timeout (if the threads started then we have a start_ns and we can also poll the syncbatters and isolate and manually kill the bad threads)

		if(pEngSpec->bRun && pEngSpec->start_ns)
		{
			elap_ns = ASTROS_PS_HRCLK_GET() - pEngSpec->start_ns;
			if(elap_ns > timeout_ns)
			{
				ASTROS_DBG_PRINT(ASTROS_DBGLVL_ERROR, "astros_sync_wait_sync_threads_done(%d) TIMEOUT threads_done = %d total_threads = %d elap_ns = %ld timeout_ns = %ld\n", INNING_PARAM(pInning, InParInningNumber), threads_done, pEngSpec->total_threads, elap_ns, timeout_ns);

				ASTROS_ASSERT(0);
			}
			else if(elap_ns > stall_ns)
			{
				if(pEngSpec->stalls == 400)
				{
					ASTROS_DBG_PRINT(ASTROS_DBGLVL_ERROR, "astros_sync_wait_sync_threads_done(%d) STALLED threads_done = %d total_threads = %d elap_ns = %ld\n", INNING_PARAM(pInning, InParInningNumber), threads_done, pEngSpec->total_threads, elap_ns);
				}
				pEngSpec->stalls++;
			}
			

		}
		else if(pEngSpec->bRun && (pEngSpec->stalls < 5))
		{
			ASTROS_BATTER_SLEEPUS(200);
			pEngSpec->stalls++;
		}
		else if(pEngSpec->bRun)
		{
			ASTROS_ASSERT(0);
		}
			 

		ASTROS_DBG_PRINT(verbose, "bDone=FALSE astros_sync_wait_sync_threads_done(%d) DONE threads_done = %d total_threads = %d stalls=%d\n", INNING_PARAM(pInning, InParInningNumber), threads_done,  pEngSpec->total_threads, pEngSpec->stalls);
		
		


	}

	

	return bDone;

}



int astros_sync_batters_up(astros_inning * pInning, astros_atbat *pAtbat)
{
    int error = 0;
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);
	int wait_sync_done_sleep_us = 10000;
	int amode = INNING_PARAM(pInning, InParAccess);

    ASTROS_DBG_PRINT(verbose, "astros_sync_batters_up(%p)\n", pAtbat);


	//TODO why did this need to be set?:  (for BAM) 
	//INNING_PARAM(pInning, InParEngineType) = ASTROS_AIOENGINE_SYNC;
	pAtbat->batters = 1;
	INNING_PARAM(pInning, InParBatterDistributionPolicy) = ASTROS_INNING_BDP_SYNC_BATS;


	
	ASTROS_DBG_PRINT(verbose, "astros_sync_batters_up(%p) START \n", pAtbat);
	
    if(astros_batters_setengine(pInning, pAtbat))
    {
        ASTROS_ASSERT(0);
	}
    else if(astros_inning_prep_target_queues(pInning, pAtbat))
    {
        ASTROS_ASSERT(0);
    }
    else if(astros_batters_distribute_targets_to_batters(pInning, pAtbat))
    {
        ASTROS_ASSERT(0);
    }

	if(ASTROS_SEQ_MODE_RANDOM != amode)
	{

		if(ASTROS_SEQ_MODE_SPLIT_STREAM_RST_ATBAT == amode)
		{
			astros_lineup_reset_target_sequentials(pInning->pvLineup, ASTROS_SEQ_MODE_SPLIT_STREAM_RST_ATBAT);
		}
		

		astros_batters_setup_sequentials(pInning, pAtbat);
	
	}

	

	ASTROS_DBG_PRINT(verbose, "astros_sync_batters_up(%p) WAIT SYNC ENGINE READY \n", pAtbat);
		
	astros_sync_wait_engine_ready(pInning);

	ASTROS_DBG_PRINT(verbose, "astros_sync_batters_up(%p) WAIT SYNC THREADS DONE \n", pAtbat);
	

	while(false == astros_sync_wait_sync_threads_done(pInning))
	{

		ASTROS_BATTER_SLEEPUS(wait_sync_done_sleep_us);
	}

	
    ASTROS_DBG_PRINT(verbose, "astros_sync_batters_up(%p) DONE\n", pAtbat);


    return error;    

}


void astros_sync_batter_cleanup(astros_inning * pInning, astros_atbat *pAtbat)
{
    //int verbose = astros_get_verbosity(ASTROS_DBGLVL_INFO);
	int i;
	sync_engine_spec * pEngSpec = astros_sync_get_eng_spec(pInning);
	astros_sync_engine_batter *pSyncBatter;
	astros_sync_engine_batter *pSyncBatterArray;
	batter *pBatter;
	aioengine *pEngine;
    astros_lineup *pLineup = pInning->pvLineup;
	int tryjoins = 0;
	int tryjoinlimit = 5;
	
	pBatter = &pLineup->gametime.batters[0];
	
	ASTROS_ASSERT(pBatter);
	
	pEngine = pBatter->pEngine;
	
	ASTROS_ASSERT(pEngine);


	
	ASTROS_ASSERT(pEngSpec);

	ASTROS_SPINLOCK_DESTROY(pEngSpec->donelock);

	ASTROS_ASSERT(pEngSpec->pvSyncBatterArray);
		
	pSyncBatterArray = pEngSpec->pvSyncBatterArray;



	for(i = 0; i < pEngSpec->total_threads; i++)
	{
		
		pSyncBatter = &pSyncBatterArray[i];

		ASTROS_ASSERT(pSyncBatter);

		if(pSyncBatter->pCCB)
		{
			astros_ccb_put(&pEngine->pPendingHead, pSyncBatter->pCCB);

		}

		if(pSyncBatter->fd)
		{

#ifdef ASTROS_SYNC_FD_PER_THREAD
			ccb *pCCB;

			pCCB = pSyncBatter->pCCB;

			if(pCCB->pTarget->fd)
			{

#ifdef ASTROS_WIN
				CloseHandle(pCCB->pTarget->fd);
#else
				close(pCCB->pTarget->fd);
#endif
				pCCB->pTarget->fd = 0;
					
			}

#else
			
			close(pSyncBatter->fd);
#endif			


		}

#ifdef ASTROS_WIN
#else

		tryjoins = 0;

		while(tryjoins < tryjoinlimit)
		{
			int test;

			test = pthread_tryjoin_np(pSyncBatter->id, NULL);

			if(test != 0)
			{
				ASTROS_DBG_PRINT(ASTROS_DBGLVL_ERROR, "sync_batter_cleanup(%d) TRYJOIN FAILED = %d attempt = %d of %d\n", i, test, tryjoins, tryjoinlimit);

				ASTROS_BATTER_SLEEPUS( (tryjoins + 1) * 10000);
			}
			else
			{
				break;
			}
			


			tryjoins++;
		}

#endif

		
	}

}

void astros_sync_inning_atbat_calculate(astros_atbat *pAtbat, astros_inning *pInning, int at_bat_idx)
{
	astros_sync_engine_batter *pSyncBatter;
	astros_sync_engine_batter *pSyncBatterArray;
	sync_engine_spec * pEngSpec = astros_sync_get_eng_spec(pInning);
	UINT64 io_count = 0;
	UINT64 one_billion = 1000000000;
	UINT64 elap_ns;
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);
	astros_latency cmd_lat;
	int i;
	
	memset(&cmd_lat, 0, sizeof(astros_latency));
	
	ASTROS_ASSERT(pEngSpec);

	ASTROS_ASSERT(pAtbat);
	
	pAtbat->fIops = 0.0;
	pAtbat->iops = 0;

	io_count = pEngSpec->total_io_count;

	
	elap_ns = pEngSpec->end_ns - pEngSpec->start_ns;

	ASTROS_FP_BEGIN()
	{
		float fElapSec = (float)elap_ns / 1000000000.0;
		float fIops = ((float) io_count) / fElapSec;
	
		pAtbat->fIops += fIops;
		
	}
	ASTROS_FP_END()
	
	if(elap_ns)
	{
		pAtbat->iops = (io_count * one_billion) / elap_ns;
	}
	ASTROS_DBG_PRINT(ASTROS_DBGLVL_INFO, "sync(%d:%d) elap_ns = %d io_count = %d iops = %ld fIops = %f\n", 
		INNING_PARAM(pInning, InParInningNumber), at_bat_idx, elap_ns, io_count, pAtbat->iops, pAtbat->fIops); 

	pSyncBatterArray = pEngSpec->pvSyncBatterArray;
	
	for(i = 0; i < pEngSpec->total_threads; i++)
	{
		pSyncBatter = &pSyncBatterArray[i];
	
		ASTROS_DBG_PRINT(verbose, "astros_sync_inning_atbat_calculate(%d) IOPS = %d	io_count = %d elap_ns = %ld\n", i, pAtbat->iops, io_count, elap_ns);
	
		
		cmd_lat.count += pSyncBatter->cmd_lat.count;
		cmd_lat.total_elap_ns += pSyncBatter->cmd_lat.total_elap_ns;
	
		if(i == 0)
		{
			cmd_lat.lo = pSyncBatter->cmd_lat.lo;
		}
		else if(pSyncBatter->cmd_lat.lo < cmd_lat.lo)
		{
			cmd_lat.lo = pSyncBatter->cmd_lat.lo;
		}
	
		if(pSyncBatter->cmd_lat.hi > cmd_lat.hi)
		{
			cmd_lat.hi = pSyncBatter->cmd_lat.hi;
		}
	
	}


	memcpy(&pAtbat->cmd_lat, &cmd_lat, sizeof(astros_latency));







#if 0
	int i;
	UINT64 elap_ns = 0;
	UINT64 io_count = 0;
	UINT64 one_billion = 1000000000;
	UINT64 first_start_ns = ASTROS_ALL_ONES64;
	UINT64 last_start_ns = 0;
	UINT64 first_end_ns = ASTROS_ALL_ONES64;
	UINT64 last_end_ns = 0;
	UINT64 start_ns;
	UINT64 end_ns;
	UINT64 inner_span_ns;
	UINT64 outer_span_ns;
	SINT64 delta_span_ns;
	SINT64 start_delta;
	SINT64 end_delta;
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);

	astros_latency cmd_lat;
	astros_latency inter_lat;

#ifdef ASTROS_INC_AVG_OFFSET
	astros_avg_offset avg_off;
	memset(&avg_off, 0, sizeof(astros_avg_offset));
#endif	
	memset(&cmd_lat, 0, sizeof(astros_latency));
	memset(&inter_lat, 0, sizeof(astros_latency));


	

	pAtbat->fIops = 0.0;
	pAtbat->iops = 0;

	ASTROS_DBG_PRINT(verbose, "astros_inning_atbat_calculate(batters = %d)\n", pAtbat->batters);
	
	


	for(i = 0; i < pAtbat->batters; i++)
	{
		elap_ns = pAtbat->pitches[i].elap_ns;	
		io_count = pAtbat->pitches[i].io_count;


		start_ns = pAtbat->pitches[i].start_ns;
		end_ns = pAtbat->pitches[i].end_ns;


		ASTROS_DBG_PRINT(verbose, "astros_inning_atbat_calculate(%d) IOPS = %d  io_count = %d elap_ns = %ld\n", i, pAtbat->iops, io_count, elap_ns);


		
		cmd_lat.count += pAtbat->pitches[i].cmd_lat.count;
		cmd_lat.total_elap_ns += pAtbat->pitches[i].cmd_lat.total_elap_ns;

		if(i == 0)
		{
			cmd_lat.lo = pAtbat->pitches[i].cmd_lat.lo;
		}
		else if(pAtbat->pitches[i].cmd_lat.lo < cmd_lat.lo)
		{
			cmd_lat.lo = pAtbat->pitches[i].cmd_lat.lo;
		}

		if(pAtbat->pitches[i].cmd_lat.hi > cmd_lat.hi)
		{
			cmd_lat.hi = pAtbat->pitches[i].cmd_lat.hi;
		}
		
		inter_lat.count += pAtbat->pitches[i].inter_lat.count;
		inter_lat.total_elap_ns += pAtbat->pitches[i].inter_lat.total_elap_ns;

		if(i == 0)
		{
			inter_lat.lo = pAtbat->pitches[i].inter_lat.lo;
		}
		else if(pAtbat->pitches[i].inter_lat.lo < inter_lat.lo)
		{
			inter_lat.lo = pAtbat->pitches[i].inter_lat.lo;
		}

		if(pAtbat->pitches[i].inter_lat.hi > inter_lat.hi)
		{
			inter_lat.hi = pAtbat->pitches[i].inter_lat.hi;
		}


#ifdef ASTROS_INC_AVG_OFFSET

		ASTROS_DBG_PRINT(ASTROS_DBGLVL_NONE, "astros_inning_atbat_calculate(%d) total_off = %lld	hi = %lld lo = %lld count = %lld\n", 
			i, pAtbat->pitches[i].avg_off.total_offset, pAtbat->pitches[i].avg_off.hi, pAtbat->pitches[i].avg_off.lo, pAtbat->pitches[i].avg_off.count);


		avg_off.count += pAtbat->pitches[i].avg_off.count;
		avg_off.total_offset += pAtbat->pitches[i].avg_off.total_offset;

		if(i == 0)
		{
			avg_off.lo = pAtbat->pitches[i].avg_off.lo;
		}
		else if(pAtbat->pitches[i].avg_off.lo < avg_off.lo)
		{
			avg_off.lo = pAtbat->pitches[i].avg_off.lo;
		}

		if(pAtbat->pitches[i].avg_off.hi > avg_off.hi)
		{
			avg_off.hi = pAtbat->pitches[i].avg_off.hi;
		}

#endif


		if(start_ns > last_start_ns)
		{
			last_start_ns = start_ns;
		}

		if(end_ns > last_end_ns)
		{
			last_end_ns = end_ns;
		}

		if(ASTROS_ALL_ONES64 == first_start_ns)
		{
			first_start_ns = start_ns;
		}
		else
		{
			if(start_ns < first_start_ns)
			{
				first_start_ns = start_ns;
			}
		}


		if(ASTROS_ALL_ONES64 == first_end_ns)
		{
			first_end_ns = end_ns;
		}
		else
		{
			if(end_ns < first_end_ns)
			{
				first_end_ns = end_ns;
			}
		}


		



		
		ASTROS_FP_BEGIN()
		{
			float fElapSec = (float)elap_ns / 1000000000.0;
			float fIops = ((float) io_count) / fElapSec;

			pAtbat->fIops += fIops;
			
		}
		ASTROS_FP_END()

		pAtbat->iops += (io_count * one_billion) / elap_ns;
		
	}


	memcpy(&pAtbat->cmd_lat, &cmd_lat, sizeof(astros_latency));
	memcpy(&pAtbat->inter_lat, &inter_lat, sizeof(astros_latency));

#ifdef ASTROS_INC_AVG_OFFSET
	memcpy(&pAtbat->avg_off, &avg_off, sizeof(astros_latency));

	ASTROS_DBG_PRINT(ASTROS_DBGLVL_NONE, "astros_inning_atbat_calculate(batters = %d) total_off = %lld  hi = %lld lo = %lld\n", pAtbat->batters, pAtbat->avg_off.total_offset, pAtbat->avg_off.hi, pAtbat->avg_off.lo);
#endif

	ASTROS_DBG_PRINT(ASTROS_DBGLVL_NONE, "astros_inning_atbat_calculate(batters = %d) INNING IOPS = %d  io_count = %d elap_ns = %ld\n", pAtbat->batters, pAtbat->iops, io_count, elap_ns);



	inner_span_ns = first_end_ns - last_start_ns;
	outer_span_ns = last_end_ns - first_start_ns;

	delta_span_ns = outer_span_ns - inner_span_ns;

	start_delta = last_start_ns - first_start_ns;
	end_delta = last_end_ns - first_end_ns;


	
	ASTROS_FP_BEGIN()
	{
		float fStartDelta = (((float)start_delta) / ((float)outer_span_ns)) * 100.0;
		float fEndDelta = (((float)end_delta) / ((float)outer_span_ns)) * 100.0;

		ASTROS_DBG_PRINT(verbose, "astros_inning_atbat_calculate(qd = %d at_bat=%d fIops = %f iops = %ld)\n", 
			INNING_PARAM(pInning, InParQDepth), pAtbat->atbat_number, pAtbat->fIops, pAtbat->iops);
		ASTROS_DBG_PRINT(verbose, "astros_inning_atbat_calculate(first_start_ns=%ld last_start_ns=%ld first_end_ns= %ld last_end_ns=%ld)\n", first_start_ns, last_start_ns, first_end_ns, last_end_ns);
		ASTROS_DBG_PRINT(verbose, "astros_inning_atbat_calculate(inner_span_ns=%ld outer_span_ns=%ld delta_span_ns= %ld last_end_ns=%ld)\n", inner_span_ns, outer_span_ns, delta_span_ns);
		ASTROS_DBG_PRINT(verbose, "astros_inning_atbat_calculate(start_delta=%ld end_delta=%ld)\n", start_delta, end_delta);


		if(ASTROS_INNING_BM_DYNAMIC != INNING_PARAM(pInning, InParBatterMode)) 
		{
			ASTROS_DBG_PRINT(ASTROS_DBGLVL_INFO, "astros_inning_atbat_calculate[%d](start_delta=%f  end_delta=%f) batters = %d iops = %d\n", at_bat_idx, fStartDelta, fEndDelta, pAtbat->batters, pAtbat->iops);

//			ASTROS_ASSERT(fStartDelta < 5.0);
		
//			ASTROS_ASSERT(fEndDelta < 5.0);

		}
		
		pAtbat->fStartDelta = fStartDelta;
		pAtbat->fEndDelta = fEndDelta;
		
		
	}
	ASTROS_FP_END()

#endif


}


