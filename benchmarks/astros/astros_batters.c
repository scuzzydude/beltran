#include "astros.h"




//UINT64 gDefTargetCapacity = ((UINT64)((UINT64)200049640000 - (UINT64)(2 * 1024 * 1024 * 1024))); //200GB






int astros_batters_on_deck(astros_lineup       * pLineup, astros_inning *pInning)
{
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);
    int i;
    int batters = 0;
    batter *pBatter;
    UINT64 start_ns = ASTROS_PS_HRCLK_GET();
    UINT64 end_ns;
	int batter_count = astros_inning_get_batters(pInning);
	const UINT64 tmo_ns = 1000000 * 100; //100ms busy waiting 

	ASTROS_DBG_PRINT(verbose, "astros_batters_on_deck(%d) \n", 0);
 
    ASTROS_SET_ATOMIC(pLineup->gametime.atomic_batter_on_deck, 0);

    ASTROS_BATTER_SLEEPUS(pLineup->called_up_sleep_us);

#ifdef BATTER_SYNC_ATOMIC
    ASTROS_BATTER_SLEEPUS(pLineup->called_up_sleep_us);
#else
    ASTROS_SPINLOCK_LOCK(pLineup->gametime.playball);
#endif
	
	ASTROS_DBG_PRINT(verbose, "astros_batters_on_deck(%d) GOT playball LOCK\n", batter_count);
    
    ASTROS_SET_ATOMIC(pLineup->gametime.atomic_batter_lock, -1);

	ASTROS_SET_ATOMIC(pLineup->gametime.atomic_batter_done, 0);


    for(i = 0; i < batter_count; i++)
    {
        pBatter = &pLineup->gametime.batters[i];

        if(pBatter->bCalledup)
        {
            if(pBatter->bOnDeck)
            {
                ASTROS_DBG_PRINT(verbose, "astros_batters_on_deck(%d) ALREADY ON DECK\n", pBatter->idx);
                ASTROS_ASSERT(0);
            }
            else
            {
            
                pBatter->bOnDeck = true;
                batters++;
            }

        }
        

    }

#ifdef BATTER_SYNC_ATOMIC
    ASTROS_BATTER_SLEEPUS(pLineup->called_up_sleep_us);

    while(ASTROS_GET_ATOMIC(pLineup->gametime.atomic_batter_on_deck) < batters)
	{
		UINT64 elap_ns;
		end_ns = ASTROS_PS_HRCLK_GET();

	
		elap_ns = end_ns - start_ns;

		if(elap_ns > tmo_ns)
		{
			ASTROS_DBG_PRINT(ASTROS_DBGLVL_ERROR, "astros_batters_on_deck(%d on Deck @ %ld ns) TMO !!!\n", batters, elap_ns);
			break;
			
		}
		


    }
#endif

    end_ns = ASTROS_PS_HRCLK_GET();

    ASTROS_DBG_PRINT(verbose, "astros_batters_on_deck(%d on Deck @ %ld ns\n", batters, end_ns - start_ns);
    


    return batters;
}




int astros_batter_sync(astros_lineup        * pLineup, batter * pBatter)                                                                                                                                                                                                                                                      
{

#ifdef BATTER_SYNC_ATOMIC
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);
	UINT64 count = 0;
	bool bWarned = false;
	const UINT64 warn_tmo_ns = 100000000;
	UINT64 start_ns, elap_ns;
#endif

    
   ASTROS_INC_ATOMIC(&pLineup->gametime.atomic_batter_on_deck);
             

#ifdef BATTER_SYNC_ATOMIC
    start_ns = ASTROS_PS_HRCLK_GET();

	while(1)
	{
		int batters = ASTROS_GET_ATOMIC(pLineup->gametime.atomic_batter_lock);
		int on_deck = ASTROS_GET_ATOMIC(pLineup->gametime.atomic_batter_on_deck);
		
		if(batters == 0)
		{
			break;
		}
		else
		{
			elap_ns = ASTROS_PS_HRCLK_GET() - start_ns;
		
			if(elap_ns > warn_tmo_ns)
			{
				if(bWarned == false)
				{
					ASTROS_DBG_PRINT(ASTROS_DBGLVL_ERROR, "batter_sync_warn(%d) %d batters = %d on_deck = %d  elap_ns = %ld!!!!\n", pBatter->idx, count, batters, on_deck, elap_ns);
					bWarned = true;
				}

			}
		

			count++;
		}
		

	}

	elap_ns = ASTROS_PS_HRCLK_GET() - start_ns;

	ASTROS_DBG_PRINT(verbose, "astros_batter_sync(%d) %d SYNCED elap_ns = %ld\n", pBatter->idx, count, elap_ns);

	
#else
   ASTROS_SPINLOCK_LOCK(pLineup->gametime.playball);
   ASTROS_SPINLOCK_UNLOCK(pLineup->gametime.playball);
#endif

   pBatter->at_bat_start_ns =   ASTROS_PS_HRCLK_GET();
   pBatter->bAtBat = true;

    
   return 0;

}

static inline bool astros_batter_done(int io_count, int io_limit, astros_lineup *pLineup)
{

    int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);

	ASTROS_DBG_PRINT(verbose, "astros_batter_done() io_count =%d io_limit=%d\n", io_count, io_limit);

	if(io_count >= io_limit)
	{
		return true;
	}
	else if(ASTROS_GET_ATOMIC(pLineup->gametime.atomic_batter_done))
	{
		return true;
	}

	return false;
}


int astros_batter_take_pitches_burst(batter *pBatter, aioengine *pEngine, astros_lineup *pLineup, astros_inning *pInning)
{
	int io_count = 0;
	int io_limit = astros_batter_get_io_limit(pBatter);
#ifdef ASTROS_IO_MULTI_COMPLETE
	int cur_count;
#endif

	if(pBatter->bAtBat)
	{
		pEngine->pfnQueue(pEngine);
	}	

    pBatter->at_bat_start_ns = ASTROS_PS_HRCLK_GET();
			

	while(pBatter->bAtBat)
	{


#ifdef ASTROS_IO_MULTI_COMPLETE
		
		cur_count = pEngine->pfnComplete(pEngine, false);

		if(cur_count)
		{
			pEngine->pfnQueue(pEngine);

			io_count += cur_count;


		}

#else		
		if(pEngine->pfnComplete(pEngine, false))
		{
	
		}
		else
		{
			pEngine->pfnQueue(pEngine);

			io_count++;
		}
#endif	
		if(astros_batter_done(io_count, io_limit, pLineup)) 
		{
			ASTROS_INC_ATOMIC(&pLineup->gametime.atomic_batter_done);
	
			pBatter->at_bat_end_ns = ASTROS_PS_HRCLK_GET();
			pBatter->at_bat_io_count = io_count;
			pBatter->bAtBat = false;
	 
			astros_inning_complete_at_bat(pInning, pBatter);

			pEngine->min_reap = 1;
			
			astros_batters_drain(pBatter, pEngine);
	
		}

	}
	return 0;
	
}

#ifdef ALTUVE
#define ALTUVE_ZOMBIE_DLY_CMDBURST 32
#define ALTUVE_ZOMBIE_DLY_CMDBURST_US 50
#endif




void astros_batters_ccb_check(astros_lineup *pLineup)
{
	ccb *pCCB;
	int i;
	pCCB = pLineup->gametime.pCCBBase;
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_INFO);
	UINT64 tmo_ns = 10000000;
	UINT64 cur_ns;
	int tmo_cnt = 0;
	
	for(i = 0; i < pLineup->total_q; i++)
	{
		if(pCCB->start_ns != 0)
		{
			if(0 == pCCB->end_ns)
			{
				cur_ns = ASTROS_PS_HRCLK_GET();

				if((pCCB->start_ns - cur_ns) > tmo_ns)
				{
					ASTROS_DBG_PRINT(verbose, "astros_batters_ccb_check() pCCB->idx = %d\n", pCCB->idx);
#ifdef ASTROS_CYGWIN


#endif
					tmo_cnt++;
				}
			}

		}
		

		pCCB++;
	}

	if(tmo_cnt)
	{
		ASTROS_ASSERT(0);
	}
	
}

#ifdef ASTROS_CYGWIN
#define ASTROS_CCB_CHECK
#endif

int astros_batter_take_pitches_sustained(batter *pBatter, aioengine *pEngine, astros_lineup *pLineup, astros_inning *pInning)
{
	int io_count = 0;
	int cur_count;
	int io_limit = astros_batter_get_io_limit(pBatter);
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);

	ASTROS_DBG_PRINT(verbose, "astros_batter_take_pitches_sustained() pBatter->idx = %d\n", pBatter->idx);

	if(pBatter->bAtBat)
	{
		
		pEngine->pfnQueue(pEngine);
	}	

    pBatter->at_bat_start_ns = ASTROS_PS_HRCLK_GET();
			
	ASTROS_DBG_PRINT(verbose, "astros_batter_take_pitches_sustained() pBatter->idx = %d START WHILE\n", pBatter->idx);

	while(pBatter->bAtBat)
	{

#ifdef ASTROS_IO_MULTI_COMPLETE


		cur_count = pEngine->pfnComplete(pEngine, false);

		if(cur_count)
		{

			ASTROS_DBG_PRINT(verbose, "astros_batter_take_pitches_sustained() pBatter->idx = %d cur_count = %d\n", pBatter->idx, cur_count);
			pEngine->pfnQueue(pEngine);

			io_count += cur_count;
		}
#ifdef ASTROS_CCB_CHECK
		else
		{
			astros_batters_ccb_check(pLineup);		
		}
#endif

#else

		if(pEngine->pfnComplete(pEngine, false))
		{
	
		}
		else
		{
			pEngine->pfnQueue(pEngine);

			io_count++;

		}
#endif	
		if(astros_batter_done(io_count, io_limit, pLineup)) 
		{
			pBatter->at_bat_end_ns = ASTROS_PS_HRCLK_GET();

			ASTROS_INC_ATOMIC(&pLineup->gametime.atomic_batter_done);
	
			pBatter->at_bat_io_count = io_count;
			pBatter->bAtBat = false;
	 
			astros_inning_complete_at_bat(pInning, pBatter);

			pEngine->min_reap = 1;
	
			astros_batters_drain(pBatter, pEngine);
	
		}

#ifdef ALTUVE_ZOMBIE_DLY_CMDBURST
		if(0 == (io_count % ALTUVE_ZOMBIE_DLY_CMDBURST))
		{
			ASTROS_BATTER_SLEEPUS(ALTUVE_ZOMBIE_DLY_CMDBURST_US);

		}
#endif



	}
	return 0;
	
}


FnBatterTakePitches astros_get_take_pitches(astros_inning *pInning)
{
	FnBatterTakePitches pfnTakePitches = NULL;

	switch(INNING_PARAM(pInning, InParInningMode))
	{
		case ASTROS_INNING_MODE_BURST:
			//ASTROS_ASSERT(0);
			pfnTakePitches = astros_batter_take_pitches_burst;
		break;


		case ASTROS_INNING_MODE_SUSTAINED :
		default:
			pfnTakePitches = astros_batter_take_pitches_sustained;
		break;


	}


	return pfnTakePitches;


}


int astros_batter_get_io_limit(batter *pBatter)
{
	astros_inning *pInning; 
	int limit = 0;
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_IOPATH);

	ASTROS_ASSERT(pBatter->pvCurrentInning);

	pInning = pBatter->pvCurrentInning;


	limit = astros_inning_get_io_limit(pInning);
	

	ASTROS_DBG_PRINT(verbose, "astros_batter_get_io_limit() mode = %d limit = %d\n", 
		INNING_PARAM(pInning, InParBatterMode), limit );	

	return limit;
	

}


void astros_batters_drain(batter *pBatter, aioengine *pEngine)
{
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);
	int count = 0;
	ccb *pCCB;
	int wait_count = 0;
	
    ASTROS_DBG_PRINT(verbose, "astros_batters_drain(pEngine->ccb_pending_count = %d)\n",pEngine->ccb_pending_count);


	while((pCCB = astros_ccb_fifo_dequeue(&pEngine->fifo)))
	{
		astros_ccb_put(&pEngine->pPendingHead, pCCB);
		

		

		pEngine->pfnSetup(pEngine,pCCB);

	}




	/* Have to call this once to get the last IO completed as it's stuck pending */	

	

	pEngine->pfnQueue(pEngine);

	while(count < pEngine->ccb_pending_count)
	{
		int cur_count;
		pEngine->min_reap = 1;

		cur_count = pEngine->pfnComplete(pEngine, true);

		count += cur_count;
		ASTROS_DBG_PRINT(verbose, "astros_batters_drain() IO DRAINED %d PENDING = %d cur_count = %d\n", count, pEngine->ccb_pending_count, cur_count);
	
		if(wait_count > 1)
		{
			ASTROS_BATTER_SLEEPUS(wait_count * 100);

			ASTROS_DBG_PRINT(verbose, "WAIT FOR DRAIN astros_batters_drain() IO DRAINED %d PENDING = %d cur_count = %d wait_count = %d\n", 
				count, pEngine->ccb_pending_count, cur_count, wait_count);

			if(wait_count > 10)
			{

				ASTROS_DBG_PRINT(ASTROS_DBGLVL_NONE, "WAIT FOR DRAIN TOO LONG BREAKING astros_batters_drain(%d) IO DRAINED %d PENDING = %d cur_count = %d wait_count = %d\n", 
					pBatter->idx, count, pEngine->ccb_pending_count, cur_count, wait_count);
				break;
			}
		}
		else
		{
			ASTROS_BATTER_SLEEPUS(wait_count + 20);
		}

		wait_count++;

	}

	ASTROS_DBG_PRINT(verbose, "astros_batters_drain() calling astros_ccb_count(&pPendingHead) \n",0);

	count = astros_ccb_count(&pEngine->pPendingHead);

    ASTROS_DBG_PRINT(verbose, "astros_batters_drain() return astros_ccb_count = %d\n",count);




}

#ifdef ASTROS_CYGWIN

DWORD_PTR BindThreadToCPU(
                            DWORD mask // 1  -  bind to cpu 0
                                       // 4  -  bind to cpu 2
                                       // 15 -  bind to cpu 0,1,2,3
                          )
{
    HANDLE th = GetCurrentThread();
    DWORD_PTR prev_mask = SetThreadAffinityMask(th, mask);
    return prev_mask;
}


void astros_set_batter_cpu(UINT32 cpu)
{
	DWORD mask = 1 << cpu;
	BindThreadToCPU(mask);

}

#endif


ASTROS_THREAD_FN_RET astros_batter_rotation(void *pvBatter)
{
    batter *pBatter = pvBatter;
    astros_lineup  *pLineup = pBatter->pvLineup;
	astros_inning *pInning;
    UINT32 cpu;
    int abcount;
    int odcount = 0;
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);
    aioengine *pEngine;
	ccb *pCCB;
	FnBatterTakePitches pfnTakePitches;
	ASTROS_THREAD_FN_RET retval = 0;


	ASTROS_DBG_PRINT(verbose, "astros_batter_rotation(%d) \n", pBatter->idx);

	ASTROS_GET_CPU(&cpu);
	
    ASTROS_DBG_PRINT(verbose, "astros_batter_rotation(%d) Precall cpu = %ld bCalledUp = %d\n", pBatter->idx, cpu, pBatter->bCalledup);


 	
    while(false == pBatter->bCalledup)
    {
		ASTROS_DBG_PRINT(verbose, "astros_batter_rotation(%d) bCalledup == FALSE sleep_us = %d\n", pBatter->idx, pLineup->called_up_sleep_us);
        ASTROS_BATTER_SLEEPUS(pLineup->called_up_sleep_us);
		ASTROS_GET_CPU(&cpu);
		ASTROS_DBG_PRINT(verbose, "astros_batter_rotation(%d) bCalledup == POST_SLEEP cpu = %d\n", pBatter->idx, cpu);
 		if(pBatter->bCleanup)
		{
			break;
		}
    }

	ASTROS_GET_CPU(&cpu);

	ASTROS_DBG_PRINT(verbose, "astros_batter_rotation(%d) Postcall cpu = %ld bCalledUp = %d\n", pBatter->idx, cpu, pBatter->bCalledup);

    pBatter->on_deck_start_ns = ASTROS_PS_HRCLK_GET();
    
	ASTROS_GET_CPU(&cpu);
	
    ASTROS_BATTER_GET_POLICY_AND_PRIORITY(pBatter, &pBatter->rt_policy, &pBatter->rt_priority);
    ASTROS_DBG_PRINT(verbose, "astros_batter_rotation(%d) CALLED UP CPU = %d rt_policy = %d rt_priority = %d on_deck_start_ns = %ld\n", pBatter->idx, cpu, pBatter->rt_policy, pBatter->rt_priority, pBatter->on_deck_start_ns);


	


    
    while(pBatter->bCalledup)
    {

        while (pBatter->bOnDeck)
        {

   //         ASTROS_DBG_PRINT(ASTROS_DBGLVL_INFO, "astros_batter_rotation(%d) ON DECK atomic_batter_on_deck = %d Zombie Parm = %d\n", pBatter->idx, ASTROS_GET_ATOMIC(pLineup->gametime.atomic_batter_on_deck), pBatter->pEngine->zombieParm);

            pEngine = pBatter->pEngine;

			while((pCCB = astros_ccb_get(&pEngine->pStartQueueHead)))
			{

				pEngine->pfnSetup(pEngine,pCCB);



				astros_ccb_put(&pEngine->pPendingHead, pCCB);
			}

			pEngine->ccb_pending_count = astros_ccb_count(&pEngine->pPendingHead);
			pEngine->min_reap = pEngine->ccb_pending_count / 2;

			if(pEngine->min_reap < 1)
			{
				pEngine->min_reap = 1;
			}

		    abcount = 0;
			pInning = pBatter->pvInning;

			ASTROS_ASSERT(pInning);

			pfnTakePitches = astros_get_take_pitches(pInning);

			ASTROS_ASSERT(pfnTakePitches);
			
            astros_batter_sync(pLineup, pBatter);

        //    pBatter->at_bat_start_ns = ASTROS_PS_HRCLK_GET();
				            
            while(pBatter->bAtBat)
            {

				if(0 == (abcount % 1000))
				{
					ASTROS_DBG_PRINT(verbose, "astros_batter_rotation(%d) ATBAT INBOX  at_bat_start_ns = %ld abcount = %d odcount = %d pvBatter=%p\n", pBatter->idx, pBatter->at_bat_start_ns, abcount, odcount, pvBatter);
				}
				pfnTakePitches(pBatter, pEngine, pLineup, pInning);

             	abcount++;
            }

            ASTROS_DBG_PRINT(verbose, "astros_batter_rotation(%d) EXIT AT_BAT at_bat_start_ns = %ld abcount = %d odcount = %d pvBatter=%p\n", pBatter->idx, pBatter->at_bat_start_ns, abcount, odcount, pvBatter);

            ASTROS_DBG_PRINT(verbose, "astros_batter_rotation(%d) tgtmask = 0x%016x\n", pBatter->idx, pBatter->tgtmask);



			pBatter->bOnDeck = false;

            odcount++;

        }



        ASTROS_BATTER_SLEEPUS(pLineup->called_up_sleep_us);
        
    }

   	pBatter->bCalledup = false;

    ASTROS_DBG_PRINT(verbose, "astros_batter_rotation(%d) EXIT\n", pBatter->idx);

	pBatter->bPinched = true;

    return retval;
}



int astros_batters_setup_sequentials(astros_inning * pInning, astros_atbat *pAtbat)
{
    int i;
	int j;
    astros_lineup *pLineup = pInning->pvLineup;
    batter *pBatter;
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);
	int batters = astros_inning_get_batters(pInning);
	//int batter_count = astros_inning_get_batters(pInning);
	//int engine = astros_lineup_get_engine(pLineup, pInning);
	target *pTarget;
	UINT64 tgtmask;
	UINT64 one64 = 1;
	UINT64 block_offset;
	UINT64 cur_offset;
	int bcount;
	int amode = INNING_PARAM(pInning, InParAccess);
	int atbat = INNING_PARAM(pInning, InParAtBat);


		

	//astros_lineup_reset_target_sequentials(pLineup, ASTROS_SEQ_MODE_SPLIT_STREAM_RST_INNING);


	//astros_batters_setup_sequentials(pInning, pAtbat);
	if(amode == ASTROS_SEQ_MODE_SPLIT_STREAM_RST_INNING)
	{
		if(0 == atbat)
		{
			ASTROS_DBG_PRINT(verbose, "_setup_seq: atbat = 0 amode = %d PROCEED\n", amode);
			
		}
		else
		{
			ASTROS_DBG_PRINT(verbose, "_setup_seq: atbat = %d amode = %d SKIPPING\n", atbat, amode);
			return 0;

		}
	}




	ASTROS_DBG_PRINT(verbose, "astros_batters_setup_sequentials(pInning=%d, pAtbat=%d) access = %d batters =%d amode = %d\n", INNING_PARAM(pInning, InParInningNumber), atbat, INNING_PARAM(pInning, InParAccess), batters, amode);
	verbose = ASTROS_DBGLVL_NONE;
	

	for(j = 0; j < INNING_PARAM(pInning, InParTargetCount); j++)
	{
		bcount = 0;
		
		pTarget = &pLineup->gametime.targets[j];

		tgtmask = (one64 << pTarget->idx);

		
		ASTROS_DBG_PRINT(verbose, "TGT(%d) : tgtmask = %llx\n", pTarget->idx, tgtmask);


    	for(i = 0; i < batters; i++)
    	{
			pBatter = &pLineup->gametime.batters[i];
		
			if(tgtmask & pBatter->tgtmask)
			{
				ASTROS_DBG_PRINT(verbose, "astros_batters_setup_sequentials() BATTER(%d) tgtmask %llx : bcount = %d\n", i, pBatter->tgtmask, bcount);
				bcount++;
			}
		
    	}


		block_offset = pTarget->capacity4kblock / bcount;
		block_offset = block_offset * 8;
		cur_offset = 0;
		
		ASTROS_DBG_PRINT(verbose, "TRT(%d) 4kblocks = %lld block_offset = %lld : bcount = %d\n", i, pTarget->capacity4kblock, block_offset, bcount);

	  	for(i = 0; i < batters; i++)
    	{
			pBatter = &pLineup->gametime.batters[i];
			if(tgtmask & pBatter->tgtmask)
			{
				pTarget->sequential_control.bat[i].next_lba = cur_offset;
				ASTROS_DBG_PRINT(verbose, "astros_batters_setup_sequentials() BATTER(%d) nextlba = %lld\n", i, cur_offset);
				cur_offset += block_offset;
				
			}
		
				    
	  	}








	}
	return 0;
	
}
	

int astros_batters_setengine(astros_inning * pInning, astros_atbat *pAtbat)
{
    int i;
    astros_lineup *pLineup = pInning->pvLineup;
    batter *pBatter;
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_INFO);
	int batters = astros_inning_get_batters(pInning);
	int engine = astros_lineup_get_engine(pLineup, pInning);



    ASTROS_DBG_PRINT(verbose, "astros_batters_setengine() max_batters = %d batters =%d engine = %d\n", INNING_PARAM(pInning, InParMaxBatters), batters, engine);

    for(i = 0; i < batters; i++)
    {
		
        pBatter = &pLineup->gametime.batters[i];
		pBatter->pvInning = pInning;
		pBatter->tgtmask = 0;
		
        astros_batters_set_ioengine(pBatter, engine);

		pBatter->pEngine->zombieParm = INNING_PARAM(pInning, InParZombieParm); 
		pBatter->pEngine->bLatency = INNING_PARAM(pInning, InParEnableLatency);

		
		pBatter->pEngine->kicks = 0;
		pBatter->pEngine->stalls = 0;

		
    }
    
    return 0;
}

int astros_batters_register(astros_inning * pInning, astros_atbat *pAtbat)
{

    int i;
    astros_lineup *pLineup = pInning->pvLineup;
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_IOPATH);
	int batters = astros_inning_get_batters(pInning);
    batter *pBatter;

    ASTROS_DBG_PRINT(verbose, "astros_batters_register(%p) max_batters = %d\n", pAtbat, INNING_PARAM(pInning, InParMaxBatters));

	

    for(i = 0; i < batters; i++)
    {
        pBatter = &pLineup->gametime.batters[i];

		ASTROS_DBG_PRINT(verbose, "astros_batters_register(%p) %d \n", pBatter, i);

        pBatter->pEngine->pfnRegister(pBatter->pEngine);

		pBatter->pvCurrentInning = pInning;
    }

    pLineup->gametime.batter_count = INNING_PARAM(pInning, InParMaxBatters);

	ASTROS_DBG_PRINT(verbose, "astros_batters_register() gametime.batter_count = %d \n", pLineup->gametime.batter_count);

    return 0;
}

int astros_inning_check_batters(astros_inning * pInning)
{
	int i;
	int at_bat;
	int outs;	
    batter *pBatter;
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);
	astros_lineup *pLineup = pInning->pvLineup;
	int batters = astros_inning_get_batters(pInning);
	UINT64 batter_sleep_us = 10000;	
	UINT64 count;
	UINT64 single_at_bat_tmo_us = 10000L * (1000000L); 

#ifdef ASTROS_CYGWIN
	single_at_bat_tmo_us = 1000000 * 5; 
#endif



	ASTROS_DBG_PRINT(verbose, "astros_inning_check_batters() batters = %d \n", batters);

	count = 0;
	
	do
	{
		at_bat = 0;
    	for(i = 0; i < batters; i++)
    	{
			pBatter = &pLineup->gametime.batters[i];
			if(pBatter->bAtBat)
			{
				at_bat++;
			}

		}

		count++;
		

	} while(at_bat < batters);

	ASTROS_BATTER_SLEEPUS(batter_sleep_us);
		
	ASTROS_DBG_PRINT(verbose, "astros_inning_check_batters() at_bat = %d count = %d\n", at_bat, count);

	count = 0;
	do
	{
		outs = 0;	
    	for(i = 0; i < batters; i++)
    	{
			pBatter = &pLineup->gametime.batters[i];
			if((pBatter->bAtBat == false) && (pBatter->bOnDeck == false))
			{
				outs++;
			}
    	}

		ASTROS_BATTER_SLEEPUS(batter_sleep_us);
		count++;

		if(count > (single_at_bat_tmo_us / batter_sleep_us))
		{
			ASTROS_DBG_PRINT(ASTROS_DBGLVL_ERROR, "astros_inning_check_batters() at_bat = %d count = %d ATBAT TIMEOUT\n", at_bat, count);
			ASTROS_ASSERT(0);
		}

	}
	while (outs < at_bat);

	ASTROS_DBG_PRINT(verbose, "astros_inning_check_batters() outs = %d \n", outs);

	




	return outs;	

}

int astros_batters_up(astros_inning * pInning, astros_atbat *pAtbat)
{
    int error = 0;
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);
    int batter_count;
	int amode = INNING_PARAM(pInning, InParAccess);
    
    ASTROS_DBG_PRINT(verbose, "astros_batters_up(%p)\n", pAtbat);


        
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
    else if(astros_batters_register(pInning, pAtbat))
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


	//this takes the .playball LOCK
    batter_count = astros_batters_on_deck(pInning->pvLineup, pInning);
        
    ASTROS_DBG_PRINT(verbose, "astros_batters_up() %d Batters on Deck \n", batter_count);
    
	//this release the .playball LOCK
    astros_lineup_first_pitch(pInning->pvLineup);
       
    ASTROS_DBG_PRINT(verbose, "astros_batters_up() %d Post first pitch \n", batter_count);
    
	astros_inning_check_batters(pInning);
	ASTROS_DBG_PRINT(verbose, "astros_batters_up(%p) RETURN\n", pAtbat);
	
    return error;    

}



int astros_callup_batters(astros_lineup *pLineup)
{
    batter *pBatter;
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);
    int i;
	int calledup_count, total_bats;
	int waits = 0;
	int const wait_freq = 100;
	unsigned long mask;
	
    for(i = 0; i < pLineup->gametime.max_batters; i++)
    {
         pBatter = &pLineup->gametime.batters[i];

		 ASTROS_DBG_PRINT(verbose, "astros_callup_batter() %d Calledup pBatter=%px \n", i, pBatter);

         pBatter->bCalledup = true;
    }    

	total_bats = i;
	
    ASTROS_DBG_PRINT(verbose, "astros_callup_batter() total_bats = %d Calledup \n", i);

	do
	{	
		ASTROS_BATTER_SLEEPUS(1000);
		calledup_count = 0;
		mask = 0;
		for(i = 0; i < pLineup->gametime.max_batters; i++)
		{
			pBatter = &pLineup->gametime.batters[i];

			if(pBatter->bCalledup)
			{
				calledup_count++;
				mask |= (1L << i);
			}
		}		

		if(0 == (waits % wait_freq))
		{
			ASTROS_DBG_PRINT(verbose, "astros_callup_batter(%d) calledup_count = %d total-bats = %d pinch_mask = %lx\n", waits, calledup_count, total_bats, mask);
		}
		waits++;

		if(waits > 500)
		{
			ASTROS_ASSERT(0);
		}
		
	} while(calledup_count < total_bats);



    return i;
}




int astros_batters_distribute_targets_to_batters(astros_inning * pInning, astros_atbat *pAtbat)
{
    int error = 0;
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);
    int dcount = 0;
    int tidx, bidx;
	int op;
    target *pTarget;
    astros_lineup *pLineup = pInning->pvLineup;
    ccb *pCCB;
    batter *pBatter;
    aioengine *pEngine;
    CCBCompletionCallback pfnCCBCallback = astros_inning_get_ccb_callback(pInning);
  
    ASTROS_DBG_PRINT(verbose, "astros_distribute_targets_to_batters(pInning->batter_distrubition_policy = %d)\n", 
		INNING_PARAM(pInning, InParBatterDistributionPolicy));




	
    switch(INNING_PARAM(pInning, InParBatterDistributionPolicy))
    {
        case ASTROS_INNING_BDP_RR:
        {
            while(dcount < pAtbat->total_ccb)
            {
				UINT64 one64 = 1;
                tidx = dcount % INNING_PARAM(pInning, InParTargetCount);

                bidx = dcount % INNING_PARAM(pInning, InParScoringBatters);

				op = INNING_PARAM(pInning, InParOperation);

				
				
                pTarget = &pLineup->gametime.targets[tidx];

				pTarget->access = INNING_PARAM(pInning, InParAccess);
	

				ASTROS_DBG_PRINT(verbose, "astros_batters_distribute()... TGTIDX = %d FN: %s BIDX = %d op = %d pTarget = %p\n", tidx, pTarget->path, bidx, op, pTarget);
					
				pBatter = &pLineup->gametime.batters[bidx];
					
				ASTROS_ASSERT(pBatter);

				pTarget->pvEngine = pBatter->pEngine;
				
					//astros_draft_blocks picks out the valid block devices
				if(astros_open_target(pLineup, pTarget->path, tidx, op))
				{
						ASTROS_ASSERT(0);
				}
                
                ASTROS_ASSERT(pTarget->pReady);
                

                pEngine = pBatter->pEngine;

                ASTROS_ASSERT(pEngine);
                                
				pEngine->pvInning = pInning;

				pCCB =  astros_ccb_get(&pTarget->pReady);


				if(ASTROS_SEQ_MODE_RANDOM == INNING_PARAM(pInning, InParAccess))
				{
					pCCB->pfnCCB = (CCB_get_lba_fn)ccb_get_random_lba;
				}
				else
				{
					pCCB->pfnCCB = (CCB_get_lba_fn)ccb_get_seq_lba;
				}


                pBatter->tgtmask |= (one64 << (UINT64)pTarget->idx); 
                
                if(pCCB)
                {
					astros_inning_ready_ccb(pInning, pCCB);
					
				
					pCCB->marker = 0xBA000000 | (bidx << 16) | pCCB->idx;


					
                    ASTROS_DBG_PRINT(verbose, "astros_distribute_targets_to_batters(%d) (pCCB->idx = %d tidx = %d bidx = %d) TGT = %d QD = %d SB = %d fn=%px\n", 
							dcount, pCCB->idx, tidx, bidx, INNING_PARAM(pInning, InParTargetCount), INNING_PARAM(pInning, InParQDepth), INNING_PARAM(pInning, InParScoringBatters), pEngine->pfnPrepCCB);
                    dcount++;


                    if(pEngine->pfnPrepCCB(pEngine, pCCB, pTarget, pfnCCBCallback ))
                    {
                        ASTROS_ASSERT(0);
                    }

                }
            }

        }
        break;
#ifndef ALTUVE
		case ASTROS_INNING_BDP_SYNC_BATS:
			error = astros_sync_distribute_targets_to_sync_batters(pInning, pAtbat);
		break;
#endif


        case ASTROS_INNING_BDP_TARGET_GRP: /* This is more like fio, put all target ccbs on the first worker ... */
        default:
            ASTROS_ASSERT(0);
        break;
    }
    



    return error;   
}

int astros_batters_set_ioengine(batter * pBatter, int engine)
{
    aioengine *pEngine;
    int error = 0;
    astros_lineup *pLineup;
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_INFO);
	
	
    ASTROS_ASSERT(engine < ASTROS_AIOENGINE_MAX);

    pBatter->pEngine = &pBatter->engineA[engine];
    pEngine = &pBatter->engineA[engine];
    pLineup = pBatter->pvLineup;
    pEngine->pvBatter = pBatter;
	pEngine->cpu = pBatter->idx;
	
	astros_batter_reset_latency(pBatter);

	ASTROS_DBG_PRINT(verbose, "astros_batters_set_ioengine(%d) engine = %d\n", pBatter->idx, engine);	

	
    if(false == pEngine->bInit)
    {

        switch(engine)
        {

#ifdef ALTUVE

            case ASTROS_AIOENGINE_KLOOP:
                pEngine->type = ASTROS_AIOENGINE_KLOOP;
                pEngine->mode = ASTROS_AIOENGINE_MODE_LINUX_KERNEL;
                pEngine->depth = pLineup->total_q;
                pEngine->pfnInit = altuve_kloop_init;
                pEngine->pfnFree = altuve_kloop_free;
                pEngine->pfnPrepCCB = altuve_kloop_prep_ccb;
 //               pEngine->pfnReset = altuve_kloop_reset;
                pEngine->pfnRegister = altuve_kloop_register;
                pEngine->pfnSetup = altuve_kloop_setup_ccb;
                pEngine->pfnQueue = altuve_kloop_queue_pending;
                pEngine->pfnComplete = altuve_kloop_complete;
            break;


            case ASTROS_AIOENGINE_ZOMBIE:
                pEngine->type = ASTROS_AIOENGINE_ZOMBIE;
                pEngine->mode = ASTROS_AIOENGINE_MODE_LINUX_KERNEL;
                pEngine->depth = pLineup->total_q;
                pEngine->pfnInit = altuve_zombie_init;
                pEngine->pfnFree = altuve_zombie_free;
                pEngine->pfnPrepCCB = altuve_zombie_prep_ccb;
//                pEngine->pfnReset = altuve_zombie_reset;
                pEngine->pfnRegister = altuve_zombie_register;
                pEngine->pfnSetup = altuve_zombie_setup_ccb;
				pEngine->pfnQueueCommand = altuve_get_queue_command();
                pEngine->pfnQueue = altuve_zombie_queue_pending;
                pEngine->pfnComplete = altuve_zombie_complete;

            break;
			

#else
#ifndef ASTROS_WIN
#ifdef ASTROS_CYGWIN
			case ASTROS_AIOENGINE_AIOWIN:
							pEngine->type = ASTROS_AIOENGINE_URING;
							pEngine->mode = ASTROS_AIOENGINE_MODE_LINUX_USER;
							pEngine->depth = pLineup->total_q;
							pEngine->pfnInit = astros_aiowin_init;
							pEngine->pfnFree = astros_aiowin_free;
							pEngine->pfnPrepCCB = astros_aiowin_prep_ccb;
			//				  pEngine->pfnReset = astros_aiowin_reset;
							pEngine->pfnRegister = astros_aiowin_register;
							pEngine->pfnSetup = astros_aiowin_setup_ccb;
							pEngine->pfnQueue = astros_aiowin_queue_pending;
							pEngine->pfnComplete = astros_aiowin_complete;
			break;


#else
#ifdef ASTROS_LIBURING


            case ASTROS_AIOENGINE_URING:
                pEngine->type = ASTROS_AIOENGINE_URING;
                pEngine->mode = ASTROS_AIOENGINE_MODE_LINUX_USER;
                pEngine->depth = pLineup->total_q;
                pEngine->pfnInit = astros_iouring_init;
                pEngine->pfnFree = astros_iouring_free;
                pEngine->pfnPrepCCB = astros_aio_prep_ccb;
//                pEngine->pfnReset = astros_iouring_reset;
                pEngine->pfnRegister = astros_iouring_register;
                pEngine->pfnSetup = astros_iouring_setup_ccb;
                pEngine->pfnQueue = astros_iouring_queue_pending;
                pEngine->pfnComplete = astros_iouring_complete;
            break;
#endif

            case ASTROS_AIOENGINE_LIBAIO:
                pEngine->type = ASTROS_AIOENGINE_LIBAIO;
                pEngine->mode = ASTROS_AIOENGINE_MODE_LINUX_USER;
                pEngine->depth = pLineup->total_q;
                pEngine->pfnInit = astros_libaio_init;
                pEngine->pfnFree = astros_libaio_free;
                pEngine->pfnPrepCCB = astros_aio_prep_ccb;
//                pEngine->pfnReset = astros_libaio_reset;
                pEngine->pfnRegister = astros_libaio_register;
                pEngine->pfnSetup = astros_libaio_setup_ccb;
                pEngine->pfnQueue = astros_libaio_queue_pending;
                pEngine->pfnComplete = astros_libaio_complete;
            break;
#endif
#else
			case ASTROS_AIOENGINE_WINAIO:
				pEngine->type = ASTROS_AIOENGINE_WINAIO;
				pEngine->mode = ASTROS_AIOENGINE_MODE_WIN_USER;
				pEngine->depth = pLineup->total_q;
				pEngine->pfnInit = astros_winaio_init;
				pEngine->pfnFree = astros_winaio_free;
				pEngine->pfnPrepCCB = astros_winaio_prep_ccb;
			//				  pEngine->pfnReset = astros_libaio_reset;
				pEngine->pfnRegister = astros_winaio_register;
				pEngine->pfnSetup = astros_winaio_setup_ccb;
				pEngine->pfnQueue = astros_winaio_queue_pending;
				pEngine->pfnComplete = astros_winaio_complete;

			break;
				


#endif

			case ASTROS_AIOENGINE_SYNC:
			pEngine->type = ASTROS_AIOENGINE_SYNC;
			pEngine->mode = ASTROS_AIOENGINE_MODE_LINUX_USER;

			pEngine->pfnFree = astros_sync_engine_free;
			pEngine->pfnInit = astros_sync_engine_init;
							
			break;

			case ASTROS_AIOENGINE_BAM_ARRAY:
			pEngine->type = ASTROS_AIOENGINE_BAM_ARRAY;
			pEngine->mode = ASTROS_AIOENGINE_MODE_LINUX_USER;

			pEngine->pfnFree = astros_sync_engine_free;
			pEngine->pfnInit = astros_sync_engine_init;
			break;

	


			



#endif
            default:
                ASTROS_ASSERT(0);
            break;

        }


   if(0 == (error = pEngine->pfnInit(pEngine)))
        {
            pEngine->bInit = true;
        }


    }
    return error;

}



void astros_batter_reset_latency(batter *pBatter)
{

	pBatter->last_submit_ns = 0;

	memset(&pBatter->cmd_lat, 0, sizeof(astros_latency));
	memset(&pBatter->inter_lat, 0, sizeof(astros_latency));
	memset(&pBatter->avg_off, 0, sizeof(astros_avg_offset ));
	
}


void astros_batter_calculate_inter_latency(ccb *pCCB)
{
	int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);
	batter *pBatter;
	UINT64 elap;

	pBatter = ((aioengine *)pCCB->pvEngine)->pvBatter;

	if(false == pBatter->bAtBat)
	{
		return;
	}

	ASTROS_DBG_PRINT(verbose, "astros_batter_calculate_inter_latency(%d) last_submit_ns = %ld pCCB->start_ns = %ld \n", 
	pCCB->idx, pBatter->last_submit_ns, pCCB->start_ns);
	

	
	if(pBatter->last_submit_ns == 0)
	{
		pBatter->last_submit_ns = pCCB->start_ns;

	}
	else
	{
		elap = pCCB->start_ns - pBatter->last_submit_ns;

		if(pBatter->inter_lat.count == 0)
		{
			pBatter->inter_lat.hi = elap;
			pBatter->inter_lat.lo = elap;
			pBatter->avg_off.hi = pCCB->offset;
			pBatter->avg_off.lo = pCCB->offset;
			
		}
		else
		{

			if(elap > pBatter->inter_lat.hi)
			{
				pBatter->inter_lat.hi = elap;
			}
			else if(elap < pBatter->inter_lat.lo)
			{
				pBatter->inter_lat.lo = elap;
			}

			if(pCCB->offset > pBatter->avg_off.hi)
			{
				pBatter->avg_off.hi = pCCB->offset;
			}
			else if(pCCB->offset < pBatter->avg_off.lo)
			{
				pBatter->avg_off.lo = pCCB->offset;
			}
				


		}

		pBatter->inter_lat.total_elap_ns += elap;
		pBatter->inter_lat.count++;

		pBatter->avg_off.total_offset += pCCB->offset;
		pBatter->avg_off.count++;
		


		pBatter->last_submit_ns = pCCB->start_ns;



		ASTROS_DBG_PRINT(ASTROS_DBGLVL_NONE, "astros_batter_calculate_inter_latency(%d) offset = %lld total_offset = %lld hi = %lld lo = %lld\n", 
			pCCB->idx, pCCB->offset, pBatter->avg_off.total_offset, pBatter->avg_off.hi, pBatter->avg_off.lo);










		



		ASTROS_DBG_PRINT(verbose, "astros_batter_calculate_inter_latency(%d) lo = %ld hi = %ld elap = %ld\n", pCCB->idx, pBatter->inter_lat.lo, pBatter->inter_lat.hi, elap);
		
	}


	


}

void astros_batter_calculate_cmd_latency(ccb *pCCB)
{
	int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);
	UINT64 elap;
	batter *pBatter;

	ASTROS_ASSERT(pCCB);

	pBatter = ((aioengine *)pCCB->pvEngine)->pvBatter;

	if(pBatter)
	{

		if(false == pBatter->bAtBat)
		{
			return;
		}
	}
	else
	{
		ASTROS_DBG_PRINT(ASTROS_DBGLVL_INFO, "astros_batter_calculate_cmd_latency() pBatter == NULL: TODO fix?, pCCB=%px pCCB->idx=%d pCCB->marker=0x%08x pvEngine=%px\n", pCCB, pCCB->idx, pCCB->marker, pCCB->pvEngine);
		//ASTROS_ASSERT(0);
		return;
	}
	
	elap = pCCB->end_ns - pCCB->start_ns;

	ASTROS_DBG_PRINT(verbose, "astros_score_calculate_latency(%d) start_ns = %ld end_ns = %ld elap = %ld pBatter = %px\n", pCCB->idx, pCCB->start_ns, pCCB->end_ns, elap, pBatter);

	if(pBatter->cmd_lat.count == 0)
	{
		pBatter->cmd_lat.hi = elap;
		pBatter->cmd_lat.lo = elap;
	}
	else 
	{
		if(elap > pBatter->cmd_lat.hi)
		{
			pBatter->cmd_lat.hi = elap;
		}
		else if(elap < pBatter->cmd_lat.lo)
		{
			pBatter->cmd_lat.lo = elap;
		}

	}

	pBatter->cmd_lat.total_elap_ns += elap;
	pBatter->cmd_lat.count++;




}



void astros_batter_force_out(aioengine *pEngine, int code)
{

	batter *pBatter;
	astros_lineup *pLineup;
	


	ASTROS_DBG_PRINT(ASTROS_DBGLVL_INFO, "astros_batter_force_out(%d) code = %d\n", pEngine->cpu, code);
	
		
	ASTROS_ASSERT(pEngine->pvBatter);

	pBatter = pEngine->pvBatter;
	pLineup = pBatter->pvLineup;

	ASTROS_ASSERT(pLineup);
	
	/* cheat code, batters exit early */
	ASTROS_INC_ATOMIC(&pLineup->gametime.atomic_batter_done);

	//TOOD: Record the code int the logging (most likley will result in at bat and inning IOPs way low)

}




