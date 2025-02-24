#include "astros.h"



int astros_umpire_pregame(astros_lineup *pLineup)
{
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_INFO);
    UINT32 master_cpu;
    pid_t pid = getpid();
    cpu_set_t set;
    CPU_ZERO(&set);
    UINT32 cpu;
    int policy;
    struct sched_param param;
    int pmax, pmin;
    const int priority_backoff = 1;


    pLineup->gametime.id = pthread_self(); 

    master_cpu = ASTROS_PS_GET_NCPUS() - 1;

	if(master_cpu > 64)
	{
		master_cpu = 64;
	}
//aster_cpu = 64; //temp

    pLineup->gametime.max_batters = master_cpu;
#ifdef ASTROS_CYGWIN
	if(master_cpu > 32)
	{
		pLineup->gametime.max_batters = 31;
	}
#endif


    //pLineup->gametime.max_batters = 1;


    
    pthread_getschedparam(pLineup->gametime.id, &policy, &param);
   
    ASTROS_GET_CPU(&cpu);
    ASTROS_DBG_PRINT(verbose, "astros_umpire_pregame() max_batters = %d current_cpu = %d pid = %d policy = %d priority = %d\n", pLineup->gametime.max_batters, cpu, pid, policy, param.sched_priority);


    CPU_SET(master_cpu, &set);

    pmin = sched_get_priority_min(pLineup->rt_policy);
    pmax = sched_get_priority_max(pLineup->rt_policy);

    pLineup->rt_priority = pmax - priority_backoff;

    ASTROS_DBG_PRINT(verbose, "astros_umpire_pregame() rt_policy = %d pmin = %d pmax = %d pLineup->rt_priority = %d\n", pLineup->rt_policy, pmin, pmax, pLineup->rt_priority);

    param.sched_priority = pLineup->rt_priority;

    pthread_setschedparam(pLineup->gametime.id, pLineup->rt_policy, &param);


#ifdef ASTROS_CYGWIN
	/* TODO:  This is a bug for CYGWIN implementation.  The batter (main threads) are created just after return from this function 
	          Setting CPU of those threads fail.  cygwin_thread_test() is just simplified version of main batter.  It allows setting CPU
	          After at this point, but if we move it to the section below, it fails. In straight up Linux 
	          We set the "MAIN" thread to the LAST CPU so that batter 0 == cpu0  but all the main thread does is check status of the test and update stats so it doesn't need a whole CPU of horsepower*/
	//					cygwin_thread_test();
#endif

#ifndef ASTROS_CYGWIN 
   /* TODO Abover, causes CPU imbalance between cygwin and linux version, unsure if it's measurable as there are too many other differences with Windows */
    if (sched_setaffinity(pid, sizeof(set), &set) == 0)
    {
        ASTROS_BATTER_SLEEPUS(100);
    }
    else
    {
		ASTROS_DBG_PRINT(ASTROS_DBGLVL_ERROR, "astros_umpire_pregame rt_priority =%d\n", pLineup->rt_priority);
        ASTROS_ASSERT(0);
    }


    ASTROS_GET_CPU(&cpu);

#if 0 //def ASTROS_CYGWIN
//TODO: See Above
//					cygwin_thread_test();
#endif

    pthread_getschedparam(pLineup->gametime.id, &policy, &param);

#endif

	if(astros_scorer_get(&pLineup->gametime.scorer_id, pLineup, master_cpu))
	{
		ASTROS_ASSERT(0);
	}


    ASTROS_DBG_PRINT(verbose, "astros_umpire_pregame() POST batter_count = %d current_cpu = %d pid = %d policy = %d priority = %d\n", pLineup->gametime.batter_count, cpu, pid, policy, param.sched_priority);

    return 0;

}

int astros_get_open_mode(int operation, astros_lineup * pLineup)
{
	int mode;
	
/*

	if(astros_is_lineup_fixed_load())
	{
			mode = O_RDWR | O_DIRECT;
	}
	else if(ASTROS_CCB_OP_WRITE == operation) 
	{
		  mode = O_WRONLY | O_DIRECT;
	}
	else
	{
		  mode = O_RDONLY | O_DIRECT;
	}
*/

	mode = O_RDWR | O_DIRECT;

	
	return mode;
}


int astros_test_ioengine_callback(ccb *pCCB)
{
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_IOPATH);
    aioengine *pEngine = pCCB->pvEngine;

    ASTROS_DBG_PRINT(verbose, "astros_test_ioengine_callback(%d) pEngine = %p\n",pCCB->idx,pEngine);

    pEngine->pfnSetup(pEngine,pCCB);

    return 0;
}


int astros_test_ioengine(astros_lineup * pLineup)
{
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_INFO);
    aioengine *pEngine;
    int io_count = 0;
    int limit = 1000000;
    ccb *pCCB;
    batter *pBatter = &pLineup->gametime.batters[0];
    target *pTarget;
    int tgtidx = 0;
    int io_submit = 0;
    UINT64 start_ns, end_ns, elap_ns;
    float fIops;


    ASTROS_DBG_PRINT(verbose, "CLOCK_REALTIME        : hr_clk_res = %d ns\n", ASTROS_PS_GET_HRCLK_RES_NS(CLOCK_REALTIME) );
    ASTROS_DBG_PRINT(verbose, "CLOCK_MONOTONIC        : hr_clk_res = %d ns\n", ASTROS_PS_GET_HRCLK_RES_NS(CLOCK_MONOTONIC) );
    ASTROS_DBG_PRINT(verbose, "CLOCK_MONOTONIC_COARSE : hr_clk_res = %d ns \n", ASTROS_PS_GET_HRCLK_RES_NS(CLOCK_MONOTONIC_COARSE) );

    
    astros_batters_set_ioengine(pBatter, ASTROS_AIOENGINE_URING);

    pEngine = pBatter->pEngine;
    
    ASTROS_DBG_PRINT(verbose, "astros_test_ioengine()\n",0);

    if(astros_open_target(pLineup, "/dev/nvme0n1", tgtidx, ASTROS_CCB_OP_READ))
    {
        ASTROS_ASSERT(0);
        
    }
    else
    {
        pTarget = &pLineup->gametime.targets[tgtidx];

        ASTROS_DBG_PRINT(verbose, "astros_test_ioengine() pTarget = %p\n", pTarget);
    }

    
    while((pCCB = astros_get_free_ccb(pLineup)))
    {
		pCCB->io_size = 4096;

		if(pEngine->pfnPrepCCB(pEngine, pCCB, pTarget, astros_test_ioengine_callback))
        {
        	ASTROS_ASSERT(0);
        }
		
    }

	pEngine->pfnRegister(pEngine);

	while((pCCB = astros_ccb_get(&pEngine->pStartQueueHead)))
	{
		pEngine->pfnSetup(pEngine,pCCB);
	}





    start_ns = ASTROS_PS_HRCLK_GET();

    while(io_count < limit)
    {
        io_submit += pEngine->pfnQueue(pEngine);

        if(pEngine->pfnComplete(pEngine, false))
        {
            ASTROS_DBG_PRINT(verbose, "astros_test_ioengine(%d) ERROR\n", io_count);
			io_count++;	
        }
        else
        {
//            ASTROS_DBG_PRINT(verbose, "astros_test_ioengine(%d) GOOD\n", io_count);
            io_count++;
        }


    }

    end_ns = ASTROS_PS_HRCLK_GET();

    elap_ns = end_ns - start_ns;
        
    ASTROS_DBG_PRINT(verbose, "io_count = %d start_ns = %ld end_ns = %ld elap_ns = %ld\n", io_count, start_ns, end_ns, elap_ns);

    fIops = (float)io_count / (float)( (float)elap_ns / 1000000000.0);
    
    ASTROS_DBG_PRINT(verbose, "IOP/s = %f\n", fIops);




    pEngine->pfnFree(pEngine);

    return 0;

}


#ifdef ALTUVE
struct sched_attr {
	__u32 size;

	__u32 sched_policy;
	__u64 sched_flags;

	/* SCHED_NORMAL, SCHED_BATCH */
	__s32 sched_nice;

	/* SCHED_FIFO, SCHED_RR */
	__u32 sched_priority;

	/* SCHED_DEADLINE */
	__u64 sched_runtime;
	__u64 sched_deadline;
	__u64 sched_period;

	/* Utilization hints */
	__u32 sched_util_min;
	__u32 sched_util_max;

};


int astros_batter_get_policy_and_priority(batter *pBatter, int *pPolicy, int * pPriority) { return 0; } 
int astros_batter_set_priority(batter *pBatter, int rt_policy) 
{ 

#if 0 /* played around with on 5.xx kernels - doesn't compile for 4.xx - may not be required */

//	struct sched_attr;
//	struct sched_param p;
//	struct sched_param p;
	//struct altuve_sched_attr atr;
	struct sched_attr atr;
	int res ;

    int verbose = astros_get_verbosity(ASTROS_DBGLVL_INFO);
	int priority = -19 + pBatter->idx;
	int min = 0;
	int max = 0;

//	max = sched_get_priority_max(SCHED_FIFO);
//	min = sched_get_priority_min(SCHED_FIFO);
	
//	p.sched_priority = 1;
	memset(&atr, 0, sizeof(struct sched_attr));

	atr.sched_policy = SCHED_FIFO;
	atr.sched_priority = 1;


	ASTROS_DBG_PRINT(verbose, "astros_batter_set_priority(%d) SET_NICE = %d min=%d max=%d\n", pBatter->idx, priority, min, max);
//	sched_set_fifo(pBatter->id);
//	sched_set_normal(pBatter->id, priority);

//	sched_setscheduler_nocheck(pBatter->id, SCHED_FIFO, NULL);

	res = sched_setattr_nocheck(pBatter->id, &atr);

	ASTROS_DBG_PRINT(verbose, "astros_batter_set_priority(%d) res = %d \n", pBatter->idx, res);


//	ASTROS_DBG_PRINT(verbose, "astros BAT(%d) sched_priority = %d\n", pBatter->idx, p.sched_priority);
	
#else
#ifdef RHEL8
//    int verbose = astros_get_verbosity(ASTROS_DBGLVL_INFO);
//	int ret;



	
//	ASTROS_DBG_PRINT(verbose, "BAT(%d) MAX_PRI = %d\n", pBatter->idx, sched_get_priority_max(SCHED_FIFO));
//	ASTROS_DBG_PRINT(verbose, "BAT(%d) MIN_PRI = %d\n", pBatter->idx, sched_get_priority_min(SCHED_FIFO));
	





#endif







#endif

	return 0; 


} 
#else

int astros_batter_set_priority(batter *pBatter, int rt_policy)
{
#ifndef ASTROS_CYGWIN
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);
    int policy;
    struct sched_param param;
    astros_lineup *pLineup = pBatter->pvLineup;
    int batter_priority;
    pthread_getschedparam(pBatter->id, &policy, &param);

/*
#define SCHED_OTHER             0
#define SCHED_FIFO              1
#define SCHED_RR                2
#ifdef __USE_GNU
# define SCHED_BATCH            3
# define SCHED_ISO              4
# define SCHED_IDLE             5
# define SCHED_DEADLINE         6
    
*/
    batter_priority = pLineup->rt_priority - pBatter->idx;
    batter_priority--;
    ASTROS_DBG_PRINT(verbose, "Batter(%d) policy = %d sched_priority = %d batter_priority = %d\n", pBatter->idx, policy, param.sched_priority, batter_priority);
    
    param.sched_priority = batter_priority;
    
    if(0 != pthread_setschedparam(pBatter->id, rt_policy, &param))
    {
        ASTROS_ASSERT(0);
    }
#endif
    return 0;
    
}
int astros_batter_get_policy_and_priority(batter *pBatter, int *pPolicy, int * pPriority)
{
#ifndef ASTROS_CYGWIN

    struct sched_param param;

    ASTROS_ASSERT(pPolicy);
    ASTROS_ASSERT(pPriority);

    if(0 == pthread_getschedparam(pBatter->id, pPolicy, &param))
    {
        *pPriority = param.sched_priority;
        return 0;
    }
    
    return -1;
#else
	return 0;
#endif	
}

#endif

#ifdef ASTROS_CYGWIN
int cygwin_thread_run = 1;
ASTROS_THREAD_FN_RET astros_cygwin_test_thread(void *pvBatter)
{
	batter *pBatter = pvBatter;
	UINT32 cpu;
	int verbose = astros_get_verbosity(ASTROS_DBGLVL_INFO);
	ASTROS_THREAD_FN_RET retval = 0;
	int count = 0;
	
	ASTROS_GET_CPU(&cpu);

	ASTROS_DBG_PRINT(verbose, "astros_cygwin_test_thread(%d) INIT cpu = %d\n", pBatter->idx, cpu);

	while(cygwin_thread_run)
	{
		usleep(500000);

		ASTROS_GET_CPU(&cpu);
		
		ASTROS_DBG_PRINT(verbose, "astros_cygwin_test_thread(%d) COUNT cpu = %d\n", pBatter->idx, cpu);

		count++;
	}	
	
	return retval;
	
	
}
void cygwin_thread_test(void)
{
	int verbose = astros_get_verbosity(ASTROS_DBGLVL_INFO);
	int thread_count = 12;
	int i;
	batter *pBatterA;
	batter *pBatter;
	int cpu;
	
	pBatterA = ASTROS_ALLOC(64, sizeof(batter) * thread_count);

	for(i = 0; i <thread_count; i++)
	{
		pBatter = &pBatterA[i];
		pBatter->idx = i;

		if(0 != ASTROS_BATTER_ROTATION_CREATE(pBatter, astros_cygwin_test_thread, i))
        {
            ASTROS_ASSERT(0);
        }
		else
		{
			ASTROS_DBG_PRINT(verbose, "cygwin_thread_test(%d) ROTATION GOOD\n", pBatter->idx);
		}

		cpu = i;

        if(ASTROS_BATTER_SET_CPU(pBatter->id, cpu))
        {
            ASTROS_ASSERT(0);
        }
		else
		{
			ASTROS_DBG_PRINT(verbose, "cygwin_thread_test(%d) SET CPU GOOD (%d)\n", pBatter->idx, cpu);
		}

	}

	usleep(5000000);
	cygwin_thread_run = 0;
	exit(0);

}

#endif

int astros_get_rt_policty()
{
	return SCHED_FIFO;    
}    

void set_ld_bind_now() 
{
    // Set the LD_BIND_NOW environment variable to 1
    if (setenv("LD_BIND_NOW", "1", 1) != 0) {
        perror("setenv failed");
    } else {
        printf("LD_BIND_NOW set to 1\n");
    }
}


int main(int argc, char **argv)
{
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_INFO);
    astros_lineup *pLineup;
	UINT64 start_ns;
	UINT64 end_ns;
	float  elap_sec = 0.0;
	float innings_per_sec = 0.0;
	struct rlimit rlim;
	int gameid;

	getrlimit(RLIMIT_STACK, &rlim);
	astros_init_game_param();

#ifdef ASTROS_SPDK
	return astros_spdk_test();
#endif
#ifdef ASTROS_CYGWIN
	astros_cygwin_init_hifreq_clock();
#endif	

	astros_parse_cli(argc, argv);

	printf("Astros Version: %s\n", astros_get_version());
	printf("RAND_MAX = %u sizeof(int) = %lu  %lu\n", RAND_MAX, sizeof(int), lrand48() );
	printf("PTHREAD_STACK_MIN = %u sizeof(int) = %lu  %lu\n", PTHREAD_STACK_MIN, sizeof(int), lrand48() );


	
	ASTROS_DBG_PRINT(verbose, "CLOCK_REALTIME		 : hr_clk_res = %d ns\n", ASTROS_PS_GET_HRCLK_RES_NS(CLOCK_REALTIME) );
	ASTROS_DBG_PRINT(verbose, "CLOCK_MONOTONIC 	   : hr_clk_res = %d ns\n", ASTROS_PS_GET_HRCLK_RES_NS(CLOCK_MONOTONIC) );
	ASTROS_DBG_PRINT(verbose, "CLOCK_MONOTONIC_COARSE : hr_clk_res = %d ns \n", ASTROS_PS_GET_HRCLK_RES_NS(CLOCK_MONOTONIC_COARSE) );

	
    ASTROS_DBG_PRINT(verbose, "main(argc = %d) sizeof long = %d MAX_STACK = %d CUR_STACK = %d sizeof(astros_lineup) = %d\n", argc, sizeof(long), rlim.rlim_cur, rlim.rlim_max, sizeof(astros_lineup));

	

	rlim.rlim_cur = rlim.rlim_max;
	setrlimit(RLIMIT_STACK, &rlim);
	

	ASTROS_DBG_PRINT(verbose, "sizeof(ASTROS_FILE_PTR) = %d\n", sizeof(ASTROS_FILE_PTR));



    start_ns = ASTROS_PS_HRCLK_GET();




    pLineup = astros_get_lineup(argc, argv);


	if(astros_field_check(pLineup))
	{
    	ASTROS_DBG_PRINT(verbose, "Innings Planned = %d\n", pLineup->innings_planned);

    	astros_lineup_run(pLineup, 0);

		if( pLineup->inning_count)
		{
		 	innings_per_sec = ((float)pLineup->inning_count) / elap_sec;
		}

		
        ASTROS_BATTER_SLEEPUS(10000);
	
    }
	else
	{
		astros_field_monitor(pLineup);
	}

	end_ns = ASTROS_PS_HRCLK_GET();
	
	elap_sec = ((float)(end_ns - start_ns)) / 1000000000.0;

	ASTROS_DBG_PRINT(verbose, "Total Sec = %f Innings = %d innings_per_sec = %f Logfile = %s\n", 
		elap_sec, pLineup->inning_count, innings_per_sec, pLineup->pScoreCard->csvlogFn);

	gameid = pLineup->gameid.gameid;

	astros_put_gameid(gameid);
	
	astros_free_lineup(pLineup);
	
    return 0;
    
}



