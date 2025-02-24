#include "astros.h"
#include "astros_bam.h"

int  astros_bam_init(int controller_count, void **ppBamTargets);

int astros_bam_start_io_kernel(void * pvBamTarget, int io_limit, int op, int access);

//#define ASTROS_BAM_FAKE_BATTER_ROTATION

ASTROS_THREAD_FN_RET astros_batter_bam_rotation(void *pvSyncBatter)
{
	ASTROS_THREAD_FN_RET retval = 0;
    astros_sync_engine_batter *pSyncBatter = pvSyncBatter;
	sync_engine_spec    *pEngSpec;
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_INFO);
	ccb *pCCB;
	int io_limit, op, access;

	astros_inning *pInning;
	
	
	pEngSpec = pSyncBatter->pEngSpec;

	ASTROS_DBG_PRINT(verbose, "astros_batter_bam_rotation(%d) \n", pSyncBatter->idx);

	ASTROS_ASSERT(pEngSpec);
	
	pInning = pEngSpec->pvInning;
	
	ASTROS_ASSERT(pInning);
		
	io_limit = astros_inning_get_io_limit(pEngSpec->pvInning);

	access = INNING_PARAM(pInning, InParAccess);
		
	op = INNING_PARAM(pInning, InParOperation);


	pCCB = pSyncBatter->pCCB;

	ASTROS_ASSERT(pCCB->pTarget);
	
	ASTROS_ASSERT(pCCB->pTarget->pvBamTargetControl);	


	ASTROS_DBG_PRINT(verbose, "astros_batter_bam_rotation(%d) pCCB = %p pCCB->pTarget = %p pCCB->pTarget->pvBamTargetControl = %p\n", pSyncBatter->idx, pCCB, pCCB->pTarget, pCCB->pTarget->pvBamTargetControl);
	

	ASTROS_INC_ATOMIC(&pEngSpec->atomic_sync_threads_ready);

	while(ASTROS_GET_ATOMIC(pEngSpec->atomic_start_sync))
	{
		ASTROS_BATTER_SLEEPUS(20);
	}

#ifdef ASTROS_BAM_FAKE_BATTER_ROTATION
	ASTROS_DBG_PRINT(verbose, "astros_batter_bam_rotation(%d) FAKE\n", pSyncBatter->idx);
	ASTROS_BATTER_SLEEPUS(10000);
#else
	ASTROS_DBG_PRINT(verbose, "astros_batter_bam_rotation(%d) io_limit = %d op = %d access = %d\n", pSyncBatter->idx, io_limit, op, access);
	astros_bam_start_io_kernel(pCCB->pTarget->pvBamTargetControl, io_limit, op, access);
#endif
	ASTROS_INC_ATOMIC(&pEngSpec->atomic_sync_batter_done);

	ASTROS_DBG_PRINT(verbose, "astros_batter_bam_rotation(%d) RETURN\n", pSyncBatter->idx);

	return retval;
	
}

void *ppBamTargets[BAM_MAX_CONTROLLERS];

int astros_field_get_bam_targets(astros_lineup *pLineup)
{
	int i;
	int tgt_count = 0;
	int controller_count = 1;
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_INFO);
	target *pTarget;
	
//	tgt_count = astros_bam_init(pLineup, 1);

	ASTROS_DBG_PRINT(verbose, "astros_field_get_bam_targets(%p)\n", pLineup);

	tgt_count = astros_bam_init(controller_count, ppBamTargets);
	
	for(i = 0; i < tgt_count; i++)
	{
	
		pTarget = &pLineup->gametime.targets[i];
		ASTROS_ASSERT(pTarget);
		pTarget->idx = i;
		pTarget->pvBamTargetControl = ppBamTargets[i];

		
		ASTROS_DBG_PRINT(verbose, "astros_field_get_bam_targets(%p) target = %d pvBamTargetControl = %p\n", pLineup, pTarget->idx, pTarget->pvBamTargetControl);
	
	}

	return tgt_count;
}




