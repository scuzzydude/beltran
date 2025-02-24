#include "astros.h"



astros_fixed_load gFixed_loads[2] = 
{
	{
		{    
			{0, 0, 0, 6, 3, 75, 8, 0, 0, 0, 0, 0, 0, 0},
		    {0, 0, 0, 0, 0, 51, 6, 6, 11, 0, 0, 0, 0, 0}
		},
		{30, 130},
		99	
	},		
	{
		{    
			{0, 0, 0, 9, 8, 52, 1, 23, 1, 0, 0, 0, 0, 0},
		    {0, 0, 0, 0, 0, 53, 12, 6, 3, 0, 0, 0, 0, 0}
		},
		{27, 47},
		98	
	}		

};

astros_fixed_load * astros_get_fixed_load(aioengine *pEngine)
{
	astros_fixed_load *pLoad;
	int load;
	
	astros_inning *pInning = pEngine->pvInning;
	
	

	load = INNING_PARAM(pInning, InParEngineType);

#ifdef ASTROS_DIM_LOAD_TYPE

	pLoad = &gFixed_loads[load];
#else
	pLoad = &gFixed_loads[0];
#endif



	return pLoad;
}



int astros_lineup_run(astros_lineup *pLineup, int level)
{
    int error = 0;
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);
    astros_dimension *pDim;
    int dim_index = pLineup->dimensions_order[level];
    
    pDim = &pLineup->dimmensions[dim_index];

    ASTROS_DBG_PRINT(verbose, "astros_lineup_run(level = %d dim_index = %d) = %s (%d) planned =%d\n", level, dim_index, pDim->name, pDim->current_idx, pLineup->innings_planned);

	
	//ASTROS_BATTER_SLEEPUS(pLineup->called_up_sleep_us);
	
    if(level < pLineup->dimensions_count)
    {
        while(pDim->current_idx <= pDim->max)
        {

			if(level == 0)
			{
				
				pLineup->current_row = pDim->current_idx;
				pLineup->current_col = 0;
			}

            astros_lineup_run(pLineup, level + 1);


            switch(pDim->inc_type)
            {
                case ASTROS_INC_TYPE_POW2:
                pDim->current_idx = pDim->current_idx * 2;

                break;
                
                case ASTROS_INC_TYPE_LINEAR:
                default:
                pDim->current_idx = pDim->current_idx + pDim->inc;
                break;
               

            }
            
        }

        pDim->current_idx = pDim->min;
        


    }
    else
    {
		if(false == astros_inning_infield_fly(pLineup))
		{

        	ASTROS_DBG_PRINT(verbose, "astros_lineup_run(level  %d == dimensions_count = %d) \n", level, pLineup->dimensions_count);
        	if(pLineup->bDryRun)
        	{
            	pLineup->innings_planned++;
				ASTROS_DBG_PRINT(verbose, "astros_lineup_run(------------------- %d --- LEVEL = %d ROW = %d COL = %d) \n", pLineup->innings_planned, level, pLineup->current_row, pLineup->current_col);
				pLineup->current_col++;
        	}
        	else
        	{
				bool bPlayBall = true;
				bool bDebugInning = false;
				ASTROS_DBG_PRINT(verbose, "astros_lineup_run(SET row = %d col =%d ) \n", pLineup->current_row, pLineup->current_col);
				if(pLineup->ppRowColMap)
				{	
					pLineup->ppRowColMap[pLineup->current_row][pLineup->current_col] = pLineup->inning_count;
				}
				
				if(pLineup->debug_inning > -1)
				{
					if(pLineup->debug_inning == pLineup->inning_count)
					{
						ASTROS_WAIT_SCORER_READY(pLineup->pScoreCard);
						pLineup->pScoreCard->bScore = false;
						bDebugInning = true;
					}
					else
					{
 						bPlayBall = false;
					}
				}
			
				if(bPlayBall)
				{
					if(pLineup->fnLogColumn)
					{
						pLineup->fnLogColumn(pLineup, pLineup->inning_count);
					}
					else
					{
            			astros_start_inning(pLineup, pLineup->inning_count);
					}
				}

				if(bDebugInning)
				{
					if(0 == ASTROS_SCORE_INNING(pLineup->inning_count, pLineup->pScoreCard ))
					{
						ASTROS_SCORE_CSVLOG(pLineup->inning_count, pLineup, pLineup->pScoreCard);
					}
					pLineup->pScoreCard->last_inning_updated = pLineup->inning_count;
				
				}
			
				pLineup->current_col++;
        	}
        	pLineup->inning_count++;
		}
		else
		{
			ASTROS_DBG_PRINT(verbose, "astros_lineup_run(INFIELD FLY--- %d --- LEVEL = %d ROW = %d COL = %d) \n", pLineup->innings_planned, level, pLineup->current_row, pLineup->current_col);
		}
    }

	




    return error;
}



int astros_warmup_lineup( astros_lineup *pLineup )
{

    return astros_lineup_setup_rotation(pLineup);
   
}


void astros_lineup_dump(astros_lineup *pLineup)
{
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_INFO);


	
		ASTROS_DBG_PRINT(verbose, "gametime_offset                            %d : %d\n", pLineup->gametime_offset, offsetof(astros_lineup, gametime_offset));
		ASTROS_DBG_PRINT(verbose, "lineup_size                                %d : %d\n", pLineup->lineup_size, offsetof(astros_lineup, lineup_size));      
		ASTROS_DBG_PRINT(verbose, "dimensions_count                           %d : %d\n", pLineup->dimensions_count, offsetof(astros_lineup, dimensions_count));
		ASTROS_DBG_PRINT(verbose, "master_cpu 						          %d : %d\n", pLineup->master_cpu, offsetof(astros_lineup, master_cpu));
/*

		int 						   dimensions_order[ASTROS_MAX_DIMENSION];
		astros_dimension			   dimmensions[ASTROS_MAX_DIMENSION]; 

		unsigned int				   inning_count;
		unsigned int				   innings_planned;
		unsigned int				   league;
		unsigned int				   max_io;
		unsigned int				   total_q;
		unsigned int				   total_data_buffer_size;
		unsigned int				   called_up_sleep_us;
		int 						   rt_policy;	 
		int 						   rt_priority;
		astros_gameid				   gameid;
	
	
		bool						   bDryRun;
		bool						   bRsv[3];
	
	
		int 						   current_row;
		int 						   current_col;
		int 						   max_row;
		int 						   max_col;
		int 						   debug_inning;
		unsigned int				   field;
	
	
		UINT32						   master_cpu;
	
		gametime_resources				 gametime;
	
		astros_scorecard			   *pScoreCard;
	
		int 						  **ppRowColMap;
	
	} astros_lineup;

	*/



}


extern int gDefaultUserModeLib;



int astros_lineup_get_engine(astros_lineup *pLineup, astros_inning * pInning)
{
#ifdef ALTUVE

	INNING_PARAM(pInning, InParEngineType) = ASTROS_AIOENGINE_ZOMBIE;

#ifdef ALTUVE_ZOMBIE_SMARTPQI
	INNING_PARAM(pInning, InParZombieParm) = ALTUVE_ZOMBIE_PQI_TYPE;
#endif
#ifdef ALTUVE_ZOMBIE_MEGARAIDSAS
	INNING_PARAM(pInning, InParZombieParm) = ALTUVE_ZOMBIE_MR_TYPE;
#endif


	
		


	return ASTROS_AIOENGINE_ZOMBIE;
#else

	ASTROS_ASSERT( (ASTROS_AIOENGINE_URING == (INNING_PARAM(pInning, InParEngineType))) || 
	                (ASTROS_AIOENGINE_LIBAIO == (INNING_PARAM(pInning, InParEngineType))) || 
		             (ASTROS_AIOENGINE_WINAIO == (INNING_PARAM(pInning, InParEngineType))) || 
				     (ASTROS_AIOENGINE_SYNC == (INNING_PARAM(pInning, InParEngineType))) ||
					(ASTROS_AIOENGINE_BAM_ARRAY == (INNING_PARAM(pInning, InParEngineType))));

	return INNING_PARAM(pInning, InParEngineType);
	        
//	return gDefaultUserModeLib;
#endif



}


void astros_lineup_alloc_databuffer(astros_lineup *pLineup)
{
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_INFO);
	int max_qd;
	int max_targets;
	int total_q;
	int max_single_buffer;
	int total_buffers;
	int ret;
	int max_bats;

	ASTROS_UNUSED(max_bats);


	max_qd		= pLineup->dimmensions[ASTROS_DIM_QDEPTH].max;
	max_targets = pLineup->dimmensions[ASTROS_DIM_TARGET_COUNT].max;
	max_single_buffer = pLineup->dimmensions[ASTROS_DIM_IOSIZE].max;

	if(ASTROS_FIXED_LOAD_IOSIZE == max_single_buffer)
	{
		max_single_buffer = (1 << 13) * 512;
	}
	
	max_bats = pLineup->dimmensions[ASTROS_DIM_FIXEDBATTERS].max;
	



	total_q = max_qd * max_targets; 	   

#if 0
	if(pLineup->dimmensions[ASTROS_DIM_FIXEDBATTERS].max >= ASTROS_BM_INNING_BM_FIO)
	{
		total_q = total_q * max_bats;
	}
#endif



#ifdef ALTUVE


#ifndef RHEL8
/* TODO: RHEL8 I implemented this to see if kernel space buffers was result of my slow zombie writes 
in 5.xx kernels.  It was not.  However, this is a useful feature, as it will mimic the same memory allocation 
as fio/astros_user, including SGLs (assume, all early testing is 4x).   Conversely, default kernel mode 
allocation is contiguous and should not be multiple segments.  This is useful as well when debugging 
PCIe protocol flow control issues 

Just need to port the appropriate 4.xx kernel calls

*/








	if(pLineup->bProxyUserDataBuffer)
	{
		int res = -1;
	 	struct page **page_ptrs;
	//	char *my_page_address;
		
	 	unsigned long uaddr =  (unsigned long)pLineup->pProxyUserDataBuffer;
		int num_pages = pLineup->total_data_buffer_size / 4096;
		int alloc_bytes;
		astros_umpire *pUmpire;
		int locked = 1;
		
		ASTROS_DBG_PRINT(verbose, "alloc_databuffer() ALTUVE: %px : total_size = %d\n", pLineup->pProxyUserDataBuffer, pLineup->total_data_buffer_size);

		ASTROS_DBG_PRINT(verbose, "alloc_databuffer() max_qd %d : %d\n", max_qd, pLineup->max_qd);
		ASTROS_DBG_PRINT(verbose, "alloc_databuffer() max_targets %d : %d\n", max_targets, pLineup->max_targets);
		ASTROS_DBG_PRINT(verbose, "alloc_databuffer() max_single_buffer %d : %d\n", max_single_buffer, pLineup->max_single_buffer);
		ASTROS_DBG_PRINT(verbose, "alloc_databuffer() total_buffers %d : %d\n", pLineup->total_buffers, pLineup->total_buffers);
		ASTROS_DBG_PRINT(verbose, "alloc_databuffer() total_q %d : %d\n", total_q, pLineup->total_q);

		ASTROS_ASSERT(max_qd == pLineup->max_qd);
		ASTROS_ASSERT(max_targets == pLineup->max_targets);
		ASTROS_ASSERT(max_single_buffer == pLineup->max_single_buffer);

		alloc_bytes =  sizeof(struct page *) * num_pages;
		
		page_ptrs = ASTROS_ALLOC(64, alloc_bytes);
		

		ASTROS_DBG_PRINT(verbose, "alloc_databuffer() CAL get_user_pages_fast()  uaddr = 0x%16lx num_pages = %d alloc_bytes =%d page_ptrs=%px\n", uaddr, num_pages, alloc_bytes, page_ptrs);


		res = access_ok(uaddr, pLineup->total_data_buffer_size);

		pUmpire = altuve_get_umpire();

		ASTROS_DBG_PRINT(verbose, "usr_context.mm = %px gameid = %d pLineup->gameid.gameid = %d\n", pUmpire->usr_context.mm, pUmpire->usr_context.gameid, pLineup->gameid.gameid);
		

		res = get_user_pages_remote(pUmpire->usr_context.mm, uaddr, num_pages, 1, page_ptrs, NULL, &locked);

		if(res == num_pages)
		{
			int i;
			
			ASTROS_DBG_PRINT(verbose, "alloc_databuffer() get_user_pages_remove() res =  %d num_pages = %d GOOD!\n", res, num_pages);

			for(i = 0; i < res; i++)
			{
				ASTROS_DBG_PRINT(verbose, "alloc_databuffer page_ptrs[%d] = %px  : %px \n", i, page_ptrs[i], page_address(page_ptrs[i]));
			}

			pLineup->gametime.pDataBufferBase = (void *)uaddr;
			pUmpire->usr_context.page_ptrs = page_ptrs;
			return;
			
		}
		else
		{
			ASTROS_DBG_PRINT(verbose, "alloc_databuffer() get_user_pages_remove() res =  %d num_pages = %d EXPECTED MATCH ERROR\n", res, num_pages);
		}
		
		

	}


#endif
#endif




	ASTROS_DBG_PRINT(verbose, "alloc_databuffer():: max_qd = %d max_targets = %d total_q = %d max_single_buffer = %d \n", max_qd, max_targets, total_q, max_single_buffer);

	pLineup->total_data_buffer_size = max_single_buffer * total_q;

#ifdef ALTUVE
#define ALTUVE_MAX_DB_SIZE (4 * 1024 * 1024)
		if(pLineup->total_data_buffer_size > ALTUVE_MAX_DB_SIZE)
		{
			pLineup->total_data_buffer_size = ALTUVE_MAX_DB_SIZE;
	
		}
	
#endif
	total_buffers = pLineup->total_data_buffer_size / max_single_buffer;
	
	ASTROS_DBG_PRINT(verbose, "PRE_ALLOC :: max_qd = %d max_targets = %d total_q = %d  max_single_buffer = %d total_data_buffer_size =%d total_buffers = %d\n", max_qd, max_targets, total_q, max_single_buffer, pLineup->total_data_buffer_size, total_buffers);


#ifdef ALTUVE

	ret = ASTROS_ALLOC_DATABUFFER(&pLineup->gametime.pDataBufferBase, max_single_buffer, pLineup->total_data_buffer_size);
#else
	pLineup->gametime.pDataBufferBase = ASTROS_ALLOC(max_single_buffer, pLineup->total_data_buffer_size);
	ASTROS_ASSERT(pLineup->gametime.pDataBufferBase);
	ret = 0;
#endif	


	ASTROS_DBG_PRINT(verbose, "pDataBufferBase = %p pLineup->total_data_buffer_size = %d K total_buffers = %d\n", pLineup->gametime.pDataBufferBase, pLineup->total_data_buffer_size / 1024, total_buffers);
	
	ASTROS_ASSERT(ret == 0);

	pLineup->total_q = total_q;
	pLineup->total_buffers = total_buffers;

	pLineup->max_row = max_qd;
	pLineup->max_targets = max_targets;
	pLineup->max_single_buffer = max_single_buffer;
	pLineup->max_qd = max_qd;
	

}






void astros_lineup_prep_databuffers(astros_lineup *pLineup)
{
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);
//	int max_qd;
//	int max_targets;
//	int total_q;
	int ccb_size;
	ccb *pCCB, *pNext;
//	int max_single_buffer;
//	int total_buffers;
//c+	int ret;	  
    unsigned char *pBuf;
	int i;
	


	astros_lineup_alloc_databuffer(pLineup);

	ccb_size = pLineup->total_q * sizeof(ccb);
	
	ASTROS_DBG_PRINT(verbose, "max_qd = %d max_targets = %d total_q = %d ccb_size = %d max_single_buffer = %d total_data_buffer_size =%d\n", 
		pLineup->max_qd, pLineup->max_targets, pLineup->total_q, ccb_size, pLineup->max_single_buffer, pLineup->total_data_buffer_size);
	
	pLineup->gametime.pCCBBase = ASTROS_ALLOC(64, ccb_size);

	pLineup->gametime.sync_batter_array_size = sizeof(astros_sync_engine_batter) * pLineup->total_q;
	
	pLineup->gametime.pvSyncBatterArray = ASTROS_ALLOC(64, pLineup->gametime.sync_batter_array_size);


	
	pCCB = pLineup->gametime.pCCBBase;
	pLineup->gametime.pCCBBase = pCCB;
	
	ASTROS_ASSERT(pCCB);
	
	memset(pCCB,0,ccb_size);
	
	pNext = NULL;
	pBuf = pLineup->gametime.pDataBufferBase;
	
	for(i = 0; i < pLineup->total_q; i++)
	{
		int buf_idx;
		
		pCCB->idx = i;

		buf_idx = i % pLineup->total_buffers;
		
		pCCB->pNext = pNext;

#ifdef ALTUVE
		if(pLineup->bProxyUserDataBuffer)
		{
			pCCB->pData = page_address(altuve_get_umpire()->usr_context.page_ptrs[i]);
		}
		else
#endif

		{
			pCCB->pData = (void *)&pBuf[buf_idx * pLineup->max_single_buffer];
		}
		pNext = pCCB;
	
		ASTROS_DBG_PRINT(verbose, "CCB(%p: %05d) pData = %px pNext = %px buf_idx = %d\n", pCCB, pCCB->idx, pCCB->pData, pCCB->pNext, buf_idx);
	
		pCCB++;
	}
	
	
	pLineup->gametime.pCCBFree = pNext;

	ASTROS_DBG_PRINT(verbose, "pCCBFree = idx (%d)\n", pNext->idx);


	if(pLineup->innings_planned > 0)
	{
		int score_card_size = sizeof(astros_scorecard) + (sizeof(astros_inning) * pLineup->innings_planned); 
	
		ASTROS_DBG_PRINT(verbose, "astros_get_lineup innings_planned = %d score_card_size = %d sizeof(astros_inning) - %d \n", pLineup->innings_planned, score_card_size, sizeof(astros_inning));


#ifdef ALTUVE
		pLineup->pScoreCard = vmalloc(score_card_size);
#else
		pLineup->pScoreCard = ASTROS_ALLOC(64, score_card_size);
		ASTROS_DBG_PRINT(ASTROS_DBGLVL_ERROR, "pLineup = %px pScorecard = %px in=%d\n", pLineup, pLineup->pScoreCard, pLineup->innings_planned);
#endif


		ASTROS_ASSERT(pLineup->pScoreCard);

		pLineup->pScoreCard->total_innings = pLineup->innings_planned;
		pLineup->pScoreCard->inning_size_bytes = sizeof(astros_inning);
		pLineup->pScoreCard->current_inning = -1;
		pLineup->pScoreCard->score_card_size = score_card_size;


	}
	else
	{
		
		ASTROS_DBG_PRINT(ASTROS_DBGLVL_ERROR, "astros_get_lineup innings_planned = %d sizeof(astros_inning) - %d \n", pLineup->innings_planned, sizeof(astros_inning));
		
		ASTROS_ASSERT(0);
	}
	pLineup->inning_count = 0;


}




	

#ifdef ALTUVE
astros_lineup * astros_get_lineup(int argc, char **argv) { return NULL; }
#else
astros_lineup * astros_get_lineup(int argc, char **argv)
{
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);
    int size = sizeof(astros_lineup);
    astros_lineup *pLineup = ASTROS_ALLOC(64, size);
    int i;
	int map_size;
	int cidx = 1;

	ASTROS_UNUSED(cidx);

    if(pLineup)
    {



        memset(pLineup, 0, size);

		pLineup->szUserTag = NULL;

		astros_get_gameid(pLineup);

		pLineup->szUserTag = astros_run_tag_ptr(pLineup);

		ASTROS_GETHOSTNAME(&pLineup->gameid.szHostname[0], 128);

		ASTROS_DBG_PRINT(ASTROS_DBGLVL_INFO, "szHostname = %s\n", pLineup->gameid.szHostname);

		ASTROS_DBG_PRINT(ASTROS_DBGLVL_INFO, "szUserTag = %s\n", pLineup->szUserTag);
	
		pLineup->field = gDefaultBaseballField;


		
        ASTROS_DBG_PRINT(verbose, "Lineup Good size = %d\n", size);

		//pLineup->debug_inning = 265;

		
		pLineup->debug_inning = -1;
		
		if(ASTROS_FIELD_KERNEL == gDefaultBaseballField)
		{
			astros_kernel_mode_dim_constraints();
		}
		else
		{
			
			astros_user_mode_contstraints();
			
		}



        memcpy(&pLineup->dimmensions[0], g_Default_dimmensions, sizeof(g_Default_dimmensions));


        for(i = 0; i < ASTROS_MAX_DIMENSION; i++)
        {
            if(strlen(pLineup->dimmensions[i].name))
            {
            
                ASTROS_DBG_PRINT(verbose, "ORDER(%d) DIM[%d]  = %s\n", i, pLineup->dimmensions[i].dim_enum, pLineup->dimmensions[i].name);
                pLineup->dimensions_order[i] = i;
                pLineup->dimensions_count++;
            }   
            else
            {
                pLineup->dimensions_order[i] = ALLONES32;
                break;
            }


        }

        for(i = 0; i < ASTROS_MAX_BATTERS; i++)
        {
            pLineup->gametime.batters[i].pvLineup = pLineup;
        }

    	pLineup->fnLogColumn = NULL;

        pLineup->league = ASTROS_LEAGUE_BLOCK;
        
        ASTROS_DBG_PRINT(verbose, "dimension_count = %d\n", pLineup->dimensions_count);
		

        pLineup->called_up_sleep_us = ASTROS_CALLED_UP_SLEEP ;

        ASTROS_SPINLOCK_INIT(pLineup->gametime.playball);

        astros_lineup_draft_players(pLineup);
        


    }
	else
	{
		ASTROS_ASSERT(0);
	}

	printf("**** ENGINE cur_indx = %d\n", pLineup->dimmensions[ASTROS_DIM_ENGINE_TYPE].current_idx);

    pLineup->bDryRun = true;
    astros_lineup_run(pLineup, 0);
	
	if(0)
	{
		ASTROS_DBG_PRINT(ASTROS_DBGLVL_INFO, "astros_lineup_run(%d) inning_count = %d inning_planned = %d\n", pLineup->bDryRun, pLineup->inning_count, pLineup->innings_planned);
		exit(0);
	}
    pLineup->bDryRun = false;

	if(ASTROS_FIELD_KERNEL == pLineup->field)
	{
		pLineup->bProxyUserDataBuffer = false;

		if(pLineup->bProxyUserDataBuffer)
		{
			astros_lineup_alloc_databuffer(pLineup);

			pLineup->pProxyUserDataBuffer = pLineup->gametime.pDataBufferBase;
		}
		else
		{
			pLineup->pProxyUserDataBuffer = NULL;
		}

		ASTROS_DBG_PRINT(ASTROS_DBGLVL_INFO, "KERNEL LINEUP(%d): pProxyUserDataBuffer = %p :  pDataBufferBase = %p\n", pLineup->bProxyUserDataBuffer,
			pLineup->pProxyUserDataBuffer,  pLineup->gametime.pDataBufferBase);

		
		return pLineup;
	}


	astros_warmup_lineup(pLineup);

	astros_lineup_prep_databuffers(pLineup);
	
	
	map_size = pLineup->innings_planned * sizeof(int) + (pLineup->current_row * sizeof(int *));

	ASTROS_DBG_PRINT(ASTROS_DBGLVL_INFO, "innings_planned = %d row = %d col = %d map_size = %d\n", pLineup->innings_planned, pLineup->current_row, pLineup->current_col, map_size);

	//pLineup->ppRowColMap = (int **) ASTROS_ALLOC(64, map_size);
	pLineup->ppRowColMap = NULL; 
	/* TODO: THe purpose of the RowColMpa is for batter scaling, rather than start with 1 batter for high-IO we can look at what the scaling was in the previous QD, which is the same column, prior row */
	/* Must have messed something up with my calculations, get malloc error below (corrupted something with the initialization of multi-dim array */

	if(pLineup->ppRowColMap)
	{

		//TODO: Moved but might have been a smissed bug
		//pLineup->current_row++;

		int *pTemp = (int *)(pLineup->ppRowColMap + pLineup->current_row);
		int j;
		
		for(i = 0; i < pLineup->current_row; i++)
		{
			pLineup->ppRowColMap[i] = pTemp;

			pTemp += pLineup->current_col;

			for(j = 0; j < pLineup->current_col; j++)
			{
				pLineup->ppRowColMap[i][j] = -1;	
			}

		}

		
		ASTROS_DBG_PRINT(ASTROS_DBGLVL_INFO, "innings_planned = %d max_row = %d max_col = %d \n", pLineup->innings_planned, pLineup->max_row, pLineup->max_col);


		pLineup->current_row = 0;
		pLineup->current_col = 0;

	}
	else
	{
		pLineup->max_row = pLineup->current_row;
		pLineup->max_col = pLineup->current_col;

	}





    return pLineup;

} 
	
#endif


int astros_free_lineup(astros_lineup *pLineup)
{

	if(pLineup)
	{
		ASTROS_SCORER_POST_FINAL(pLineup, pLineup->pScoreCard);

		if(pLineup->ppRowColMap)
		{
			ASTROS_FREE(pLineup->ppRowColMap);
		}

		if(pLineup->gametime.pDataBufferBase && (pLineup->bProxyUserDataBuffer == false))
		{
			ASTROS_FREE_DATABUFFER(pLineup->gametime.pDataBufferBase, pLineup->total_data_buffer_size);
		}

		
		ASTROS_FREE_SCORECARD(pLineup->pScoreCard);


	}

    ASTROS_SPINLOCK_DESTROY(pLineup->gametime.playball);

    return 0;
}

//#define CYGWIN_DEBUG_ROTATION

int astros_lineup_setup_rotation(astros_lineup *pLineup)
{
    int err = 0;
    batter *pBatter;
#ifdef CYGWIN_DEBUG_ROTATION
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_INFO);
	void * pVfn = astros_cygwin_test_thread;
#else
	void * pVfn = astros_batter_rotation;
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);
#endif
    int i;
	int cpu;
	pLineup->rt_policy = astros_get_rt_policty();    
    
		


    if(ASTROS_UMPIRE_PREGAME(pLineup))
    {
        ASTROS_ASSERT(0);
    }


    ASTROS_DBG_PRINT(verbose, "astros_setup_rotation() Post Pregame(%d) LU=%px batter_size = %d offsetofGT=%d\n", pLineup->gametime.max_batters, pLineup, sizeof(batter), offsetof(astros_lineup, gametime));

    
    for(i = 0; i < pLineup->gametime.max_batters; i++)
    {
        pBatter = &pLineup->gametime.batters[i];
        pBatter->idx = i;
        pBatter->pvLineup = pLineup;
    


		//cpu = (ASTROS_PS_GET_NCPUS() - 1) - (i + 1);

#if 0 //def ASTROS_CYGWIN
		cpu = (i % pLineup->gametime.max_batters) + 1;
#else
		cpu = i;
#endif
		pBatter->cpu = cpu;

	ASTROS_DBG_PRINT(verbose, "astros_setup_rotation() batter(%d) %px CPU = %d\n", i, pBatter, cpu);



		
#ifdef ALTUVE_ZOMBIE_WORKQUEUES
	ASTROS_BATTER_ROTATION_CREATE(pBatter, astros_batter_rotation, cpu);

#else
		if(0 != ASTROS_BATTER_ROTATION_CREATE(pBatter, pVfn, cpu))
        {
            ASTROS_ASSERT(0);
        }
#endif        
        if(ASTROS_BATTER_SET_CPU(pBatter->id, cpu))
        {
#ifdef CYGWIN_DEBUG_ROTATION
			ASTROS_DBG_PRINT(verbose, "astros_setup_rotation() SET_CPU ERROR! idx = %d cpu = %d\n", pBatter->idx, cpu);
#else
            ASTROS_ASSERT(0);
#endif
        }

		
        
        if(ASTROS_BATTER_SET_PRIORITY(pBatter,pLineup->rt_policy))
        {
            ASTROS_ASSERT(0);
        }


    }

#ifdef ASTROS_CYGWIN
	astros_cygwin_payofff_umpire(pLineup);
#endif


    ASTROS_BATTER_SLEEPUS(100);
#ifdef CYGWIN_DEBUG_ROTATION
    ASTROS_BATTER_SLEEPUS(2000000);
	exit(0);
#endif
    astros_callup_batters(pLineup);

    return err;
}


int astros_lineup_draft_blocks(astros_lineup *pLineup)
{

#if 0
    DIR *dir;
	struct dirent *d;
	struct blkdev_cxt cxt = {};
    int count = 0;
    

    int error = 0;
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_INFO);
    

    /* taken from lsblk.c source */ 
    ASTROS_DBG_PRINT(verbose, "astros_draft_blocks = %d\n", 0);

	if (!(dir = opendir(_PATH_SYS_BLOCK)))
	{
		return EXIT_FAILURE;
    }

    
	while ((d = xreaddir(dir))) 
	{

        ASTROS_DBG_PRINT(verbose, "%d) astros_draft_blocks = %s\n", count, d->d_name);

	/*
		if (set_cxt(&cxt, NULL, NULL, d->d_name))
			goto next;

		if (is_maj_excluded(cxt.maj) || !is_maj_included(cxt.maj))
			goto next;

		process_blkdev(&cxt, NULL, 1, NULL);
    	next:
	    	reset_blkdev_cxt(&cxt);

    */

        count++;
        
	}

	closedir(dir);


    exit(0);

    return error;
#endif
return 0;


}


int astros_lineup_draft_players(astros_lineup *pLineup)
{
    int error = 0;

    
    switch(pLineup->league)
    {

        case ASTROS_LEAGUE_BLOCK:
        default :
           // error = astros_lineup_draft_blocks( pLineup);

        break;

    }




    return error;

}


void astros_lineup_first_pitch(astros_lineup      * pLineup)
{

    int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);

	ASTROS_DBG_PRINT(verbose, "astros_lineup_first_pitch(%d) call UNLOCK playball\n", 0);

#ifdef BATTER_SYNC_ATOMIC
#else
    ASTROS_SPINLOCK_UNLOCK(pLineup->gametime.playball);
#endif
	ASTROS_DBG_PRINT(verbose, "astros_lineup_first_pitch(%d) call UNLOCKED playball\n", 0);

    ASTROS_SET_ATOMIC(pLineup->gametime.atomic_batter_lock, 0);

}    





void astros_lineup_reset_target_sequentials(astros_lineup                    *pLineup, int mode)
{
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);
	target *pTarget;
	int i;

	ASTROS_DBG_PRINT(verbose, "astros_lineup_reset_target_sequentials(%px) %d\n", pLineup, mode);

	for(i = 0; i < ASTROS_MAX_TARGETS; i++)
	{
		pTarget = &pLineup->gametime.targets[i];

		memset(&pTarget->sequential_control, 0, sizeof(seq_control));

		pTarget->sequential_control.mode = mode;
	}
	

}










