#include "astros.h"


void astros_inning_cleanup_atbat(astros_inning * pInning, astros_atbat *pAtbat)
{
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);

    int i;
    astros_lineup *pLineup = pInning->pvLineup;
    batter *pBatter;
	ccb *pCCB;
	int ccb_count = 0;

    ASTROS_DBG_PRINT(verbose, "astros_inning_cleanup_atbat(target_count = %d qdepth = %d inning_number = %d)\n", 
		INNING_PARAM(pInning, InParTargetCount), INNING_PARAM(pInning, InParQDepth), INNING_PARAM(pInning, InParInningNumber));

    for(i = 0; i < pAtbat->batters; i++)
    {
        pBatter = &pLineup->gametime.batters[i];
		ASTROS_DBG_PRINT(verbose, "astros_inning_cleanup_atbat(%d) pfnFree\n", i);

		ccb_count = 0;

		if(pBatter->pEngine)
		{
		

			while( (pCCB = astros_ccb_get(&pBatter->pEngine->pPendingHead)))
			{
				ASTROS_DBG_PRINT(verbose, "astros_inning_cleanup_atbat(%d) pfnFree CCB->idx =%d count = %d\n", i, pCCB->idx, ccb_count);

				astros_put_free_ccb((astros_lineup *)pInning->pvLineup, pCCB);

				ccb_count++;
			}
			ASTROS_DBG_PRINT(verbose, "astros_inning_cleanup_atbat(%d) pfnFree ccb_count = %d\n", i, ccb_count);

			pBatter->pEngine->pfnFree(pBatter->pEngine);


			memset(pBatter->pEngine, 0, sizeof(aioengine));
		}
    }

	ASTROS_DBG_PRINT(verbose, "astros_inning_cleanup_atbat ccb_count = (%d) \n", ccb_count);


}


void astros_inning_cleanup_inning(astros_inning * pInning)
{

	if(pInning->pPitches)
	{
#ifndef ALTUVE_ZOMBIE		
		ASTROS_FREE(pInning->pPitches);
#endif
	}

	ASTROS_FREE(pInning);

}

void astros_signs_kstats_save_historgram(astros_kernel_stats *pKStats, astros_inning *pInning, int at_bat_idx)
{
#ifdef ASTROS_GET_KSTATS

    int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);
	char histpath[128];
	astros_lineup *pLineup = pInning->pvLineup;
	int i, j, k, idx;
	astros_time_histogram *pHist;
	FILE *fd;
	UINT64 last_ns;
	
	sprintf(histpath, "hist/%d.%d.%d.hist.csv", pLineup->gameid.gameid, INNING_PARAM(pInning, InParInningNumber), at_bat_idx);

	fd = fopen(histpath, "w");

	if(fd)
	{
		ASTROS_DBG_PRINT(verbose, "astros_score_open_csvlog() OPENED fn = %s\n", histpath);
		
	}	
	else
	{
		ASTROS_ASSERT(0);
	}


	ASTROS_DBG_PRINT(verbose, "astros_signs_kstats_save_historgram(%s)\n", histpath);
	
	for(i = 0; i < ASTROS_MAX_BATTERS; i++)
	{
		for(j = 0; j < 2; j++)
		{
			if(j)
			{
				fprintf(fd, "COMPLETIONS,%d,%d",i, at_bat_idx);
				pHist = &pKStats->batterStats[i].hCompletions;
			}
			else
			{
				fprintf(fd, "SUBMITS,%d,%d",i, at_bat_idx);
				pHist = &pKStats->batterStats[i].hSubmits;
				
			}

			last_ns = 0;	


			if(pHist)
			{
				for(k = 0; k < ASTROS_TIME_HISTORGRAM_CNT; k++)
				{
					idx = (pHist->idx + k) % ASTROS_TIME_HISTORGRAM_CNT;
				
					ASTROS_DBG_PRINT(verbose, "%d:%d:%d][%d] ns = %ld\n", i,j,k, idx, pHist->ns[idx]);
					
					fprintf(fd, ",%ld", pHist->ns[idx]);


					ASTROS_ASSERT(last_ns <= pHist->ns[idx]);
					
					last_ns = pHist->ns[idx];

					
					
				}


			}

			fprintf(fd,"\n");


		}


	}

	if(fd)
	{
		fclose(fd);
	}



#endif

}

#ifdef ASTROS_GET_KSTATS
astros_kernel_stats kStats;
#endif

void astros_inning_atbat_calculate(astros_atbat *pAtbat, astros_inning *pInning, int at_bat_idx)
{
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
#ifdef ASTROS_GET_KSTATS
	int total_submits = 0;
	int other_submits = 0;
	int total_comps = 0;
	int other_comps = 0;
	int total_irqs = 0;
	int irq_responses = 0;
	bool bGoodKStats = false;
#endif



#ifdef ASTROS_INC_AVG_OFFSET
	astros_avg_offset avg_off;
	memset(&avg_off, 0, sizeof(astros_avg_offset));
#endif	
	memset(&cmd_lat, 0, sizeof(astros_latency));
	memset(&inter_lat, 0, sizeof(astros_latency));


	

	pAtbat->fIops = 0.0;
	pAtbat->iops = 0;

	ASTROS_DBG_PRINT(verbose, "astros_inning_atbat_calculate(batters = %d)\n", pAtbat->batters);

#ifdef ASTROS_GET_KSTATS
	if(astros_signs_read_kstats(pInning->pvLineup, &kStats, sizeof(kStats)))
	{
		ASTROS_DBG_PRINT(ASTROS_DBGLVL_ERROR, "ERROR read kstats %d\n", at_bat_idx);

	}
	else
	{
		astros_signs_kstats_save_historgram(&kStats, pInning, at_bat_idx);

		bGoodKStats = true;
	}
#endif
		


	
	

	
	for(i = 0; i < pAtbat->batters; i++)
	{
		elap_ns = pAtbat->pitches[i].elap_ns;	
		io_count = pAtbat->pitches[i].io_count;


		start_ns = pAtbat->pitches[i].start_ns;
		end_ns = pAtbat->pitches[i].end_ns;


		ASTROS_DBG_PRINT(verbose, "astros_inning_atbat_calculate(%d) IOPS = %d  io_count = %d elap_ns = %ld\n", i, pAtbat->iops, io_count, elap_ns);

#ifdef ASTROS_GET_KSTATS
		if(bGoodKStats)
		{
			total_submits += kStats.batterStats[i].submit_count;
			other_submits += kStats.batterStats[i].other_cpu_submit;	

			total_comps += kStats.batterStats[i].complete_count;
			other_comps += kStats.batterStats[i].other_cpu_complete;

			total_irqs += kStats.batterStats[i].irq_count;
			irq_responses += kStats.batterStats[i].irq_total_responses;
			

		}
#endif



		
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



#ifdef ASTROS_GET_KSTATS
	if(total_submits)
	{
		pAtbat->kstats.fOtherBatSubmit = ((float)other_submits / (float)total_submits) * 100.0;
	
	}
	if(total_comps)
	{
		pAtbat->kstats.fOtherBatComplete = ((float)other_comps / (float)total_comps) * 100.0;
	}

	if(total_submits)
	{
		pAtbat->kstats.fIrqPercentSubmits = ((float)total_irqs / (float)total_submits) * 100.0;
	}

	if(total_irqs)
	{
		pAtbat->kstats.fRespPerIrq = (float)irq_responses / (float)total_irqs;

	}
	
#endif

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
		UINT64 iBps = (UINT64)pAtbat->iops * (UINT64)INNING_PARAM(pInning, InParIoSize);
		
		ASTROS_DBG_PRINT(verbose, "astros_inning_atbat_calculate(qd = %d at_bat=%d fIops = %f iops = %ld)\n", 
			INNING_PARAM(pInning, InParQDepth), pAtbat->atbat_number, pAtbat->fIops, pAtbat->iops);
		ASTROS_DBG_PRINT(verbose, "astros_inning_atbat_calculate(first_start_ns=%ld last_start_ns=%ld first_end_ns= %ld last_end_ns=%ld)\n", first_start_ns, last_start_ns, first_end_ns, last_end_ns);
		ASTROS_DBG_PRINT(verbose, "astros_inning_atbat_calculate(inner_span_ns=%ld outer_span_ns=%ld delta_span_ns= %ld last_end_ns=%ld)\n", inner_span_ns, outer_span_ns, delta_span_ns);
		ASTROS_DBG_PRINT(verbose, "astros_inning_atbat_calculate(start_delta=%ld end_delta=%ld)\n", start_delta, end_delta);


		if(ASTROS_INNING_BM_DYNAMIC != INNING_PARAM(pInning, InParBatterMode)) 
		{
			ASTROS_DBG_PRINT(ASTROS_DBGLVL_INFO, "astros_inning_atbat_calculate[%d](start_delta=%f  end_delta=%f) batters = %d iops = %d Bps = %ld\n", at_bat_idx, fStartDelta, fEndDelta, pAtbat->batters, pAtbat->iops, iBps);

//			ASTROS_ASSERT(fStartDelta < 5.0);
		
//			ASTROS_ASSERT(fEndDelta < 5.0);

		}
		
		pAtbat->fStartDelta = fStartDelta;
		pAtbat->fEndDelta = fEndDelta;
		
		
	}
	ASTROS_FP_END()


}


int astros_inning_get_last_dynamic_batter(astros_inning * pInning)
{
	int batters = 1;
	int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);

	astros_lineup *pLineup = pInning->pvLineup;
	astros_scorecard *pScoreCard = pLineup->pScoreCard;
	astros_inning *pTempInning;
	int i;
	int last_inning = INNING_PARAM(pInning, InParInningNumber);
	int row = INNING_PARAM(pInning, InParRow);
	int col = INNING_PARAM(pInning, InParCol);


	ASTROS_DBG_PRINT(verbose, "astros_inning_get_last_dynamic_batter(%d) row = %d col = %d\n", last_inning, row, col);

	if(row > 1)
	{


		for(i = 0; i < last_inning; i++)
		{
			int tRow;	
			int tCol;
			
			pTempInning = &pScoreCard->innings[i];
			tRow = INNING_PARAM(pTempInning, InParRow);
			tCol = INNING_PARAM(pTempInning, InParCol); 
		
			ASTROS_DBG_PRINT(verbose, "astros_inning_get_last_dynamic_batter(%d)\n", i, tRow, tCol);
			
			if(col == tCol)
			{
				if(tRow == (row - 1))
				{
					batters = INNING_PARAM(pTempInning, InParScoringBatters); 

					if(batters > 1)
					{
						batters--;  //we want to scale up, but not start all the way from 1 
					}


					ASTROS_DBG_PRINT(verbose, "astros_inning_get_last_dynamic_batter(%d) FOUND LAST=%d scoring_batters = %d\n", INNING_PARAM(pTempInning, InParInningNumber), batters);
				}


			}

	
		


		
		}	

	}

	return batters;
	

}



void astros_inning_score(astros_inning * pInning)
{
	int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);
	astros_lineup *pLineup = pInning->pvLineup;
	astros_scorecard *pScoreCard = pLineup->pScoreCard;
	int ioff;
	

	ASTROS_ASSERT(pScoreCard);

	ioff = ((char *)(&pScoreCard->innings[INNING_PARAM(pInning, InParInningNumber)])) - ((char *)pScoreCard);
	
    ASTROS_DBG_PRINT(verbose, "astros_inning_score(inning =%d qd = %d io_size = %d io_count = %d measure_at_bats = %d ioff/of %d:%d)(%d)\n", 
		INNING_PARAM(pInning, InParInningNumber), INNING_PARAM(pInning, InParQDepth), INNING_PARAM(pInning, InParIoSize),  
		INNING_PARAM(pInning, InParSustainedIos), INNING_PARAM(pInning, InParMeasuringAtBats), ioff, pScoreCard->score_card_size, sizeof(astros_inning));

		
	memcpy(&pScoreCard->innings[INNING_PARAM(pInning, InParInningNumber)], pInning, sizeof(astros_inning));

	

	pScoreCard->current_inning = INNING_PARAM(pInning, InParInningNumber);



}


#define ASTROS_INN_SCALE_ST_CONTINUE 0
#define ASTROS_INN_SCALE_ST_BREAK    1
#define ASTROS_INN_SCALE_ST_DONE     2
#define ASTROS_INN_SCALE_POW2        3
#define ASTROS_INN_SCALE_ST_OOO      4


#define ASTROS_INN_SCALE_METHOD_SIMPLE 0 
#define ASTROS_INN_SCALE_METHOD_EXP    1 

int astros_inning_scaling_simple(astros_inning * pInning, int incoming_state, int max_batters, int *pScoring_batters)
{
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);
	int state = ASTROS_INN_SCALE_ST_CONTINUE;
	int scale_at_bat = INNING_PARAM(pInning, InParScaleAtBat);
	int current_at_bat = INNING_PARAM(pInning, InParAtBat);
	int scaling_at_bats = current_at_bat - scale_at_bat;
	int i;
	int iops;
    astros_atbat *pAtbat;
	int max_iops = -1;
	int max_iops_idx = -1;
	int last;

	
    ASTROS_DBG_PRINT(verbose, "astros_inning_scaling_simple(inning =%d qd = %d io_size = %d) scale_at_bat = %d current_at_bat = %d scaling_at_bats = %d scoring_batters = %d\n", 
    	INNING_PARAM(pInning, InParInningNumber), INNING_PARAM(pInning, InParQDepth), INNING_PARAM(pInning, InParIoSize), scale_at_bat, current_at_bat, scaling_at_bats, *pScoring_batters);

	if(scaling_at_bats > 1)
	{

		for(i = scale_at_bat; i < scaling_at_bats; i++)
		{
			pAtbat = &pInning->atbats[i];
			iops = pAtbat->iops;
			
			if(iops > max_iops)
			{
				max_iops = iops;
				max_iops_idx = i;

			}
			
			ASTROS_DBG_PRINT(verbose, "astros_inning_scaling_simple(%d) iops = %d max_iops = %d max_iops_idx = %d\n", i, iops, max_iops, max_iops_idx); 

		}

		last = i - 1;	

		ASTROS_DBG_PRINT(verbose, "astros_inning_scaling_simple() LAST  = %i\n", last); 

		if((max_iops_idx < last) || 
		(*pScoring_batters >= max_batters ))
		{
			pAtbat = &pInning->atbats[max_iops_idx];
			*pScoring_batters = pAtbat->batters;
			state = ASTROS_INN_SCALE_ST_DONE;
			
			INNING_PARAM(pInning, InParScalingBestAtBat) = max_iops_idx;
		}
		else 
		{
			*pScoring_batters = (*pScoring_batters + 1);
		}

	}
	else
	{
		*pScoring_batters = (*pScoring_batters + 1);
	}
	

	return state;
}



int get_next_power_of_two(int val)
{
	int next;
	int temp = 0;
	int orig = val;
	int verbose = astros_get_verbosity(ASTROS_DBGLVL_INFO);

	while(val)
	{
		val = val >> 1;
		temp++;
	}

	next = 1 << (temp);

	if(next == orig)
	{
		next = next << 1;
	}



	ASTROS_DBG_PRINT(verbose, "get_next_power_of_two(%d) next = %d\n", orig, next);

	return next;

}




int astros_inning_scaling_exp(astros_inning * pInning, int incoming_state, int max_batters, int *pScoring_batters)
{
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);
	int state = ASTROS_INN_SCALE_ST_CONTINUE;
	int scale_at_bat = INNING_PARAM(pInning, InParScaleAtBat);
	int current_at_bat = INNING_PARAM(pInning, InParAtBat);
	int scaling_at_bats = current_at_bat - scale_at_bat;
	int i;
	int iops;
    astros_atbat *pAtbat;
	int max_iops = -1;
	int max_iops_idx = -1;
	int last;

	
    ASTROS_DBG_PRINT(verbose, "astros_inning_scaling_exp(inning =%d qd = %d io_size = %d) scale_at_bat = %d current_at_bat = %d scaling_at_bats = %d scoring_batters = %d\n", 
    	INNING_PARAM(pInning, InParInningNumber), INNING_PARAM(pInning, InParQDepth), INNING_PARAM(pInning, InParIoSize), scale_at_bat, current_at_bat, scaling_at_bats, *pScoring_batters);

	if(scaling_at_bats > 1)
	{

		for(i = scale_at_bat; i < scaling_at_bats; i++)
		{
			pAtbat = &pInning->atbats[i];
			iops = pAtbat->iops;
			
			if(iops > max_iops)
			{
				max_iops = iops;
				max_iops_idx = i;

			}
			
			ASTROS_DBG_PRINT(verbose, "astros_inning_scaling_exp(%d) iops = %d max_iops = %d max_iops_idx = %d\n", i, iops, max_iops, max_iops_idx); 

		}

		last = i - 1;	

		ASTROS_DBG_PRINT(verbose, "astros_inning_scaling_exp() LAST  = %i\n", last); 

		if(*pScoring_batters >= max_batters )
		{
			pAtbat = &pInning->atbats[max_iops_idx];
			*pScoring_batters = pAtbat->batters;
			state = ASTROS_INN_SCALE_ST_DONE;
			
			INNING_PARAM(pInning, InParScalingBestAtBat) = max_iops_idx;
		}
		else if(max_iops_idx < last)
		{
			pAtbat = &pInning->atbats[max_iops_idx];

			if(pAtbat->scale_scratch == 0)
			{	
				pAtbat->scale_scratch = 1;
				
				*pScoring_batters = get_next_power_of_two(pAtbat->batters);
				state = ASTROS_INN_SCALE_POW2;
				
			}
			else
			{

				*pScoring_batters = pAtbat->batters;
				state = ASTROS_INN_SCALE_ST_DONE;
			}

		}
		else 
		{
			*pScoring_batters = (*pScoring_batters + 1);
		}

	}
	else
	{
		*pScoring_batters = (*pScoring_batters + 1);
	}
	
	ASTROS_DBG_PRINT(ASTROS_DBGLVL_INFO, "astros_inning_scaling_exp(%d) state = %d scoring_batters  = %d\n", current_at_bat, state, *pScoring_batters); 

	return state;
}


int astros_inning_scaling(astros_inning * pInning, int incoming_state, int max_batters, int *pScoring_batters)
{
//	int method = ASTROS_INN_SCALE_METHOD_SIMPLE;
	int method = ASTROS_INN_SCALE_METHOD_EXP;
	int state = ASTROS_INN_SCALE_ST_OOO;



	

	switch(method)
	{
		case ASTROS_INN_SCALE_METHOD_EXP:
		state = astros_inning_scaling_exp(pInning, incoming_state, max_batters, pScoring_batters);
		break;
		

	
		case ASTROS_INN_SCALE_METHOD_SIMPLE:
		default:
			state = astros_inning_scaling_simple(pInning, incoming_state, max_batters, pScoring_batters);
		break;

	}


	return state;
	

}


void astros_inning_bat_ready(astros_atbat *pAtbat, astros_inning * pInning, int scoring_batters, int size)
{
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);

	pAtbat->batters = scoring_batters;

	ASTROS_ASSERT(pInning->pPitches);
	
	pAtbat->pitches = pInning->pPitches;

	memset(pAtbat->pitches, 0, size);

	ASTROS_DBG_PRINT(verbose, "astros_inning_bat_ready(pAtbat = %px) pitches = %px batters=%d\n", pAtbat, pAtbat->pitches, scoring_batters);		

}
				

int astros_start_inning_async(astros_lineup *pLineup, int inning_number, astros_inning * pInning)
{
    int error = 0;
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);
    astros_inning * pPrecon;
	astros_atbat *pAtbat;
	int total_queue;
	int measuring_at_bats;
	int scoring_batters;	
	int i;
	int pitches_size = sizeof(astros_pitches) * ASTROS_MAX_BATTERS;

	ASTROS_ASSERT(pLineup == pInning->pvLineup);

	ASTROS_ASSERT(pLineup->pScoreCard);

	scoring_batters = INNING_PARAM(pInning, InParFixedBatters);

	total_queue = INNING_PARAM(pInning, InParTargetCount) * INNING_PARAM(pInning, InParQDepth);

	measuring_at_bats = INNING_PARAM(pInning, InParMeasuringAtBats);

	pInning->pPitches = ASTROS_ALLOC(64, pitches_size);

	ASTROS_ASSERT(pInning->pPitches);
	




		
    ASTROS_DBG_PRINT(verbose, "astros_start_inning_async(inning =%d qd = %d io_size = %d qd_trough = %d burst_percent = %d op = %d)\n", inning_number, 
		INNING_PARAM(pInning, InParQDepth), INNING_PARAM(pInning, InParIoSize), 
		INNING_PARAM(pInning, InParQueueDepthTrough), INNING_PARAM(pInning, InParBurstPercent), INNING_PARAM(pInning, InParOperation));
    
    if((pPrecon = astros_inning_get_precon(pLineup, pInning)))
    {
        ASTROS_ASSERT(0);
    }



	if(  (ASTROS_BM_INNING_BM_BATPERTARGET == INNING_PARAM(pInning, InParBatterMode)) || 
		 (ASTROS_BM_INNING_BM_FIO  == INNING_PARAM(pInning, InParBatterMode)) )
	{
		scoring_batters = scoring_batters * INNING_PARAM(pInning, InParTargetCount);

		ASTROS_DBG_PRINT(verbose, "BM_BATPERTARGET(%d) scoring_batters =%d max_batters=%d\n", 
			inning_number, scoring_batters, pLineup->gametime.max_batters);

		if(scoring_batters > pLineup->gametime.max_batters)
		{

			scoring_batters = pLineup->gametime.max_batters;
		}


	}
    else if(ASTROS_INNING_BM_DYNAMIC == INNING_PARAM(pInning, InParBatterMode))
    {
		int state = ASTROS_INN_SCALE_ST_OOO;
		int max_scaling_at_bats = ASTROS_MAX_ATBATS - measuring_at_bats;
		int max_batters = pLineup->gametime.max_batters;
		int atbmax = max_batters;

		
		ASTROS_DBG_PRINT(verbose, "astros_start_inning_async(%d) max_scaling_at_bats =%d total_queue=%d max_batters=%d\n", inning_number, max_scaling_at_bats, total_queue, max_batters);

		pInning->bScaling = true;
		

		if(total_queue > 1)
		{
			
			scoring_batters = astros_inning_get_last_dynamic_batter(pInning);

			if(scoring_batters > 1)
			{
				scoring_batters--;
			}


			atbmax = MIN(max_scaling_at_bats - 1, max_batters - 1);
			atbmax = MIN(atbmax, total_queue);

			ASTROS_DBG_PRINT(verbose, "astros_start_inning_async(%d) atbmax = %d\n", atbmax); 

			while(scoring_batters <= atbmax)
			{
				
				INNING_PARAM(pInning, InParScoringBatters) = scoring_batters;
			
				pAtbat = &pInning->atbats[INNING_PARAM(pInning, InParAtBat)];
				
				pAtbat->batters = scoring_batters;

				astros_inning_bat_ready(pAtbat, pInning, scoring_batters, pitches_size);
				
				astros_batters_up(pInning, pAtbat);
				
				astros_inning_atbat_calculate(pAtbat, pInning, INNING_PARAM(pInning, InParAtBat));
				
				ASTROS_DBG_PRINT(verbose, "astros_start_inning_async(%d) SCALING batters =%d iops = %d atbmax = %d\n", inning_number, pAtbat->batters, pAtbat->iops, atbmax);
				
				astros_inning_cleanup_atbat(pInning, pAtbat);

				INNING_PARAM(pInning, InParAtBat)++;

				state = astros_inning_scaling(pInning, state, atbmax, &scoring_batters);

				if((state == ASTROS_INN_SCALE_ST_CONTINUE) || (state ==  ASTROS_INN_SCALE_POW2))
				{

				}
				else if(state == ASTROS_INN_SCALE_ST_DONE)
				{
				
					ASTROS_DBG_PRINT(verbose, "astros_start_inning_async(%d) SCALING DONE = scoring_batters = %d\n", inning_number, scoring_batters);
					break;

				}

				
			}
			



		}

		pInning->bScaling = false;


    }



	ASTROS_DBG_PRINT(ASTROS_DBGLVL_INFO, "*** (inning = %d of %d qd = %d - bats = %d - op = %s engine = %d access = %d TGTS %d ** psz=%d lat_en=%d mode=%d fixed_bats=%d burst_per=%d load=%d iosize(b)=%d ioab=%d\n", 
		inning_number, 
		pLineup->innings_planned, 
		INNING_PARAM(pInning, InParQDepth), 
		scoring_batters, 
		((INNING_PARAM(pInning, InParOperation) == ASTROS_CCB_OP_WRITE) ? "write" : "read" ), INNING_PARAM(pInning, InParEngineType), INNING_PARAM(pInning, InParAccess), INNING_PARAM(pInning, InParTargetCount), pitches_size, 
		INNING_PARAM(pInning, InParEnableLatency), INNING_PARAM(pInning, InParInningMode), INNING_PARAM(pInning, InParFixedBatters), INNING_PARAM(pInning, InParBurstPercent), INNING_PARAM(pInning, InParLoadType),  INNING_PARAM(pInning, InParIoSize), INNING_PARAM(pInning, InParSustainedIos) / 1024);

	

	INNING_PARAM(pInning, InParScoringBatters) = scoring_batters;

	INNING_PARAM(pInning, InParScoreAtBat) = INNING_PARAM(pInning, InParAtBat);

	for(i = 0; i < measuring_at_bats; i++)
	{
 

		ASTROS_ASSERT(INNING_PARAM(pInning, InParAtBat) < ASTROS_MAX_ATBATS);


    	pAtbat = &pInning->atbats[INNING_PARAM(pInning, InParAtBat)];


    	pAtbat->batters = scoring_batters;

		ASTROS_DBG_PRINT(verbose, "astros_start_inning_async(SCORE total_queue = %d batters = %d measuring_atbat_idx = %d pInning->atbat = %d)\n", 
			total_queue, pAtbat->batters, i, INNING_PARAM(pInning, InParAtBat));


				


		if(pAtbat->batters > total_queue)
		{
			scoring_batters = total_queue;
		}

		astros_inning_bat_ready(pAtbat, pInning, scoring_batters, pitches_size);

		ASTROS_DBG_PRINT(verbose, "astros_start_inning_async(actual_batters = %d total_queue =  %d)\n", pAtbat->batters, total_queue);
	

    	astros_batters_up(pInning, pAtbat);

		ASTROS_DBG_PRINT(verbose, "astros_start_inning_async batters_up return =%d\n", inning_number); 

		astros_inning_atbat_calculate(pAtbat, pInning, INNING_PARAM(pInning, InParAtBat));


    	astros_inning_cleanup_atbat(pInning, pAtbat);

		INNING_PARAM(pInning, InParAtBat)++;

		ASTROS_BATTER_SLEEPUS(100);
		
	}

	astros_inning_score(pInning);
	
	astros_inning_cleanup_inning(pInning);


	

    return error;
}


//**************************** SYNC **********************************************************
//**************************** SYNC **********************************************************
//**************************** SYNC **********************************************************

#ifndef ALTUVE

int astros_start_inning_sync(astros_lineup *pLineup, int inning_number, astros_inning * pInning)
{
    int error = 0;
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_INFO);
//    astros_inning * pInning;
    astros_inning * pPrecon;
	astros_atbat *pAtbat;
	int total_queue;
	int measuring_at_bats;
	int scoring_batters;	
	int i;
	int pitches_size = sizeof(astros_pitches) * ASTROS_MAX_BATTERS;
	
//    pInning = astros_get_inning(pLineup, inning_number);
	INNING_PARAM(pInning, InParEngineType) = pLineup->dimmensions[ASTROS_DIM_ENGINE_TYPE].current_idx;

	ASTROS_DBG_PRINT(verbose, "astros_start_inning_sync() ENGINE TYPE = %d\n", INNING_PARAM(pInning, InParEngineType));



//	ASTROS_ASSERT(pInning);

	ASTROS_ASSERT(pLineup == pInning->pvLineup);

	ASTROS_ASSERT(pLineup->pScoreCard);

	scoring_batters = INNING_PARAM(pInning, InParFixedBatters);

	total_queue = INNING_PARAM(pInning, InParTargetCount) * INNING_PARAM(pInning, InParQDepth);

	measuring_at_bats = INNING_PARAM(pInning, InParMeasuringAtBats);

	pInning->pPitches = ASTROS_ALLOC(64, pitches_size);

	ASTROS_ASSERT(pInning->pPitches);
	




		
    ASTROS_DBG_PRINT(verbose, "astros_start_inning_sync(inning =%d qd = %d io_size = %d qd_trough = %d burst_percent = %d op = %d)\n", inning_number, 
		INNING_PARAM(pInning, InParQDepth), INNING_PARAM(pInning, InParIoSize), 
		INNING_PARAM(pInning, InParQueueDepthTrough), INNING_PARAM(pInning, InParBurstPercent), INNING_PARAM(pInning, InParOperation));
    
    if((pPrecon = astros_inning_get_precon(pLineup, pInning)))
    {
        ASTROS_ASSERT(0);
    }





	ASTROS_DBG_PRINT(ASTROS_DBGLVL_INFO, "*** (inning = %d of %d qd = %d - bats = %d - op = %s engine = %d *** Target %d **SYNC** psz=%d lat_en=%d mode=%d fixed_bats=%d burst_per=%d\n", 
		inning_number, 
		pLineup->innings_planned, 
		INNING_PARAM(pInning, InParQDepth), 
		scoring_batters, 
		((INNING_PARAM(pInning, InParOperation) == ASTROS_CCB_OP_WRITE) ? "write" : "read" ), INNING_PARAM(pInning, InParEngineType), INNING_PARAM(pInning, InParTargetCount), pitches_size, 
		INNING_PARAM(pInning, InParEnableLatency), INNING_PARAM(pInning, InParInningMode), INNING_PARAM(pInning, InParFixedBatters), INNING_PARAM(pInning, InParBurstPercent) );

	

	INNING_PARAM(pInning, InParScoringBatters) = scoring_batters;

	INNING_PARAM(pInning, InParScoreAtBat) = INNING_PARAM(pInning, InParAtBat);

	for(i = 0; i < measuring_at_bats; i++)
	{

		ASTROS_ASSERT(INNING_PARAM(pInning, InParAtBat) < ASTROS_MAX_ATBATS);


    	pAtbat = &pInning->atbats[INNING_PARAM(pInning, InParAtBat)];


    	pAtbat->batters = scoring_batters;

		ASTROS_DBG_PRINT(verbose, "astros_start_inning_sync(SCORE total_queue = %d batters = %d measuring_atbat_idx = %d pInning->atbat = %d)\n", 
			total_queue, pAtbat->batters, i, INNING_PARAM(pInning, InParAtBat));

		if(pAtbat->batters > total_queue)
		{
			scoring_batters = total_queue;
		}

		astros_inning_bat_ready(pAtbat, pInning, scoring_batters, pitches_size);


		ASTROS_DBG_PRINT(verbose, "astros_start_inning_sync(actual_batters = %d total_queue =  %d)\n", pAtbat->batters, total_queue);
	

    	astros_sync_batters_up(pInning, pAtbat);

		

		ASTROS_DBG_PRINT(verbose, "astros_start_inning_sync batters_up return =%d\n", inning_number); 

		astros_sync_inning_atbat_calculate(pAtbat, pInning, INNING_PARAM(pInning, InParAtBat));


		astros_sync_batter_cleanup(pInning, pAtbat);

		

    	astros_inning_cleanup_atbat(pInning, pAtbat);

		INNING_PARAM(pInning, InParAtBat)++;

		ASTROS_BATTER_SLEEPUS(100);

		
	}

	astros_inning_score(pInning);
	
	astros_inning_cleanup_inning(pInning);

	
	ASTROS_DUMP_MEMORY("VmSize:", "END INNING");
	

    return error;
}
#endif

int astros_start_inning(astros_lineup *pLineup, int inning_number)
{
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);
	
	int bat_mode;
	astros_inning * pInning;
		
	pInning = astros_get_inning(pLineup, inning_number);
	
	ASTROS_ASSERT(pInning);

	bat_mode = INNING_PARAM(pInning, InParBatterMode);

	if(ASTROS_SEQ_MODE_SPLIT_STREAM_RST_INNING == INNING_PARAM(pInning, InParAccess))
	{
		astros_lineup_reset_target_sequentials(pLineup, ASTROS_SEQ_MODE_SPLIT_STREAM_RST_INNING);

	}
		
	ASTROS_DBG_PRINT(verbose, "astros_start_inning batmode=%d fixbat=%d\n", bat_mode, g_Default_dimmensions[ASTROS_DIM_FIXEDBATTERS].current_idx );

#ifndef ALTUVE
	if(ASTROS_BM_SYNC_BATTERS == bat_mode)
	{
		return astros_start_inning_sync(pLineup, inning_number, pInning);
	}
	else
#endif
#ifdef NVIDIA_BAM
	if(ASTROS_BM_BAM == bat_mode)
	{
		return astros_start_inning_sync(pLineup, inning_number, pInning);
	}
	else
#else	
	if(ASTROS_BM_BAM == bat_mode)
	{
		ASTROS_DBG_PRINT(ASTROS_DBGLVL_ERROR, "astros not compiled with NVIDIA_BAM, get another version\n");
		ASTROS_ASSERT(0);
	}
#endif
	{
		return astros_start_inning_async(pLineup, inning_number, pInning);
	}
	
	
}




bool astros_inning_infield_fly(astros_lineup *pLineup)
{
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_IOPATH);

	/* An infield fly is to prevent unnecessary innings of paramaters that are of no value based on another parameter */
	bool bInfieldFly = false;

	if(pLineup->dimmensions[ASTROS_DIM_INNING_MODE].current_idx == ASTROS_INNING_MODE_SUSTAINED)
	{
		if(pLineup->dimmensions[ASTROS_DIM_BURST_PERCENT].current_idx != 0)
		{
			ASTROS_DBG_PRINT(verbose, "astros_inning_infield_fly() INFIELD FLY INNING_MODE = %d BURST_PERCENT = %d\n", 
				pLineup->dimmensions[ASTROS_DIM_INNING_MODE].current_idx,
				pLineup->dimmensions[ASTROS_DIM_BURST_PERCENT].current_idx);
			bInfieldFly = true;
		}
	}
#if 0//def ASTROS_DIM_LOAD_TYPE
	if(pLineup->dimmensions[ASTROS_DIM_IOSIZE].current_idx != ASTROS_FIXED_LOAD_IOSIZE)
	{
		if(pLineup->dimmensions[ASTROS_DIM_LOAD_TYPE].current_idx != 0)
		{
			bInfieldFly = true;
		}


	}
#endif


	return bInfieldFly;

}


CCBCompletionCallback astros_inning_get_ccb_callback(astros_inning *pInning)
{
	CCBCompletionCallback pfnCCBCallback = astros_ccb_callback_sustained;

	

	switch(INNING_PARAM(pInning, InParInningMode))
	{
		case ASTROS_INNING_MODE_BURST:
			pfnCCBCallback = astros_ccb_callback_burst;
			break;


		default:
			break;
			
	}

	
	return pfnCCBCallback;
	
}

astros_inning * astros_get_inning(astros_lineup *pLineup, int inning_number)
{

    astros_inning * pInning;
    int size = sizeof(astros_inning);
    int i;
	int temp;
	
    pInning = ASTROS_ALLOC(64, size);

    ASTROS_ASSERT(pInning);

    pInning->pvLineup = pLineup;
	INNING_PARAM(pInning, InParInningNumber) = inning_number;
    INNING_PARAM(pInning, InParQDepth) = pLineup->dimmensions[ASTROS_DIM_QDEPTH].current_idx;
    INNING_PARAM(pInning, InParIoSize) = pLineup->dimmensions[ASTROS_DIM_IOSIZE].current_idx;
    INNING_PARAM(pInning, InParTargetCount)	= pLineup->dimmensions[ASTROS_DIM_TARGET_COUNT].current_idx;
#ifdef ASTROS_DIM_ATBAT_COUNT
	INNING_PARAM(pInning, InParMeasuringAtBats) = pLineup->dimmensions[ASTROS_DIM_ATBAT_COUNT].current_idx;
#else
	INNING_PARAM(pInning, InParMeasuringAtBats) = DEF_AB_CNT;
#endif


    INNING_PARAM(pInning, InParSustainedIos) =  pLineup->dimmensions[ASTROS_DIM_IO_COUNT].current_idx;

	if(GAME_PARAM(GmParConstantDataLength))
	{
		ASTROS_DBG_PRINT(ASTROS_DBGLVL_NONE, "Inning(%d) IoSize= %d (bytes) CDL = %d (kilobytes) sustainedIos = %d\n", inning_number, INNING_PARAM(pInning, InParIoSize), GAME_PARAM(GmParConstantDataLength), INNING_PARAM(pInning, InParSustainedIos));

		INNING_PARAM(pInning, InParSustainedIos) = GAME_PARAM(GmParConstantDataLength) / (INNING_PARAM(pInning, InParIoSize) / 1024);

		ASTROS_DBG_PRINT(ASTROS_DBGLVL_NONE, "(POST-CALC)Inning(%d) IoSize= %d (bytes) CDL = %d (kilobytes) sustainedIos = %d\n", inning_number, INNING_PARAM(pInning, InParIoSize), GAME_PARAM(GmParConstantDataLength), INNING_PARAM(pInning, InParSustainedIos));
		
		
	}



	INNING_PARAM(pInning, InParBurstPercent) = pLineup->dimmensions[ASTROS_DIM_BURST_PERCENT].current_idx;
	INNING_PARAM(pInning, InParInningMode) = pLineup->dimmensions[ASTROS_DIM_INNING_MODE].current_idx;
	INNING_PARAM(pInning, InParFixedBatters) = pLineup->dimmensions[ASTROS_DIM_FIXEDBATTERS].current_idx;
    INNING_PARAM(pInning, InParOperation) = pLineup->dimmensions[ASTROS_DIM_OPERATION].current_idx;
    INNING_PARAM(pInning, InParBatterMode) = pLineup->dimmensions[ASTROS_DIM_BAT_MODE].current_idx;
 	INNING_PARAM(pInning, InParAccess)  = pLineup->dimmensions[ASTROS_DIM_ACCESS_TYPE].current_idx;

    INNING_PARAM(pInning, InParBatterMode) = pLineup->dimmensions[ASTROS_DIM_BAT_MODE].current_idx;

	INNING_PARAM(pInning, InParEnableLatency) = pLineup->dimmensions[ASTROS_DIM_LAT_STATS].current_idx;
	INNING_PARAM(pInning, InParEngineType) = pLineup->dimmensions[ASTROS_DIM_ENGINE_TYPE].current_idx;
	INNING_PARAM(pInning, InParTargetCount)  = pLineup->dimmensions[ASTROS_DIM_TARGET_COUNT].current_idx;
#ifdef ASTROS_DIM_LOAD_TYPE
	INNING_PARAM(pInning, InParLoadType)  = pLineup->dimmensions[ASTROS_DIM_LOAD_TYPE].current_idx;
#endif		

	ASTROS_DBG_PRINT(ASTROS_DBGLVL_NONE, "Inning(%d) MODE = %d\n", inning_number, INNING_PARAM(pInning, InParInningMode));
	ASTROS_DBG_PRINT(ASTROS_DBGLVL_NONE, "Inning(%d) Engine = %d\n", inning_number, INNING_PARAM(pInning, InParEngineType));



	temp = (INNING_PARAM(pInning, InParBurstPercent) % 100);

	temp = INNING_PARAM(pInning, InParQDepth) * temp;
	temp = temp / 100;

	INNING_PARAM(pInning, InParQueueDepthTrough) = temp;
    INNING_PARAM(pInning, InParMaxBatters) = pLineup->gametime.max_batters;
	INNING_PARAM(pInning, InParRow) = pLineup->current_row;
	INNING_PARAM(pInning, InParCol) = pLineup->current_col;
	

	for(i = 0; i < ASTROS_MAX_ATBATS; i++)
	{
		astros_atbat *pAtbat;

    	pAtbat = &pInning->atbats[INNING_PARAM(pInning, InParAtBat)];
		
		pAtbat->atbat_number = i;
	}
	

    return pInning;
}

astros_inning * astros_inning_get_precon(astros_lineup *pLineup, astros_inning * pInning)
{
    return NULL;
}

astros_atbat * astros_inning_get_at_bat(astros_inning * pInning)
{
	astros_atbat *pAtBat;	

	pAtBat = &pInning->atbats[INNING_PARAM(pInning, InParAtBat)];

	return pAtBat;
	
}

void astros_inning_complete_at_bat(astros_inning * pInning,  batter *pBatter)
{

	astros_atbat *pAtbat; 
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);

	UINT64 elap_ns = pBatter->at_bat_end_ns - pBatter->at_bat_start_ns;

	pAtbat = astros_inning_get_at_bat(pInning);

	ASTROS_ASSERT(pAtbat->pitches);
		

	pAtbat->pitches[pBatter->idx].elap_ns = elap_ns;
	pAtbat->pitches[pBatter->idx].start_ns = pBatter->at_bat_start_ns;
	pAtbat->pitches[pBatter->idx].end_ns = pBatter->at_bat_end_ns;
	pAtbat->pitches[pBatter->idx].io_count = pBatter->at_bat_io_count;
	

	memcpy(&pAtbat->pitches[pBatter->idx].cmd_lat, &pBatter->cmd_lat, sizeof(astros_latency));
	memcpy(&pAtbat->pitches[pBatter->idx].inter_lat, &pBatter->inter_lat, sizeof(astros_latency));
#ifdef ASTROS_INC_AVG_OFFSET
	memcpy(&pAtbat->pitches[pBatter->idx].avg_off, &pBatter->avg_off, sizeof(astros_avg_offset));
#endif
		



#ifdef ALTUVE
    ASTROS_DBG_PRINT(verbose, "astros_inning_complete_at_bat(inning = %d batter = %d at_bat = %d io_count = %d)\n", 
		INNING_PARAM(pInning, InParInningNumber), pBatter->idx, INNING_PARAM(pInning, InParAtBat), pBatter->at_bat_io_count);

#else
	float fElapSec = elap_ns / 1000000000.0;
    float fIops = (float)pBatter->at_bat_io_count / fElapSec;
    ASTROS_DBG_PRINT(verbose, "astros_inning_complete_at_bat(inning = %d batter = %d at_bat = %d fIops = %f io_count = %d)\n", 
		INNING_PARAM(pInning, InParInningNumber), pBatter->idx, INNING_PARAM(pInning, InParAtBat), fIops, pBatter->at_bat_io_count);
#endif
}
					
int astros_inning_get_batters(astros_inning * pInning)
{
#if 1
	return astros_inning_get_at_bat(pInning)->batters;
#else
	return INNING_PARAM(pInning, InParScoringBatters);
#endif


}
   

/* Single Batter up - one pass of a sweep */

int astros_inning_prep_target_queues(astros_inning * pInning, astros_atbat *pAtbat)
{
    int error = 0;
    int i, j;
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);
    ccb *pCCB;
    target *pTarget;
    astros_lineup *pLineup = pInning->pvLineup;
	int limit = INNING_PARAM(pInning, InParQDepth);
	
    
	pAtbat->total_ccb = 0;

	if(ASTROS_BM_INNING_BM_FIO == INNING_PARAM(pInning, InParBatterMode))
	{
		limit = limit * INNING_PARAM(pInning, InParFixedBatters);
	}
		
	
	ASTROS_DBG_PRINT(verbose, "astros_prep_target_queues(target_count = %d qdepth = %d)\n", 
			INNING_PARAM(pInning, InParTargetCount), INNING_PARAM(pInning, InParQDepth));

    for(i = 0; i < INNING_PARAM(pInning, InParTargetCount); i++)
    {
        pTarget = &pLineup->gametime.targets[i];

        for(j = 0; j < limit; j++)
        {
            pCCB = astros_get_free_ccb(pInning->pvLineup);

			ASTROS_ASSERT(pCCB);

			pCCB->io_size = INNING_PARAM(pInning, InParIoSize);

			pCCB->block_count = pCCB->io_size / 512;
			

            astros_ccb_put(&pTarget->pReady, pCCB);
           
            pAtbat->total_ccb++;
        }
    }

    ASTROS_DBG_PRINT(verbose, "astros_prep_target_queues(pAtbat->total_ccb = %d)\n", pAtbat->total_ccb);

    return error;
}


void astros_inning_ready_ccb(astros_inning *pInning, ccb *pCCB)
{
	ASTROS_ASSERT(pCCB);
	ASTROS_ASSERT(pInning);

	pCCB->op = INNING_PARAM(pInning, InParOperation);

}



int astros_inning_get_io_limit(astros_inning *pInning)
{
	int limit = 0;


	switch(INNING_PARAM(pInning, InParBatterMode))
	{
		case ASTROS_BM_SYNC_BATTERS :
		case ASTROS_INNING_BM_DYNAMIC :
		case ASTROS_INNING_BM_FIXED:
		case ASTROS_BM_INNING_BM_BATPERTARGET:
		case ASTROS_BM_INNING_BM_FIO:
		default:
			limit = INNING_PARAM(pInning, InParSustainedIos);
		break;

	
	
	}


	return limit;


}

					


