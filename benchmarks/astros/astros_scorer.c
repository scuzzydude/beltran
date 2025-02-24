
#include "astros.h"
#include "math.h"
/* astros_score NOT to be included in altuve driver, so no special handling of FP operations */







void astros_scorer_put_footer(FILE *fptr, astros_scorecard *pScoreCard)
{
	fprintf(fptr, "\n\nGAME_STATS\n");

	fprintf(fptr,"Elapsed Seconds, %f\n", pScoreCard->fElap);
	fprintf(fptr,"Total_Innings, %d\n", pScoreCard->total_innings);
	fprintf(fptr,"Innings_Per_Sec, %f\n", pScoreCard->fInPerSec);
	fprintf(fptr,"InningSizeBytes, %d\n", pScoreCard->inning_size_bytes);
	fprintf(fptr,"ScoreCardSize, %d\n", pScoreCard->score_card_size);
	

}

int astros_scorer_post_final(astros_lineup *pLineup, astros_scorecard *pScoreCard)
{
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_INFO);
	

	ASTROS_DBG_PRINT(verbose, "astros_scorer_post_final() last_inning_updated =%d current_inning = %d \n", pScoreCard->last_inning_updated, pScoreCard->current_inning);
	

	while(pScoreCard->last_inning_updated < 0)
	{
		ASTROS_BATTER_SLEEPUS(50000);
	}

	while((pScoreCard->last_inning_updated < pScoreCard->current_inning))
	{

		ASTROS_DBG_PRINT(verbose, "astros_scorer_post_final() WHILE last_inning_updated =%d current_inning = %d \n", pScoreCard->last_inning_updated, pScoreCard->current_inning);
		ASTROS_BATTER_SLEEPUS(50000);
	}

	pScoreCard->end_ns = ASTROS_PS_HRCLK_GET();


	pScoreCard->fElap = ((float)(pScoreCard->end_ns - pScoreCard->start_ns)) / 1000000000.0;
	pScoreCard->fInPerSec = (float)pScoreCard->total_innings / pScoreCard->fElap;

		
	astros_scorer_put_footer(pScoreCard->csvlogFptr, pScoreCard);
	

	fclose(pScoreCard->csvlogFptr);

	ASTROS_DBG_PRINT(verbose, "astros_scorer_post_final() DONE last_inning_updated =%d current_inning = %d LOGFILE = %s \n", 
		pScoreCard->last_inning_updated, pScoreCard->current_inning, pScoreCard->csvlogFn);

	astros_score_lineup_columnated(pLineup);
	
	return 0;
}

int astros_score_inning(int inning_number, astros_scorecard *pScoreCard)
{
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);
	int error = 0;
	int i;
	astros_inning *pInning;
	int start_at_bat;
	int end_at_bat;
	int count;
	SINT32 iops = 0;	
	float fTemp;
	float fCulled;
	float fDistance;
	int iops_hi = 0;
	int iops_lo = -1;
	int idx_hi = -1;
	int idx_lo = -1;
	SINT32 total_iops = 0;
	SINT32 abiops;
	int ioff;
	astros_latency cmd_lat;
	astros_latency inter_lat;
	astros_avg_offset avg_off;
#ifdef	ASTROS_GET_KSTATS
	astros_at_bat_kstats_summary kstats;
#endif
	int sbats = 0;

	pInning = &pScoreCard->innings[inning_number];

	ASTROS_DBG_PRINT(verbose, "astros_score_inning(%d) pInning->inning_number =%d \n", inning_number, INNING_PARAM(pInning, InParInningNumber));


	start_at_bat =	INNING_PARAM(pInning, InParScoreAtBat);
	end_at_bat = start_at_bat + INNING_PARAM(pInning, InParMeasuringAtBats);
		
	count = 0;
	total_iops = 0;
	for(i = start_at_bat; i < end_at_bat; i++)
	{
		abiops = pInning->atbats[i].iops; 
		total_iops += abiops;

		if(abiops > iops_hi)
		{
			iops_hi = abiops;
			idx_hi = i;
		}

		if(-1 == iops_lo)
		{
			iops_lo = abiops;
			idx_lo = i;
		}
		else if(abiops < iops_lo)
		{
			iops_lo = abiops;
			idx_lo = i;
		}
		


		count++;
	}


	if(count)
	{
		iops = total_iops / count;
	}

	pInning->box_score.fStats[BxScrHiIops] = iops_hi;
	pInning->box_score.fStats[BxScrLoIops] = iops_lo;


	pInning->box_score.fStats[BxScrIops] = iops;
	pInning->box_score.fStats[BxScrBps] = (float)iops * (float)INNING_PARAM(pInning, InParIoSize);

	pInning->box_score.fStats[BxScrInning] = (float)INNING_PARAM(pInning, InParInningNumber);

	

	
	if(count > 3)
	{
		pInning->box_score.fStats[BxScrCulledIops] = (float)((total_iops - (iops_hi + iops_lo)) / (count - 2));
	}


	ioff = ((char *)(&pScoreCard->innings[INNING_PARAM(pInning, InParInningNumber)])) - ((char *)pScoreCard);
	


	

	ASTROS_DBG_PRINT(ASTROS_DBGLVL_INFO, "astros_score_inning(%d) Mean iops = %d iops_lo = %d iops_hi = %d fCulledIops =%f ioff/of %d:%d)(%d)\n", 
		inning_number, iops, iops_lo, iops_hi, pInning->box_score.fStats[BxScrCulledIops], ioff, pScoreCard->score_card_size, sizeof(astros_inning));

	fTemp = 0.0;
	fCulled = 0.0;

	for(i = start_at_bat; i < end_at_bat; i++)
	{

		abiops = pInning->atbats[i].iops; 

		fDistance = ((float)abiops - (float)iops);
		fDistance = fDistance * fDistance;

		fTemp += fDistance;

		if((i != idx_hi) && (i != idx_lo))
		{
			fDistance = ((float)abiops - pInning->box_score.fStats[BxScrCulledIops]);
			fDistance = fDistance * fDistance;

			fCulled += fDistance;
		}


		ASTROS_DBG_PRINT(verbose, "astros_score_inning(%d) atbat=%d fDistance=%f fTemp=%f\n", inning_number, i, fDistance, fTemp);
		
	}

	pInning->box_score.fStats[BxScrStdDevIops] = sqrt(fTemp);
	pInning->box_score.fStats[BxScrCulledStdDevIops] = sqrt(fCulled);


	
	
	if(pInning->box_score.fStats[BxScrIops] > 0.0)
	{
		pInning->box_score.fStats[BxScrIopsDeviationPercent]  = (pInning->box_score.fStats[BxScrStdDevIops] / pInning->box_score.fStats[BxScrIops]) * 100.0;
	}
	if(pInning->box_score.fStats[BxScrCulledIops] > 0.0)
	{
		pInning->box_score.fStats[BxScrCulledIopsDeviationPercent]  = (pInning->box_score.fStats[BxScrCulledStdDevIops]  / pInning->box_score.fStats[BxScrCulledIops]) * 100.0;
	}



	memset(&cmd_lat, 0, sizeof(astros_latency));
	memset(&inter_lat, 0, sizeof(astros_latency));
	memset(&avg_off, 0, sizeof(astros_avg_offset));


#ifdef	ASTROS_GET_KSTATS
	memset(&kstats, 0, sizeof(astros_at_bat_kstats_summary));
#endif

	for(i = start_at_bat; i < end_at_bat; i++)
	{
		


#ifdef	ASTROS_GET_KSTATS
		kstats.fIrqPercentSubmits += pInning->atbats[i].kstats.fIrqPercentSubmits;
		kstats.fOtherBatSubmit += pInning->atbats[i].kstats.fOtherBatSubmit;
		kstats.fOtherBatComplete += pInning->atbats[i].kstats.fOtherBatComplete;
		kstats.fRespPerIrq += pInning->atbats[i].kstats.fRespPerIrq;
#endif
	
		cmd_lat.count += pInning->atbats[i].cmd_lat.count;
		cmd_lat.total_elap_ns += pInning->atbats[i].cmd_lat.total_elap_ns;

		if(i == 0)
		{
			cmd_lat.lo = pInning->atbats[i].cmd_lat.lo;
		}
		else if(pInning->atbats[i].cmd_lat.lo < cmd_lat.lo)
		{
			cmd_lat.lo = pInning->atbats[i].cmd_lat.lo;
		}

		if(pInning->atbats[i].cmd_lat.hi > cmd_lat.hi)
		{
			cmd_lat.hi = pInning->atbats[i].cmd_lat.hi;
		}
		
		inter_lat.count += pInning->atbats[i].inter_lat.count;
		inter_lat.total_elap_ns += pInning->atbats[i].inter_lat.total_elap_ns;

		if(i == 0)
		{
			inter_lat.lo = pInning->atbats[i].inter_lat.lo;
		}
		else if(pInning->atbats[i].inter_lat.lo < inter_lat.lo)
		{
			inter_lat.lo = pInning->atbats[i].inter_lat.lo;
		}

		if(pInning->atbats[i].inter_lat.hi > inter_lat.hi)
		{
			inter_lat.hi = pInning->atbats[i].inter_lat.hi;
		}

#ifdef ASTROS_INC_AVG_OFFSET

		avg_off.count += pInning->atbats[i].avg_off.count;
		
		avg_off.total_offset += pInning->atbats[i].avg_off.total_offset;

		if(i == 0)
		{
			avg_off.lo = pInning->atbats[i].avg_off.lo;
		}
		else if(pInning->atbats[i].avg_off.lo < avg_off.lo)
		{
			avg_off.lo = pInning->atbats[i].avg_off.lo;
		}

		if(pInning->atbats[i].avg_off.hi > avg_off.hi)
		{
			avg_off.hi = pInning->atbats[i].avg_off.hi;
		}
#endif


		sbats++;

	}


#ifdef ASTROS_GET_KSTATS
	if(sbats)
	{

		pInning->box_score.fStats[BxScrOtherBatSubmit] = kstats.fOtherBatSubmit / (float)sbats;
		pInning->box_score.fStats[BxScrOtherBatComplete] = kstats.fOtherBatComplete / (float)sbats;
		pInning->box_score.fStats[BxScrIrqPercentSubmits] = kstats.fIrqPercentSubmits / (float)sbats;
		pInning->box_score.fStats[BxScrRespPerIrq] = kstats.fRespPerIrq / (float)sbats;

	}
#endif
	if(cmd_lat.count)
	{
		float us = (float)cmd_lat.total_elap_ns / (float)cmd_lat.count;
		us = us / 1000.0;
		pInning->box_score.fStats[BxScrCmdLatAvgUs] = us;
	}
	else
	{
		pInning->box_score.fStats[BxScrCmdLatAvgUs] = 0;
	}
	pInning->box_score.fStats[BxScrCmdLatHiUs] = (float)cmd_lat.hi / 1000.0;
	pInning->box_score.fStats[BxScrCmdLatLoUs] = (float)cmd_lat.lo / 1000.0;
	
	if(inter_lat.count)
	{
		float us = (float)inter_lat.total_elap_ns / (float)inter_lat.count;
		us = us / 1000.0;
		pInning->box_score.fStats[BxScrInterLatAvgUs] = us;
	}
	else
	{
		pInning->box_score.fStats[BxScrInterLatAvgUs] = 0;
	}
	pInning->box_score.fStats[BxScrInterLatHiUs] = (float)inter_lat.hi / 1000.0;
	pInning->box_score.fStats[BxScrInterLatLoUs] = (float)inter_lat.lo / 1000.0;


#ifdef ASTROS_INC_AVG_OFFSET
	if(avg_off.count)
	{
		float favgoff = (float)avg_off.total_offset / (float)avg_off.count;
		pInning->box_score.fStats[BxScrAvgOffset] = favgoff; 
	}
	else
	{
		pInning->box_score.fStats[BxScrAvgOffset] = 0; 

	}

	pInning->box_score.fStats[BxScrAvgOffsetHi] = (float)avg_off.hi ;
	pInning->box_score.fStats[BxScrAvgOffsetLo] = (float)avg_off.lo ;

#endif

	pInning->box_score.fStats[BxScrScoringBatters]  = (float)INNING_PARAM(pInning, InParScoringBatters) ;


	return error;

}

int astros_score_open_csvlog(astros_lineup *pLineup)
{	
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_INFO);
	int error = 0;
	FILE *fptr;
	char fn[512];
	int i;
//	sprintf(fn, "%s/astros-%s-%d.csv", pLineup->gameid.logpath, pLineup->gameid.szHostname, pLineup->gameid.gameid);


	if(pLineup->bUserTag)
	{
		sprintf(fn, "%s/%s.csv", pLineup->gameid.logpath, pLineup->szUserTag);
	}
	else
	{
		sprintf(fn, "%s/%d-astros-%s-%s.csv", pLineup->gameid.logpath, pLineup->gameid.gameid, pLineup->szUserTag, pLineup->gameid.szHostname);
	}
	

	
	ASTROS_DBG_PRINT(verbose, "astros_score_open_csvlog() csvlog fn = %s = %d\n", fn, pLineup->gameid.gameid);

	fptr = fopen(fn, "w");

	if(fptr)
	{
		ASTROS_DBG_PRINT(verbose, "astros_score_open_csvlog() OPENED fn = %s\n", fn);
		pLineup->pScoreCard->csvlogFptr = fptr;
		strcpy(pLineup->pScoreCard->csvlogFn, fn);

		fprintf(fptr, "GAMEID, FIELD, USER_TAG, Series, Sequence, AstrosVersion");
		
		for(i = 0; i < InParMaxEnum; i++)
		{
			fprintf(fptr, ",%s", InningParamStr(i));
		}


		for(i = 0; i < BxScrMaxEnum; i++)
		{
			fprintf(fptr, ",%s", BoxScoreStatStr(i));
		}

		for(i = 0; i < ASTROS_MAX_ATBATS; i++)
		{
			fprintf(fptr,",AtBat(%d)_batters", i);
			fprintf(fptr,",AtBat(%d)_iops", i);

			fprintf(fptr,",AtBat(%d)_fStartDelta",i);
			fprintf(fptr,",AtBat(%d)_fEndDelta",i);

		}



		fprintf(fptr,"\n");

		pLineup->pScoreCard->csvlogFptr = fptr;
	

	}
	


	return error;
}

void astros_score_csvlog(int inning_number, astros_lineup *pLineup, astros_scorecard *pScoreCard)
{
	FILE *fptr;
	astros_inning *pInning;
	astros_atbat *pAtBat;
	int i;
	
	if(pLineup->pScoreCard->csvlogFptr)
	{
		pInning = &pScoreCard->innings[inning_number];

		fptr = pLineup->pScoreCard->csvlogFptr;

		fprintf(fptr, "%d, %d,%s,%d,%d,%s", pLineup->gameid.gameid, pLineup->field, pLineup->szUserTag, g_series_number, g_sequence_number, astros_get_version());
				

		

		for(i = 0; i < InParMaxEnum; i++)
		{
			fprintf(fptr, ",%d", INNING_PARAM(pInning, i));
		}
		

		for(i = 0; i < BxScrMaxEnum; i++)
		{
			fprintf(fptr, ",%f", pInning->box_score.fStats[i]);
		}


		for(i = 0; i < ASTROS_MAX_ATBATS; i++)
		{

			pAtBat = &pInning->atbats[i];
		

			
			fprintf(fptr,",%d", pAtBat->batters);
			fprintf(fptr,",%d", pAtBat->iops);
			fprintf(fptr,",%f", pAtBat->fStartDelta);
			fprintf(fptr,",%f", pAtBat->fEndDelta);
			
		}



		fprintf(fptr,"\n");



		

	}
}

void astros_wait_scorer_ready(astros_scorecard *pScoreCard)
{
	while(false == pScoreCard->bScore)
	{
		ASTROS_BATTER_SLEEPUS(1000);
	}

}

void * astros_score_keeping(void *pvLineup)
{
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_INFO);
	astros_lineup *pLineup = pvLineup;
	astros_scorecard *pScoreCard = NULL;
	UINT32 cpu;
	int i;
	
	ASTROS_GET_CPU(&cpu);
 
	ASTROS_DBG_PRINT(verbose, "astros_score_keeping() created pLineup=%p cpu = %d\n", pLineup, cpu);



	while(NULL == pScoreCard)
	{
		pScoreCard = pLineup->pScoreCard;
		ASTROS_BATTER_SLEEPUS(100);

	}

	pScoreCard->start_ns = ASTROS_PS_HRCLK_GET();

	astros_score_open_csvlog(pLineup);

	ASTROS_DBG_PRINT(verbose, "astros_score_keeping() created pScoreCard=%p cpu = %d\n", pScoreCard, cpu);

	pScoreCard->current_inning = -1;	

	pScoreCard->bScore = true;

	while(pScoreCard->bScore)
	{

		ASTROS_BATTER_SLEEPUS(100000);

		if(pScoreCard->current_inning < 0)
		{

		}
		else if(pScoreCard->last_inning_updated < pScoreCard->current_inning)
		{
			int start_inning = pScoreCard->last_inning_updated;
			int end_inning = pScoreCard->current_inning;		

			ASTROS_DBG_PRINT(verbose, "astros_score_keeping() current_inning  = %d total_innings = %d last_inning_updated = %d\n", pScoreCard->current_inning, pScoreCard->total_innings, pScoreCard->last_inning_updated);
		
			for(i = start_inning; i <= end_inning; i++)
			{
				if(i < 0)
				{
					pScoreCard->last_inning_updated++;
				}
				else if(0 == astros_score_inning(i, pScoreCard))
				{
					astros_score_csvlog(i, pLineup, pScoreCard);

					pScoreCard->last_inning_updated++;
				}
			}


		}
		

	}

	
	return NULL;
}



int  astros_scorer_get(ASTROS_BATTER_JERSEY * pScorer_id, astros_lineup *pLineup, int master_cpu)
{

    int verbose = astros_get_verbosity(ASTROS_DBGLVL_INFO);
	int error = 0; 

	ASTROS_DBG_PRINT(verbose, "astros_scorer_get() =%px\n", pScorer_id);

#if 0
	pScorer_id = CreateThread(NULL, 0, astros_score_keeping, pLineup, 0, NULL);

#endif

#ifdef ASTROS_WIN
		ASTROS_ASSERT(0);
#else
	
		cpu_set_t set;
	
		CPU_ZERO(&set);
		
		error = pthread_create(pScorer_id, NULL, astros_score_keeping, pLineup);
#endif



#if 0	
	if(0 == error)
	{
		ASTROS_DBG_PRINT(verbose, "astros_scorer_get() created pLineup=%p\n", pLineup);

		CPU_SET(master_cpu, &set);
		
    	error = pthread_setaffinity_np(*pScorer_id, sizeof(cpu_set_t), &set);
    
	}
#endif	
	return error;
}






int astros_score_column(void *pvLineup, int inning)
{
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);
	astros_lineup *pLineup = pvLineup;
	astros_inning *pInning;
	int qd;
	
		
	ASTROS_ASSERT(pLineup->fptr_col);
	
	ASTROS_ASSERT(pLineup->pScoreCard);

	pInning = &pLineup->pScoreCard->innings[inning];

	ASTROS_ASSERT(inning == INNING_PARAM(pInning, InParInningNumber));
	
	qd = INNING_PARAM(pInning, InParQDepth);

	ASTROS_DBG_PRINT(verbose, "astros_score_column(%d) COL/ROW (%d:%d) QD = %d\n", inning, pLineup->current_col, pLineup->current_row, qd);

	if(0 == pLineup->current_col)
	{
		fprintf(pLineup->fptr_col, "\n%d,", qd);
	}

	fprintf(pLineup->fptr_col, "%f,", pInning->box_score.fStats[pLineup->LogColBxScr]);


	return 0;
}


bool astros_col_header_decode(FILE *fp, astros_inning *pInning, int bx_scr_idx)
{
	bool bFound = true;
	int val = INNING_PARAM(pInning, bx_scr_idx);
	char *str = "???";
	
	switch(bx_scr_idx)
	{
		case InParOperation:
		{
			if(ASTROS_CCB_OP_WRITE == val)
			{
				str = "write";
			}
			else if(ASTROS_CCB_OP_READ == val)
			{
				str = "read";
			}
			else 
			{
				str = "unknown";
			}


			fprintf(fp, "%s,", str);
		}
		break;

		case InParEngineType:
		{
			if(ASTROS_AIOENGINE_URING == val)
			{
				str = "uring";
			}
			else if(ASTROS_AIOENGINE_LIBAIO == val)
			{
				str = "libaio";
			}
			else if(ASTROS_AIOENGINE_ZOMBIE == val)
			{
				str = "zombie";
			}
		

			fprintf(fp, "%s,", str);

		}
		break;

		case InParZombieParm:
		{

				if(ALTUVE_ZOMBIE_PQI_TYPE == val)
				{
					str = "smartpqi";
				}
				else if(ALTUVE_ZOMBIE_MR_TYPE == val)
				{
					str = "megaraid_sas";
				}
				else
				{
					str = "???";
				}
			
			
				fprintf(fp, "%s,", str);
			
		}
		break;












		
		default:
			bFound = false;
		break;
	}

	return bFound;

}

int astros_score_column_header(void *pvLineup, int inning)
{
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);
	astros_lineup *pLineup = pvLineup;
	astros_inning *pInning;
	int qd;
	
		
	ASTROS_ASSERT(pLineup->fptr_col);
	
	ASTROS_ASSERT(pLineup->pScoreCard);

	pInning = &pLineup->pScoreCard->innings[inning];

	ASTROS_ASSERT(inning == INNING_PARAM(pInning, InParInningNumber));
	
	qd = INNING_PARAM(pInning, InParQDepth);


	if(1 == pLineup->current_row)
	{
		ASTROS_DBG_PRINT(verbose, "astros_score_column_header COL/ROW (%d:%d) QD = %d\n", inning, pLineup->current_col, pLineup->current_row, qd);

		if(0 == pLineup->current_col)
		{
			fprintf(pLineup->fptr_col, "\n%s,", InningParamStr(pLineup->LogColBxScr));
		}

		if(astros_col_header_decode(pLineup->fptr_col, pInning, pLineup->LogColBxScr))
		{

		}
		else
		{
			fprintf(pLineup->fptr_col, "%d,", INNING_PARAM(pInning, pLineup->LogColBxScr));
		}
	}


	return 0;
}


void astros_score_log_headers(astros_lineup *pLineup, int bxScr, int inning)
{
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);
	int i;
	
	ASTROS_ASSERT(pLineup->fptr_col);

	
	ASTROS_DBG_PRINT(verbose, "astros_score_log_headers(%d:%d)\n", bxScr, inning);
	
	fprintf(pLineup->fptr_col, "\nBoxScoreStat, %s\n", BoxScoreStatStr(bxScr));
	
	for(i = 0; i < InParMaxEnum; i++)
	{
		pLineup->inning_count = 0;
		
		pLineup->fnLogColumn = astros_score_column_header;
		pLineup->LogColBxScr = i;	
		astros_lineup_run(pLineup, 0);

	}
	fprintf(pLineup->fptr_col, "\nUSER_TAG,%s", pLineup->szUserTag);
	fprintf(pLineup->fptr_col, "\nTAG");

}




void astros_score_sum_columns(FILE *fptr, astros_lineup *pLineup)
{
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_INFO);
	int i;
	astros_inning *pInning;
	int csize = sizeof(astros_score_column_summary) * pLineup->max_col;
	astros_score_column_summary *pColSum;

	pColSum = ASTROS_ALLOC(64, csize);

	ASTROS_ASSERT(pColSum);

	memset(pColSum, 0, csize);
	
	
	ASTROS_DBG_PRINT(verbose, "astros_score_sum_columns(%px) max_col = %d max_row = %d inning_count = %d csize = %d\n", 
		pLineup, pLineup->max_col, pLineup->max_row, pLineup->inning_count, csize);

	for(i = 0; i < pLineup->inning_count; i++)
	{
		int col, row;
		
		pInning = &pLineup->pScoreCard->innings[i];

		ASTROS_ASSERT(pInning);

		col = INNING_PARAM(pInning, InParCol);
		row = INNING_PARAM(pInning, InParRow);

		ASTROS_ASSERT(col < pLineup->max_col);
		ASTROS_ASSERT(row <= pLineup->max_row);

	

		pColSum[col].fStats[ColSumTotalIops] += pInning->box_score.fStats[BxScrIops] ;

		if(INNING_PARAM(pInning, InParQDepth) > pColSum[col].fStats[ColSumMaxQueueDepth])
		{
			pColSum[col].fStats[ColSumMaxQueueDepth] = INNING_PARAM(pInning, InParQDepth);
		}
		
		pColSum[col].fStats[ColSumTotalRow]++;
		
		



		ASTROS_DBG_PRINT(verbose, "astros_score_sum_columns(inning = %d) %px (%d,%d)\n", i, pInning, col, row );

		

	}


	fprintf(fptr, "\n\nColumn Summary\n");

	for(i = 0; i < ColSumMaxEnum; i++)
	{
		int j;
		
		fprintf(fptr, "%s",  ColSumStatStr(i));
		
		for(j = 0; j < pLineup->max_col; j++)
		{
			fprintf(fptr, ",%f", pColSum[j].fStats[i]);
		}



		fprintf(fptr, "\n");
		
	}

	



}



void astros_score_lineup_columnated(astros_lineup *pLineup)
{
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);
	char fn[512];
	FILE *fptr;
	int i;
	int inning_count = 0;

	if(pLineup->bUserTag)
	{
		sprintf(fn, "%s/astros-col-%s.csv", pLineup->gameid.logpath, pLineup->szUserTag);
	}
	else
	{
		sprintf(fn, "%s/astros-col-%s-%d.csv", pLineup->gameid.logpath, pLineup->gameid.szHostname, pLineup->gameid.gameid);
	}


	ASTROS_DBG_PRINT(verbose, "astros_score_lineup_columnated(%s)\n", fn);

	fptr = fopen(fn, "w");

	if(fptr)
	{
		
		inning_count = pLineup->inning_count;
		
			
		for(i = 0; i < BxScrMaxEnum; i++)
		{
			ASTROS_DBG_PRINT(verbose, "astros_score_lineup_columnated(%d) : ENUM START %d\n", i);

			pLineup->fptr_col = fptr;

			astros_score_log_headers(pLineup, i, i);
			
			ASTROS_DBG_PRINT(verbose, "astros_score_lineup_columnated(%d) : HEADER DONE %d\n", i, i);

			pLineup->inning_count = 0;
			pLineup->fnLogColumn = astros_score_column;
			pLineup->LogColBxScr = i;
			
			astros_lineup_run(pLineup, 0);
			
			fprintf(pLineup->fptr_col, "\n");

			ASTROS_DBG_PRINT(verbose, "astros_score_lineup_columnated(%d) ENUM DONE\n", i);

		}

		


		astros_scorer_put_footer(pLineup->fptr_col, pLineup->pScoreCard);

//		astros_score_sum_columns(pLineup->fptr_col, pLineup);
		


		pLineup->inning_count = inning_count;
				


		pLineup->fnLogColumn = NULL;

		fclose(fptr);
	}	



}













