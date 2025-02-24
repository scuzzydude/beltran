#include "astros.h"

/*************************************************************************************************************/
/* Globals */
/*************************************************************************************************************/
int astros_field_get_bam_targets(astros_lineup *pLineup);


int g_GameParam[GameParMaxEnum];

#define ASTROS_VERSION_MAJOR 1
#define ASTROS_VERSION_MINOR 0
#define ASTROS_VERSION_BUILD 6

char szAstrosVersion[32] = { 0 };

#define DEF_K_IOPS 4   
#define MAX_K_IOPS 4
#define DEF_MAX_QD 64


#define DEF_MIN_QD 1
#define DEF_MAX_TGT 1
#define DEF_MIN_TGT 1

#define DEF_MAX_BAT 1
#define DEF_MIN_BAT 1



#define DEF_ACCESS_TYPE_MIN ASTROS_SEQ_MODE_RANDOM
//#define	DEF_ACCESS_TYPE_MIN ASTROS_SEQ_MODE_SPLIT_STREAM_RST_INNING
//#define	DEF_ACCESS_TYPE_MIN ASTROS_SEQ_MODE_SPLIT_STREAM_RST_ATBAT


#define DEF_ACCESS_TYPE_MAX ASTROS_SEQ_MODE_RANDOM 
//#define DEF_ACCESS_TYPE_MAX ASTROS_SEQ_MODE_SPLIT_STREAM_RST_INNING 
//#define DEF_ACCESS_TYPE_MAX ASTROS_SEQ_MODE_SPLIT_STREAM_RST_ATBAT






#define DEF_BAT_SCALE		 ASTROS_INC_TYPE_POW2
//#define DEF_BAT_SCALE		 ASTROS_INC_TYPE_LINEAR


/* Field */
#define DEF_BB_FIELD ASTROS_FIELD_USER
//#define DEF_BB_FIELD ASTROS_FIELD_KERNEL

/* Operation */

#define DEF_MIN_OP ASTROS_CCB_OP_WRITE
//#define DEF_MIN_OP ASTROS_CCB_OP_READ
//#define DEF_MAX_OP ASTROS_CCB_OP_WRITE
#define DEF_MAX_OP ASTROS_CCB_OP_READ


/* Burst */
#define DEF_MIN_MODE ASTROS_INNING_MODE_SUSTAINED
//#define DEF_MAX_MODE ASTROS_INNING_MODE_BURST
#define DEF_MAX_MODE ASTROS_INNING_MODE_SUSTAINED


#define DEF_MIN_LAT_STATS_MIN ASTROS_DIM_LAT_STATS_ENB
#define DEF_MIN_LAT_STATS_MAX ASTROS_DIM_LAT_STATS_ENB


int gDefaultUserModeLib = ASTROS_AIOENGINE_LIBAIO;
int gDefaultBaseballField = DEF_BB_FIELD;

#ifdef ASTROS_WIN
#define DEF_ENGINE_TYPE_MIN ASTROS_AIOENGINE_WINAIO
#define DEF_ENGINE_TYPE_MAX ASTROS_AIOENGINE_WINAIO
#else
#define DEF_ENGINE_TYPE_MIN ASTROS_AIOENGINE_URING
//#define DEF_ENGINE_TYPE_MIN ASTROS_AIOENGINE_LIBAIO
//#define DEF_ENGINE_TYPE_MAX ASTROS_AIOENGINE_LIBAIO
#define DEF_ENGINE_TYPE_MAX ASTROS_AIOENGINE_URING
#endif




#ifdef ASTROS_WIN
#define DEF_BATTERMODE_MIN ASTROS_BM_SYNC_BATTERS
#define DEF_BATTERMODE_MAX ASTROS_BM_SYNC_BATTERS
#else
#define DEF_BATTERMODE_MIN ASTROS_BM_INNING_BM_BATPERTARGET
#define DEF_BATTERMODE_MAX ASTROS_BM_INNING_BM_BATPERTARGET

#endif
//#define DEF_BATTERMODE_MIN ASTROS_INNING_BM_DYNAMIC
//#define DEF_BATTERMODE_MAX ASTROS_INNING_BM_DYNAMIC

//#define DEF_BATTERMODE_MIN ASTROS_BM_SYNC_BATTERS
//#define DEF_BATTERMODE_MAX ASTROS_BM_SYNC_BATTERS


//#define ASTROS_FIXED_LOAD 





#define DEF_MIN_IO_SIZE_K 4
#define DEF_MAX_IO_SIZE_K 4




astros_dimension g_Default_dimmensions[ASTROS_MAX_DIMENSION] = {
	{
    	ASTROS_DIM_QDEPTH,
    	DEF_MIN_QD,
    	DEF_MIN_QD,
    	DEF_MAX_QD, 
    	1,
    	ASTROS_INC_TYPE_LINEAR,
    	//ASTROS_INC_TYPE_POW2,
    	"QueueDepth"
    
 	}, //0
#ifdef ASTROS_FIXED_LOAD
	{
    	ASTROS_DIM_IOSIZE,
    	ASTROS_FIXED_LOAD_IOSIZE,
    	ASTROS_FIXED_LOAD_IOSIZE,
    	ASTROS_FIXED_LOAD_IOSIZE,
    	ASTROS_FIXED_LOAD_IOSIZE,
    	ASTROS_INC_TYPE_POW2,
    	"IOSize"
    
 	}, //1
#else
	{
    	ASTROS_DIM_IOSIZE,
    	(1024 * DEF_MIN_IO_SIZE_K),
    	(1024 * DEF_MIN_IO_SIZE_K),
    	(1024 * DEF_MAX_IO_SIZE_K),
    	4096,
    	ASTROS_INC_TYPE_POW2,
    	"IOSize"
    
 	}, //1
#endif
	{
    	ASTROS_DIM_TARGET_COUNT,
   	 	DEF_MIN_TGT,
    	DEF_MIN_TGT,
    	DEF_MAX_TGT,
    	1,
    	ASTROS_INC_TYPE_LINEAR,
    	"TargetCount"
    
 	},//2
	{
		 ASTROS_DIM_OPERATION,
		 DEF_MIN_OP,
		 DEF_MIN_OP,
		 DEF_MAX_OP, 
		 1,
		 ASTROS_INC_TYPE_LINEAR,
		 "Operation"
			 
	},//3

 	{
	   	ASTROS_DIM_IO_COUNT,
	   	1024 * DEF_K_IOPS,
	   	1024 * DEF_K_IOPS,
	   	1024 * MAX_K_IOPS, //262144,
	   	1024,
	   	ASTROS_INC_TYPE_POW2,
	   	"AtIOCount"
			   
  	},//4
	{
		  ASTROS_DIM_BURST_PERCENT,
		  0,
		  0,
		  90, 
		  10,
		  ASTROS_INC_TYPE_LINEAR,
		  "BurstPercent"
			  
	 },//5

	 {
		  ASTROS_DIM_INNING_MODE,
		  DEF_MIN_MODE,
		  DEF_MIN_MODE,
		  DEF_MAX_MODE, 
		  1,
		  ASTROS_INC_TYPE_LINEAR,
		  "InningMode"
			  
	 },//6

	{
		 ASTROS_DIM_FIXEDBATTERS,
		 DEF_MIN_BAT,
		 DEF_MIN_BAT,
		 DEF_MAX_BAT, 
		 1,
		 DEF_BAT_SCALE,	
		 "FixedBatters"
			 
	},//7



#ifdef ASTROS_DIM_LOAD_TYPE
#ifdef ASTROS_FIXED_LOAD 

		{
			 ASTROS_DIM_LOAD_TYPE,
			 0,
			 0,
			 1, 
			 1,
			 ASTROS_INC_TYPE_LINEAR,
			 "LoadType"
				 
		},//13
#else
	
	{
		 ASTROS_DIM_LOAD_TYPE,
		 0,
		 0,
		 0, 
		 1,
		 ASTROS_INC_TYPE_LINEAR,
		 "LoadType"
			 
	},//13

#endif
#else

	{
		ASTROS_DIM_ATBAT_COUNT,
		DEF_AB_CNT,
		DEF_AB_CNT,
		DEF_AB_CNT,
		1,
		ASTROS_INC_TYPE_LINEAR,
		"AtBatCount"
		
 	},//8

#endif

	{
		 ASTROS_DIM_BAT_MODE,
		 DEF_BATTERMODE_MIN,
		 DEF_BATTERMODE_MIN,
		 DEF_BATTERMODE_MAX, 
		 1,
		 ASTROS_INC_TYPE_LINEAR,
		 "BatterMode"
			 
	},//9

	{
		 ASTROS_DIM_ACCESS_TYPE,
		 DEF_ACCESS_TYPE_MIN,
		 DEF_ACCESS_TYPE_MIN,
		 DEF_ACCESS_TYPE_MAX, 
		 1,
		 ASTROS_INC_TYPE_LINEAR,
		 "AccessType"
			 
	},//10

	
	{
		 ASTROS_DIM_LAT_STATS,
		 DEF_MIN_LAT_STATS_MIN,
		 DEF_MIN_LAT_STATS_MIN,
		 DEF_MIN_LAT_STATS_MAX, 
		 1,
		 ASTROS_INC_TYPE_LINEAR,
		 "LatencyStats"
			 
	},//11
	{
		 ASTROS_DIM_ENGINE_TYPE,
		 DEF_ENGINE_TYPE_MIN,
		 DEF_ENGINE_TYPE_MIN,
		 DEF_ENGINE_TYPE_MAX, 
		 1,
		 ASTROS_INC_TYPE_LINEAR,
		 "EngineType"
			 
	},//12

};

#define MAX_RUN_TAG_LEN 128
char g_szRunTag[MAX_RUN_TAG_LEN];
unsigned int g_series_number = 0;
unsigned int g_sequence_number = 0;
/*************************************************************************************************************/
/* End Globals */
/*************************************************************************************************************/

/*************************************************************************************************************/
/* Function */
/*************************************************************************************************************/
void astros_init_game_param(void)
{
	/* This unit is kilobytes */

	GAME_PARAM(GmParConstantDataLength) = 0;
	//GAME_PARAM(GmParConstantDataLength) = 8192;
	

}


char * astros_get_version(void)
{

	if(0 == strlen(szAstrosVersion))
	{

		sprintf(szAstrosVersion, "v%02d.%02d.%03d", ASTROS_VERSION_MAJOR, ASTROS_VERSION_MINOR, ASTROS_VERSION_BUILD);

	}

	return szAstrosVersion;

}


char * astros_run_tag_ptr(astros_lineup *pLineup)
{
	int verbose = ASTROS_DBGLVL_INFO;

	if(0 == strlen(g_szRunTag))
	{
		ASTROS_DBG_PRINT(verbose, "astros_run_tag_ptr(%d) \n", pLineup->gameid.gameid);			
	    pLineup->bUserTag = false;
	
		sprintf(g_szRunTag, "LOCALRUN%05d", pLineup->gameid.gameid);
	}
	else
	{
	    pLineup->bUserTag = true;
	}

	return &g_szRunTag[0];
}

void astros_cli_xxx_handle(void *pvArg)
{
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_INFO);
	int val;
	int error = 0;
	struct arg_int *pArg = (struct arg_int *)pvArg;
	struct arg_str *psArg = (struct arg_str *)pvArg;
	const char *sval = psArg->sval[0];

	ASTROS_ASSERT(pArg->count == 1);
	
	val = pArg->ival[0];
	

	if(0 == strcmp(pArg->hdr.longopts, "qmin"))
	{

	   	g_Default_dimmensions[ASTROS_DIM_QDEPTH].current_idx = val;
		g_Default_dimmensions[ASTROS_DIM_QDEPTH].min = val;
	
	}
	else if(0 == strcmp(pArg->hdr.longopts, "qmax"))
	{
		g_Default_dimmensions[ASTROS_DIM_QDEPTH].max = val;
	}
	else if(0 == strcmp(pArg->hdr.longopts, "qinc"))
	{
		if((1 == val) || (0 == val) )
		{
			g_Default_dimmensions[ASTROS_DIM_QDEPTH].inc_type = val;
		}
		else
		{
			printf("Bad qinc val = %d\n", val);
			error++;
		}
	}
	else if(0 == strcmp(pArg->hdr.longopts, "batmin"))
	{
	   	g_Default_dimmensions[ASTROS_DIM_FIXEDBATTERS].current_idx = val;
		g_Default_dimmensions[ASTROS_DIM_FIXEDBATTERS].min = val;
	}
	else if(0 == strcmp(pArg->hdr.longopts, "batmax"))
	{
		g_Default_dimmensions[ASTROS_DIM_FIXEDBATTERS].max = val;
	}
	else if(0 == strcmp(pArg->hdr.longopts, "batinc"))
	{
		if((1 == val) || (0 == val) )
		{
			g_Default_dimmensions[ASTROS_DIM_FIXEDBATTERS].inc_type = val;
		}
		else
		{
			printf("Bad batinc val = %d\n", val);
			error++;
		}
	}
	else if(0 == strcmp(pArg->hdr.longopts, "ioengine"))
	{
		
		ASTROS_DBG_PRINT(verbose, "ioengine = %s\n", sval);

		if(0 == strcmp(sval, "sync"))
		{
			g_Default_dimmensions[ASTROS_DIM_BAT_MODE].current_idx = ASTROS_BM_SYNC_BATTERS;
			g_Default_dimmensions[ASTROS_DIM_BAT_MODE].min = ASTROS_BM_SYNC_BATTERS;
			g_Default_dimmensions[ASTROS_DIM_BAT_MODE].max = ASTROS_BM_SYNC_BATTERS;

			g_Default_dimmensions[ASTROS_DIM_ENGINE_TYPE].current_idx = ASTROS_AIOENGINE_SYNC;
			g_Default_dimmensions[ASTROS_DIM_ENGINE_TYPE].min = ASTROS_AIOENGINE_SYNC;
			g_Default_dimmensions[ASTROS_DIM_ENGINE_TYPE].max = ASTROS_AIOENGINE_SYNC;
		

			ASTROS_DBG_PRINT(verbose, "ioengine = SYNC %s\n", sval);
		}
		else if(0 == strcmp(sval, "libaio"))
		{
			g_Default_dimmensions[ASTROS_DIM_BAT_MODE].current_idx = ASTROS_BM_INNING_BM_BATPERTARGET;
			g_Default_dimmensions[ASTROS_DIM_BAT_MODE].min = ASTROS_BM_INNING_BM_BATPERTARGET;
			g_Default_dimmensions[ASTROS_DIM_BAT_MODE].max = ASTROS_BM_INNING_BM_BATPERTARGET;

			g_Default_dimmensions[ASTROS_DIM_ENGINE_TYPE].current_idx = ASTROS_AIOENGINE_LIBAIO;
			g_Default_dimmensions[ASTROS_DIM_ENGINE_TYPE].min = ASTROS_AIOENGINE_LIBAIO;
			g_Default_dimmensions[ASTROS_DIM_ENGINE_TYPE].max = ASTROS_AIOENGINE_LIBAIO;

		}
		else if(0 == strcmp(sval, "liburing"))
		{
			g_Default_dimmensions[ASTROS_DIM_BAT_MODE].current_idx = ASTROS_BM_INNING_BM_BATPERTARGET;
			g_Default_dimmensions[ASTROS_DIM_BAT_MODE].min = ASTROS_BM_INNING_BM_BATPERTARGET;
			g_Default_dimmensions[ASTROS_DIM_BAT_MODE].max = ASTROS_BM_INNING_BM_BATPERTARGET;

			g_Default_dimmensions[ASTROS_DIM_ENGINE_TYPE].current_idx = ASTROS_AIOENGINE_URING;
			g_Default_dimmensions[ASTROS_DIM_ENGINE_TYPE].min = ASTROS_AIOENGINE_URING;
			g_Default_dimmensions[ASTROS_DIM_ENGINE_TYPE].max = ASTROS_AIOENGINE_URING;

		}
		else if(0 == strcmp(sval, "zombie"))
		{
			g_Default_dimmensions[ASTROS_DIM_ENGINE_TYPE].current_idx = ASTROS_AIOENGINE_ZOMBIE;
			g_Default_dimmensions[ASTROS_DIM_ENGINE_TYPE].min = ASTROS_AIOENGINE_ZOMBIE;
			g_Default_dimmensions[ASTROS_DIM_ENGINE_TYPE].max = ASTROS_AIOENGINE_ZOMBIE;

			gDefaultBaseballField = ASTROS_FIELD_KERNEL;
		}
		else if(0 == strcmp(sval, "bam"))
		{
			g_Default_dimmensions[ASTROS_DIM_BAT_MODE].current_idx = ASTROS_BM_BAM;
			g_Default_dimmensions[ASTROS_DIM_BAT_MODE].min = ASTROS_BM_BAM;
			g_Default_dimmensions[ASTROS_DIM_BAT_MODE].max = ASTROS_BM_BAM;

			g_Default_dimmensions[ASTROS_DIM_ENGINE_TYPE].current_idx = ASTROS_AIOENGINE_BAM_ARRAY;
			g_Default_dimmensions[ASTROS_DIM_ENGINE_TYPE].min = ASTROS_AIOENGINE_BAM_ARRAY;
			g_Default_dimmensions[ASTROS_DIM_ENGINE_TYPE].max = ASTROS_AIOENGINE_BAM_ARRAY;
			
			gDefaultBaseballField = ASTROS_FIELD_BAM;

		}
		

	}
	else if(0 == strcmp(pArg->hdr.longopts, "blockmin"))
	{

		if(0 == val)
		{
			g_Default_dimmensions[ASTROS_DIM_IOSIZE].current_idx = 512;
			g_Default_dimmensions[ASTROS_DIM_IOSIZE].min = 512;
		}
		else
		{
		 	g_Default_dimmensions[ASTROS_DIM_IOSIZE].current_idx = val * 1024;
			g_Default_dimmensions[ASTROS_DIM_IOSIZE].min = val * 1024;
		}
	}
	else if(0 == strcmp(pArg->hdr.longopts, "blockmax"))
	{
		if(0 == val)
		{
			g_Default_dimmensions[ASTROS_DIM_IOSIZE].max = 512;
		}
		else
		{
			g_Default_dimmensions[ASTROS_DIM_IOSIZE].max = val * 1024;
		}

	}
	else if(0 == strcmp(pArg->hdr.longopts, "ops"))
	{
//		char *sval = psArg->sval[0];
		
		ASTROS_DBG_PRINT(verbose, "ops = %s\n", sval);

		if(0 == strcmp(sval, "write"))
		{
			g_Default_dimmensions[ASTROS_DIM_OPERATION].current_idx = ASTROS_CCB_OP_WRITE;
			g_Default_dimmensions[ASTROS_DIM_OPERATION].min = ASTROS_CCB_OP_WRITE;
			g_Default_dimmensions[ASTROS_DIM_OPERATION].max = ASTROS_CCB_OP_WRITE;

		}
		else if(0 == strcmp(sval, "read"))
		{
			g_Default_dimmensions[ASTROS_DIM_OPERATION].current_idx = ASTROS_CCB_OP_WRITE;
			g_Default_dimmensions[ASTROS_DIM_OPERATION].min = ASTROS_CCB_OP_READ;
			g_Default_dimmensions[ASTROS_DIM_OPERATION].max = ASTROS_CCB_OP_READ;

		}
		else if(0 == strcmp(sval, "both"))
		{
			g_Default_dimmensions[ASTROS_DIM_OPERATION].current_idx = ASTROS_CCB_OP_WRITE;
			g_Default_dimmensions[ASTROS_DIM_OPERATION].min = ASTROS_CCB_OP_WRITE;
			g_Default_dimmensions[ASTROS_DIM_OPERATION].max = ASTROS_CCB_OP_READ;

		}
		else
		{
			
			printf("Bad ops val = %s\n", sval);
			error++;
		}

	}
	else if(0 == strcmp(pArg->hdr.longopts, "access"))
	{
//		char *sval = psArg->sval[0];
		
		ASTROS_DBG_PRINT(verbose, "access = %s\n", sval);

		if(0 == strcmp(sval, "seq"))
		{
			g_Default_dimmensions[ASTROS_DIM_ACCESS_TYPE].current_idx = ASTROS_SEQ_MODE_SPLIT_STREAM_RST_INNING;
			g_Default_dimmensions[ASTROS_DIM_ACCESS_TYPE].min = ASTROS_SEQ_MODE_SPLIT_STREAM_RST_INNING;
			g_Default_dimmensions[ASTROS_DIM_ACCESS_TYPE].max = ASTROS_SEQ_MODE_SPLIT_STREAM_RST_INNING;

		}
		else if(0 == strcmp(sval, "random"))
		{
			g_Default_dimmensions[ASTROS_DIM_ACCESS_TYPE].current_idx = ASTROS_SEQ_MODE_RANDOM;
			g_Default_dimmensions[ASTROS_DIM_ACCESS_TYPE].min = ASTROS_SEQ_MODE_RANDOM;
			g_Default_dimmensions[ASTROS_DIM_ACCESS_TYPE].max = ASTROS_SEQ_MODE_RANDOM;

		}
		else
		{
			
			printf("Bad access val = %s\n", sval);
			error++;
		}

	}
	else if(0 == strcmp(pArg->hdr.longopts, "tgtmin"))
	{
	   	g_Default_dimmensions[ASTROS_DIM_TARGET_COUNT].current_idx = val;
		g_Default_dimmensions[ASTROS_DIM_TARGET_COUNT].min = val;
	}
	else if(0 == strcmp(pArg->hdr.longopts, "tgtmax"))
	{
		g_Default_dimmensions[ASTROS_DIM_TARGET_COUNT].max = val;
	}
	else if(0 == strcmp(pArg->hdr.longopts, "tgtinc"))
	{
		if((1 == val) || (0 == val) )
		{
			g_Default_dimmensions[ASTROS_DIM_TARGET_COUNT].inc_type = val;
		}
		else
		{
			printf("Bad tgtinc val = %d\n", val);
			error++;
		}
	}
	else if(0 == strcmp(pArg->hdr.longopts, "tgtset"))
	{
	   	g_Default_dimmensions[ASTROS_DIM_TARGET_COUNT].current_idx = val;
		g_Default_dimmensions[ASTROS_DIM_TARGET_COUNT].min = val;
		g_Default_dimmensions[ASTROS_DIM_TARGET_COUNT].max = val;
	}
	else if(0 == strcmp(pArg->hdr.longopts, "tag"))
	{
//		char *sval = psArg->sval[0];

		strncpy(g_szRunTag, sval, (MAX_RUN_TAG_LEN - 1));
		
	}
	else if(0 == strcmp(pArg->hdr.longopts, "ioab"))
	{
		val = val * 1024;
	   	g_Default_dimmensions[ASTROS_DIM_IO_COUNT].current_idx = val;
		g_Default_dimmensions[ASTROS_DIM_IO_COUNT].min = val;
		g_Default_dimmensions[ASTROS_DIM_IO_COUNT].max = val;
	}
	else if(0 == strcmp(pArg->hdr.longopts, "burstmin"))
	{
	   	g_Default_dimmensions[ASTROS_DIM_BURST_PERCENT].current_idx = val;
		g_Default_dimmensions[ASTROS_DIM_BURST_PERCENT].min = val;
	}
	else if(0 == strcmp(pArg->hdr.longopts, "burstmax"))
	{
	   	if(0 == g_Default_dimmensions[ASTROS_DIM_BURST_PERCENT].current_idx)
	   	{
			g_Default_dimmensions[ASTROS_DIM_BURST_PERCENT].current_idx = val;
	   	}
	
		g_Default_dimmensions[ASTROS_DIM_BURST_PERCENT].max = val;
	}
	else if(0 == strcmp(pArg->hdr.longopts, "burstinc"))
	{
		if((val > 0) && (val < 101) )
		{
			g_Default_dimmensions[ASTROS_DIM_BURST_PERCENT].inc_type = val;
		}
		else
		{
			printf("Bad burstinc val = %d\n", val);
			error++;
		}
	}
	else if(0 == strcmp(pArg->hdr.longopts, "series"))
	{
		g_series_number = val;
	}
	else if(0 == strcmp(pArg->hdr.longopts, "sequence"))
	{
		g_sequence_number = val;
	}
	else if(0 == strcmp(pArg->hdr.longopts, "cdl"))
	{
		GAME_PARAM(GmParConstantDataLength) = val;
	}


	

	if(error)
	{
		printf("Bad CLI Param Value\n");
		exit(0);

	}
	
}


void astros_parse_cli(int argc, char **argv)
{
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);
	struct arg_int *qmin, *qmax, *qinc;
	struct arg_int *batmin, *batmax, *batinc;
	struct arg_int *tgtmin, *tgtmax, *tgtinc, *tgtset;
	struct arg_int *blockmin, *blockmax;
	struct arg_int *burstmin, *burstmax, *burstinc;
	struct arg_str *ioengine;
	struct arg_str *ops;
	struct arg_str *access;
	struct arg_str *tag;
	struct arg_lit *help;
	struct arg_lit *zombie;
	struct arg_lit *version;
	struct arg_end *end;
	struct arg_int *ioab;
	struct arg_int *sequence;
	struct arg_int *series;
	struct arg_int *cdl;
	
	int nerrors;
	char progname[] = "astros";
	int i;
	
	void *iargtable[] = 
	{
			help	= arg_litn("?hH", "help", 0, 1, "display this help and exit"),
			zombie	= arg_litn("zZ", "zombie", 0, 1, "zombie mode"),
			qmin	= arg_int0(NULL, "qmin", "<n>", "Minimun QDepth (1-512)"),
			qmax	= arg_int0(NULL, "qmax", "<n>", "Maximum QDepth (1-512)"),
			qinc	= arg_int0(NULL, "qinc", "<n>", "QDepth Incrementer (0 = Linear, 1=Power of 2)"),
			batmin	= arg_int0(NULL, "batmin", "<n>", "Minimun Batters (1-MAXCPU)"),
			batmax	= arg_int0(NULL, "batmax", "<n>", "Maximum Batters (1-MAXCUP)"),
			batinc	= arg_int0(NULL, "batinc", "<n>", "Batter Incrementer (0 = Linear, 1=Power of 2)"),
			tgtmin	= arg_int0(NULL, "tgtmin", "<n>", "Minimum Targets (update <hostname>.tgtlst)"),
			tgtmax	= arg_int0(NULL, "tgtmax", "<n>", "Maximum Targets (update <hostname>.tgtlst)"),
			tgtinc	= arg_int0(NULL, "tgtinc", "<n>", "Target Incrementer (0 = Linear, 1=Power of 2)"),
			tgtset	= arg_int0(NULL, "tgtset", "<n>", "Target Set  (single target count for all sweeps)"),
		ioengine    = arg_str0(NULL, "ioengine", "<str>", "IOEngine [sync,libaio,liburing,zombie,bam]"),
		blockmin	= arg_int0(NULL, "blockmin", "<n>", "Minimun IoSize (kB) (0-256) 0=512b"),
		blockmax	= arg_int0(NULL, "blockmax", "<n>", "Maximum IoSize (kB) (0-256)"),
			ops    	= arg_str0(NULL, "ops", "<str>", "operations [write,read,both]"),
			access 	= arg_str0(NULL, "access", "<str>", "access [seq,random]"),
			tag 	= arg_str0(NULL, "tag", "<str>", "user tag [up to 128 chars, no whitespace]"),
		    ioab    = arg_int0(NULL, "ioab", "<n>", "IO per At Bat (k)"),
			burstmax	= arg_int0(NULL, "burstmax", "<n>", "Burst Percetage Max (0-100%)"),
			burstmin	= arg_int0(NULL, "burstmin", "<n>", "Burst Percentage Min(0-100%)"),
			burstinc	= arg_int0(NULL, "burstmax", "<n>", "Burst Percent Increment (0-100)"),
			sequence	= arg_int0(NULL, "sequence", "<n>", "Sequence Number (for scripting)"),
			series 	= arg_int0(NULL,     "series", "<n>",   "Series   Number (for scripting)"),
			cdl 	= arg_int0(NULL,     "cdl", "<n>",   "Contstant Data length  (kilobytes) - scale IOSize but keep total data constant per at bat"),
			version	= arg_litn("v", "version", 0, 1, "version number"),
			end 	= arg_end(20),
	};

	g_szRunTag[0] = 0;


	
	nerrors = arg_parse(argc,argv,iargtable);

	ASTROS_DBG_PRINT(verbose, "astros_parse_cli(%d) nerrors=%d\n", argc, nerrors);

	if(help->count > 0)
	{
		  arg_print_glossary(stdout, iargtable, "  %-25s %s\n");
		  exit(0);
	}


	if (nerrors > 0)
    {
        /* Display the error details contained in the arg_end struct.*/
        arg_print_errors(stdout, end, progname);
        printf("Try '%s --help' for more information.\n", progname);
    }
	else
	{	
		int ac = sizeof(iargtable) / sizeof(void *);
		
		ASTROS_DBG_PRINT(verbose, "iargtable size = %d\n", ac);

		for(i = 0; i < (ac - 1); i++)
		{
			struct arg_int * pArg = iargtable[i];
			
			ASTROS_DBG_PRINT(verbose, "%d))) ARG SET count = %d = %s\n", i, pArg->count, pArg->hdr.longopts); 


			if(pArg->count && ( 0 == strcmp(pArg->hdr.longopts, "zombie")))
			{
				printf("Zombie Mode Set\n");
				gDefaultBaseballField = ASTROS_FIELD_KERNEL;
		
			}
			else if(pArg->count && (0 == strcmp(pArg->hdr.longopts, "version")))
			{
				printf("Astros Version: %s\n", astros_get_version());
				exit(0);
			}
			else if(pArg->count)
			{
				ASTROS_DBG_PRINT(verbose, "%d) * ARG SET count = %d = %s\n", i, pArg->count, pArg->hdr.longopts);	

				astros_cli_xxx_handle(pArg);

			}
			
		}

	}

	
	
}
			



void astros_kernel_mode_dim_constraints(void)
{
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_INFO);
	int i;
	astros_dimension *pDim;
	int fixups = 0;
	

    ASTROS_DBG_PRINT(verbose, "astros_lineup_kernel_mode_dim_constraints(%d) =  \n", ASTROS_MAX_DIMENSION);
	
	for(i = 0; i < ASTROS_MAX_DIMENSION; i++)
	{
		pDim = &g_Default_dimmensions[i];

		switch(pDim->dim_enum)
		{
			case ASTROS_DIM_ENGINE_TYPE:
			{

	
				if((ASTROS_AIOENGINE_URING == pDim->min) || (ASTROS_AIOENGINE_LIBAIO == pDim->min))
				{
					pDim->min = ASTROS_AIOENGINE_ZOMBIE;
					fixups++;
				}
				if((ASTROS_AIOENGINE_URING == pDim->max) || (ASTROS_AIOENGINE_LIBAIO == pDim->max))
				{
					pDim->max = ASTROS_AIOENGINE_ZOMBIE;
					fixups++;
				}
				if((ASTROS_AIOENGINE_URING == pDim->current_idx) || (ASTROS_AIOENGINE_LIBAIO == pDim->current_idx))
				{
					pDim->current_idx = ASTROS_AIOENGINE_ZOMBIE;
					fixups++;
				}
				
				ASTROS_DBG_PRINT(verbose, "astros_lineup_kernel_mode_dim_constraints ASTROS_DIM_ENGINE_TYPE fixups = %d	\n", fixups);

			}

			default:
			break;



		}

		

	}
	




}
void astros_user_mode_contstraints(void)
{
#if ASTROS_WIN
	astros_win_init();
#else

    int verbose = astros_get_verbosity(ASTROS_DBGLVL_INFO);
	int i;
	astros_dimension *pDim;
	int fixups = 0;
	struct utsname uts;
	int kernel_major = 0;

	
    ASTROS_DBG_PRINT(verbose, "astros_user_mode_contstraints(%d) =  \n", ASTROS_MAX_DIMENSION);
#ifdef ASTROS_CYGWIN
	ASTROS_UNUSED(uts);
	//FAKE IT
	kernel_major = 4;
#else
	if(0 == uname(&uts))
	{
		ASTROS_DBG_PRINT(verbose, "sysname: %s\n", uts.sysname);
		ASTROS_DBG_PRINT(verbose, "nodename: %s\n", uts.nodename);
		ASTROS_DBG_PRINT(verbose, "release: %s\n", uts.release);
		ASTROS_DBG_PRINT(verbose, "version: %s\n", uts.version);
		ASTROS_DBG_PRINT(verbose, "machine: %s\n", uts.machine);
		ASTROS_DBG_PRINT(verbose, "domainname: %s\n", uts.domainname);

		if(uts.release[0] == '5')
		{
			kernel_major = 5;
		}
		else if(uts.release[0] == '4')
		{
			kernel_major = 4;
		}
		else
		{
			ASTROS_ASSERT(0);
		}

		


	}
	else
	{
		kernel_major = 4;
	}
#endif
	ASTROS_DBG_PRINT(verbose, "kernel_major: %d\n", kernel_major);

	if(kernel_major == 4)
	{
	
	  for(i = 0; i < ASTROS_MAX_DIMENSION; i++)
	  {
		pDim = &g_Default_dimmensions[i];

		switch(pDim->dim_enum)
		{
			case ASTROS_DIM_ENGINE_TYPE:
			{

	
				if((ASTROS_AIOENGINE_URING == pDim->min))
				{
					pDim->min = ASTROS_AIOENGINE_LIBAIO;
					fixups++;
				}
				if((ASTROS_AIOENGINE_URING == pDim->max))
				{
					pDim->max = ASTROS_AIOENGINE_LIBAIO;
					fixups++;
				}
				if((ASTROS_AIOENGINE_URING == pDim->current_idx))
				{
					pDim->current_idx = ASTROS_AIOENGINE_LIBAIO;
					fixups++;
				}
				
				ASTROS_DBG_PRINT(verbose, "stros_user_mode_linux_contstraints ASTROS_DIM_ENGINE_TYPE fixups = %d	\n", fixups);

			}

			default:
			break;



		}

		

	  }
	
	}
#endif

}





#if 0
char *ltrim(char *s)
{
	ASTROS_ASSERT(s);
	
    while(isspace(*s)) s++;
    return s;
}

char *rtrim(char *s)
{
    char* back = s + strlen(s);
	ASTROS_ASSERT(s);

    while(isspace(*--back));
    *(back+1) = '\0';
    return s;
}

char *trim(char *s)
{
	ASTROS_ASSERT(s);
    return rtrim(ltrim(s)); 
}
#endif




#if 0 //def ASTROS_CYGWIN
ASTROS_FD astros_open_file(char *path, int mode)
{
	HANDLE hn;
	
	wchar_t  wszPath[128];
	
	MultiByteToWideChar(CP_ACP, MB_PRECOMPOSED, path, -1, wszPath, 128);

	hn = CreateFile(&wszPath,  (GENERIC_READ | GENERIC_WRITE), (FILE_SHARE_READ | FILE_SHARE_WRITE), NULL, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, NULL);
	
	return hn;



}
#else
ASTROS_FD astros_open_file(char *path, int mode)
{
	return open(path, mode);
}
#endif


#ifndef ASTROS_CYGWIN

int astros_open_target_file(astros_lineup * pLineup, char *path, int targetidx, int operation)
{
	int mode = astros_get_open_mode(operation, pLineup);

	return astros_open_file(path, mode);
	
}

int astros_open_target(astros_lineup * pLineup, char *path, int targetidx, int operation)
{
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);
    target *pTarget;
    int error = 0;
    int mode;
	
    ASTROS_ASSERT(targetidx < ASTROS_MAX_TARGETS);
    
    pTarget = &pLineup->gametime.targets[targetidx];

	mode =  astros_get_open_mode(operation, pLineup);

	  ASTROS_DBG_PRINT(verbose, "astros_open_target(%d) atomic_qd = %d\n", targetidx, ASTROS_GET_ATOMIC(pTarget->atomic_qd));

	  ASTROS_SET_ATOMIC(pTarget->atomic_qd, 0);

	  
	  if(pTarget->fd)
	  {
			if(pTarget->mode == mode)
			{
				ASTROS_DBG_PRINT(verbose, "astros_open_target(%d) OPEN ALREADY in mode = %d '%s' GOOD fd = %d OP = %d\n", targetidx, mode, path, pTarget->fd, operation);
				return error;
			}
			else
			{

				close(pTarget->fd);
				ASTROS_BATTER_SLEEPUS(100);
			
				pTarget->fd = 0;
			}
	  }	


	pTarget->fd =  astros_open_file(path, mode);



    strcpy(pTarget->path, path);

    if(pTarget->fd > 0)
    {
        ASTROS_DBG_PRINT(verbose, "astros_open_target(%d) '%s' GOOD fd = %d OP = %d\n", targetidx, path, pTarget->fd, operation);
		pTarget->mode = mode;

    }
    else
    {
        ASTROS_DBG_PRINT(ASTROS_DBGLVL_ERROR, "astros_open_target(%d) '%s' FAILED fd = %d\n", targetidx, path, pTarget->fd);
        error =  pTarget->fd;   
    }
   return error;

}
#endif
int astros_test_rotation_sync(astros_lineup        *pLineup)
{
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_INFO);
    int i;
    UINT64 hi = 0;
    UINT64 low = 0;
    UINT64 cur, diff;
#if 0
    const int timer_samples = 10;
    UINT64 delta_ns[timer_samples];
#endif
    int batter_count;

    pLineup->gametime.batter_count = 11;

    batter_count = astros_batters_on_deck(pLineup, NULL);
    
    batter *pBatter;
    
    ASTROS_DBG_PRINT(verbose, "astros_test_rotation_sync() %d Batters on Deck \n", batter_count);

    astros_lineup_first_pitch(pLineup);

    ASTROS_DBG_PRINT(verbose, "astros_test_rotation_sync() atomic_batter_lock = %d \n", ASTROS_GET_ATOMIC(pLineup->gametime.atomic_batter_lock));

  
    ASTROS_BATTER_SLEEPUS(100000);

    

    
    for(i = 0; i < pLineup->gametime.batter_count; i++)
    {
        pBatter = &pLineup->gametime.batters[i];
        cur = pBatter->at_bat_start_ns;


        if(pBatter->bAtBat)
        {
                    
            ASTROS_DBG_PRINT(verbose, "astros_test_rotation_sync() Batter = %d start_ns = %ld \n", pBatter->idx, cur);

            if(0 == low)
            {
                low = cur;
            }
            else if(cur < low)
            {
                low = cur;
            }

            if(cur > hi)
            {
                hi = cur;
            }
        
        }
        else
        {
            ASTROS_DBG_PRINT(verbose, "astros_test_rotation_sync() Batter = %d NOT AT BAT (%d:%d:%d)\n", pBatter->idx, pBatter->bCalledup, pBatter->bOnDeck, pBatter->bAtBat);

        }

    }

    diff = hi - low;
    
    ASTROS_DBG_PRINT(verbose, "astros_test_rotation_sync() low = %ld hi = %ld diff = %ld\n", low, hi, diff);

    hi = 0;

    for(i = 0; i < pLineup->gametime.batter_count; i++)
    {
        pBatter = &pLineup->gametime.batters[i];
        cur = pBatter->at_bat_start_ns;

        hi += (cur - low);
        
        ASTROS_DBG_PRINT(verbose, "astros_test_rotation_sync(%d) diff = %ld\n", pBatter->idx, cur - low);
    }

    ASTROS_DBG_PRINT(verbose, "astros_test_rotation_sync(%d) average start spread = %f ns\n", (float)((float)hi / (float)  pLineup->gametime.batter_count) );
    

    



    return 0;
}



void astros_get_gameid(astros_lineup *pLineup)
{
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_INFO);
	char *fn = "astros_gameid.bin";
	FILE *fptr;

	fptr = fopen(fn, "rb+");


	char *logdir = "logs";
	if (0 != access(logdir, F_OK)) 
	{
		ASTROS_DBG_PRINT(verbose, "astros_get_gameid() Directory %s DOES NOT EXIST\n", logdir); 		
		if(0 == mkdir(logdir, 0777))
		{
			
		}
		else
		{
			ASTROS_DBG_PRINT(verbose, "astros_get_gameid() Could not create directory %s using CWD\n", logdir); 		
			logdir = "";

		}
	}
	else
	{
		ASTROS_DBG_PRINT(verbose, "astros_get_gameid() Directory %s EXISTS\n", logdir); 		
	}
	strcpy(pLineup->gameid.logpath, logdir);
	

	if(fptr)
	{
		ASTROS_DBG_PRINT(verbose, "astros_get_gameid() OPEN %s\n", fn);	

		if(fread(&pLineup->gameid, sizeof(astros_gameid), 1, fptr))
		{
			ASTROS_DBG_PRINT(verbose, "astros_get_gameid() game_id =%d path = %s\n", pLineup->gameid.gameid, pLineup->gameid.logpath); 

		}
		pLineup->gameid.gameid++;
		
		fseek(fptr, 0, SEEK_SET);

		fwrite(&pLineup->gameid, sizeof(astros_gameid), 1, fptr);

		ASTROS_DBG_PRINT(verbose, "astros_get_gameid() WRITTEN game_id =%d path = %s\n", pLineup->gameid.gameid, pLineup->gameid.logpath); 
		


		
		
	}
	else
	{
		ASTROS_DBG_PRINT(verbose, "astros_get_gameid() NOT EXIST %s\n", fn);			

		pLineup->gameid.gameid = 1;

		

		fptr = fopen(fn, "wb");

		if(fptr)
		{
			ASTROS_DBG_PRINT(verbose, "astros_get_gameid() CREATED %s\n", fn);			

			fwrite(&pLineup->gameid, sizeof(astros_gameid), 1, fptr);
		}

	}
	if(fptr)
	{
		fclose(fptr);
	}
}


#ifdef ASTROS_CYGWIN 
int astros_load_block_device_details(target *pTarget)
{
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_INFO);

	HANDLE hDevice = INVALID_HANDLE_VALUE;	// handle to the drive to be examined 
	BOOL bResult	= FALSE;				 // results flag
	DWORD junk 	= 0;					 // discard results
	wchar_t  wszPath[128];
	GET_LENGTH_INFORMATION li;
	UINT64 sb64;
	
	MultiByteToWideChar(CP_ACP, MB_PRECOMPOSED, pTarget->path, -1, wszPath, 128);



	
	hDevice = CreateFileW(wszPath, 		 // drive to open
						   GENERIC_READ | GENERIC_WRITE,				 // no access to the drive
						   FILE_SHARE_READ | // share mode
						   FILE_SHARE_WRITE, 
						   NULL,			 // default security attributes
						   OPEN_EXISTING,	 // disposition
						   0,				 // file attributes
						   NULL);			 // do not copy file attributes



	
	if (hDevice == INVALID_HANDLE_VALUE)	// cannot open the drive
	{
		return (FALSE);
	}

	bResult = DeviceIoControl(hDevice,                       // device to be queried
                            IOCTL_DISK_GET_LENGTH_INFO, // operation to perform
                            NULL, 0,                       // no input buffer
                            &li, sizeof(li),            // output buffer
                            &junk,                         // # bytes returned
                            (LPOVERLAPPED) NULL);          // synchronous I/O



	if(bResult)
	{



		sb64 = li.Length.QuadPart;

		sb64 = sb64 - (1024 * 1024);
		
		pTarget->size_bytes = sb64;


		ASTROS_DBG_PRINT(verbose, "astros_load_block_device_details(%d)WIN OPENED %s size_bytes = %lld\n", pTarget->idx, pTarget->path, pTarget->size_bytes);
		pTarget->capacity4kblock = pTarget->size_bytes / 4096;

		ASTROS_DBG_PRINT(verbose, "astros_load_block_device_details(%d)WIN %s capacity4kblock = %lld\n", pTarget->idx, pTarget->path, pTarget->capacity4kblock);


	}


	CloseHandle(hDevice);


	return 0;
	
}

#else
int astros_load_block_device_details(target *pTarget)
{

    int verbose = astros_get_verbosity(ASTROS_DBGLVL_INFO);
	int fd;
	off_t size_bytes;
		

	ASTROS_DBG_PRINT(verbose, "astros_load_block_device_details(%d)\n", pTarget->idx);

	fd = open(pTarget->path, O_RDONLY);
	



	if(fd)
	{

		size_bytes = lseek(fd, 0, SEEK_END);	

		ASTROS_DBG_PRINT(verbose, "astros_load_block_device_details(%d) OPENED %s size_bytes = %lld\n", pTarget->idx, pTarget->path, size_bytes);

		pTarget->size_bytes = size_bytes - (1024 * 1024);

		pTarget->capacity4kblock = size_bytes / 4096;

		ASTROS_DBG_PRINT(verbose, "astros_load_block_device_details(%d) %s capacity4kblock = %lld\n", pTarget->idx, pTarget->path, pTarget->capacity4kblock);

		close(fd);

		
	}
	else
	{
		ASTROS_DBG_PRINT(verbose, "astros_load_block_device_details(%d) ERROR OPEN %s\n", pTarget->idx, pTarget->path);
		ASTROS_ASSERT(0);
	}



	return 0;
	
}
#endif			


int load_target_list_file(astros_lineup *pLineup)
{
	int tgt_count = 0;
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_INFO);
	FILE *fd;
	char *line = NULL;
	size_t len;
	int rb;
	int idx = 0;
	int slen;
	target *pTarget;
	char fn[ASTROS_HOSTNAME_SIZE + 10];
//	char hn[32];



	
	sprintf(fn, "%s.tgtlst", pLineup->gameid.szHostname);


	fd = fopen(fn, "r");

	ASTROS_ASSERT(pLineup);

	if(fd)
	{
		ASTROS_DBG_PRINT(verbose, "load_target_list_file(%d) OPENED %s\n", fd, fn);
		while(-1 != (rb = getline(&line, &len, fd)))
		{

			ASTROS_DBG_PRINT(verbose, "load_target_list_file(%d) rb = %d line = %s\n", fd, rb, line);

#ifndef ASTROS_WIN
#ifdef ASTROS_CYGWIN
			if(NULL == strstr(line, "PhysicalDrive"))
#else
			if(NULL == strstr(line, "/dev/"))
#endif
			{
				ASTROS_DBG_PRINT(verbose, "load_target_list_file NULL : %d\n", idx);
				break;
			}
			else
#endif
			{
				slen = strlen(line);



				ASTROS_DBG_PRINT(verbose, "load_target_list_file(%d) [%d](%s) : %d:%d\n", fd, idx, line, len, slen);
				
				ASTROS_ASSERT(idx < ASTROS_MAX_TARGETS);
				  
				pTarget = &pLineup->gametime.targets[idx];

				pTarget->idx = idx;
	
				ASTROS_ASSERT(slen < 64);

				//line = trim(line);

#ifndef ASTROS_WIN
				{
					int nonp = 0;
					int i;
					for(i = 0; i < len; i++)
					{
						if(isprint(line[i]))
						{
							nonp++;
						}
						else
						{
							break;
						}

					}


					ASTROS_DBG_PRINT(verbose, "NONP = [%d]\n", nonp);
									
					strncpy(pTarget->path, line, nonp);
				

				}
#else
				strncpy(pTarget->path, line, slen);
#endif	

				ASTROS_DBG_PRINT(verbose, "PATH = [%s]\n", pTarget->path);

				astros_load_block_device_details(pTarget);
				



				free(line);

				line = NULL;

				idx++;			
			}

			
		}
		
		ASTROS_DBG_PRINT(verbose, "load_target_list_file(%d) FCLOSE %s\n", fd, fn);

		fclose(fd);


	}
	else
	{
		ASTROS_DBG_PRINT(verbose, "load_target_list_file(%d) FAILED OPEN %s\n", fd, fn);
		
	}

	tgt_count = idx;

	return tgt_count;
}


#define TODO_WHAT_LINEUP_REQURIES 0

int astros_field_get_targets(astros_lineup *pLineup)
{
	int tgt_count = 0;
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_INFO);

	ASTROS_DBG_PRINT(verbose, "astros_field_get_targets(%d)\n", pLineup->league);

    if(ASTROS_LEAGUE_BLOCK == pLineup->league)
    {
		tgt_count = load_target_list_file(pLineup);

		if(tgt_count > TODO_WHAT_LINEUP_REQURIES)
		{		


		}

	}



	ASTROS_DBG_PRINT(verbose, "astros_field_get_targets(%d) tgt_count = %d\n", pLineup->league, tgt_count);

	
	return tgt_count;

}
		

#ifndef ASTROS_CYGWIN
#define ASTROS_IOCTL
#endif

#ifdef ASTROS_IOCTL
#define CCISS_ALTUVE_SETRUNNUM _IOW(CCISS_IOC_MAGIC, 27, int)
#define CCISS_IOC_MAGIC 'B'


int astros_ioctl_setrunnum(int runnum)
{
#ifdef ASTROS_WIN
#else	
	int verbose = astros_get_verbosity(ASTROS_DBGLVL_INFO);
	char * devnm = "/dev/sg1";
	int fd;
	int res;
	
	ASTROS_DBG_PRINT(verbose, "astros_ioctl_setrunnum(%d) %s\n",runnum, devnm);


	
	
	fd = open(devnm, O_RDWR);

	if(fd < 0)
	{
		ASTROS_DBG_PRINT(verbose, "astros_ioctl_setrunnum(%d) %s: fd = %d\n",runnum, devnm, fd);
	}
	else
	{
		res = ioctl(fd, CCISS_ALTUVE_SETRUNNUM, (int32_t*) &runnum); 
		ASTROS_DBG_PRINT(verbose, "astros_ioctl_setrunnum(%d) %s: fd = %d DONE res = %d\n",runnum, devnm, fd, res);

	}
#endif

	return 0;
		

}
#endif

int astros_field_check(astros_lineup *pLineup)
{
	bool bRun = false;
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_INFO);

	if(ASTROS_FIELD_KERNEL == pLineup->field)
	{
		ASTROS_DBG_PRINT(verbose, "astros_field_check(ASTROS_FIELD_KERNEL)%d\n",0);

#ifdef ASTROS_IOCTL
		astros_ioctl_setrunnum(pLineup->gameid.gameid); 	
#endif
		
		if(astros_signs_write_lineup(pLineup))
		{
		}
	}
	if(ASTROS_FIELD_BAM == pLineup->field)
	{
		ASTROS_DBG_PRINT(verbose, "astros_field_check(ASTROS_FIELD_BAM)%d\n",0);

#ifdef NVIDIA_BAM

		if(astros_field_get_bam_targets(pLineup))
		{
			bRun = true;
		}
		else
		{
			ASTROS_ASSERT(0);
		}

#else
		ASTROS_DBG_PRINT(ASTROS_DBGLVL_ERROR, "Astros NOT COMPILED FOR BAM");
#endif
	}
	else
	{
		astros_field_get_targets(pLineup);
		
		ASTROS_DBG_PRINT(verbose, "astros_field_check(ASTROS_FIELD_USER)%d\n",0);
		bRun = true;
	}
	return bRun;
}

int astros_field_gamewrapup(astros_lineup *pLineup)
{
	int i;
	int error = 0;
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_INFO);
	astros_scorecard *pScorecard;

	ASTROS_ASSERT(pLineup);

	pScorecard = pLineup->pScoreCard;

	ASTROS_ASSERT(pScorecard);

	astros_score_open_csvlog(pLineup);

	ASTROS_ASSERT(pScorecard->csvlogFptr);


	for(i = 0; i < pScorecard->current_inning; i++)
	{
		ASTROS_DBG_PRINT(verbose, "astros_field_gamewrapup(%d)\n", i);

		if(0 == astros_score_inning(i, pScorecard))
		{
			astros_score_csvlog(i, pLineup, pScorecard);

		}
	}

	ASTROS_DBG_PRINT(verbose, "astros_field_gamewrapup(%d) Done with forloop\n", i);
	
	astros_score_lineup_columnated(pLineup);

	ASTROS_DBG_PRINT(verbose, "astros_field_gamewrapup(%d) EXIT\n", error);

	return error;

}

int astros_field_monitor(astros_lineup *pLineup)
{
	astros_scorecard *pScorecard;
	int sclen = sizeof(astros_scorecard);
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);
	int count = 0;
	int wait_us = 1000 * 1000 * 1;
	int error = 0;
	UINT64 start_ns;
	int last_inning_update = -1;
	int last_inning;
	
	pScorecard = ASTROS_ALLOC(64, sclen);
	
	ASTROS_ASSERT(pScorecard);


	start_ns = ASTROS_PS_HRCLK_GET();


	
	do
	{
		ASTROS_BATTER_SLEEPUS(wait_us);		

		ASTROS_DBG_PRINT(verbose, "astros_field_monitor(%d)[%d]\n", sclen, count);
		
		if(astros_signs_read_scorecard(pLineup, sclen, pScorecard))
		{
			ASTROS_DBG_PRINT(ASTROS_DBGLVL_ERROR, "ERror READ SCORECARD(%d)\n", sclen);	

		}
		else
		{
			ASTROS_DBG_PRINT(verbose, "Scorecard Good sclen =%d score_card_size = %d inning = %d : innings_planned = %d\n", sclen, 
				pScorecard->score_card_size, pScorecard->current_inning, pLineup->innings_planned);	


			last_inning = pScorecard->current_inning - 1;	
			
		    if((last_inning > -1) && (last_inning != last_inning_update))
		    {
				float percent = 0.0;
				double fElap;
				
				if(pLineup->innings_planned)
				{
					percent = (float)last_inning / (float)pLineup->innings_planned;
					percent = percent * 100;
				}

				fElap = ((float)ASTROS_PS_HRCLK_GET() - (float)start_ns) / 1000000000.0;

				printf("Inning(%d of %d) %.02f Percent  Elap = %f sec\n", last_inning, pLineup->innings_planned, percent, fElap );
				last_inning_update = last_inning;
			}

			

			



			if(pScorecard->current_inning == (pLineup->innings_planned))
			{

				if(sclen < pScorecard->score_card_size)
				{
					sclen = pScorecard->score_card_size;
				
					ASTROS_FREE(pScorecard);
				
					pScorecard = NULL;
				
					pScorecard = ASTROS_ALLOC(64, sclen);
				
					ASTROS_ASSERT(pScorecard);

					if(astros_signs_read_scorecard(pLineup, sclen, pScorecard))
					{
						ASTROS_DBG_PRINT(ASTROS_DBGLVL_ERROR, "ERror READ SCORECARD(%d) END\n", sclen);	
					}
		
				}

			
				ASTROS_DBG_PRINT(verbose, "Scorecard done(%d:%d)\n", pScorecard->current_inning, pLineup->innings_planned); 
				pLineup->pScoreCard = pScorecard;

				pLineup->pScoreCard->start_ns = start_ns;
				astros_field_gamewrapup(pLineup);
				ASTROS_DBG_PRINT(ASTROS_DBGLVL_INFO, "astros_field_monitor RETURN from astros_field_gamewrapup()(%d)\n", 0);

				break;
			}



		}
		count++;

	} while(1);

	ASTROS_DBG_PRINT(ASTROS_DBGLVL_INFO, "astros_field_monitor EXIT(%d)\n", 0);

	return error;

}



int ccb_queue_test(void)
{
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_INFO);
	int count = 32;
	ccb *pCCBBase = ASTROS_ALLOC(64, sizeof(ccb) * count);
	int i;
	ccb *pCCB;
	ccb_queue cQ;
#if(CCB_Q_TYPE == CCB_Q_TYPE_LOCKED)
	int waiting_count;
	ccb **pCCBList;
	ccb_list cL;

	astros_ccb_list_init(&cL, count, "CCBQINIT");

#endif	

	astros_ccb_queue_init(&cQ, count, "CCBQINIT");

	for(i = 0; i < count; i++)
	{
		pCCB = &pCCBBase[i];
		pCCB->idx = i;

		astros_ccb_queue_enqueue(&cQ, pCCB);

	}

	ASTROS_DBG_PRINT(verbose, "dq FULL count = %d\n", astros_ccb_queue_count(&cQ));

#if(CCB_Q_TYPE == CCB_Q_TYPE_LOCKED)

	pCCBList = astros_ccb_queue_multi_dequeue(&cQ, &waiting_count);

	for(i = 0; i < waiting_count; i++)
	{
		pCCB = pCCBList[i];

		ASTROS_DBG_PRINT(verbose, "dq CCB idx = %d count = %d\n", pCCB->idx, astros_ccb_queue_count(&cQ));
	
	}



#else



	while((pCCB = astros_ccb_queue_dequeue(&cQ)))
	{
		
		ASTROS_DBG_PRINT(verbose, "dq CCB idx = %d count = %d\n", pCCB->idx, astros_ccb_queue_count(&cQ));

	}
	pCCB = astros_ccb_queue_dequeue(&cQ);
	
	ASTROS_DBG_PRINT(verbose, "dq DONE = %p\n", pCCB);


#endif
		

	i = count;

	while(i)
	{
		i--;

		pCCB = &pCCBBase[i];

		astros_ccb_queue_enqueue(&cQ, pCCB);

	}

#if(CCB_Q_TYPE == CCB_Q_TYPE_LOCKED)
	pCCBList = astros_ccb_queue_multi_dequeue(&cQ, &waiting_count);

	for(i = 0; i < waiting_count; i++)
	{
		pCCB = pCCBList[i];

		ASTROS_DBG_PRINT(verbose, "dq CCB idx = %d count = %d\n", pCCB->idx, astros_ccb_queue_count(&cQ));
	
	}

	for( i = 0; i < count; i++)
	{
		pCCB = &pCCBBase[i];

		astros_ccb_list_enqueue(&cL, pCCB);
		
		ASTROS_DBG_PRINT(verbose, "list CCB idx = %d count = %d\n", pCCB->idx, astros_ccb_list_count(&cL));

	}



	while((pCCB = astros_ccb_list_dequeue(&cL)))
	{
		
		ASTROS_DBG_PRINT(verbose, "list DQ CCB idx = %d count = %d\n", pCCB->idx, astros_ccb_list_count(&cL));

	}





#else
	while((pCCB = astros_ccb_queue_dequeue(&cQ)))
	{
		
		ASTROS_DBG_PRINT(verbose, "dq CCB idx = %d : count = %d \n", pCCB->idx, astros_ccb_queue_count(&cQ));

	}
#endif
	
	return 0;



}


int precheck_cli(int argc, char **argv)
{
	int precheck = 0;

	if(argc > 1)
	{
		if(argv[1][0] == '-')
		{
			switch(argv[1][1])
			{
				case 'c':
				{
					char *dumpfn = "cmdstatdump.txt";
					
					precheck = 1;
					if(argc > 2)
					{
						
						dumpfn = argv[2];
					}
					astros_signs_dump_cmdstats(dumpfn);
				
					break;
				}

				
				default:
					break;
				


			}


		}
	}
	
		
	return precheck;

}


void astros_put_gameid(int gameid)
{
	FILE *fd;

	fd = fopen("astros.last.game", "w");
	
	if(fd)
	{

		fprintf(fd, "%d\n", gameid);

		fclose(fd);
	}


}

