#ifndef _ASTROS_H
#define _ASTROS_H



#ifdef ASTROS_CYGWIN
#include "Windows.h"
#endif

#define SINT8  int8_t


#define SINT16 int16_t
#define SINT32 int32_t
#define SINT64 int64_t
#define UINT8  uint8_t
#define UINT16 uint16_t
#define UINT32 uint32_t
#ifndef UINT64 
#define UINT64 uint64_t
#endif

#ifdef ALTUVE_ZOMBIE_LOOPBACK
#define ALTUVE_LOOPBACK_THREADED
#endif

#ifdef ASTROS_WIN
#include "astros_ps_linux_common.h"
#include <argtable3.h>
#include "astros_ps_windows.h"

#else
#ifdef ASTROS_CYGWIN


#else
#define ASTROS_LIBAIO
#define ASTROS_LIBURING
#endif
#include "astros_ps_linux_common.h"
#endif
#ifdef ALTUVE
#include "astros_ps_linux_kernel.h"
#else
#include "argtable3.h"
#include "astros_ps_linux_user.h"
#endif
#endif

#define ASTROS_UNUSED(x) (void)(x)




#define DEF_AB_CNT 7

/* This is the default ordering of the sweep parameters */
/* But designed to be dynamically orderable */

#define ASTROS_DIM_QDEPTH        0
#define ASTROS_DIM_IOSIZE        1 
#define ASTROS_DIM_TARGET_COUNT  2 
#define ASTROS_DIM_OPERATION     3
#define ASTROS_DIM_IO_COUNT      4
#define ASTROS_DIM_BURST_PERCENT 5
#define ASTROS_DIM_INNING_MODE   6
#define ASTROS_DIM_FIXEDBATTERS  7
//#define ASTROS_DIM_ATBAT_COUNT   8
#define ASTROS_DIM_LOAD_TYPE     8
#define ASTROS_DIM_BAT_MODE      9
#define ASTROS_DIM_ACCESS_TYPE   10
#define ASTROS_DIM_LAT_STATS     11
#define ASTROS_DIM_ENGINE_TYPE   12


/* This barfs when we go above 13 - James Harden is telling me something */
/* It has to do with the dimensions and not the raw number of innings because we can go really high (well over 10000) */
/* Overriding some that are fixed and rarely change, we can move those to globals but at some point I should figure it out */
/* Usually, inning count calculation in dry run comes up with total innings == 0 */
#define ASTROS_MAX_DIMENSION     13


#define ASTROS_INC_TYPE_LINEAR 0
#define ASTROS_INC_TYPE_POW2   1


#define ASTROS_DIM_LAT_STATS_DIS 0 
#define ASTROS_DIM_LAT_STATS_ENB 1



static inline int astros_get_verbosity(int local)
{


    //todo - have a global variable for setting that can override all local;
    return local;
    //return ASTROS_DBGLVL_ERROR;
}



typedef struct
{
    UINT32          dim_enum;
    UINT32          current_idx;
    UINT32          min;
    UINT32          max;
    UINT32          inc;
    UINT32          inc_type;
    char            name[32];    


} astros_dimension;

extern astros_dimension g_Default_dimmensions[ASTROS_MAX_DIMENSION];

#define ALLONES32 0xFFFFFFFF


#define ASTROS_MAX_BATTERS          64
#define ASTROS_MAX_TARGETS          128
#define ASTROS_MAX_CCBS             4096



#define ASTROS_LEAGUE_BLOCK         0


#define ASTROS_AIOENGINE_URING      0
#define ASTROS_AIOENGINE_LIBAIO     1
#define ASTROS_AIOENGINE_AIOWIN     1              /* Used so don't have to change scripting between Linux and Cygwin */ 
#define ASTROS_AIOENGINE_KBLOCK     2
#define ASTROS_AIOENGINE_BMM        3
#define ASTROS_AIOENGINE_FW_RS      4
#define ASTROS_AIOENGINE_FW_PAL     5
#define ASTROS_AIOENGINE_KLOOP      6
#define ASTROS_AIOENGINE_ZOMBIE     7
#define ASTROS_AIOENGINE_ZOMBIE_MR  8
#define ASTROS_AIOENGINE_ZOMBIE_PQI 9
#define ASTROS_AIOENGINE_SYNC       10
#define ASTROS_AIOENGINE_WINAIO     11
#define ASTROS_AIOENGINE_BAM_ARRAY  12



#define ASTROS_AIOENGINE_MAX        13


#define ALTUVE_ZOMBIE_PQI_TYPE      0
#define ALTUVE_ZOMBIE_MR_TYPE       1



#define ASTROS_AIOENGINE_MODE_LINUX_USER    0
#define ASTROS_AIOENGINE_MODE_LINUX_KERNEL  1
#define ASTROS_AIOENGINE_MODE_FW            2
#define ASTROS_AIOENGINE_MODE_WIN_USER      3




#define ASTROS_CCB_OP_WRITE                 0
#define ASTROS_CCB_OP_READ                  1


#define ASTROS_CCB_ACCESS_RANDOM            0
#define ASTROS_CCB_ACCESS_SEQUENTIAL        1


typedef struct __seq_ctrl
{
	int mode;
	
	struct
	{
		UINT64 next_lba;
		
	} bat[ASTROS_MAX_BATTERS];

} seq_control;

typedef struct _astros_target
{
    ASTROS_FD  fd;
    int  idx;
	int  mode;
    char path[64];
	int  access;
	UINT64 capacity4kblock;
	UINT64 size_bytes;
    struct __command_control_block  *pReady;
    struct __command_control_block  *pDone;
    ASTROS_ATOMIC       atomic_qd;

	void *pvZombieDevice;
	void *pvZombieParent;

	seq_control sequential_control; 

	void *pvEngine;

#ifdef NVIDIA_BAM
	void *pvBamTargetControl;
#endif


} target;

typedef int (*CCBCompletionCallback)(void *pvCCB);
typedef void (*CCB_get_lba_fn)(void *pvCCB);




#define CCB_Q_TYPE_ATOMIC 0
#define CCB_Q_TYPE_LOCKED 1

#define CCB_Q_TYPE CCB_Q_TYPE_ATOMIC

#define ALTUVE_ZOMBIE_SENT_DONE_CCB_SYNC 


typedef struct __command_control_block
{

#ifdef ASTROS_LINUX_KERNEL_LISTS
		struct list_head queue_list;
#else
		struct __command_control_block	*pListNext;
#endif	

    int                    idx;
    int                    tgtidx;
    int                    io_size;
    UINT64                 offset;
	UINT64                 lba;
	int                    block_count;
    int                    op;
	unsigned int           marker;

    void *                 pData;
    target                 *pTarget;


	UINT64                 start_ns;
	UINT64                 end_ns;

#ifdef ALTUVE_ZOMBIE_SENT_DONE_CCB_SYNC
	UINT32                 sent;
	UINT32                 done;
	UINT32                 inflight;
#endif



    CCBCompletionCallback pfnCCBCallback;
	CCB_get_lba_fn        pfnCCB;
	
    void                            *pvEngine;
    struct __command_control_block  *pNext;
    union
    {
        struct
        {
            int iov_idx;
            struct io_uring_sqe *pSqe;
        } iouring;
		struct
		{
			void *pvScsi;
			
		} smartpqi_zombie;
#ifdef ASTROS_LIBAIO
		struct
		{
			struct iocb aIocb;
			struct iovec iov;
		} libaio;
#endif
#ifdef ASTROS_WIN
		struct
		{
			OVERLAPPED stOverlapped;			
		} winaio;
#endif
#ifdef ASTROS_CYGWIN
		struct
		{
			OVERLAPPED stOverlapped;			
		} aiowin;
#endif

    } engine_scratch;


} ccb;


typedef struct _ccb_q_
{
	int  max_count;
	char *szName;

#if(CCB_Q_TYPE == CCB_Q_TYPE_ATOMIC)
	ASTROS_ATOMIC cur_count;
	int  tail;
	int  head;
	ccb **ppCCBqueue;

#elif (CCB_Q_TYPE == CCB_Q_TYPE_LOCKED)
	ccb **ppCCBqueue[2];
	int qidx[2];
	int beat;
    ASTROS_SPINLOCK     ccbqlock;

#endif
	unsigned int sig;




}  ccb_queue;

typedef struct
{
#ifdef ASTROS_LINUX_KERNEL_LISTS
		struct list_head queue_list;
#else

#ifdef ASTROS_SIMPLE_LIST
	ccb **ccbA;
#else
	ccb *pHead;
	ccb *pTail;
#endif
	
#endif
	int count;
	char *szName;
	int verbose;

} ccb_list;



typedef int (*IOEngineInitFunction)(void *pvEngine);
typedef int (*IOEngineFreeFunction)(void *pvEngine);
typedef int (*IOEnginePrepCCBFunction)(void *pvEngine, ccb *pCCB,  target *pTarget, void *pvfnCallback );
typedef int (*IOEngineResetFunction)(void *pvEngine);
typedef int (*IOEngineRegisterFunction)(void *pvEngine);
typedef int (*IOEngineSetupCCBFunction)(void *pvEngine, ccb *pCCB);
typedef int (*IOEngineQueuePending)(void *pvEngine);
typedef int (*IOEngineGetCompletion)(void *pvEngine, bool bDrain);





typedef struct
{
	  ASTROS_ATOMIC		  atomic_sync_threads_ready;
	  ASTROS_ATOMIC 	  atomic_io_count;
	  ASTROS_ATOMIC 	  atomic_sync_batter_done;
	  ASTROS_ATOMIC       atomic_start_sync;
	  
	  ASTROS_SPINLOCK	  donelock;

	  UINT64              start_ns;
	  UINT64              end_ns;

	  int                 total_io_count;
	  int                 total_threads;
	  int                 stalls;
	  void               *pvInning;

	  bool               bRun;
	  bool               bBam;
	  
	  void               *pvSyncBatterArray;
	  void               *pvEngine;

} sync_engine_spec;


typedef struct 
{
	ccb *pHead;
	ccb *pTail;

} astros_ccb_fifo;

typedef struct
{

	ccb_list ccb_list;
    ASTROS_SPINLOCK     ccb_list_lock;

} astros_ccb_list_locked;

#define ALTUVE_ZOMBIE_SINGLE_PATH_CROSS_COMPLETE


typedef struct
{


#ifdef ALTUVE_ZOMBIE_SINGLE_PATH_CROSS_COMPLETE
	astros_ccb_list_locked done;
#else
    ccb_queue done;
	astros_ccb_list_locked checked_pitches;
#endif

	ccb_list pending;

#ifdef ALTUVE_ZOMBIE_SENT_DONE_CCB_SYNC
	ccb *pPendingCurrent;
	int stalled_pending_checks;
#endif
	

} zombie_engine_spec;



typedef struct 
{
	UINT64 total_elap_ns;
	UINT64 count;
	UINT64 hi;
	UINT64 lo;

} astros_latency;


typedef struct 
{
	UINT64 total_offset;
	UINT64 count;
	UINT64 hi;
	UINT64 lo;

} astros_avg_offset;



#define ASTROS_MAX_CMDSTATS 128

typedef struct
{
	unsigned int cmd_len[4];
	unsigned int block_pow_2[2][14];
	//unsigned int block_4k_or_less[2][14];
	unsigned int running_nonpow2_block_cnt[2];
	unsigned int rnning_nonpow2_cmds[2];
	unsigned int nonrwcmd;
	unsigned int total_blocks[2];
	unsigned int totalrw[2];
	unsigned int ops[256];

} astros_single_cmdstat;


typedef struct
{
	astros_single_cmdstat stats[ASTROS_MAX_CMDSTATS];
	
} astros_cmd_stats;


	

typedef struct
{
    int type;
    int mode; 
    int depth;
	int min_reap;

	int stalls;
	int kicks;

	int cpu;
    
    bool bInit;
	bool bLatency;
	
    union
    {
#ifdef ASTROS_WIN
		winaio_engine_spec winaio;
		sync_engine_spec   sync;
#else /*!ASTROS_WIN */
#ifdef ALTUVE
        zombie_engine_spec zombie;
#else /* !ALTUVE */

#ifdef ASTROS_LIBURING
        iouring_engine_spec iouring;
#endif
		//kloop_engine_spec kloop;
        //kloop_engine_spec smartpqi;
#ifdef ASTROS_LIBAIO
  		libaio_engine_spec libaio;
#endif
#ifdef ASTROS_CYGWIN 
		ASTROS_FD hCompletionPort;
#endif
		sync_engine_spec   sync;


#endif /*!END ASTROS_WIN */		
    } engineSpecific;

    IOEngineInitFunction     pfnInit;
    IOEngineFreeFunction     pfnFree;
    IOEnginePrepCCBFunction  pfnPrepCCB; 
//never used    IOEngineResetFunction    pfnReset;
    IOEngineRegisterFunction pfnRegister;
    IOEngineSetupCCBFunction pfnSetup;
    IOEngineQueuePending     pfnQueue;
    IOEngineGetCompletion    pfnComplete;

	ZOMBIE_QCMD_FN           pfnQueueCommand;


	int 				     ccb_pending_count;
    ccb                      *pPendingHead;
	ccb                      *pStartQueueHead;

	astros_ccb_fifo          fifo;


	void                     *pvInning;
	void                     *pvBatter;
	
	int                      zombieParm;


} aioengine;

#define ASTROS_BATTER_NAME_LEN 64

typedef ASTROS_THREAD_FN_RET (*batterThreadFn)(void *pvData);

typedef struct __astros_batter
{
    int idx;
	int cpu;

    ASTROS_BATTER_JERSEY      id;
    bool bCalledup;
    bool bOnDeck;
    bool bAtBat;
	bool bPinched;
	bool bCleanup;
	
	
    int rt_policy;
    int rt_priority;

	UINT64             tgtmask;
    UINT64             on_deck_start_ns;
    UINT64             at_bat_start_ns;
	UINT64             at_bat_end_ns;
	int                at_bat_io_count;

	astros_latency     cmd_lat;
	astros_latency     inter_lat;
	astros_avg_offset  avg_off;
	
	
	UINT64             last_submit_ns;
	
	batterThreadFn rotateFn;

	
    void *pvLineup;           
	void *pvCurrentInning;
	void *pvInning;
	
	char                            batter_name[ASTROS_BATTER_NAME_LEN];

    aioengine                      *pEngine;
    aioengine                       engineA[ASTROS_AIOENGINE_MAX];
} batter;



typedef struct 
{
    ccb *pCCBBase;
    ccb *pCCBFree;

	void *pvSyncBatterArray;
	int  sync_batter_array_size;



    int max_batters;

    void *pDataBufferBase;

    ASTROS_BATTER_JERSEY            id;
    ASTROS_BATTER_JERSEY            scorer_id;
        

    ASTROS_ATOMIC       atomic_batter_on_deck;
    ASTROS_ATOMIC       atomic_batter_lock;
	ASTROS_ATOMIC		atomic_batter_done;

    ASTROS_SPINLOCK     playball;

    int                batter_count;

	
    batter batters[ASTROS_MAX_BATTERS];
    target targets[ASTROS_MAX_TARGETS];

} gametime_resources;


#define ASTROS_MAX_ATBATS 16

//#define ASTROS_INC_AVG_OFFSET

typedef struct
{
	UINT64 start_ns;
	UINT64 end_ns;
	UINT64 elap_ns;
	UINT32 io_count;

	
	astros_latency cmd_lat;
	astros_latency inter_lat;
#ifdef ASTROS_INC_AVG_OFFSET
	astros_avg_offset avg_off;
#endif

} astros_pitches;


//#define ASTROS_KERNEL_STATS

#ifdef ASTROS_KERNEL_STATS
#ifndef ALTUVE
#define ASTROS_GET_KSTATS
#endif
#endif


typedef struct
{
	float fOtherBatSubmit;
	float fOtherBatComplete;
	float fIrqPercentSubmits;
	float fRespPerIrq;
	
} astros_at_bat_kstats_summary;


typedef struct
{
    bool          bPrecon;
    bool          bRsvd[3];


    int           total_ccb;
    int           batters;
	int           atbat_number;
	int           scale_scratch;
   
    UINT32        iops;
	float         fIops;
	float         fStartDelta;
	float 		  fEndDelta;

	astros_pitches *pitches;
	
//	astros_pitches pitches[ASTROS_MAX_BATTERS];

	astros_latency cmd_lat;
	astros_latency inter_lat;

#ifdef ASTROS_INC_AVG_OFFSET
	astros_avg_offset avg_off;
#endif
#ifdef ASTROS_KERNEL_STATS
	astros_at_bat_kstats_summary kstats;
#endif

} astros_atbat;


#define ASTROS_KERNEL_STATS_LBA_MASK 0xFFFL
#define ASTROS_KERNEL_STATS_LBA_MARKER_STRIDE 8
#define GET_KSTAT_LBA_MARKER(_cpu) (_cpu * ASTROS_KERNEL_STATS_LBA_MARKER_STRIDE) 
#define GET_KSTAT_LBA_CPU(lba) ((lba & ASTROS_KERNEL_STATS_LBA_MASK) / ASTROS_KERNEL_STATS_LBA_MARKER_STRIDE)

#define ASTROS_TIME_HISTORGRAM_CNT 512

typedef struct 
{
	UINT32 idx;

	UINT64 ns[ASTROS_TIME_HISTORGRAM_CNT];

} astros_time_histogram;
	
	
typedef struct
{
		int submit_count;
		int other_cpu_submit;
	
		int complete_count;
		int other_cpu_complete;
	
		int irq_count;
		int irq_total_responses;
		
		astros_time_histogram hSubmits;
		astros_time_histogram hCompletions;
		
	
} astros_kernel_batter_stats;
	
typedef struct
{
		astros_kernel_batter_stats batterStats[ASTROS_MAX_BATTERS];
	
} astros_kernel_stats;


#define ASTROS_ALL_ONES64 0xFFFFFFFFFFFFFFFF


#define ASTROS_INNING_MODE_SUSTAINED 0
#define ASTROS_INNING_MODE_BURST     1
#define ASTROS_INNING_MODE_LOAD      2

#define ASTROS_INNING_BM_FIXED           0
#define ASTROS_INNING_BM_DYNAMIC         1
#define ASTROS_BM_INNING_BM_BATPERTARGET 2
#define ASTROS_BM_INNING_BM_FIO          3
#define ASTROS_BM_SYNC_BATTERS           4
#define ASTROS_BM_BAM                    5


#define ASTROS_INNING_BDP_RR          0
#define ASTROS_INNING_BDP_TARGET_GRP  1 
#define ASTROS_INNING_BDP_SYNC_BATS   2

#define ASTROS_SEQ_MODE_RANDOM                         0x000
#define	ASTROS_SEQ_MODE_SPLIT_STREAM_RST_INNING        0x001 
#define	ASTROS_SEQ_MODE_SPLIT_STREAM_RST_ATBAT         0x002 


#define BOX_SCORE_STAT_NAME \
	__Cx(BxScrIops) \
	__Cx(BxScrInning) \
	__Cx(BxScrStdDevIops) \
	__Cx(BxScrIopsDeviationPercent) \
	__Cx(BxScrCulledIops) \
	__Cx(BxScrCulledStdDevIops) \
	__Cx(BxScrCulledIopsDeviationPercent) \
	__Cx(BxScrHiIops) \
	__Cx(BxScrLoIops) \
	__Cx(BxScrCmdLatAvgUs) \
	__Cx(BxScrCmdLatHiUs) \
    __Cx(BxScrCmdLatLoUs) \
    __Cx(BxScrInterLatAvgUs) \
    __Cx(BxScrInterLatHiUs) \
    __Cx(BxScrInterLatLoUs) \
	__Cx(BxScrScoringBatters) \
	__Cx(BxScrOtherBatSubmit) \
	__Cx(BxScrOtherBatComplete) \
	__Cx(BxScrIrqPercentSubmits) \
	__Cx(BxScrRespPerIrq) \
	__Cx(BxScrBps) \









#define __Cx(x) x, 
enum BoxScoreStatEnum { BOX_SCORE_STAT_NAME BxScrMaxEnum};
#undef __Cx
#define __Cx(x) #x,
static const char * const BoxScoreStatStr[] = { BOX_SCORE_STAT_NAME };
#define BoxScoreStatStr(x) &BoxScoreStatStr[x][5] 


typedef struct
{
	float fStats[BxScrMaxEnum];
} astros_box_score;


#define COL_SUM_STAT_NAME \
	__Colx(ColSumTotalIops) \
	__Colx(ColSumTotalRow) \
	__Colx(ColSumMaxQueueDepth) \


#define __Colx(x) x, 
enum ColSumStatEnum {COL_SUM_STAT_NAME ColSumMaxEnum};
#undef __Colx
#define __Colx(x) #x,
static const char * const ColSumStatStr[] = { COL_SUM_STAT_NAME };
#define ColSumStatStr(x) &ColSumStatStr[x][6] 


typedef struct 
{
	float fStats[ColSumMaxEnum];

} astros_score_column_summary;
	
	
/******************************************************************************************************/
/** INNING_PARAMETERS                                                                                **/
/******************************************************************************************************/

#define INNING_PARAMETER_NAME \
	__Ix(InParInningNumber) \
	__Ix(InParQDepth) \
	__Ix(InParIoSize) \
	__Ix(InParTargetCount) \
	__Ix(InParOperation) \
	__Ix(InParAccess) \
	__Ix(InParAtBat) \
	__Ix(InParSustainedIos) \
	__Ix(InParInningMode) \
	__Ix(InParBatterMode) \
	__Ix(InParMaxBatters) \
	__Ix(InParBatterDistributionPolicy) \
	__Ix(InParPreConAtBat) \
	__Ix(InParScoreAtBat) \
	__Ix(InParScaleAtBat) \
	__Ix(InParMeasuringAtBats) \
	__Ix(InParScoringBatters) \
	__Ix(InParRow) \
	__Ix(InParCol) \
	__Ix(InParScalingBestAtBat) \
	__Ix(InParBurstPercent) \
	__Ix(InParQueueDepthTrough) \
	__Ix(InParFixedBatters) \
	__Ix(InParField) \
	__Ix(InParZombieParm) \
	__Ix(InParEnableLatency) \
	__Ix(InParEngineType) \
	__Ix(InParLoadType) \
 

#define __Ix(x) x, 
enum InningParamEnum { INNING_PARAMETER_NAME InParMaxEnum};
#undef __Ix
#define __Ix(x) #x,
static const char * const InningParameterStr[] = { INNING_PARAMETER_NAME };
#define InningParamStr(x) &InningParameterStr[x][5] 
#define INNING_PARAM(_pInn, _idx) _pInn->InningParam[_idx]
	
/******************************************************************************************************/
/** GAME_PARAMETERS                                                                                  **/  
/******************************************************************************************************/
#define GAME_PARAMETER_NAME \
	__Gx(GmParConstantDataLength) \

#define __Gx(x) x, 
enum GameParamEnum { GAME_PARAMETER_NAME GameParMaxEnum};
#undef __Gx
#define __Gx(x) #x,
static const char * const GameParameterStr[] = { GAME_PARAMETER_NAME };
#define GameParamStr(x) &GameParameterStr[x][5] 
extern int g_GameParam[GameParMaxEnum];
#define GAME_PARAM(_idx) g_GameParam[_idx]


 
typedef struct
{

	int InningParam[InParMaxEnum];	

    void *pvLineup;

	bool bScaling;
	astros_pitches *pPitches;
	
		
	astros_box_score box_score;

    astros_atbat atbats[ASTROS_MAX_ATBATS];

} astros_inning;



typedef struct
{
	int total_innings;
	int current_inning;
	int last_inning_updated;
	int max_atbats;
	int inning_size_bytes;
	int score_card_size;
	UINT64 start_ns;
	UINT64 end_ns;


	float fElap; 
	float fInPerSec;
		
	bool bScore;
	bool bRsvd[3];

	ASTROS_FILE_PTR  csvlogFptr;
	char    csvlogFn[156];

	astros_inning innings[0];

} astros_scorecard;



typedef struct 
{
	UINT32 new_game;


} astros_control;


#define ASTROS_HOSTNAME_SIZE 128

typedef struct
{
	int gameid;
	char logpath[128];
	char szHostname[ASTROS_HOSTNAME_SIZE];

} astros_gameid;


#define ASTROS_FIELD_MASK_INSERTION_POINT   0xF
#define ASTROS_FIELD_USER                     0
#define ASTROS_FIELD_KERNEL                   1
#define ASTROS_FIELD_FIO                      2 //for commong charting
#define ASTROS_FIELD_BAM                      3 

typedef int (*LULogColumn)(void *pvLineup, int inning);


typedef struct 
{
	int                            gametime_offset;
	int                            lineup_size;
    int                            dimensions_count;
    int                            dimensions_order[ASTROS_MAX_DIMENSION];
    astros_dimension               dimmensions[ASTROS_MAX_DIMENSION]; 
    unsigned int                   inning_count;
    unsigned int                   innings_planned;
    unsigned int                   league;
    unsigned int                   max_io;
    unsigned int                   total_q;
    unsigned int                   total_data_buffer_size;
    unsigned int                   called_up_sleep_us;
    int                            rt_policy;    
    int                            rt_priority;
    astros_gameid                  gameid;

	int                            max_qd;
	int                            max_targets; 
	int                            max_single_buffer;
	int                            total_buffers;



    bool                           bDryRun;
    bool                           bProxyUserDataBuffer;
	bool                           bRsv[2];
	
	void                          *pProxyUserDataBuffer;


	int                            current_row;
	int                            current_col;
	int                            max_row;
	int                            max_col;
	int                            debug_inning;
	unsigned int                   field;


	LULogColumn                    fnLogColumn;
	int                            LogColBxScr;
	ASTROS_FILE_PTR                fptr_col;
	

    UINT32                         master_cpu;

    gametime_resources               gametime;

	astros_scorecard               *pScoreCard;

	int                           **ppRowColMap;

	char                           *szUserTag;
	bool                            bUserTag;


} astros_lineup;

#define MAX_SCORECARD_HISTORY 4




typedef struct
{
	bool bWaitForLineup;
	bool bPlayBall;
	bool bRainedOut;


	struct task_struct *			id;
	struct workqueue_struct *     wqid;
	
	astros_lineup *pCurLineup;
	astros_scorecard *pPrevScoreCard[MAX_SCORECARD_HISTORY];
	int            prev_sc_idx;


	
    ASTROS_SPINLOCK     scorelock;
	
	struct
	{
		int num_scsicmd;
		int size_scsi_cmd;
		void *pvScsiCmds;
		char *pCdbs;
		void *pvReqs;
		UINT8 *pSense;
	} zombie;

	struct
	{
		int gameid;
		struct task_struct *tsk; 
		struct mm_struct *mm;
		struct page **page_ptrs;
	} usr_context;

	void  * pvCmdHist;
	void  * pvCmdStat;

	int    zombie_tgt_cnt;
	
	target ZombieTargets[ASTROS_MAX_TARGETS];
    void *pvLoopback;

	void *pvKernelStats;


} astros_umpire;



#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))




typedef struct
{
	int                  idx;
	int                  fd;
	ASTROS_BATTER_JERSEY id;
	ccb                 *pCCB;
	sync_engine_spec    *pEngSpec;
	
	
    UINT64             start_ns;
	UINT64             end_ns;

	
	astros_latency     cmd_lat;


} astros_sync_engine_batter;


#define ASTROS_FIXLOAD_POW2_SLOTS 14

typedef struct
{
	
	int pow2_loads[2][ASTROS_FIXLOAD_POW2_SLOTS];
	int odd_block_midpoint[2];
	int percent_write;
	
} astros_fixed_load;

#define ASTROS_FIXED_LOAD_IOSIZE 0xBA1717






int astros_ccb_find_and_remove(ccb **pHead, ccb *pCCB);

extern unsigned int g_series_number;
extern unsigned int g_sequence_number;


/* astros.c */
int astros_open_target(astros_lineup * pLineup, char *path, int targetidx, int operation);
int astros_umpire_pregame(astros_lineup *pLineup);

void astros_get_gameid(astros_lineup *pLineup);
void astros_kernel_mode_dim_constraints(void);
void astros_user_mode_contstraints(void);
int astros_open_target_file(astros_lineup * pLineup, char *path, int targetidx, int operation);
astros_fixed_load * astros_get_fixed_load(aioengine *pEngine);
char * astros_run_tag_ptr(astros_lineup *pLineup);
void astros_parse_cli(int argc, char **argv);
int astros_field_check(astros_lineup *pLineup);
int astros_field_monitor(astros_lineup *pLineup);
void astros_put_gameid(int gameid);
char * astros_get_version(void);
void astros_init_game_param(void);
ASTROS_THREAD_FN_RET astros_cygwin_test_thread(void *pvBatter);
void cygwin_thread_test(void);
int astros_cygwin_payofff_umpire(astros_lineup *pLineup);
void astros_cygwin_init_hifreq_clock(void);











/* astros_batters.c */
int astros_batter_sync(astros_lineup        * pLineup, batter * pBatter);
int astros_batters_on_deck(astros_lineup       * pLineup, astros_inning * pInning);
ASTROS_THREAD_FN_RET astros_batter_rotation(void *pvBatter);
int astros_batter_set_priority(batter *pBatter, int rt_policy);
int astros_batter_get_policy_and_priority(batter *pBatter, int *pPolicy, int * pPriority);
int astros_batters_setengine(astros_inning * pInning, astros_atbat *pAtbat);
int astros_batters_up(astros_inning * pInning, astros_atbat *pAtbat);
int astros_batters_register(astros_inning * pInning, astros_atbat *pAtbat);
typedef int (*FnBatterTakePitches)(batter *pBatter, aioengine *pEngine, astros_lineup *pLineup, astros_inning *pInning);
int astros_batter_get_io_limit(batter *pBatter);
void astros_batters_drain(batter *pBatter, aioengine *pEngine);
int astros_batters_distribute_targets_to_batters(astros_inning * pInning, astros_atbat *pAtbat);
int astros_batters_set_ioengine(batter * pBatter, int engine);
void astros_batter_force_out(aioengine *pEngine, int code);
int astros_batters_setup_sequentials(astros_inning * pInning, astros_atbat *pAtbat);




/* astros_inning.c */
void astros_inning_cleanup_atbat(astros_inning * pInning, astros_atbat *pAtbat);
int astros_start_inning(astros_lineup *pLineup, int inning_number);
astros_inning * astros_get_inning(astros_lineup *pLineup, int inning_number);
astros_inning * astros_inning_get_precon(astros_lineup *pLineup, astros_inning * pInning);
void astros_inning_complete_at_bat(astros_inning * pInning,  batter *pBatter);
astros_atbat * astros_inning_get_at_bat(astros_inning * pInning);
int astros_inning_get_batters(astros_inning * pInning);
bool astros_inning_infield_fly(astros_lineup *pLineup);
CCBCompletionCallback astros_inning_get_ccb_callback(astros_inning *pInning);
int astros_inning_prep_target_queues(astros_inning * pInning, astros_atbat *pAtbat);
void astros_inning_ready_ccb(astros_inning *pInning, ccb *pCCB);
int astros_inning_get_io_limit(astros_inning *pInning);
void astros_inning_bat_ready(astros_atbat *pAtbat, astros_inning * pInning, int scoring_batters, int size);
void astros_inning_score(astros_inning * pInning);
void astros_inning_cleanup_inning(astros_inning * pInning);








/* astros_ccb.c */
void ccb_get_seq_lba(ccb *pCCB);
int astros_ccb_count(ccb **pHead);
ccb * astros_get_free_ccb(astros_lineup *pLineup);
void astros_ccb_put(ccb **pHead, ccb *pCCB);
ccb * astros_ccb_get(ccb **pHead);
void astros_put_free_ccb(astros_lineup *pLineup, ccb *pCCB);
int astros_ccb_callback_sustained(void *pvCCB);
int astros_ccb_callback_burst(void *pvCCB);
void astros_ccb_fifo_enqueue (astros_ccb_fifo *pFifo, ccb *pCCB);
ccb * astros_ccb_fifo_dequeue (astros_ccb_fifo *pFifo);
void astros_ccb_queue_init(ccb_queue *pQ, int count, char *szName);
void astros_ccb_queue_enqueue(ccb_queue *pQ, ccb *pCCB);
ccb * astros_ccb_queue_dequeue(ccb_queue *pQ);
void astros_ccb_queue_free(ccb_queue *pQ);
ccb ** astros_ccb_queue_multi_dequeue(ccb_queue *pQ, int *pCount);

void astros_ccb_list_init_locked(astros_ccb_list_locked *pLockedFifo);
void astros_ccb_list_free_locked(astros_ccb_list_locked *pLockedFifo);
void astros_ccb_list_enqueue_locked (astros_ccb_list_locked *pLockedFifo, ccb *pCCB);
ccb * astros_ccb_list_dequeue_locked (astros_ccb_list_locked *pLockedFifo);
void astros_ccb_set_fixed_load(ccb *pCCB, aioengine *pEngine);



/* astros_win.c */
int astros_winaio_init(void *pvEngine);
int astros_winaio_free(void *pvEngine);
int astros_winaio_setup_ccb(void *pvEngine, ccb *pCCB);
int astros_winaio_queue_pending(void *pvEngine);
int astros_winaio_complete(void *pvEngine, bool bDrain);
int astros_winaio_reset(void *pvEngine);
int astros_winaio_register(void *pvEngine);
int astros_winaio_prep_ccb(void *pvEngine, ccb *pCCB, target *pTarget, void * pvfnCallback);







#if(CCB_Q_TYPE == CCB_Q_TYPE_ATOMIC)
static inline int astros_ccb_queue_count(ccb_queue *pQ) { return ASTROS_GET_ATOMIC(pQ->cur_count); }
#else
static inline int astros_ccb_queue_count(ccb_queue *pQ)
{
	int beat = pQ->beat & 1;

	return pQ->qidx[beat];

}

#endif
static inline int astros_ccb_list_count(ccb_list *pL) { return pL->count; }
void astros_ccb_list_init(ccb_list *pL, int count, char *szName);
void astros_ccb_list_enqueue(ccb_list *pL, ccb *pCCB);
ccb * astros_ccb_list_dequeue(ccb_list *pL);
void astros_ccb_list_free(ccb_list *pL);


void ccb_get_random_lba(ccb *pCCB);



/* astros_lineup.c */
int astros_warmup_lineup( astros_lineup *pLineup );
astros_lineup * astros_get_lineup(int argc, char **argv);
int astros_free_lineup(astros_lineup *pLineup);
int astros_lineup_setup_rotation(astros_lineup *pLineup);
int astros_callup_batters(astros_lineup *pLineup);
int astros_lineup_run(astros_lineup *pLineup, int level);
int astros_lineup_cleanup(astros_lineup *pLineup);
int astros_lineup_draft_players(astros_lineup *pLineup);
void astros_lineup_first_pitch(astros_lineup      * pLineup);
int astros_lineup_get_engine(astros_lineup *pLineup, astros_inning * pInning);
void astros_lineup_prep_databuffers(astros_lineup *pLineup);
void astros_lineup_reset_target_sequentials(astros_lineup                    *pLineup, int mode);


/* astros_scorer.c */
int  astros_scorer_get(ASTROS_BATTER_JERSEY * pScorer_id, astros_lineup *pLineup, int master_cpu);
int astros_scorer_post_final(astros_lineup *pLineup, astros_scorecard *pScoreCard);
int astros_score_inning(int inning_number, astros_scorecard *pScoreCard);
void astros_score_csvlog(int inning_number, astros_lineup *pLineup, astros_scorecard *pScoreCard);
void astros_wait_scorer_ready(astros_scorecard *pScoreCard);
void astros_score_csvlog(int inning_number, astros_lineup *pLineup, astros_scorecard *pScoreCard);
int astros_score_open_csvlog(astros_lineup *pLineup);
void astros_score_lineup_columnated(astros_lineup *pLineup);
void astros_batter_calculate_cmd_latency(ccb *pCCB);
void astros_batter_reset_latency(batter *pBatter);
void astros_batter_calculate_inter_latency(ccb *pCCB);



/* astros_signs.c */

int astros_signs_write_lineup(astros_lineup *pLineup);
int astros_signs_read_scorecard(astros_lineup *pLineup, int length, astros_scorecard *pScorecard);
int astros_signs_read_kstats(astros_lineup *pLineup, astros_kernel_stats *pKstats, int length);
int astros_signs_dump_cmdstats(char *dumpfn);



/* altuve_main.c */
int altuve_umpire_pregame(astros_lineup *pLineup);
astros_umpire * altuve_get_umpire(void);

/* astros_sync_batters.c */
int astros_sync_batters_up(astros_inning * pInning, astros_atbat *pAtbat);
int astros_sync_engine_free(void *pvEngine);
int astros_sync_distribute_targets_to_sync_batters(astros_inning * pInning, astros_atbat *pAtbat);
int astros_sync_engine_init(void *pvEngine);
void astros_sync_batter_cleanup(astros_inning * pInning, astros_atbat *pAtbat);
void astros_sync_inning_atbat_calculate(astros_atbat *pAtbat, astros_inning *pInning, int at_bat_idx);
void astros_sync_wait_engine_ready(astros_inning * pInning);

/* astros_spdk */
int astros_spdk_test(void);



#define ASTROS_CALLED_UP_SLEEP       100000
#define ASTROS_CALLED_UP_SYNC_SLEEP 1000000






#include "astros_linux_aio.h"
#ifdef ALTUVE
#include "altuve_aio.h"
#else
extern int gDefaultBaseballField;
#endif

#ifdef ALTUVE_ZOMBIE
#define ALTUVE_ZOMBIE_ALLOC_CDB_SIZE 32
#define ALTUVE_ZOMBIE_MAX_SCSICMD     (4096 * 2)



static inline struct scsi_cmnd * altuve_zombie_idx_scsi(int index, astros_umpire *pUmpire)
{
	int bidx = pUmpire->zombie.size_scsi_cmd * index;
	UINT8 *ptr =  pUmpire->zombie.pvScsiCmds;

	return (struct scsi_cmnd *) &ptr[bidx];


}

static inline UINT8 * altuve_zombie_idx_cdb(int index, astros_umpire *pUmpire)
{
	int bidx = ALTUVE_ZOMBIE_ALLOC_CDB_SIZE * index;
	UINT8 *ptr =  pUmpire->zombie.pCdbs;

	return &ptr[bidx];

}

#define AZ_CCB_SCSI(_idx, _pUmpire) altuve_zombie_idx_scsi(_idx, _pUmpire)
#define AZ_CDB_SCSI(_idx, _pUmpire) altuve_zombie_idx_cdb(_idx, _pUmpire)
#define AZ_ESCRATCH_CCB_SCSI(_pCCB) _pCCB->engine_scratch.smartpqi_zombie.pvScsi


#define ALL_ONES_64BIT 0xFFFFFFFFFFFFFFFF




//#define ALTUVE_ZOMBIE_SMARTPQI

#ifdef ALTUVE_ZOMBIE_SMARTPQI
int altuve_zombie_init_smartpqi(astros_umpire *pUmpire);
void altuve_zombie_cleanup_smartpqi(astros_umpire *pUmpire);
#endif

//#define ALTUVE_ZOMBIE_MEGRAIDSAS
#ifdef ALTUVE_ZOMBIE_MEGARAIDSAS
int altuve_zombie_init_megaraidsas(astros_umpire *pUmpire);
void altuve_zombie_cleanup_megaraidsas(astros_umpire *pUmpire);
#endif


void altuve_zombie_get_targets(astros_lineup *pLineup, astros_umpire *pUmpire);




#endif



static inline bool astros_is_lineup_fixed_load(astros_lineup *pLineup)
{
	


	if(ASTROS_FIXED_LOAD_IOSIZE == pLineup->dimmensions[ASTROS_DIM_IOSIZE].current_idx)
	{
		return true;
	}
	
	return false;
	
}

static inline bool astros_is_fixed_load(aioengine *pEngine)
{
	astros_inning *pInning = (astros_inning *)pEngine->pvInning;
	astros_lineup *pLineup = (astros_lineup *)pInning->pvLineup;
	
	return astros_is_lineup_fixed_load(pLineup);
	
}


#define ASTROS_IO_MULTI_COMPLETE

extern UINT64 gDefTargetCapacity;

#define TWOPOINT5HUNDREDGB4KBLOCKS 61035156
#define TWOHUNDREDGB4KBLOCKS (78643200 / 3) * 2
#define THREEHUNDREDGB4KBLOCKS 78643200 
#define TWOTB4KBLOCKS 498073600
#define ONETB4KBLOCKS 498073600 / 2

//#define DEF_4KBLOCKS TWOTB4KBLOCKS
//#define DEF_4KBLOCKS ONETB4KBLOCKS
//#define DEF_4KBLOCKS THREEHUNDREDGB4KBLOCKS
//#define DEF_4KBLOCKS TWOPOINT5HUNDREDGB4KBLOCKS
#define DEF_4KBLOCKS 45400000



#define ASTROS_ERR_ZOMBIE_CCB_DONE_STALL           1 
#define ASTROS_ERR_LIBAIO_IO_GET_EVENT_STALL       2


#define BATTER_SYNC_ATOMIC

#define ASTROS_FB_BLOCK_SIZE       4096


/* astros_win.c or astros_linux.c */
int astros_get_open_mode(int operation, astros_lineup * pLineup);
int astros_get_rt_policty(void);




int astros_win_rotation_create(void * pVfn, void * pvBatter);
int astros_win_batter_set_cpu(ASTROS_BATTER_JERSEY jersey, int cpu);

/* astros_bam.cu */

ASTROS_THREAD_FN_RET astros_batter_bam_rotation(void *pvSyncBatter);





#endif /* _ASTROS_H */

