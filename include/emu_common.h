#ifndef __EMU_COMMON_H
#define __EMU_COMMON_H



#include "emu_stats.h"






//no reason this can't be 64K, just need to massage some stuff
#define BAM_EMU_MAX_QUEUES 1024

//#define	KERNEL_DBG_ARRAY

#ifdef KERNEL_DBG_ARRAY
#define BA_DBG_SET(__p, __idx, __val) if(0 == (blockIdx.x * blockDim.x + threadIdx.x))  __p->debugA[__idx] = __val 
#else
#define BA_DBG_SET(__p, __idx, __val)
#endif

#define BA_DBG_IDX_MARK_RUN  0
#define BA_DBG_IDX_RUN_COUNT 1

#define BA_DBG_VAL_MARK_RUN  0xBABABABA




#define BAM_EMU_TARGET_DISABLE                      0
#define BAM_EMU_TARGET_ENABLE              0x00000001
#define BAM_EMU_TARGET_LATENCY             0x00000002
#define BAM_EMU_TARGET_AGGREGATION         0x00000004
#define BAM_EMU_TARGET_AGG_CONT_BITMASK    0x0000FF00
#define BAM_EMU_TARGET_AGG_CONT_BITSHIFT            8



#define BAM_EMU_HOST_ASSERT(__assert) if(!(__assert))do { \
	printf("BAM_EMU_HOST_ASSERT @LINE=%d in %s\n", __LINE__, __FILE__); 	\
	printf("\n\n***** EXIT(0) ***"); \
	exit(0); } while(0)

#define BAM_EMU_DEVICE_ASSERT(__assert) if(!(__assert))do { \
	printf("BAM_EMU_DEVICE_ASSERT @LINE=%d in %s\n", __LINE__, __FILE__); 	\
	printf("\n\n***** EXIT(0) ***"); \
	assert(0);\
	} while(0)


#define BAM_EMU_DEVICE_DBG_ASSERT_ENABLE 1  
#define BAM_EMU_DEVICE_ASSERT_DBG(__assert) if(BAM_EMU_DEVICE_DBG_ASSERT_ENABLE) BAM_EMU_DEVICE_ASSERT(__assert)



#define BAM_EMU_DBGLVL_NONE    0
#define BAM_EMU_DBGLVL_IOPATH  1
#define BAM_EMU_DBGLVL_ERROR   2
#define BAM_EMU_DBGLVL_INFO    3
#define BAM_EMU_DBGLVL_DETAIL  4
#define BAM_EMU_DBGLVL_VERBOSE 5

#define BAM_EMU_DBGLVL_COMPILE BAM_EMU_DBGLVL_ERROR

#define BAM_DBG_CODE_PATH_ALL             0xFFFFFFFFFFFFFFFFUL
#define BAM_DBG_CODE_PATH_H_CREATE_Q      0x1
#define BAM_DBG_CODE_PATH_H_EMU_RPC       0x2
#define BAM_DBG_CODE_PATH_H_INIT_EMU      0x4
#define BAM_DBG_CODE_PATH_H_START_EMU     0x8
#define BAM_DBG_CODE_PATH_H_UPDATEDQ      0x10
#define BAM_DBG_CODE_PATH_H_EMU_THREAD    0x20
#define BAM_DBG_CODE_PATH_H_CLEANUP_EMU   0x40
#define BAM_DBG_CODE_PATH_H_INIT_MAPPER   0x80
#define BAM_DBG_CODE_PATH_H_LAT_PSIZE     0x100
#define BAM_DBG_CODE_PATH_H_INIT_AGG      0x200
#define BAM_DBG_CODE_PATH_H_INIT_VENDOR   0x400
#define BAM_DBG_CODE_PATH_H_STATS_DUMP    0x800
#define BAM_DBG_CODE_PATH_H_EMU_CTRL      0x1000



#define BAM_DBG_CODE_DEVICE_OFFSET        32UL
#define BAM_DBG_CODE_MACRO_DEVICE(_cd) ((uint64_t)_cd << BAM_DBG_CODE_DEVICE_OFFSET)
#define BAM_DBG_CODE_PATH_D_KER_QSTRM     BAM_DBG_CODE_MACRO_DEVICE(0x1) 
#define BAM_DBG_CODE_PATH_D_SQ_CHECK      BAM_DBG_CODE_MACRO_DEVICE(0x2)
#define BAM_DBG_CODE_PATH_D_SQ_PROCESS    BAM_DBG_CODE_MACRO_DEVICE(0x4)
#define BAM_DBG_CODE_PATH_D_NVME_SUB      BAM_DBG_CODE_MACRO_DEVICE(0x8)
#define BAM_DBG_CODE_PATH_D_NVME_EXE      BAM_DBG_CODE_MACRO_DEVICE(0x10)
#define BAM_DBG_CODE_PATH_D_NVME_LOOP     BAM_DBG_CODE_MACRO_DEVICE(0x20)
#define BAM_DBG_CODE_PATH_D_CQ_DRAIN      BAM_DBG_CODE_MACRO_DEVICE(0x40)
#define BAM_DBG_CODE_PATH_D_INIT_Q_PAIR   BAM_DBG_CODE_MACRO_DEVICE(0x80)
#define BAM_DBG_CODE_PATH_D_GET_Q_PAIR    BAM_DBG_CODE_MACRO_DEVICE(0x100)
#define BAM_DBG_CODE_PATH_D_MAPPER        BAM_DBG_CODE_MACRO_DEVICE(0x200) 
#define BAM_DBG_CODE_PATH_D_LATENCY       BAM_DBG_CODE_MACRO_DEVICE(0x400) 
#define BAM_DBG_CODE_PATH_D_CULL          BAM_DBG_CODE_MACRO_DEVICE(0x800) 
#define BAM_DBG_CODE_PATH_D_COMP          BAM_DBG_CODE_MACRO_DEVICE(0x1000) 
#define BAM_DBG_CODE_PATH_D_LAT_RECURSE   BAM_DBG_CODE_MACRO_DEVICE(0x2000)
#define BAM_DBG_CODE_PATH_D_AGG           BAM_DBG_CODE_MACRO_DEVICE(0x4000)
#define BAM_DBG_CODE_PATH_D_VENDOR        BAM_DBG_CODE_MACRO_DEVICE(0x8000)






#define BAM_EMU_DEFAULT_CODE_PATH_VERBOSITY 0//BAM_DBG_CODE_PATH_ALL
static uint64_t gCodePathVerbosity = BAM_EMU_DEFAULT_CODE_PATH_VERBOSITY;
__device__ static uint64_t gDeviceCodePathVerbosity = BAM_EMU_DEFAULT_CODE_PATH_VERBOSITY;


__host__  static inline void bam_emu_dbg_printf(int verbose, const char *fmt, ...)
{
    va_list args;
   
    if(verbose >=  BAM_EMU_DBGLVL_COMPILE )
    {  

    	printf("BAM_HOST_EMU:") ;       
        va_start(args, fmt);
        vprintf(fmt, args);
        va_end(args);

    }
}

__host__ __device__ static inline int bam_get_verbosity(int local, uint64_t code_path)
{
#ifdef __CUDA_ARCH__
	if(code_path & gDeviceCodePathVerbosity)
#else
	if(code_path & gCodePathVerbosity)
#endif

	{
		return BAM_EMU_DBGLVL_ERROR;
	}
	else
	{
		//TO TURN OFF ALL debug print return 0;
    	return local;
//		return BAM_EMU_DBGLVL_ERROR;

	}

}

#define BAM_EMU_HOST_DBG_PRINT(__verbose, __format, ...) do { if( __verbose >= BAM_EMU_DBGLVL_COMPILE) bam_emu_dbg_printf(__verbose, __format, __VA_ARGS__); } while (0)
//Kludge because of vprintf in device code
//Not big deal these should only be used for debug anyway
#define BAM_EMU_DEV_DBG_PRINT1(__verbose, __format, _v1) do                { if( __verbose >= BAM_EMU_DBGLVL_COMPILE) printf(__format, _v1); } while (0)
#define BAM_EMU_DEV_DBG_PRINT2(__verbose, __format, _v1, _v2) do           { if( __verbose >= BAM_EMU_DBGLVL_COMPILE) printf(__format, _v1, _v2); } while (0)
#define BAM_EMU_DEV_DBG_PRINT3(__verbose, __format, _v1, _v2, _v3) do      { if( __verbose >= BAM_EMU_DBGLVL_COMPILE) printf(__format, _v1, _v2, _v3); } while (0)
#define BAM_EMU_DEV_DBG_PRINT4(__verbose, __format, _v1, _v2, _v3, _v4) do { if( __verbose >= BAM_EMU_DBGLVL_COMPILE) printf(__format, _v1, _v2, _v3, _v4); } while (0)


//**********************************************************************************************************
//*** Emulator Control ***
//**********************************************************************************************************

//controls emulation compile and benchmark enablement
#define BAM_EMU_COMPILE 

//runs emalator from kernel threads launched by applicaiton (block benchmark only for now)
#define BAM_RUN_EMU_IN_BAM_KERNEL


//Runs the kernel from host thread
#define BAM_EMU_TARGET_HOST_THREAD



//TODO: Async Memcopy when Queues are configured seemed to either stall the EMU threads or never complete
//Want to eventually have the emulator running so that the ADMIN queues can be created by the emulator and 
//don't need to be faked out in libnvm.  For now, to move forward, bring up the emulator after the queues are configured
#define BAM_EMU_START_EMU_POST_Q_CONFIG
#define BAM_EMU_START_EMU_IN_APP_LAYER



/* Early Simple Loopback w/o simulated latency or transfer */
//#define BAM_EMU_TGT_SIMPLE_MODE_NVME_LOOPBACK

#define BAM_EMU_USE_DEVICE_CONSTANTS

//**********************************************************************************************************
//*** Doorbells ***
//**********************************************************************************************************


#define EMU_DB_MEM_MAPPED_FILE        1  
#define EMU_DB_MEM_ATOMIC_MANAGED     2  
#define EMU_DB_MEM_ATOMIC_DEVICE      3

#define BAM_EMU_DOORBELL_TYPE         EMU_DB_MEM_MAPPED_FILE
//#define BAM_EMU_DOORBELL_TYPE         EMU_DB_MEM_ATOMIC_MANAGED
//#define BAM_EMU_DOORBELL_TYPE         EMU_DB_MEM_ATOMIC_DEVICE


//**********************************************************************************************************
//*** Kernel control memory locations  ***
//**********************************************************************************************************

//Use stack structure (context/register) for QControl
//The value represents the max_queues_per_thread, as this uses static (register) structure in kernel

//TODO: Doesn't make much difference, needs review.
//#define BAM_EMU_USE_KCONTEXT_Q_CTRL  1






//**********************************************************************************************************
//*** Emulator Common Structures  ***
//**********************************************************************************************************

typedef struct
{
	uint32_t clock_rate_khz;
	
} bam_emu_constants;


#if 0//def  BAM_EMU_USE_DEVICE_CONSTANTS
extern __constant__ bam_emu_constants g_device_EmuConstants;


__device__	  inline  int64_t  NS_Clock2() 
{
	  auto							TimeSinceEpoch_ns  =  cuda::std::chrono::duration_cast<cuda::std::chrono::nanoseconds>( cuda::std::chrono::system_clock::now().time_since_epoch() );
	  return  static_cast<int64_t>( TimeSinceEpoch_ns.count() );
}
__device__	  inline  int64_t  NS_Clock() 
{
	uint64_t cycles = clock64();
	uint64_t ns = NS_Clock2();
	
	BAM_EMU_DEV_DBG_PRINT3(BAM_EMU_DBGLVL_ERROR, "NS_CLOCK cycles = %ld clock_khz = %d clock2 = %ld\n", cycles, g_device_EmuConstants.clock_rate_khz,NS_Clock2());

	return ns;
}


#else
__device__	  inline  int64_t  NS_Clock() 
{
	  auto							TimeSinceEpoch_ns  =  cuda::std::chrono::duration_cast<cuda::std::chrono::nanoseconds>( cuda::std::chrono::system_clock::now().time_since_epoch() );
	  return  static_cast<int64_t>( TimeSinceEpoch_ns.count() );
}
#endif
__device__ __host__ inline float get_GBs_per_sec(uint64_t elap_ns, int bytes)
{

	float gbs = (float)bytes / (float)elap_ns;

	return gbs;
}





typedef union
{
	nvm_cmd_t    nvme_cmd;
	//reserve for new command formats for storage next
	//assume that they will be smaller or equal to 64-byte NVMe (which is overkill)

} storage_next_command;

#define STORAGE_NEXT_CONTEXT_LEVELS 7 
typedef struct 
{
	storage_next_command *pCmd;
	//Used by each level or mapped as an implementation structure
	//or a pointer holder if mapped to device/emulator managed memory
	uint64_t    storage_implementation_context[STORAGE_NEXT_CONTEXT_LEVELS];  
} storage_next_emuluator_context;

//Macros, incase we want to change the API later


#define SN_CONTEXT_TAG(_context) (_context->pCmd->nvme_cmd.dword[0] >> 16)
#define SN_CONTEXT_OP(_context) (_context->pCmd->nvme_cmd.dword[0] & 0x7F)
#define SN_CONTEXT_LBA(_context) ((uint64_t)(((uint64_t)_context->pCmd->nvme_cmd.dword[11] << 32) | _context->pCmd->nvme_cmd.dword[10]))
#define SN_CONTEXT_NUM_BLOCKS(_context) ((_context->pCmd->nvme_cmd.dword[12] & 0xFFFF) + 1)
#define SN_OP_READ NVM_IO_READ


#define EMU_CONTEXT storage_next_emuluator_context
#define EMU_COMPONENT_NAME_LEN 64



#define EMU_MODEL_TYPE_INVALID          0
#define EMU_MODEL_TYPE_LATENCY          1
#define EMU_MODEL_TYPE_AGGREGATION      2
#define EMU_MODEL_TYPE_VENDOR           3



typedef struct 
{
	uint32_t uModelType;
	uint32_t block_size;	
	char szModelName[EMU_COMPONENT_NAME_LEN];

	BufferPtr d_model_private;

	void *pvDevPrivate;
	void *pvHostPrivate;

	
} bam_emu_target_model;



#define EMU_MAP_TYPE_INVALID 0
#define EMU_MAP_TYPE_DIRECT  1 


typedef struct 
{
	char szMapName[EMU_COMPONENT_NAME_LEN];

	uint32_t     uMapType;

	bam_emu_target_model model;
		
	
} bam_emu_mapper;




typedef struct
{	

#if(BAM_EMU_DOORBELL_TYPE == EMU_DB_MEM_ATOMIC_DEVICE) 
	simt::atomic<uint32_t, simt::thread_scope_device> atomic_db;
	uint8_t pad0[28];
#endif

	uint16_t 			q_number;  
	uint16_t 			q_size;
	uint16_t            q_size_minus_1;
	uint8_t 			cq;
	uint8_t 			enabled;

	uint32_t            rollover;

	
	uint64_t 			ioaddr;
	volatile uint32_t            *db;

	uint32_t            head;
	uint32_t            tail;

	void *              pEmuQ;
	

} bam_emulated_queue;

typedef struct
{
	bam_emulated_queue    sQ;
	bam_emulated_queue    cQ;

	uint16_t                   qp_enabled;
	uint16_t                   q_number;

	void                  *pvThreadContext;

	storage_next_emuluator_context *pContext;
	
} bam_emulated_queue_pair;


__device__ inline bam_emulated_queue_pair* get_bam_emulated_queue_pair_from_pvThreadContext(void **ppvThreadContext)
{
    return (bam_emulated_queue_pair *)((char *)ppvThreadContext - offsetof(bam_emulated_queue_pair, pvThreadContext));
}


typedef struct 
{

		uint64_t                numEmuThreads;
		uint32_t 				numQueues;
		volatile int            bRun;
		int                     bDone;
		int                     nOneShot;
		int                     rsvd;
		char                    szName[32];
		uint64_t                thread_count;

		uint32_t                debugA[32];

#if(BAM_EMU_DOORBELL_TYPE == EMU_DB_MEM_ATOMIC_MANAGED) 
		struct
		{
			simt::atomic<uint32_t, simt::thread_scope_device> cq_db;
    		uint8_t pad0[28];
    		simt::atomic<uint32_t, simt::thread_scope_device> sq_db;
    		uint8_t pad1[28];
   

		} atomic_doorbells[BAM_EMU_MAX_QUEUES];
#endif


	BufferPtr                       d_mapper;

	bam_emu_mapper                 *pDevMapper;  //device copy

	emu_stats                      *pStats;
		
} bam_emulated_target_control;
	
typedef struct
{
	DmaPtr              target_q_mem[2];
} bam_emu_qmem;



typedef struct
{
	cudaStream_t            tgtStream;
	cudaStream_t            queueStream;
	cudaStream_t            bamStream;

	bam_emulated_queue_pair        *queuePairs;

	DmaPtr d_queue_q_mem;
	DmaPtr d_target_control_mem;

	bam_emulated_target_control    *pTgt_control; //managed, shared with device
	bam_emulated_queue_pair        *pDevQPairs;
	bam_emu_qmem                    devQMem[BAM_EMU_MAX_QUEUES];
	bam_emu_mapper                 mapper;
	
} bam_target_emulator;


typedef struct 
{
	char name[64];
	
	uint64_t emulationTargetFlags;
	int id;
	int g_size;
	int b_size;
	int shared_size;
	
	bam_target_emulator   tgt;
	nvm_ctrl_t            *pCtrl;
	uint32_t 	          cudaDevice;
	pthread_t          	  emu_threads[3];
	int                   bRun;
	uint32_t              sectorSize;
	uint32_t              loopbackMask;
	
	
} bam_host_emulator;



#define EMU_SUBMIT_GOOD  0
#define EMU_SUBMIT_QFULL 1 
#define EMU_WAIT_CQHEAD  2
#define EMU_CQ_EMPTY     3 


//#define BAM_EMU_QTHREAD_ONE_SHOT

//using SCSI terminology, data-in from Host -> target
#define BAM_EMU_DATA_IN  0
#define BAM_EMU_DATA_OUT 1
typedef ulonglong4 emu_copy_type;

typedef uint32_t (*fnModelPrivateInit)(bam_host_emulator *pEmu, bam_emu_target_model *pModel);

#endif



