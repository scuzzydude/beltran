
#include <cuda.h>
#include <nvm_ctrl.h>
#include <nvm_types.h>
#include <nvm_queue.h>
#include <nvm_util.h>
#include <nvm_admin.h>
#include <nvm_error.h>
#include <nvm_cmd.h>
#include <string>
#include <stdexcept>
#include <vector>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <ctrl.h>
#include <buffer.h>
#include <event.h>
#include <queue.h>
#include <nvm_parallel_queue.h>
#include <nvm_io.h>
#include <page_cache.h>
#include <util.h>
#include <iostream>
#include <fstream>
#include <cstdarg>
#include <iostream>

#include "astros_bam.h"

#define BAM_CODE_PATH_INIT     0x00000001
#define BAM_CODE_PATH_HOST_IO  0x00000002

uint64_t gAstrosCodePathVerbosity = BAM_CODE_PATH_INIT;

#define BAM_ASTROS_HOST_ASSERT(__assert) if(!(__assert))do { \
printf("BAM_ASTROS_HOST_ASSERT @LINE=%d in %s\n", __LINE__, __FILE__); 	\
usleep(1000); \
printf("\n\n***** EXIT(0) ***"); \
exit(0); } while(0)


static inline void bam_astros_dbg_printf(int verbose, const char *fmt, ...)
{

    va_list args;
   
    if(verbose >=  BAM_ASTROS_DBGLVL_COMPILE )
    {   
    	printf("ASTROS_BAM:") ;       
        va_start(args, fmt);
        vprintf(fmt, args);
        va_end(args);
    }

}


//#define BAM_ASTROS_DBG_PRINT(__verbose, __format, ...) do { if( __verbose >= BAM_ASTROS_DBGLVL_COMPILE) bam_astros_dbg_printf(__verbose, __format, __VA_ARGS__); } while (0)\

#define BAM_ASTROS_DBG_PRINT(__verbose, __format, ...) 

#define BAM_ASTROS_DBG_PRINT_HEX(__verbose, __str, __hex) if( __verbose >= BAM_ASTROS_DBGLVL_COMPILE) \
	 std::cout << "ASTROS_BAM:" << __str << std::hex << __hex << std::endl

#define BAM_ASTROS_DBG_PRINT_DEC(__verbose, __str, __hex) if( __verbose >= BAM_ASTROS_DBGLVL_COMPILE) \
	 std::cout << "ASTROS_BAM:" << __str << std::dec << __hex << std::endl




static inline int bam_astros_get_verbosity(int code_path, int local)
{

	if(code_path & gAstrosCodePathVerbosity)
	{
		return BAM_ASTROS_DBGLVL_ERROR;
	}
	else
	{
    //todo - have a global variable for setting that can override all local;
    return local;
    //return ASTROS_DBGLVL_ERROR;
	}
}


const char* const libnvm_device_names[] = {"/dev/libnvm0"};







typedef struct
{
	Controller*  pBamCtrl;
	page_cache_t *pHostPageCache;
	int           cuda_device;
	int           page_size;
	uint64_t      num_pages;
	char          szDevicePath[BAM_MAX_DEVICE_PATH];
} bam_target;




typedef struct
{
	std::vector<Controller*> ctrls;
	int         controller_count;

} bam_config;

bam_config gBamConfig;

extern "C" int  astros_bam_init(int controller_count, void **ppBamTargets);
int astros_bam_init(int controller_count, void **ppBamTargets)

{
	int verbose = bam_astros_get_verbosity(BAM_CODE_PATH_INIT, BAM_ASTROS_DBGLVL_INFO);
	int             tgt_count = 0;
	int 			nvmNamespace = 1;
	int 			nDevices = 0;
	int 			nvmeQueueDepth = 1024;
	int 			nvmeNumQueues = 1;
	int             cuda_device = 0;
	bam_target       *pBamTarget;
	Controller       *pCtrl;
	cudaDeviceProp	prop;
	uint64_t page_size = 512;
	uint64_t n_pages = (10L  * (uint64_t)(1024 * 1024 * 1024)) / page_size;

	BAM_ASTROS_DBG_PRINT(verbose, "astros_bam_init(%d, %p)\n", controller_count, ppBamTargets);


	gBamConfig.controller_count = 0;	
	

	if(cudaSuccess == cudaGetDeviceCount(&nDevices))
	{

		for (int i = 0; i < nDevices; i++)
		{
			cudaGetDeviceProperties(&prop, i);
			printf("Device Number: %d\n", i);
			printf("  Device name: %s\n", prop.name);
			printf("  Memory Clock Rate (KHz): %d\n", 
				prop.memoryClockRate);
			printf("  Memory Bus Width (bits): %d\n", 
				prop.memoryBusWidth);
			printf("  Peak Memory Bandwidth (GB/s): %f\n\n", 
				2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
		}
		
		try 
		{

			for (size_t i = 0; i < controller_count; i++)
			{
				pBamTarget = (bam_target *)aligned_alloc(64, sizeof(bam_target));

				BAM_ASTROS_HOST_ASSERT(pBamTarget);
			
				strcpy(pBamTarget->szDevicePath, libnvm_device_names[i]);

				pCtrl = new Controller(libnvm_device_names[i], nvmNamespace, 0, nvmeQueueDepth, nvmeNumQueues);

				gBamConfig.ctrls.push_back(pCtrl);
				gBamConfig.controller_count++;
				
				pBamTarget->page_size = page_size;
				pBamTarget->cuda_device = cuda_device;
				pBamTarget->pHostPageCache = new page_cache_t(page_size, n_pages, cuda_device, gBamConfig.ctrls[0][0], (uint64_t) 64, gBamConfig.ctrls);
				pBamTarget->num_pages = n_pages;

				ppBamTargets[i] = pBamTarget;

				tgt_count++;
			}

		}

		catch (const error & e)
		{
			BAM_ASTROS_DBG_PRINT(BAM_ASTROS_DBGLVL_ERROR, "astros_bam_init() Unexpected error: %s\n", e.what());
			exit(0);
		}

	}
	else
	{
		BAM_ASTROS_DBG_PRINT(BAM_ASTROS_DBGLVL_ERROR,"ERROR : astros_bam_init() cudaGetDeviceCOunt(%d)\n", 0);
	}

	return tgt_count;	
}



#if 1

extern "C" int astros_bam_start_io_kernel(void * pvBamTarget, int io_limit, int op, int access);





int astros_bam_start_io_kernel(void * pvBamTarget, int io_limit, int op, int access)
{
	int verbose = bam_astros_get_verbosity(BAM_CODE_PATH_HOST_IO, BAM_ASTROS_DBGLVL_NONE);
	bam_target *	pBamTarget;
	int 			page_size;
	uint64_t		n_elems;

		std::cout << "astros_bam_start_io_kernel" << std::endl;

	BAM_ASTROS_DBG_PRINT_HEX(verbose,"bam_start_io_kernel(%p)\n", pvBamTarget);
	

	
	range_t<uint64_t> *pRange;

	pBamTarget			= (bam_target *)pvBamTarget;

	BAM_ASTROS_HOST_ASSERT(pBamTarget);
	BAM_ASTROS_HOST_ASSERT(pBamTarget->pHostPageCache);

	page_size			= pBamTarget->page_size;

	page_cache_t *	d_pc = (page_cache_t *) pBamTarget->pHostPageCache->d_pc_ptr;



	n_elems = pBamTarget->num_pages;

#if 1
//	BAM_ASTROS_DBG_PRINT(verbose,"bam_start_io_kernel(%ld) n_elems\n", n_elems);

	uint64_t		t_size = n_elems * sizeof(uint64_t);

#if 1
	try 
	{

//		BAM_ASTROS_DBG_PRINT(verbose,"bam_start_io_kernel(%p) TRY\n", pvBamTarget);

		std::cout << "h_range" << std::endl;

		
	//	range_t <uint64_t> h_range((uint64_t) 0, (uint64_t) n_elems, (uint64_t) 0, (uint64_t) (t_size / page_size),
	//		 (uint64_t) 0, (uint64_t) page_size, pBamTarget->pHostPageCache, pBamTarget->cuda_device);

	pRange = new range_t<uint64_t>((uint64_t) 0, (uint64_t) n_elems, (uint64_t) 0, (uint64_t) (t_size / page_size),
			 (uint64_t) 0, (uint64_t) page_size, pBamTarget->pHostPageCache, pBamTarget->cuda_device);
			 
#if 0	

		range_t <uint64_t> *d_range = (range_t <uint64_t> *) h_range.d_range_ptr;

		std::vector <range_t<uint64_t>*> vr(1);
		vr[0]				= &h_range;

		//(const uint64_t num_elems, const uint64_t disk_start_offset, const std::vector<range_t<T>*>& ranges, Settings& settings)
		array_t <uint64_t> a(n_elems, 0, vr, pBamTarget->cuda_device);
#endif


	}
	catch (const error & e)
	{
		//BAM_ASTROS_DBG_PRINT(BAM_ASTROS_DBGLVL_ERROR, "astros_bam_start_io_kernel() Unexpected error: %s\n", e.what());
		return 1;
	}
#endif
	BAM_ASTROS_HOST_ASSERT(0);
#endif
	BAM_ASTROS_HOST_ASSERT(0);

	return 0;
}
#endif





