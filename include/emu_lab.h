#ifndef __EMU_LAB_H
#define __EMU_LAB_H

/* Helper functions for lab infrastructure, not directly tied to EMU but utilized by main */

#define BA_LAB_LOGFILENAME "bam_block.csv"

static bool ACp_file_exists (char *filename) 
{
  	struct stat   buffer;   
  	return (stat (filename, &buffer) == 0);
}


static void emu_lab_log_to_csv(char *filename, Settings *pSettings, double elapsed, double iops, double bandwidth, uint64_t ios, uint64_t total_data_bytes)
{	
	int bHeader = 1;
	FILE *fout;

	if(ACp_file_exists(filename ))
	{
		fout = fopen(filename , "a");
		bHeader = 0;
	}
	else
	{
		fout = fopen(filename , "w");
		bHeader = 1;
		
	}

	if(fout)
	{
		if(bHeader)
		{
			fprintf(fout, "USER_TAG,series,sequence,access,operation,Elaps,iops,bps,QDepth,IoSize,BAM_Requests,BAM_QDepth,BAM_Pages,BAM_Cuda_BlkSize\n");
		}

		printf("bandwidth2 %f\n", bandwidth * 1000000000.0);	
		printf("iops2 %f\n", iops);

		fprintf(fout,"%s,%d,%d,%d,%d,%f,%f,%f,%ld,%ld,%ld,%ld,%ld,%ld\n",
		pSettings->user_tag,
		pSettings->series,
		pSettings->sequence,
		(pSettings->random ? 0 : 1),
		(pSettings->accessType),
		elapsed / 1000000.0, //ticks in micro ??
		iops,
		bandwidth * 1000000000.0,
		pSettings->numThreads,
		pSettings->pageSize,
		pSettings->numReqs,
		pSettings->queueDepth,
		pSettings->numPages,
		pSettings->blkSize
		);




		fclose(fout);
	}

}


#endif
