
#include "astros.h"



FILE * astros_signs_get_fd(int field, char *szSign, char *szMode)
{
	FILE *fd = NULL;	
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);
	char path[128];

	switch(field)
	{
		case ASTROS_FIELD_KERNEL:
			sprintf(path, "/proc/altuve/%s", szSign);
			fd = fopen(path, szMode);
			ASTROS_DBG_PRINT(verbose, "astros_signs_get_fd(%d:%s) PATH = %s MODE = %s FD = %p\n", field, szSign, path, szMode, fd);
			break;
			
		default:
			break;

	}
	


	return fd;
}

int astros_signs_write_control(astros_control *pControl, int field)
{
	FILE *fd = astros_signs_get_fd(field, "control", "wb");
	int len = sizeof(astros_control);
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_INFO);
	int bw;
	int error = 0;
	
	if(fd)
	{
		bw = fwrite(pControl, len, 1, fd);

		if(!bw)
		{
			error = 1;
		}
			
		ASTROS_DBG_PRINT(verbose, "astros_signs_write_control() LEN = %d BW = %d\n", len, bw);

		fclose(fd);

	}

	return error;

}

int astros_signs_write_lineup(astros_lineup *pLineup)
{
	int error = 0;
	FILE *fd = astros_signs_get_fd(pLineup->field, "lineup", "wb");
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_INFO);
	int len;
	int bw;
	bool bSendControl = false;
	astros_control aControl;


	
	if(fd)
	{
		//len = sizeof(astros_lineup);
		len = offsetof(astros_lineup, gametime);


		pLineup->gametime_offset = len;

		bw = fwrite(pLineup, len, 1, fd);

		if(bw)
		{
			bSendControl = true;
		}
		
		ASTROS_DBG_PRINT(verbose, "astros_signs_write_lineup() LEN = %d BW = %d\n", len, bw);

		fclose(fd);

	}

	if(bSendControl)
	{
		memset(&aControl, 0, sizeof(aControl));

		aControl.new_game = pLineup->gameid.gameid;

		astros_signs_write_control(&aControl, pLineup->field);

	}

	

	return error;

}

int astros_signs_read_scorecard(astros_lineup *pLineup, int length, astros_scorecard *pScorecard)
{
	int error = 0;

	FILE *fd = astros_signs_get_fd(pLineup->field, "scorecard", "rb");
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);
	int br;

	ASTROS_DBG_PRINT(verbose, "astros_signs_read_scorecard(%px, %d, %px, %d)\n", pLineup, length, pScorecard, fd);
	

	if(fd)
	{

		br = fread(pScorecard, length, 1, fd);

		ASTROS_DBG_PRINT(verbose, "astros_signs_read_scorecard() %d\n", br);

		fclose(fd);
	}
	else
	{
		error = 1;
	}



	return error;
	
	

}


int astros_signs_read_kstats(astros_lineup *pLineup, astros_kernel_stats *pKstats, int length)
{

	int error = 0;
	int field = ASTROS_FIELD_KERNEL;
	FILE *fd = astros_signs_get_fd(field, "kstats", "rb");
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);
	int br;

	ASTROS_DBG_PRINT(verbose, "astros_signs_read_kstats(%px, %d,  %d)\n", pLineup, length, fd);
	

	if(fd)
	{

		br = fread(pKstats, length, 1, fd);

		ASTROS_DBG_PRINT(verbose, "astros_signs_read_kstats() %d\n", br);

		fclose(fd);
	}
	else
	{
		error = 1;
	}



	return error;


}


int astros_signs_dump_cmdstats(char *dumpfn)
{
	int error = 0;
	int field = ASTROS_FIELD_KERNEL;
	FILE *fd = astros_signs_get_fd(field, "cmdstats", "rb");
    int verbose = astros_get_verbosity(ASTROS_DBGLVL_NONE);
	FILE *outfd;
	astros_single_cmdstat tStat;
	astros_cmd_stats *pStats;
	int len =  sizeof(astros_cmd_stats);
	int i;
	int br;
	int j, k = 0;
	int cpucnt = 0;
	
	ASTROS_UNUSED(br);
	

	ASTROS_DBG_PRINT(verbose, "astros_signs_dump_cmdstats(%d, %s)\n",  fd, dumpfn);

	outfd = fopen(dumpfn, "w");

	memset(&tStat,0, sizeof(astros_single_cmdstat));

	pStats = ASTROS_ALLOC(64, len);
	

	if(fd && outfd && pStats)
	{

		ASTROS_DBG_PRINT(verbose, "astros_signs_dump_cmdstats(%d, %s) FILES GOOD\n",  fd, dumpfn);

/*

#define ASTROS_MAX_CMDSTATS 128
		
		typedef struct
		{
			unsigned int cmd_len[4];
			unsigned int block_pow_2[2][14];
			unsigned int running_nonpow2_block_cnt[2];
			unsigned int rnning_nonpow2_cmds[2];
			unsigned int nonrwcmd;
			unsigned int total_blocks[2];
			unsigned int totalrw[2];
			
		
		} astros_single_cmdstat;

*/
		br = fread(pStats, len, 1, fd);


	//	fprintf("CPU,cmd_len[0],cmd_len[1],cmd_len[1],cmd_len[2],")

	//	cmd_len[0],cmd_len[1],cmd_len[1],cmd_len[2]

			fprintf(outfd, "0,1 INDEX is 0 = WRITE, 1 = READ\n");

			fprintf(outfd, "CPU,");
		
			for(j = 0; j < 4; j++)
			{
		
				fprintf(outfd, "cmd_len[%d],", j);
			}
		
			for(k = 0; k < 2; k++)
			{
		
				for(j = 0; j < 14; j++)
				{
					fprintf(outfd,"block_pow_2[%d][%d],", k, j);
				}
		
			}
		
			for(k = 0; k < 2; k++)
			{
				fprintf(outfd, "running_nonpow2_block_cnt[%d],", k);
			}
		
			for(k = 0; k < 2; k++)
			{
				fprintf(outfd, "rnning_nonpow2_cmds[%d],", k);
			}
		
			for(k = 0; k < 2; k++)
			{
		
				fprintf(outfd, "non_pow2_avg[%d],", k);
			}
		
			fprintf(outfd, "non_rw_cmds,");

			for(k = 0; k < 2; k++)
			{
				fprintf(outfd, "total_blocks[%d],", k);
			}
		
			for(k = 0; k < 2; k++)
			{
				fprintf(outfd, "total_rw[%d],", k);
			}
		
			for(k = 0; k < 2; k++)
			{
				fprintf(outfd, "average_rw_block[%d],", k);
			}

			for(k = 0; k < 256; k++)
			{
				fprintf(outfd, "op[%02x],", k);		
			}
			fprintf(outfd, "\n");
	


		for(i = 0; i < ASTROS_MAX_CMDSTATS; i++)
		{

			if(pStats->stats[i].totalrw[0] || pStats->stats[i].totalrw[1])
			{
				cpucnt++;

				ASTROS_DBG_PRINT(verbose, "astros_signs_dump_cmdstats(%d, %s) CPU has valid = %d\n",  fd, dumpfn, i);

				fprintf(outfd, "%d,", i);

				for(j = 0; j < 4; j++)
				{

					fprintf(outfd, "%d,", pStats->stats[i].cmd_len[j]);
					tStat.cmd_len[j] += pStats->stats[i].cmd_len[j];							
				}

				for(k = 0; k < 2; k++)
				{

					for(j = 0; j < 14; j++)
					{
						fprintf(outfd,"%d,", pStats->stats[i].block_pow_2[k][j]);
						tStat.block_pow_2[k][j] += pStats->stats[i].block_pow_2[k][j];
					}


				}

				for(k = 0; k < 2; k++)
				{
					fprintf(outfd, "%d,", pStats->stats[i].running_nonpow2_block_cnt[k]);
					tStat.running_nonpow2_block_cnt[k] += pStats->stats[i].running_nonpow2_block_cnt[k];
				}

				for(k = 0; k < 2; k++)
				{
					fprintf(outfd, "%d,", pStats->stats[i].rnning_nonpow2_cmds[k]);
					tStat.rnning_nonpow2_cmds[k] += pStats->stats[i].rnning_nonpow2_cmds[k];
				}

				for(k = 0; k < 2; k++)
				{
					float favg = 0.0;

					if(pStats->stats[i].rnning_nonpow2_cmds[k])
					{
						favg = (float)pStats->stats[i].running_nonpow2_block_cnt[k] / (float)pStats->stats[i].rnning_nonpow2_cmds[k];
					}

					fprintf(outfd, "%f,", favg);
				}


				fprintf(outfd, "%d,",  pStats->stats[i].nonrwcmd);
				tStat.nonrwcmd += pStats->stats[i].nonrwcmd;

				for(k = 0; k < 2; k++)
				{
					fprintf(outfd, "%d,", pStats->stats[i].total_blocks[k]);
					tStat.total_blocks[k] += pStats->stats[i].total_blocks[k];
				}

				for(k = 0; k < 2; k++)
				{
					fprintf(outfd, "%d,", pStats->stats[i].totalrw[k]);
					tStat.totalrw[k] += pStats->stats[i].totalrw[k];
				}

				for(k = 0; k < 2; k++)
				{

					float favg = 0.0;

					if(pStats->stats[i].totalrw[k])
					{
						favg = (float)pStats->stats[i].total_blocks[k] / (float)pStats->stats[i].totalrw[k];

					}
				
					fprintf(outfd, "%f,", favg);
				}

				for(k = 0; k < 256; k++)
				{
					fprintf(outfd, "%d,", pStats->stats[i].ops[k]);		
					tStat.ops[k] += pStats->stats[i].ops[k];
				}
	
				

				fprintf(outfd, "\n");
	



			}
				
				
					
				
			}


			if(1)
			{

				ASTROS_DBG_PRINT(verbose, "astros_signs_dump_cmdstats(%d, %s) CPU has valid = %d\n",  fd, dumpfn, i);

				fprintf(outfd, "TOTAL(%d),", cpucnt);

				for(j = 0; j < 4; j++)
				{

					fprintf(outfd, "%d,", tStat.cmd_len[j]);
				}

				for(k = 0; k < 2; k++)
				{

					for(j = 0; j < 14; j++)
					{
						fprintf(outfd,"%d,", tStat.block_pow_2[k][j]);
					}


				}

				for(k = 0; k < 2; k++)
				{
					fprintf(outfd, "%d,", tStat.running_nonpow2_block_cnt[k]);
				}

				for(k = 0; k < 2; k++)
				{
					fprintf(outfd, "%d,", tStat.rnning_nonpow2_cmds[k]);
				}

				for(k = 0; k < 2; k++)
				{
					float favg = 0.0;

					if(tStat.rnning_nonpow2_cmds[k])
					{
						favg = (float)tStat.running_nonpow2_block_cnt[k] / tStat.rnning_nonpow2_cmds[k];
					}

					fprintf(outfd, "%f,", favg);
				}


				fprintf(outfd, "%d,",  tStat.nonrwcmd);


				for(k = 0; k < 2; k++)
				{
					fprintf(outfd, "%d,", tStat.total_blocks[k]);
				}

				for(k = 0; k < 2; k++)
				{
					fprintf(outfd, "%d,", tStat.totalrw[k]);
				}

				for(k = 0; k < 2; k++)
				{

					float favg = 0.0;

					if(tStat.totalrw[k])
					{
						favg = (float)tStat.total_blocks[k] / (float)tStat.totalrw[k];

					}
				
					fprintf(outfd, "%f,", favg);
				}

				for(k = 0; k < 256; k++)
				{
					fprintf(outfd, "%d,", tStat.ops[k] );		
				}
				

				fprintf(outfd, "\n");
			}




		fclose(outfd);
		fclose(fd);



	}
	else
	{
		error = 1;
	}

	
	return error;

}






