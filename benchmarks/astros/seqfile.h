#ifndef _SEQHEADER_H
#define _SEQHEADER_H


typedef struct
{
	unsigned short idx;
	unsigned short segment;
	unsigned int   byteoff;
	unsigned int   blockoff;
	unsigned int   lba;
	unsigned int   tl;
	unsigned int   marker;
	
} seqheader;

#endif

