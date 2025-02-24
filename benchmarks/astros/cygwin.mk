CC=gcc
CFLAGS=-I. -I../libaio/src -I./argtable -I./spdk/include
LIBS=-pthread 
#CFLAGS += -g -Wall -O2 -D_GNU_SOURCE -lm -L../liburing/src/ -laio
#CFLAGS += -Wall -O2 -D_GNU_SOURCE -laio -lm -L../liburing/src/ -L../libaio/src 
CFLAGS += -Wall -O2 -D_GNU_SOURCE -lm -DASTROS_CYGWIN  


#CFLAGS += -DASTROS_SPDK

astros: astros.o astros_batters.o astros_inning.o astros_ccb.o astros_lineup.o astros_scorer.o astros_signs.o astros_sync_batters.o argtable3.o astros_linux.o astros_spdk.o astros_cygwin_aio.c
	$(CC) -o astros astros.o astros_batters.o astros_inning.o astros_ccb.o astros_lineup.o astros_scorer.o astros_signs.o astros_sync_batters.o argtable3.o astros_linux.o astros_spdk.o astros_cygwin_aio.c $(CFLAGS) $(LIBS)
	

clean:
	rm -f astros *.o


     
