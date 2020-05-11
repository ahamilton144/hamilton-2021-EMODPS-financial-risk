# Makefile for DPS portfolio optimization problem
CXX = mpiCC
CXXFLAGS = -c -Wall
INC = -I /opt/ohpc/pub/libs/gnu8/openmpi3/boost/1.70.0/include
LIBS = -lm
UNAME_S = $(shell uname -s)

ifneq (, $(findstring SunOS, $(UNAME_S)))
    LIBS += -lnsl -lsocket -lresolv
endif

all: PortDPS_borgms PortDPS_retest

PortDPS_borgms: portDPS_borgms_main.o moeaframework.o borgms.o mt19937ar.o
	$(CXX) portDPS_borgms_main.o moeaframework.o borgms.o mt19937ar.o -o PortDPS_borgms

PortDPS_retest: portDPS_retest_main.o moeaframework.o borgms.o mt19937ar.o
	$(CXX) portDPS_retest_main.o moeaframework.o borgms.o mt19937ar.o -o PortDPS_retest

portDPS_borgms_main.o: main.cpp ./../borg/moeaframework.h ./../boostutil.h ./../borg/borgms.h
	$(CXX) $(CXXFLAGS) $(INC) main.cpp -o portDPS_borgms_main.o

portDPS_retest_main.o: main.cpp ./../borg/moeaframework.h ./../boostutil.h ./../borg/borgms.h
	$(CXX) $(CXXFLAGS) $(INC) main_retest.cpp -o portDPS_retest_main.o

moeaframework.o: ./../borg/moeaframework.c ./../borg/moeaframework.h
	$(CXX) $(CXXFLAGS) ./../borg/moeaframework.c

borgms.o: ./../borg/borgms.c ./../borg/borgms.h ./../borg/borg.h
	$(CXX) $(CXXFLAGS) ./../borg/borgms.c $(LIBS)

mt19937ar.o: ./../borg/mt19937ar.c ./../borg/mt19937ar.h
	$(CXX) $(CXXFLAGS) ./../borg/mt19937ar.c $(LIBS)

clean:
	rm -rf *.o
	rm PortDPS_borgms
	rm PortDPS_retest
