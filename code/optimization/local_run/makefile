# the compiler to use
CXX      = g++

# compiler flags:
#  -g    adds debugging information to the executable file
#  -Wall turns on most, but not all, compiler warnings
#INC	= -I /usr/include/boost 
CXXFLAGS = -g -Wall
RM      = rm -rf
#LIBS	= -lm

default: all

all: DPS_borg

DPS_borg: main.cpp lib_moeaf_borg_mt.a
	$(CXX) $(CXXFLAGS) main.cpp lib_moeaf_borg_mt.a -o DPS_borg
	@echo "DPS_borg Build complete"

# DPS_borg_main.o: main.cpp moeaframework.h boostutil.h borg.h
# 	$(CXX) $(CXXFLAGS) $(INC) main.cpp -o DPS_borg_main.o

# moeaframework.o: moeaframework.c moeaframework.h
# 	$(CXX) $(CXXFLAGS) moeaframework.c -o moeaframework.o

# borg.o: borg.c borg.h 
# 	$(CXX) $(CXXFLAGS) borg.c $(LIBS) -o borg.o

# mt19937ar.o: mt19937ar.c mt19937ar.h
# 	$(CXX) mt19937ar.c $(LIBS) -o mt19937ar.o

clean:
	$(RM) *.o
	$(RM) DPS_borg
	@echo "Clean complete"
