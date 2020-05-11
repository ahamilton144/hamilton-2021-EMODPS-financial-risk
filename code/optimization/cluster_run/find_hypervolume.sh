#/bin/bash
filr=$1
filw=$2

java -cp MOEAFramework-2.12-Demo.jar HypervolumeEval ${filr} >> ${filw}
