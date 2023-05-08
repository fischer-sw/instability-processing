#!/bin/bash

function do_queue_info {
	#check_arg
	QUEUE=$1
	QUEUED=$(qstat -i | grep $QUEUE | wc -l)
	RUNNING=$(qstat -r | grep $QUEUE | wc -l)
	echo Info for queue: ${QUEUE} Queued: ${QUEUED} Running: ${RUNNING} 
}

function do_option {

if  [ "$#" == 0 ]; then
	echo "Options -j for user jobs info and -q for queue infos."
fi

while [ -n "$1" ]; do # while loop starts
	case "$1" in
	-j) do_jobs_info ;;
	-q) do_queue_info defq ; do_queue_info rome; do_queue_info intel; do_queue_info intel_32; do_queue_info mem768; do_queue_info milan ;;
	*) echo "Option $1 not recognized. -j and -q are allowed." ;; # In case you typed a different option other than a,b,c
	esac
	shift
done

}

function do_jobs_info {
	echo Queued jobs ...
	squeue --start -u $USER
	echo Running jobs ...
	qstat -r -u $USER
} 

do_option $*
