#!/bin/bash

RES_DIR=%results_path%
RAW_DIR=%raw_data_path%

function do_job {
    cd "$1"
	
	local TMP_PATH="$RES_DIR/finger_data/$1/ratio"
	if [ -d "$TMP_PATH" ]; then
		FILES=$(ls "$TMP_PATH" | wc -l)
	else
		FILES=0
	fi
	FILES=0
	if [ $FILES == 0 ]; then

		echo "Checking if Job $1 is already started ..."
		N_JOB_ID=$(qstat -u $USER | grep "$1" | wc -l)
		echo Found $N_JOB_ID Jobs matching "$1"
		if [[ $N_JOB_ID == 0 ]]; then
			echo "starting to run job $1 ..."
			dos2unix job.sh
			JOB_ID=$(sbatch --parsable job.sh)
			echo "Job $1 submitted"
			NAME=$(grep "^#SBATCH -J" ./job.sh | cut -d" "  -f3)
			echo Job Name = ${NAME}
			echo JOB_ID ${JOB_ID}
		else
			echo Job "$1" already running or in queue
		fi
		
	else
		echo "Already calculated data for case $TMP_DIR"
	fi
    cd ~/insta_hpc/
}

function get_job_id {
	 local name=$1
   qstat -u $USER | grep $name | cut -d" " -f1
}

function do_option {

if  [ "$#" == 0 ]; then
	echo "Options -c for running a specific case, -a for running all jobs that have not generated data so far and -p for postprocessing of a specific case and -prep to prepare all cases"
fi

while [ -n "$1" ]; do # while loop starts
	case "$1" in
	-prep) do_prep_all ;;
	-j) do_job $2 ;;
	-a) do_run_all ;;
	*) echo "Option $1 not recognized. -prep, -j, and -a are allowed." ;; # In case you typed a different option other than a,b,c
	esac
	shift
done

}

function do_prep_all {

	echo "Preparing all cases ..."
        cd ~/insta_hpc
        JOBS=$(find . -nowarn -type d -maxdepth 1 -mindepth 1)
        for d in $JOBS
    	do
			TMP_DIR=$(cut -d '/' -f2 <<< $d)
			dos2unix ~/insta_hpc/$TMP_DIR/job.sh
    	done
        echo "Preparation finished"

}

function do_run_all {

	JOBS=$(find . -nowarn -type d -maxdepth 1 -mindepth 1)
	for d in $JOBS
    do
		TMP_DIR=$(cut -d '/' -f2 <<< $d)
		TMP_PATH="$RES_DIR/final_data/$TMP_DIR/instabilities"
		if [ -d $TMP_PATH ]; then
			FILES=$(ls "$TMP_PATH" | wc -l)
		else
			FILES=0
		fi 
		echo "--------------------"
		echo "Starting job processing $d"
		echo "--------------------"

		if [ $FILES == 0 ]; then
			do_job $TMP_DIR
		else
			echo case $d already processed.
		fi
    done
	qstat -u ${USER}
}

do_option $*
