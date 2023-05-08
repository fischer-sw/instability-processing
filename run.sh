#!/bin/bash

RES_DIR=%results_path%
RAW_DIR=%raw_data_path%

function do_job {
    cd $1
    
	FILES=$(ls $RES_DIR/$d/instabilities | wc -l)
	if [ $FILES == 0 ]; then

		N_JOB_ID=$(qstat -u $USER | grep $1 | wc -l)
		
		if [ $N_JOB_ID == 0 ]; then
			dos2unix job.sh
			JOB_ID=$(sbatch --parsable job.sh)
				NAME=$(grep "^#SBATCH -J" ./job.sh | cut -d" "  -f3)
			echo Name = ${NAME}
			echo SIM JOB_ID ${JOB_ID}
		else
			echo Job already running or in queue
		fi
		
	else
		echo "Already calculated data for case $1"
		#dos2unix post.sh
		#echo "Starting post processing"
		#sbatch post.sh
	fi
    cd -
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
	*) echo "Option $1 not recognized. -prep, -c, -a and -p are allowed." ;; # In case you typed a different option other than a,b,c
	esac
	shift
done

}

function do_prep_all {

	echo "Preparing all cases ..."
        cd ~/insta_hpc
        JOBS=$(find . -nowarn -type d -maxdepth 1 -mindepth 1)
		echo JOBS =  $JOBS
        for d in $JOBS
    	do
			dos2unix ~/insta_hpc/$d/job.sh
    	done
        echo "Preparation finished"

}


function do_clean {
	echo "Cleaning cases ..."
	cd ~/Jobs
	JOBS=$(find . -nowarn -type d -maxdepth 1 -mindepth 1)
	for d in $JOBS
    	do
		cd ~/Jobs/$d
		FILES=$(ls | wc -l)	
		if [ $FILES == 0 ]; then
			rmdir ~/Jobs/$d
		else
			DATAFILES=$(ls ./Data | wc -l)
			GZ=$(ls | grep .gz | wc -l)
			if [ $DATAFILES == 0 ] || [ $GZ == 0 ]; then
				echo Removing $d
				rm -r ~/Jobs/$d/*
				rmdir ~/Jobs/$d
				echo Finished removing $d
			fi
		fi
		
		done
	echo "Cleaning finished"
}

function do_run_all {

	JOBS=$(find . -nowarn -type d -maxdepth 1 -mindepth 1)
	for d in $JOBS
    do	
		if [ -d "$DIR" ]; then
			FILES=$(ls $RES_DIR/final_data/$d/instabilities | wc -l)
		else
			FILES=0
		fi
		# echo $d $FILES
		if [ $FILES == 0 ]; then
			do_job $d
		fi
    done
	qstat -u ${USER}
}

do_option $*
