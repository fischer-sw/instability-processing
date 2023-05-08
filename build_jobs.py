import logging
import shutil
import sys
import os
import re
import json
from methods import get_config, get_all_cases

# setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(message)s")


def build_job(config, case):
    """
    Function creates files and directories for the slurm job to run 
    """

    logging.info(f"Preparing hpc_job for case {case}")
    # adapt config to run on hpc
    config["cases"] = [cas]
    config["images"] = []
    config["debug"] = False
    config["new_files"] = False
    config["save_intermediate"] = True

    config["results_path"] = "/".join(["","bigdata", "FWDT", "DFischer"]+ config["results_path"].split("\\")[6:])
    config["data_path"] = "/".join(["","bigdata", "FWDT", "DFischer"]+ config["data_path"].split("\\")[6:])
    config["raw_data_path"] = "/".join(["","bigdata", "FWDT", "DFischer"]+ config["raw_data_path"].split("\\")[6:])
    config["hpc_job_conf"]["job_name"] = cas

    # create job dir of not already there
    job_base_path = os.path.join(config["hpc_job_dir"])
    job_dir_path = os.path.join(job_base_path, case)
    if os.path.exists(job_dir_path):
        logging.info(f"Job dir for case {case} already exsist")
    else:
        logging.info(f"Creating directory for case {case}")
        os.makedirs(job_dir_path)

    # writing config file to job directory
    cfg_path = os.path.join(job_dir_path, "config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)  

    # copy methods.py and requirements.txt to job dir
    shutil.copy(os.path.join(sys.path[0], "methods.py"), job_dir_path)
    shutil.copy(os.path.join(sys.path[0], "requirements.txt"), job_dir_path)

    # read job template
    template_path = os.path.join(sys.path[0], "job_template.sh")
    with open(template_path) as f:
        template = f.readlines()

    # build job.sh from template
    tmp_template = []
    for line in template:
        m = re.findall(r'\%(.*?)\%', line)
        if m != []:
            for ele in m:
            
                if ele in config["hpc_job_conf"]:
                    line = line.replace(f"%{ele}%", str(config["hpc_job_conf"][ele]))
                    logging.debug(f"line = {line}")
                else:
                    logging.error(f"Element {ele} not defined in cases.json.")
                    exit()
        
        tmp_template.append(line)
    
    # write job.sh to target location
    template_target = os.path.join(job_dir_path, "job.sh")
    with open(template_target, "w") as f:
        f.writelines(tmp_template)

    logging.info(f"Finished job preparation for case {case}")

if __name__ == "__main__":
    config = get_config()
    for cas in config["cases"]:
        build_job(config, cas)


    res_path = "/".join(["","bigdata", "FWDT", "DFischer"]+ config["results_path"].split("\\")[6:])
    dat_path = "/".join(["","bigdata", "FWDT", "DFischer"]+ config["data_path"].split("\\")[6:])
    raw_path = "/".join(["","bigdata", "FWDT", "DFischer"]+ config["raw_data_path"].split("\\")[6:])
    

    run_path = os.path.join(sys.path[0], "run.sh")
    job_base_path = os.path.join(config["hpc_job_dir"])

    with open(run_path) as f:
        run_template = f.readlines()

    # build job.sh from template
    tmp_template = []
    for line in run_template:
        m = re.findall(r'\%(.*?)\%', line)
        if m != []:
            for ele in m:
            
                if ele in config.keys():
                    line = line.replace(f"%{ele}%", str(config[ele]))
                    logging.debug(f"line = {line}")
                else:
                    logging.error(f"Element {ele} not defined in cases.json.")
                    exit()
        
        tmp_template.append(line)
    
    # write job.sh to target location
    template_target = os.path.join(job_base_path, "run.sh")
    with open(template_target, "w") as f:
        f.writelines(tmp_template)

    # copy stat.sh
    if os.path.exists(os.path.join(config["hpc_job_dir"], "stat.sh")) is False:
        shutil.copy(os.path.join(sys.path[0], "stat.sh"), job_base_path)