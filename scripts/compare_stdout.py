import os, sys, re, string, operator, math, datetime, time, signal, subprocess, shutil, glob, pkgutil
import params as p
import common_functions as cf 

def compare_stdout():
	gold_class = list(0.0 for i in range(0, 10))
	out_class  = list(0.0 for i in range(0, 10))	
	if os.path.isfile(golden_stdout_fname) and os.path.isfile(stdout_fname): 
		logf = open(golden_stdout_fname, "r")
		logg = open(stdout_fname, "r")
		for line in logf:
			if "Zero" in line:		
				gold_class[0] = float(line.split(':')[0].split('%')[0].strip()) 
			if "One" in line:		
				gold_class[1] = float(line.split(':')[0].split('%')[0].strip())
			if "Two" in line:		
				gold_class[2] = float(line.split(':')[0].split('%')[0].strip())
			if "Three" in line:		
				gold_class[3] = float(line.split(':')[0].split('%')[0].strip())
			if "Four" in line:		
				gold_class[4] = float(line.split(':')[0].split('%')[0].strip())
			if "Five" in line:		
				gold_class[5] = float(line.split(':')[0].split('%')[0].strip())
			if "Six" in line:		
				gold_class[6] = float(line.split(':')[0].split('%')[0].strip())
			if "Seven" in line:		
				gold_class[7] = float(line.split(':')[0].split('%')[0].strip())
			if "Eight" in line:		
				gold_class[8] = float(line.split(':')[0].split('%')[0].strip())
			if "Nine" in line:		
				gold_class[9] = float(line.split(':')[0].split('%')[0].strip())
		logf.close()
		for line in logg:
			if "Zero" in line:		
				out_class[0] = float(line.split(':')[0].split('%')[0].strip()) 
			if "One" in line:		
				out_class[1] = float(line.split(':')[0].split('%')[0].strip())
			if "Two" in line:		
				out_class[2] = float(line.split(':')[0].split('%')[0].strip())
			if "Three" in line:		
				out_class[3] = float(line.split(':')[0].split('%')[0].strip())
			if "Four" in line:		
				out_class[4] = float(line.split(':')[0].split('%')[0].strip())
			if "Five" in line:		
				out_class[5] = float(line.split(':')[0].split('%')[0].strip())
			if "Six" in line:		
				out_class[6] = float(line.split(':')[0].split('%')[0].strip())
			if "Seven" in line:		
				out_class[7] = float(line.split(':')[0].split('%')[0].strip())
			if "Eight" in line:		
				out_class[8] = float(line.split(':')[0].split('%')[0].strip())
			if "Nine" in line:		
				out_class[9] = float(line.split(':')[0].split('%')[0].strip())
		logg.close()

	##Find the CNN results
	gold_highest = -1	
	for i in range(0, 9):            
    		if (gold_class[i] > gold_highest):    
    	        	gold_highest = gold_class[i]    
    	       		gold_highest_pos = i

	out_highest = -1	
	for i in range(0, 9):            
    	    if (out_class[i] > out_highest):    
    	       		out_highest = out_class[i]    
    	        	out_highest_pos = i
	if (gold_highest_pos != out_highest_pos):
				return 2 #CRITICAL FAULT  	
	
	for i in range(0, 9):
			if (out_class[i] != gold_class[i]):
					return 1 #differnecte percentages! NOT CRITICAL FAULT since outcome is still correct

	return 0
			


def set_env_variables(): # Set directory paths 

	p.set_paths() # update paths 
	global stdout_fname, golden_stdout_fname, stderr_fname, injection_seeds_file, new_directory, injrun_fname, stdoutdiff_fname, stderrdiff_fname , script_fname, sdc_fname, inj_run_logname, report_fname
	
	#new_directory = p.NVBITFI_HOME + "/logs/" + app + "/" + app + "-group" + igid + "-model" + bfm + "-icount" + icount
		
	new_directory = p.script_dir[app] + "/logs" 
	if not os.path.isdir(new_directory): os.system("mkdir -p " + new_directory)	
		
	stdout_fname = p.script_dir[app] + "/" + p.stdout_file
	golden_stdout_fname = p.script_dir[app] + "/" + p.golden_stdout_file	
	

	stderr_fname = p.script_dir[app] + "/" + p.stderr_file
	injection_seeds_file =p.NVBITFI_HOME  + "/pf_injector/" + p.injection_seeds
	injrun_fname =p.NVBITFI_HOME  + "/pf_injector/" + p.inj_run_log	
	stdoutdiff_fname=p.NVBITFI_HOME +"/scripts/" + p.stdout_diff_log
	stderrdiff_fname=p.NVBITFI_HOME +"/scripts/" +p.stderr_diff_log
	script_fname =p.script_dir[app]  + "/" + p.run_script_pf  
	sdc_fname =p.script_dir[app]  + "/" + p.sdc_check_pf

	inj_run_logname =p.NVBITFI_HOME +"/pf_injector/" + p.inj_run_log
	
	#report_fname = p.script_dir[app] + "/" + p.report
	report_fname = new_directory + "/" + p.report	




def main(): 

	global app	
	#app = sys.argv[1]
	for app in p.apps: 
		set_env_variables()
		comparison = compare_stdout()
		print("Result is (0 masked 1 Safe 2 Critical): %d" %comparison)

if __name__ == "__main__" :
    main()
