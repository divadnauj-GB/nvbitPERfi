import os, sys, re, string, operator, math, datetime, time, signal, subprocess, shutil, glob, pkgutil
import params as p
import common_functions as cf 
before = -1

###############################################################################
# Check for timeout and kill the job if it has passed the threshold
###############################################################################
def is_timeout(app, pr): # check if the process is active every 'factor' sec for timeout threshold 
	factor = 0.5
	retcode = None
	tt = p.TIMEOUT_THRESHOLD * p.apps[app][3] # p.apps[app][2] = expected runtime
	if tt < 10: tt = 10

	to_th = tt / factor
	while to_th > 0:
		retcode = pr.poll()
		if retcode is not None:
			break
		to_th -= 1
		time.sleep(factor)

	if to_th == 0:
		os.killpg(pr.pid, signal.SIGINT) # pr.kill()
		print ("timeout")
		return [True, pr.poll()]
	else:
		return [False, retcode]

def get_dmesg_delta(dm_before, dm_after):
	llb = dm_before.splitlines()[-1] # last lin
	pos = dm_after.find(llb)
	return str(dm_after[pos+len(llb)+1:])

def cmdline(command):
	process = subprocess.Popen(args=command, stdout=subprocess.PIPE, shell=True)
	return process.communicate()[0]

def print_heart_beat(nj):
	global before
	if before == -1:
		before = datetime.datetime.now()
	if (datetime.datetime.now()-before).seconds >= 60:
		print ("Jobs so far: %d" %nj)
		before = datetime.datetime.now()

def get_seconds(td):
	return (td.microseconds + (td.seconds + td.days * 24 * 3600) * 10**6) / float(10**6)

###############################################################################
# Classify error injection result based on stdout, stderr, application output,
# exit status, etc.
###############################################################################
def classify_injection(app, retcode, dmesg_delta):

	#stdout_str = "" 
	#if os.path.isfile(singleout_fname): 
	#	stdout_str = str(open(singleout_fname).read())

	#if p.detectors and "- 43, Ch 00000010, engmask 00000101" in dmesg_delta and "- 13, Graphics " not in dmesg_delta and "- 31, Ch 0000" not in dmesg_delta: # this is specific for error detectors 
	#	return p.DMESG_XID_43
	#DUE fault
	if retcode != 0:
		return p.NON_ZERO_EC

 	#inj_log_str = ""
	#if os.path.isfile(injrun_fname): 
	#	inj_log_str = str(open(injrun_fname, "r").read())
	#else:
	#	print "fail"
	#if "ERROR FAIL Detected Signal SIGKILL" in inj_log_str: 
	#	if p.verbose: print ("Detected SIGKILL: %s, %s, %s, %s, %s, %s" %(igid, kname, kcount, iid, opid, bid))
	#	return p.OTHERS
	#if "Error not injected" in inj_log_str or "ERROR FAIL in kernel execution; Expected reg value doesn't match;" in inj_log_str: 
	#	print (inj_log_str)
	#	if p.verbose: print ("Error Not Injected: %s, %s, %s, %s, %s, %s" %(igid, kname, kcount, iid, opid, bid))
	#	return p.OTHERS
	#if "Error: misaligned address" in stdout_str: 
	#	return p.STDOUT_ERROR_MESSAGE
	#if "Error: an illegal memory access was encountered" in stdout_str: 
	#	return p.STDOUT_ERROR_MESSAGE
	#if "Error: misaligned address" in str(open(stderr_fname).read()): # if error is found in the log standard err 
	#	return p.STDOUT_ERROR_MESSAGE
	

	#SDC fault or Masked are identified by compare_stdout() function
		
	code = compare_stdout()
	print("code %d"%code)
	if code == 1:
		return p.SAFE
	if code == 2:
		return p.CRITICAL
	else:
		return p.MASKED_OTHER

		

def receiveSignal():
	print ("segegv/n")

###############################################################################
# Run the actual injection run 
###############################################################################
def run_injections():
	[ threadID, reg, mask, SM, stuck_at] = ["", "", "", "",""]
	fName = p.script_dir[app] + "/injectionsRF" +  ".txt"
	total_jobs = 0
	start = datetime.datetime.now()  
	i = 0
	if os.path.isfile(inj_run_logname): 
		logf = open(fName, "r")
		for line in logf:		
			threadID = line.split(' ')[0].strip()
			reg = line.split(' ')[1].strip()
			mask = line.split(' ')[2].strip()
			SM = line.split(' ')[3].strip()
			stuck_at = line.split(' ')[4].strip()
			
			print("%s %s %s %s %s" %( threadID, reg, mask, SM, stuck_at))
			f = open(injection_seeds_file,"w")
			f.write("%s\n%s\n%s\n%s\n%s\n" %(threadID, reg, mask, SM, stuck_at))
			f.close()
			cwd = os.getcwd()
			dmesg_before = cmdline("dmesg")
			
			total_jobs += 1
			
			
			pr = subprocess.Popen(script_fname, shell=True, executable='/bin/bash', preexec_fn=os.setsid) # run the injection job
			[timeout_flag, retcode] = is_timeout(app, pr)
			print("retcode %d"%retcode)

			# Record kernel error messages (dmesg)
			dmesg_after = cmdline("dmesg")
			dmesg_delta = get_dmesg_delta(dmesg_before, dmesg_after)
			dmesg_delta = dmesg_delta.replace("\n", "; ").replace(":", "-")
	
			#if timeout_flag:
			#	[threadID,reg,mask,SMID,stuckat] = get_inj_info()
			#	ret_cat = p.TIMEOUT 
			[threadID,reg,mask,SMID,stuckat] = get_inj_info()
			ret_cat = list(0 for i in range(0, 100))
			
			for k in range(0,100):
				newfile = open(singleout_fname, "w+")
				newfile2= open(singlegolden_fname, "w+")
				with open(golden_stdout_fname, "r") as file1:
					for j, line1 in enumerate(file1):
						if (j >= (12*k +1) and j<= (12*k+11)):
							print(k)
	 						newfile2.write(line1)
						if (j == (12*k )) :
							name = line1
							print("name %s"%name)
				with open(stdout_fname, "r") as file0:
					for i, line0 in enumerate(file0):			
							#with open(report_fname,"a") as f:
							#	f.write(line0)
							#f.close()
						if (i >= (12*k+1) and i<= (12*k+11)):
	 						newfile.write(line0)
						if (i ==(12*k)):
							f = line0
							pos = i
				#with open(golden_stdout_fname, "r") as file1:
				#	for j, line1 in enumerate(file1):
				#		if (j == 12*k):
				#			name = line1
				#		if (line1 == f):
				#			for l, line2 in enumerate(file1):
				#				if (l >= pos and l<= (pos+11)):
				#					print(pos)
	 			#					newfile2.write(line2)

				ret_cat[k] = classify_injection(app, retcode, dmesg_delta)
				print(ret_cat[k])
				print(p.CAT_STR[int(ret_cat[k])-1])
				#sec = input('Let us wait for user input.\n')
				if ret_cat[k] ==  5:					
					threadID = line.split(' ')[0].strip()
					reg = line.split(' ')[1].strip()
					mask = line.split(' ')[2].strip()
					SMID = line.split(' ')[3].strip()
					stuckat = line.split(' ')[4].strip()
				number = name.split('/')[7].strip()
				if os.path.isfile(report_fname):
				 with open(report_fname,"a") as f:
					f.write("image %s, threadID %s, reg %s, mask %s, SMID %s,stuckat %s; Outcome %s (code %s)\n" %(number,threadID,reg,mask,SMID,stuckat,p.CAT_STR[ret_cat[k]-1], ret_cat[k]))
				else:
				 with open(report_fname,"w+") as f:
					f.write("image %s, threadID %s,  reg %s, mask %s, SMID %s,stuckat %s; Outcome %s (code %s)\n" %(number,threadID,reg,mask,SMID,stuckat,p.CAT_STR[ret_cat[k]-1], ret_cat[k]))

				print ("image threadID %s, reg %s, mask %s, SMID %s,stuckat %s; Outcome %s (code %s)\n" %(threadID,reg,mask,SMID,stuckat,p.CAT_STR[ret_cat[k]-1], ret_cat[k]))

				os.chdir(cwd) # return to the main dir
				print_heart_beat(total_jobs)
						
			sec = input('Let us wait for user input.\n')


			seconds = str(get_seconds(datetime.datetime.now() - start))
			minutes = int(((float(seconds))//60)%60)
			hours = int((float(seconds))//3600)
			print("Elapsed %s:%s" %(str(hours),str(minutes)))
			
		logf.close()
		if os.path.isfile(report_fname):
			 with open(report_fname,"a") as f:
				f.write("Simulation time: %s:%s\n" %(str(hours),str(minutes)))
	return ret_cat



def get_inj_info():
	[threadID,reg,mask,SMID,stuckat] = ["","", "", "", ""]
	if os.path.isfile(inj_run_logname): 
		logf = open(inj_run_logname, "r")
		for line in logf:
			if "thread :" in line:
					threadID = line.split(';')[1].split('thread : ')[1].strip()
			if "Register :" in line:
					reg = line.split(';')[2].split('Register : ')[1].strip()
			if "Mask :" in line:		
					mask = line.split(';')[3].split('Mask : ')[1].strip()
			if "SMID :" in line:
					SMID = line.split(';')[4].split('SMID : ')[1].strip()
			if "Stuck at :" in line:
					stuckat = line.split(';')[5].split('Stuck at : ')[1].strip()
		logf.close()
	return [threadID,reg,mask,SMID,stuckat]


########Compare the results: Golden_stdout.txt and stdout.txt
def compare_stdout():	
	gold_class = list(0.0 for i in range(0, 10))
	out_class  = list(0.0 for i in range(0, 10))	
	list_elements = ['Zero','One','Two','Three','Four','Five','Six','Seven','Eight','Nine']
	if os.path.isfile(singlegolden_fname) and os.path.isfile(singleout_fname): 
		logf = open(singlegolden_fname, "r")
		logg = open(singleout_fname, "r")
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
	gold_highest_pos = -1
	for i in range(0, 9):            
    		if (gold_class[i] > gold_highest and str(gold_class[i]) != 'nan'):    
    	        	gold_highest = gold_class[i]    
    	       		gold_highest_pos = i

	out_highest = -1	
	out_highest_pos = -1
	for i in range(0, 9):            
    	    if (out_class[i] > out_highest and str(out_class[i]) != 'nan'):    
    	       		out_highest = out_class[i]    
    	        	out_highest_pos = i
	if (gold_highest_pos != out_highest_pos):
				return 2 #CRITICAL FAULT  	
	
	for i in range(0, 9):
			if (out_class[i] != gold_class[i]):
					return 1 #differnecte percentages! NOT CRITICAL FAULT since outcome is still correct

	return 0


###############################################################################
# Set enviroment variables for run_script
###############################################################################
def set_env_variables(): # Set directory paths 

	p.set_paths() # update paths 
	global stdout_fname,golden_stdout_fname, stderr_fname, injection_seeds_file, new_directory, injrun_fname, stdoutdiff_fname, stderrdiff_fname , script_fname, sdc_fname, inj_run_logname, report_fname, singleout_fname, singlegolden_fname
	
	#new_directory = p.NVBITFI_HOME + "/logs/" + app + "/" + app + "-group" + igid + "-model" + bfm + "-icount" + icount
		
	new_directory = p.script_dir[app] + "/logs" 
	if not os.path.isdir(new_directory): os.system("mkdir -p " + new_directory)	
		
	singleout_fname = p.script_dir[app] + "/out.txt"
	singlegolden_fname = p.script_dir[app] + "/golden.txt"
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
	
	report_fname = new_directory + "/" + p.report






def main(): 
	global app	
	for app in p.apps: 
		set_env_variables()
		err_cat = run_injections()

if __name__ == "__main__" :
    main()
