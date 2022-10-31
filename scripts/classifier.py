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

	stdout_str = "" 
	if os.path.isfile(stdout_fname): 
		stdout_str = str(open(stdout_fname).read())

	if p.detectors and "- 43, Ch 00000010, engmask 00000101" in dmesg_delta and "- 13, Graphics " not in dmesg_delta and "- 31, Ch 0000" not in dmesg_delta: # this is specific for error detectors 
		return p.DMESG_XID_43

        # in case an application exits with non-zero exit status be default, we make an exception here. 
#	if "bmatrix" in app and "Application Done" not in stdout_str: # exception for the matrixMul app
	if retcode != 0:
		return p.NON_ZERO_EC

 	inj_log_str = ""
	if os.path.isfile(injrun_fname): 
		inj_log_str = str(open(injrun_fname, "r").read())
	else:
		print "fail"
	if "ERROR FAIL Detected Signal SIGKILL" in inj_log_str: 
		if p.verbose: print ("Detected SIGKILL: %s, %s, %s, %s, %s, %s" %(igid, kname, kcount, iid, opid, bid))
		return p.OTHERS
	if "Error not injected" in inj_log_str or "ERROR FAIL in kernel execution; Expected reg value doesn't match;" in inj_log_str: 
		print (inj_log_str)
		if p.verbose: print ("Error Not Injected: %s, %s, %s, %s, %s, %s" %(igid, kname, kcount, iid, opid, bid))
		return p.OTHERS
	if "Error: misaligned address" in stdout_str: 
		return p.STDOUT_ERROR_MESSAGE
	if "Error: an illegal memory access was encountered" in stdout_str: 
		return p.STDOUT_ERROR_MESSAGE
	if "Error: misaligned address" in str(open(stderr_fname).read()): # if error is found in the log standard err 
		return p.STDOUT_ERROR_MESSAGE

	os.system(sdc_fname) # perform SDC check
	#pr = subprocess.Popen(sdc_fname, shell=True, executable='/bin/bash', preexec_fn=os.setsid);
	if  os.path.isfile(stdoutdiff_fname) and os.path.isfile(stderrdiff_fname):
		if  os.path.getsize(stdoutdiff_fname) == 0 and os.path.getsize(stderrdiff_fname) == 0: # no diff is observed
			if "ERROR FAIL in kernel execution" in inj_log_str: 
				return p.MASKED_KERNEL_ERROR # masked_kernel_error
			else:
				return p.MASKED_OTHER # if not app specific error, mark it as masked
		elif os.path.getsize(stdoutdiff_fname) != 0 and os.path.getsize(stderrdiff_fname) == 0:
			if "Xid" in dmesg_delta:
				return p.DMESG_STDOUT_ONLY_DIFF
			elif "ERROR FAIL in kernel execution" in inj_log_str: 
				return p.SDC_KERNEL_ERROR
			else:
				return p.STDOUT_ONLY_DIFF
		elif os.path.getsize(stderrdiff_fname) != 0 and os.path.getsize(stdoutdiff_fname) == 0:
			if "Xid" in dmesg_delta:
				return p.DMESG_STDERR_ONLY_DIFF
			elif "ERROR FAIL in kernel execution" in inj_log_str: 
				return p.SDC_KERNEL_ERROR
			else:
				return p.STDERR_ONLY_DIFF
		else:
			if p.verbose: 
				print ("Other from here")
			return p.OTHERS
	else: # one of the files is not found, strange
		print (" %s, %s not found" %(stdoutdiff_fname, stderrdiff_fname))
		return p.OTHERS



def receiveSignal():
	print ("segegv/n")

###############################################################################
# Run the actual injection run 
###############################################################################
def run_one_injection_job():
	[kn, threadID, reg, mask, SM, stuck_at] = ["","", "", "", "",""]
	fName = p.script_dir[app] + "/injectionsRF" +  ".txt"
	total_jobs = 0
	start = datetime.datetime.now()  
	if os.path.isfile(inj_run_logname): 
		logf = open(fName, "r")
		for line in logf:		
			kn = line.split(' ')[0].strip()
			threadID = line.split(' ')[1].strip()
			#ctaID = line.split(' ')[2].strip()
			reg = line.split(' ')[2].strip()
			mask = line.split(' ')[3].strip()
			SM = line.split(' ')[4].strip()
			stuck_at = line.split(' ')[5].strip()
			print("%s %s %s %s %s %s" %(kn, threadID, reg, mask, SM, stuck_at))
			f = open(injection_seeds_file,"w")
			f.write("%s\n%s\n%s\n%s\n%s\n" %(threadID, reg, mask, SM, stuck_at))
			f.close()
			cwd = os.getcwd()
			dmesg_before = cmdline("dmesg")
			
			total_jobs += 1
	
	#pr = os.system(script_fname)	
	#[timeout_flag, retcode] = is_timeout(app, pr)
	#if p.verbose: print ("App runtime: " + str(get_seconds(datetime.datetime.now() - start_main)))
			pr = subprocess.Popen(script_fname, shell=True, executable='/bin/bash', preexec_fn=os.setsid) # run the injection job
			[timeout_flag, retcode] = is_timeout(app, pr)
	#retval = pr.returncode
	#[timeout_flag, retcode] = (0,0)
	#signal.signal(signal.SIGSEGV, receiveSignal)
	#print ("%s ecco %s" %(timeout_flag,retcode))

	# Record kernel error messages (dmesg)
			dmesg_after = cmdline("dmesg")
			dmesg_delta = get_dmesg_delta(dmesg_before, dmesg_after)
			dmesg_delta = dmesg_delta.replace("\n", "; ").replace(":", "-")
	#print ("ciao %s" %(dmesg_delta))
	#ret_cat = classify_injection(dmesg_delta)
	#[inst_type, kn, laneID, injMask] = get_inj_info()
	
			if timeout_flag:
				[kn,threadID,reg,mask,SMID,stuckat] = get_inj_info()
				ret_cat = p.TIMEOUT 
			else:
				[kn,threadID,reg,mask,SMID,stuckat] = get_inj_info()
				ret_cat = classify_injection(app, retcode, dmesg_delta)
			if ret_cat ==  5:
					kn = line.split(' ')[0].strip()
					threadID = line.split(' ')[1].strip()
					reg = line.split(' ')[2].strip()
					mask = line.split(' ')[3].strip()
					SMID = line.split(' ')[4].strip()
					stuckat = line.split(' ')[5].strip()
			if os.path.isfile(report_fname):
			 with open(report_fname,"a") as f:
				f.write("kernel name %s; threadID %s, reg %s, mask %s, SMID %s,stuckat %s; Outcome %s (code %s)\n" %(kn,threadID,reg,mask,SMID,stuckat,p.CAT_STR[ret_cat-1], ret_cat))
			else:
			 with open(report_fname,"w+") as f:
				f.write("kernel name %s; threadID %s,  reg %s, mask %s, SMID %s,stuckat %s; Outcome %s (code %s)\n" %(kn,threadID,reg,mask,SMID,stuckat,p.CAT_STR[ret_cat-1], ret_cat))

			print ("kernel name %s; threadID %s, reg %s, mask %s, SMID %s,stuckat %s; Outcome %s (code %s)\n" %(kn,threadID,reg,mask,SMID,stuckat,p.CAT_STR[ret_cat-1], ret_cat))

			os.chdir(cwd) # return to the main dir
			print_heart_beat(total_jobs)
			
			sec = input('Let us wait for user input.\n')
	# print (ret_cat)
			#elapsed = datetime.datetime.now() - start			
			#seconds = elapsed.seconds
			seconds = str(get_seconds(datetime.datetime.now() - start))
			minutes = int(((float(seconds))//60)%60)
			hours = int((float(seconds))//3600)
			#seconds = int(seconds)
			print("Elapsed %s:%s" %(str(hours),str(minutes)))
			#print ("App runtime: " + str(get_seconds(datetime.datetime.now() - start)))
			#print(start)
	#record_result(inj_mode, igid, bfm, app, kname, kcount, iid, opid, bid, ret_cat, pc, inst_type, tid, injBID, get_seconds(elapsed), dmesg_delta, value_str, icount)

	#if get_seconds(elapsed) < 0.5: time.sleep(0.5)
	#if not p.keep_logs:
	#	shutil.rmtree(new_directory, True) # remove the directory once injection job is done
		logf.close()
	return ret_cat



def get_inj_info():
	[kn,threadID,reg,mask,SMID,stuckat] = ["","","", "", "", ""]
	if os.path.isfile(inj_run_logname): 
		logf = open(inj_run_logname, "r")
		for line in logf:
			if "inspecting:" in line:		
					kn = line.split(';')[0].split('inspecting: ')[1].strip()
			if "thread :" in line:
					threadID = line.split(';')[1].split('thread : ')[1].strip()
			#if "CtaID :" in line:		
			#		ctaID = line.split(';')[2].split('CtaID : ')[1].strip()
			if "Register :" in line:
					reg = line.split(';')[2].split('Register : ')[1].strip()
			if "Mask :" in line:		
					mask = line.split(';')[3].split('Mask : ')[1].strip()
			if "SMID :" in line:
					SMID = line.split(';')[4].split('SMID : ')[1].strip()
			if "Stuck at :" in line:
					stuckat = line.split(';')[5].split('Stuck at : ')[1].strip()
		logf.close()
	return [kn,threadID,reg,mask,SMID,stuckat]




###############################################################################
# Set enviroment variables for run_script
###############################################################################
#[stdout_fname, stderr_fname, injection_seeds_file, new_directory] = ["", "", "", ""]
def set_env_variables(): # Set directory paths 

	p.set_paths() # update paths 
	global stdout_fname, stderr_fname, injection_seeds_file, new_directory, injrun_fname, stdoutdiff_fname, stderrdiff_fname , script_fname, sdc_fname, inj_run_logname, report_fname
	
	#new_directory = p.NVBITFI_HOME + "/logs/" + app + "/" + app + "-group" + igid + "-model" + bfm + "-icount" + icount
		
	new_directory = p.script_dir[app] + "/logs" 
	if not os.path.isdir(new_directory): os.system("mkdir -p " + new_directory)	
		
	stdout_fname = p.script_dir[app] + "/" + p.stdout_file
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
	# print ("run_one_injection.py: kname=%s, argv[8]=%s" %(sys.argv[5], sys.argv[8])
	# check if paths exit
#	if not os.path.isdir(p.NVBITFI_HOME): print ("Error: Regression dir not found!")
	if not os.path.isdir(p.NVBITFI_HOME + "/logs/results"): os.system("mkdir -p " + p.NVBITFI_HOME + "/logs/results") # create directory to store summary
#[inj_mode, igid, bfm, app, kname, kcount, iid, opid, bid, icount] = sys.argv[1:]
	global app	
	#app = sys.argv[1]
	for app in p.apps: 
		set_env_variables()
		err_cat = run_one_injection_job()

if __name__ == "__main__" :
    main()
