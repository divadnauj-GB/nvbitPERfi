import os, sys, re, string, operator, math, datetime, time, signal, subprocess, shutil, glob, pkgutil
import params as p
import common_functions as cf 
before = -1



def cudaerrcode():
	if os.path.isfile(injrun_fname): 
		inj_log_str = open(injrun_fname, "r")
	else:
		print "fail"
	errmsg="N/A"
	for line in inj_log_str:
		if "ERROR FAIL in kernel execution " in line:
			#print(str(line).find("injNumActivations"))
			idx=str(line).find("ERROR FAIL in kernel execution :")+31
			errmsg=(str(line)[idx:len(str(line))].strip())
			errmsg=errmsg.strip(';')
			errmsg=errmsg.strip(')')
			errmsg=errmsg.strip('(')
			errmsg=errmsg.lstrip('(')
			#print(num_inj)
	return errmsg

def num_act():
	if os.path.isfile(injrun_fname): 
		inj_log_str = open(injrun_fname, "r")
	else:
		print "fail"
	#print "test"
	for line in inj_log_str:
		if "injNumActivations" in line:
			#print(str(line).find("injNumActivations"))
			idx=str(line).find("injNumActivations:")+19
			num_inj=int(str(line)[idx:len(str(line))].strip())
			#print(num_inj)
	return num_inj

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
		print("countdown %ss ret code %s"%(tt,retcode))
		return [True, pr.poll()]
	else:
		print("countdown %ss ret code %s"%(tt-(to_th/2),retcode))
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
def classify_injection(app, timeout_flag, retcode, dmesg_delta):

	if timeout_flag:
		return p.TIMEOUT
	
	
	stdout_str = "" 
	if os.path.isfile(stdout_fname): 
		stdout_str = str(open(stdout_fname).read())

	if p.detectors and "- 43, Ch 00000010, engmask 00000101" in dmesg_delta and "- 13, Graphics " not in dmesg_delta and "- 31, Ch 0000" not in dmesg_delta: # this is specific for error detectors 
		return p.DMESG_XID_43
	#DUE fault
	

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
	if "ERROR FAIL in kernel execution " in inj_log_str:
		if retcode != 0: 
			return p.CUDAERROR
		else:
			return p.STDOUT_ERROR_MESSAGE
	if "Error: misaligned address" in str(open(stderr_fname).read()): # if error is found in the log standard err 
		return p.STDOUT_ERROR_MESSAGE
	
	if retcode != 0:		
		return p.NON_ZERO_EC

	#SDC fault or Masked are identified by compare_stdout() function

	return retcode
	code = compare_stdout()

	#pr = subprocess.Popen(sdc_fname, shell=True, executable='/bin/bash', preexec_fn=os.setsid);
	#if  os.path.isfile(stdoutdiff_fname) and os.path.isfile(stderrdiff_fname):
	#	if  os.path.getsize(stdoutdiff_fname) == 0 and os.path.getsize(stderrdiff_fname) == 0: # no diff is observed
	#		if "ERROR FAIL in kernel execution" in inj_log_str: 
	#			return p.MASKED_KERNEL_ERROR # masked_kernel_error
	#		else:
	#			return p.MASKED_OTHER # if not app specific error, mark it as masked
	#	elif os.path.getsize(stdoutdiff_fname) != 0 and os.path.getsize(stderrdiff_fname) == 0:
	#		if "Xid" in dmesg_delta:
	#			return p.DMESG_STDOUT_ONLY_DIFF
	#		elif "ERROR FAIL in kernel execution" in inj_log_str: 
	#			return p.SDC_KERNEL_ERROR
	#		else:
	#			return p.STDOUT_ONLY_DIFF
	#	elif os.path.getsize(stderrdiff_fname) != 0 and os.path.getsize(stdoutdiff_fname) == 0:
	#		if "Xid" in dmesg_delta:
	#			return p.DMESG_STDERR_ONLY_DIFF
	#		elif "ERROR FAIL in kernel execution" in inj_log_str: 
	#			return p.SDC_KERNEL_ERROR
	#		else:
	#			return p.STDERR_ONLY_DIFF
	#	else:
	#		if p.verbose: 
	#			print ("Other from here")
	#		return p.OTHERS
	#else: # one of the files is not found, strange
	#	print (" %s, %s not found" %(stdoutdiff_fname, stderrdiff_fname))
	#	return p.OTHERS



def receiveSignal():
	print ("segegv/n")

###############################################################################
# Run the actual injection run 
###############################################################################
def run_injections():

	[ threadID, reg, mask, SM, stuck_at] = ["", "", "", "",""]
	fName = p.script_dir[app] + "/faults.list"
	
	HW_fault=open(p.NVBITFI_HOME+"/pf_injector/"+p.injection_HW,'r')
	HW_fault_loc=HW_fault.readlines()
	HW_fault.close()

	SMID=int(HW_fault_loc[0])
	subSMID=int(HW_fault_loc[1])
	LaneID=int(HW_fault_loc[2])
	Th=float(HW_fault_loc[3])
	

	logs_foler = new_directory+("/report_per_fault")
	os.system("mkdir -p %s "%(logs_foler))

	report_fname = new_directory + ("/fault_sim_report.txt") 

	total_jobs = 0
	start = datetime.datetime.now()  
	i = 0
	if not os.path.isfile(inj_run_logname): 
		file=open(inj_run_logname,'w')
		file.close()

	if os.path.isfile(inj_run_logname): 
		logf = open(fName, "r")
		for line in logf:		
			injection_file_name=line.split("/")[-1].strip()
			os.system("cp %s %s" %(p.script_dir[app]+"/syndroms/"+injection_file_name, p.NVBITFI_HOME+"/pf_injector/"+p.injection_seeds))
			

			print("%s " %(injection_file_name))

			cwd = os.getcwd()
			dmesg_before = cmdline("dmesg")			
			total_jobs += 1
			tick = datetime.datetime.now()  
			pr = subprocess.Popen(script_fname, shell=True, executable='/bin/bash', preexec_fn=os.setsid) # run the injection job
			[timeout_flag, retcode] = is_timeout(app, pr)
	
			tock = str(get_seconds(datetime.datetime.now() - tick))

			# Record kernel error messages (dmesg)
			dmesg_after = cmdline("dmesg")
			dmesg_delta = get_dmesg_delta(dmesg_before, dmesg_after)
			dmesg_delta = dmesg_delta.replace("\n", "; ").replace(":", "-")
			
			gold_f_name=open(golden_stdout_fname,'r')
			faul_f_name=open(stdout_fname,'r')
			faul_f_lines=faul_f_name.readlines()
			gold_f_list=gold_f_name.readlines()
			gold_f_name.close()
			faul_f_name.close()
			image_names=[]
			for idx,line in enumerate(gold_f_list):
				if ".png" in line:
					image_names=line.split("/")[-1].strip()													
					#print(image_names)
					ret_cat = classify_injection(app, timeout_flag, retcode, dmesg_delta)			
					if ret_cat==0:
						Golden=[]
						for i in range(0,10):
							Golden.append(gold_f_list[idx+i+1])
							#print(Golden[i])

						Faulty=[]
						for jdx,value in enumerate(faul_f_lines):
							if image_names in value:								
								for i in range(0,10):
									Faulty.append(faul_f_lines[jdx+i+1])
									#print(Faulty[i])
								break
						ret_cat=compare_stdout(Golden, Faulty)
						
					if ret_cat==p.CUDAERROR:
						print_message="DUE: cuERR["+cudaerrcode()+"]"
					else:
						print_message=p.CAT_STR[ret_cat-1]

					act=num_act()

					if os.path.isfile(report_fname):
						with open(report_fname,"a") as f:
							f.write("Fault ID: %s, ImageID: %s, num_activation: %s, Outcome %s, (code %s), inj_time: %s\n" %(injection_file_name,image_names,act,print_message, ret_cat, tock))
					else:
						with open(report_fname,"w+") as f:
							f.write("Fault ID: %s, ImageID: %s,  num_activation: %s,  Outcome %s, (code %s), inj_time: %s\n" %(injection_file_name,image_names,act,print_message, ret_cat, tock))

					print("Fault ID: %s, ImageID: %s,  num_activation: %s,  Outcome %s, (code %s), inj_time: %s\n" %(injection_file_name,image_names,act,print_message, ret_cat, tock))

			os.chdir(cwd) # return to the main dir
			print_heart_beat(total_jobs)
			
			tmp_foler = p.script_dir[app]+"/logs/tmp"			
			os.system("mkdir -p %s "%(tmp_foler))

			os.system("cp %s %s"%(p.script_dir[app]+"/syndroms/"+injection_file_name,tmp_foler))
			os.system("mv %s %s"%(p.script_dir[app]+"/stdout.txt ",tmp_foler))
			os.system("mv %s %s"%(p.script_dir[app]+"/stderr.txt ",tmp_foler))
			os.system("cp %s %s"%(p.script_dir[app]+"/golden_stdout.txt ",tmp_foler))
			os.system("cp %s %s"%(p.script_dir[app]+"/golden_stderr.txt ",tmp_foler))
			os.system("mv %s %s"%(p.NVBITFI_HOME+"/pf_injector/"+p.inj_run_log,tmp_foler))
			os.system("cp %s %s"%(p.NVBITFI_HOME+"/pf_injector/"+p.injection_HW,tmp_foler))

			#sec = input('Let us wait for user input.\n')
			tar_file = injection_file_name.split('.')[0].strip()+".tar.gz"
			
			os.chdir(tmp_foler)
			os.system("tar czfP %s %s "%(logs_foler+"/"+tar_file, "."))
			os.system("rm -r %s"%(tmp_foler))

			os.chdir(cwd)
			
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
def compare_stdout(Golden, Faulty):
	gold_class = []
	gold_pred = []
	out_class  = []
	out_pred =[]

	cnt=0
	crit=0
	for indx in range(0,len(Golden)):
		Gpart=Golden[indx].split(':')
		Fpart=Faulty[indx].split(':')
		gold_pred.append(Gpart[0].strip('%'))
		gold_class.append(Gpart[1].strip())
		out_pred.append(Fpart[0].strip('%'))
		out_class.append(Fpart[1].strip())
		if(gold_pred[indx]!=out_pred[indx] or gold_class[indx]!=out_class[indx]):
			cnt+=1
		if("nan" in out_pred[indx]) or ("NAN" in out_pred[indx])or ("inf" in out_pred[indx]) or ("INF" in out_pred[indx]):
			crit+=1

	if cnt==0:
		return p.MASKED_OTHER
	elif crit!=0:
		return p.CRITICAL
	else:
		if gold_class[0]==out_class[0] and (out_pred[0]!="nan" or out_pred[0]!="NAN"):
			if float(gold_pred[0])!=float(out_pred[0]):
				return p.SAFE
			else:
				return p.MASKED_OTHER
		else:
			return p.CRITICAL

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
	global stdout_fname,golden_stdout_fname, stderr_fname, injection_seeds_file, new_directory, injrun_fname, stdoutdiff_fname, stderrdiff_fname , script_fname, sdc_fname, inj_run_logname, report_fname
	
	#new_directory = p.NVBITFI_HOME + "/logs/" + app + "/" + app + "-group" + igid + "-model" + bfm + "-icount" + icount
		
	new_directory = p.script_dir[app] + "/logs" 
	if not os.path.isdir(new_directory): os.system("mkdir -p " + new_directory)	
		
	stdout_fname = p.script_dir[app] + "/" + p.stdout_file
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
	
	#report_fname = new_directory + "/" + p.report

	fault_list = p.script_dir[app] + "/faults.list"
	HW_faulty_file=p.NVBITFI_HOME +"/pf_injector/"+p.injection_HW






def main(): 
	global app	
	for app in p.apps: 
		set_env_variables()
		err_cat = run_injections()

if __name__ == "__main__" :
    main()
