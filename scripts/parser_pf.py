import os, sys, re, string, operator, math, datetime, time, signal, subprocess, shutil, glob, pkgutil
import params as p
import common_functions as cf 



def set_env_variables(): # Set directory paths 

	p.set_paths() # update paths 
	global report_fname, op_report, profile_name_file
	
	new_directory = p.script_dir[app] + "/logs"
	report_fname = new_directory + "/" + p.report
	op_report =  new_directory + "/" + app +"_" + p.op_report
	profile_name_file = p.script_dir[app] + "/" + p.nvbit_profile_log_RF

def parse_result():
	[masked,due,sdc] = [0, 0, 0]
	if os.path.isfile(report_fname):
		logf = open(report_fname, "r")
		for line in logf:
			if "code" in line:
				n = line.split('code ')[1].split(')')[0].strip()			
				if n == '1' or n == '3' or n == '2':
					masked +=1
				elif n == '4' or n == '5' or n== '6' or n== '7' or n == '8' or n == '9' or n== '10' or n== '11' or n == '12' or n == '13' or n== '14' or n== '15' :
					due +=1				
				elif n == '16' :
					sdc +=1
					
						
		logf.close()
	print ("For the %s ->Faults classification: masked %s due %s sdc %s" %(app,masked,due,sdc))
	return 	[masked,due,sdc]


def save_results(masked,due,sdc,reg_faulty,n_faultsRF,n_faults_reg):
			injected_faults = int(masked) + int(due) + int(sdc) 
			nfault_onereg = injected_faults / (len(reg_faulty))
			coverage = float(sdc) / float (injected_faults)
			formatted_float = "{:.2%}".format(coverage)
			with open(op_report,"w+") as f:
				f.write("Kernel Name: %s\nTotal number of faults in Register File: %s\nNumber of injected Faults: %d\n" %(app,n_faultsRF,injected_faults))
				f.write("Faults classification:\nMasked %s; DUE %s; SDC %s\n" %(masked,due,sdc))
				f.write("Global Fault coverage (SDC/Injected Faults Number): %s\n\n\n" %(formatted_float))
				f.write("Number of possible faults for each register: %d\nNumber of injected faults in each register: %d\n\n" %(n_faults_reg,nfault_onereg))
				f.write( "Register  Detected Faults(SDC)  Fault Coverage\n")
				for i in range(len(reg_faulty)):
					fault_coverage = float(reg_faulty[i]) / float(nfault_onereg) 
					formatted_float = "{:.2%}".format(fault_coverage)
					f.write( "  %-*s  %-*s  %s\n" % (12,i,14,reg_faulty[i],formatted_float))
				
#check if there is at least a detected fault for each register	
def special_check():
	logf = open(profile_name_file, "r")
	for line in logf:
		if "maxregs" in line:		
			maxregs = line.split(';')[2].split('maxregs: ')[1].split('(')[0].strip()
		if "n_faultsRF" in line:
			n_faultsRF = line.split('= ')[1].strip()
	n_faults_reg = int(n_faultsRF) / int(maxregs)
	reg_faulty = list(0 for i in range(0, int(maxregs)))	
	if os.path.isfile(report_fname):
		logf = open(report_fname, "r")
		for line in logf:
			if "code 16" in line:
				if "reg" in line:
						reg = line.split('reg ')[1].split(',')[0].strip()
						reg_faulty[int(reg)] += 1
	return [reg_faulty,n_faultsRF,n_faults_reg]
	
def main(): 
	global app, kn, inst_type, injMask , op_report
	for app in p.apps: 
		set_env_variables()
		[masked,due,sdc] = parse_result()
		[reg_faultyN,n_faultsRF,n_faults_reg] = special_check()
		save_results(masked,due,sdc,reg_faultyN,n_faultsRF,n_faults_reg)


if __name__ == "__main__" :
    main()
