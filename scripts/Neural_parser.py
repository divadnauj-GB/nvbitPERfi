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
	[masked,due,sdc_safe,sdc_critical] = [0, 0, 0, 0]
	if os.path.isfile(report_fname):
		logf = open(report_fname, "r")
		for line in logf:
			if "code" in line:
				n = line.split('code ')[1].split(')')[0].strip()			
				if n == '1' or n == '3' or n == '2':
					masked +=1
				elif n == '4' or n == '5' or n== '6' or n== '7' or n == '8' or n == '9' or n== '10' or n== '11' or n == '12' or n == '13' or n== '14' or n== '15' :
					due +=1				
				elif n == '20' :
					sdc_safe +=1
				elif n == '21' :
					sdc_critical +=1	
						
		logf.close()
	print ("For the %s ->Faults classification: masked %s due %s sdc_safe %s sdc_critical %s" %(app,masked,due,sdc_safe,sdc_critical))
	return 	[masked,due,sdc_safe,sdc_critical]




def save_results(masked,due,sdc_safe,sdc_critical,reg_faulty,n_faultsRF,n_faults_reg,index,sim_time,n_fault_in_reg):
			injected_faults = int(masked) + int(due) + int(sdc_safe) + int(sdc_critical) 
			total_sdc = int(sdc_safe) + int(sdc_critical)
			nfault_onereg = injected_faults / (len(reg_faulty))
			
			coverage = float(sdc_safe + sdc_critical) / float (injected_faults)
			formatted_float = "{:.2%}".format(coverage)
			n_threads = nfault_onereg / 2
			threads_tot= int(n_faultsRF)/(len(reg_faulty) * 32 * 2)
			reg = injected_faults / threads_tot
			
			nfault_casetwo= threads_tot
			

			with open(op_report,"w+") as f:
				#case 1				
				#f.write("Fault list generation approach: For each register(%d) injections are perfomed in %d threads(out of the %d used by kernels) testing a pseudo-random single bit stuck at 1 and stuck at 0.\n\n" %(len(reg_faulty),n_threads,threads_tot))	
				#case 2
				f.write("Fault list generation approach: For each thread(%d) injections are perfomed in %d registers(out of the %d used by kernels) testing a pseudo-random single bit stuck at 1 and stuck at 0.\n\n" %(threads_tot,reg,len(reg_faulty)))

				f.write("Fault attributes legend: faults are classified as \n- SDC-safe if the injected fault still make the Neural Network produce the correct guess with a different percentage(in each of the guessing classes) with respect to fault-free result;\n- SDC-critical if the final guess is wrong;\n- Masked if the fault doesn't affect the guess and the associated percentage neither;\n- DUE if a segmentation fault occurs and the result is not obtained at all\n\n") 
				f.write("----------------------------------------------------------------------------------\n")	
				f.write("Application name: %s LeNet\nNumber of kernels: %d\nMax number of threads: %d\nNumber of registers: %d (32 bits wide)\nTotal number of possible stuck-at faults in Register File: %s\n\nNumber of injected Faults: %d\nSimulation time (hours:minutes): %s\n\n" %(app,int(index),threads_tot,len(reg_faulty),n_faultsRF,injected_faults,sim_time))				 
				f.write("Faults classification:\nMasked  DUE   SDC (safe + critical)\n")
				f.write( "%-*s  %-*s  %-*s(%s + %s)\n\n" % (6,masked,4,due,4,total_sdc,sdc_safe,sdc_critical))
				f.write("Global Fault coverage (SDCs/Injected Faults): %s\n\n" %(formatted_float))
				f.write("Number of possible stuck-at faults for each register: %d\n"%n_faults_reg)
				#case 1
				#f.write("Number of injected faults in each register: %d\n" %nfault_onereg)
				#case 2
				#f.write("Number of injected faults in each selected register: %d\n" %nfault_casetwo)

				f.write("----------------------------------------------------------------------------------\n")				
				f.write("Detailed results for each register\nThread Id  Register  Number of injected faults     DUE           MASKED    SDC_safe  SDC_critical \n")
				f.close()
				parse_result_threads(reg_faulty,threads_tot)

				#for i in range(len(reg_faulty)):
					#case 1					
					#fault_coverage = float(reg_faulty[i]) / float(nfault_onereg) 
					#case 2					
					#fault_coverage = float(reg_faulty[i]) / float(nfault_casetwo)
					#valide always
				#	if n_fault_in_reg[i] != 0:
				#		fault_coverage = float(reg_faulty[i]) / float(n_fault_in_reg[i])
				#		formatted_float = "{:.2%}".format(fault_coverage)
				#	else:
				#		formatted_float = "{:.2%}".format(0.0)					
				#	f.write( "  %-*s  %-*s  %s\n" % (12,i,15,reg_faulty[i],formatted_float))

#check if there is at least a detected fault for each register	
def special_check():
	logf = open(profile_name_file, "r")
	maximum = -1
	for line in logf:
		if "maxregs" in line:		
			maxregs = line.split(';')[5].split('maxregs: ')[1].strip()
			maxregs = int(maxregs)
			if (maxregs > maximum):
				maximum = maxregs
		if "n_faultsRF" in line:
			n_faultsRF = line.split('= ')[1].strip()
		if "index" in line:
			index = line.split(';')[1].split('index: ')[1].strip()
	n_faults_reg = int(n_faultsRF) / int(maximum)
	reg_faulty = list(0 for i in range(0, int(maximum)))
	n_fault_in_reg = list(0 for i in range(0, int(maximum)))	
	if os.path.isfile(report_fname):
		logf = open(report_fname, "r")
		for line in logf:
			if "code 16" in line or "code 20" in line or "code 21" in line:
				if "reg" in line:
						reg = line.split('reg ')[1].split(',')[0].strip()
						reg_faulty[int(reg)] += 1
			if "reg" in line:
				numb = line.split('reg ')[1].split(',')[0].strip()
				n_fault_in_reg[int(numb)] += 1
			if "Simulation time" in line:
				sim_time = line.split('Simulation time: ')[1].strip()
	return [reg_faulty,n_faultsRF,n_faults_reg,index,sim_time,n_fault_in_reg]


def parse_result_one_thread(max_regs,thread_id):

	reg_faults_list = list(0 for i in range(0,int(len(max_regs))))
	masked_list = list(0 for i in range(0,int(len(max_regs))))
	due_list = list(0 for i in range(0,int(len(max_regs))))
	sdc_safe_list = list(0 for i in range(0,int(len(max_regs))))
	sdc_critical_list = list(0 for i in range(0,int(len(max_regs))))

	if os.path.isfile(report_fname):
		logf = open(report_fname, "r")
		#print("ciao")
		for line in logf:
			#print(line)
			if "threadID" in line:
				ID = line.split('threadID ')[1].split(',')[0].strip()
				#print(ID)
				if (int(ID) == thread_id):
					reg_numb = line.split('reg ')[1].split(',')[0].strip()
					reg_faults_list[int(reg_numb)] += 1 #in that register number a fault is injected
					#print(ID)					
					#print(reg_numb)
					n = line.split('code ')[1].split(')')[0].strip()			
					if n == '1' or n == '3' or n == '2':
						masked_list[int(reg_numb)] +=1
					elif n == '4' or n == '5' or n== '6' or n== '7' or n == '8' or n == '9' or n== '10' or n== '11' or n == '12' or n == '13' or n== '14' or n== '15' :
						due_list[int(reg_numb)]  +=1				
					elif n == '20' :
						sdc_safe_list[int(reg_numb)]  +=1
					elif n == '21' :
						sdc_critical_list[int(reg_numb)]  +=1	
		logf.close()
		#print(reg_faults_list)
	return [reg_faults_list,masked_list,due_list,sdc_safe_list,sdc_critical_list]

def parse_result_threads(max_regs,num_threads):
	with open(op_report,"a") as f:
		thread_id = 0
		while thread_id < num_threads:
			[reg_faults_list,masked_list,due_list,sdc_safe_list,sdc_critical_list] = parse_result_one_thread(max_regs,thread_id)
			j= 0	
			while j < len(reg_faults_list):
				if reg_faults_list[j] != 0:
					mask_percentage = float(masked_list[j]) / float(reg_faults_list[j])
					due_percentage =  float(due_list[j]) / float(reg_faults_list[j])
					sdc_safe_percentage = float(sdc_safe_list[j]) / float(reg_faults_list[j])
					sdc_critical_percentage = float(sdc_critical_list[j]) / float(reg_faults_list[j])
					#fault_coverage = float(reg_faulty[i]) / float(n_fault_in_reg[i])
					formatted_float_mask = "{:.2%}".format(mask_percentage)
					formatted_float_due = "{:.2%}".format(due_percentage)
					formatted_float_sdc_safe = "{:.2%}".format(sdc_safe_percentage)
					formatted_float_sdc_critical = "{:.2%}".format(sdc_critical_percentage)
				
				else:
					formatted_float_mask = "{:.2%}".format(0.0)
					formatted_float_due = "{:.2%}".format(0.0)
					formatted_float_sdc_safe = "{:.2%}".format(0.0)
					formatted_float_sdc_critical = "{:.2%}".format(0.0)
					#formatted_float = "{:.2%}".format(0.0)	
				#print(j)
				#f.write("%s " %j)				
				#f.write( " %-*s %-*s  %-*s %-*s %-*s %-*s %-*s\n" % (10,thread_id,10,j,26,reg_faults_list[j],12,formatted_float_due,8,formatted_float_mask,8,formatted_float_sdc_safe,8,formatted_float_sdc_critical))
				if reg_faults_list[j] != 0:
					f.write( " %-*s %-*s %-*s %s %-*s %s %-*s %s %-*s %s %-*s\n" % (10,thread_id,10,j,26,reg_faults_list[j],due_list[j],12,formatted_float_due,masked_list[j],8,formatted_float_mask,sdc_safe_list[j],8,formatted_float_sdc_safe,sdc_critical_list[j],8,formatted_float_sdc_critical))
				#f.write( " %s        %s          %s          %s      %s        %s         %s\n" % (thread_id,j,reg_faults_list[j],formatted_float_due,formatted_float_mask,formatted_float_sdc_safe,formatted_float_sdc_critical))
				j += 1
			thread_id += 1
	f.close()
			
def main(): 
	global app, kn, inst_type, injMask , op_report
	for app in p.apps: 
		set_env_variables()
		[masked,due,sdc_safe,sdc_critical] = parse_result()
		print ("%s %s %s %s\n" %(masked,due,sdc_safe,sdc_critical))
		[reg_faultyN,n_faultsRF,n_faults_reg,index,sim_time,n_fault_in_reg] = special_check()
		save_results(masked,due,sdc_safe,sdc_critical,reg_faultyN,n_faultsRF,n_faults_reg,index,sim_time,n_fault_in_reg)


if __name__ == "__main__" :
    main()
