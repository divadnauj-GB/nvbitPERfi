# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



import sys, re, string, os, operator, math, datetime, random
import params as p
import common_functions as cf 

MAX_INJ = p.NUM_INJECTIONS
verbose = False

##############Read the profile file to collect information
def read_profile(file_name):
	[maxregs, num_threads, n_devices] = ["", "", ""]
	if os.path.isfile(file_name): 
		logf = open(file_name, "r")
		maximum = -1
		for line in logf:
			if "maxregs" in line:		
					maxregs = line.split(';')[5].split('maxregs: ')[1].strip()
			if "num_threads" in line:
					num_threads = line.split(';')[3].split('num_threads: ')[1].strip()
			if "n_devices" in line:
					n_devices = line.split(';')[4].split('n_devices: ')[1].strip()
			maxregs = int(maxregs)
			if (maxregs > maximum):
				maximum = maxregs
		logf.close()		
		maxregs = int(maximum)
		num_threads = int(num_threads)
		n_devices = int(n_devices)
		Nfaults = 2 * 32 * maxregs * num_threads * n_devices 
		string = "n_faultsRF = " + 	str(Nfaults)
		logf = open(file_name, "a")
		logf.write(string + "\n")
		logf.close()
	return [maxregs,  num_threads, n_devices]

##############Generate the fault list according to the gathered information
def generate_fault_list(app, num_injections, maxregs,  num_threads, n_devices):
	maxregs = int(maxregs)
	num_threads = int(num_threads)
	n_devices = int(n_devices)

	fName = p.script_dir[app] + "/injectionsRF" +  ".txt"
	print (fName)
	logf = open(fName, "w")
	logf.close()
	
#####commented part used for random fault list generation	
	#i = 0
	#reg = 0
	#while  num_injections > 0: 
			
	#	threadID = random.randint(0, num_threads -1)
	#	reg = random.randint(0, maxregs -1)
	#	if i == 31:
	#			i = 0 		
	#	mask = 1<<i 
	#	i = i + 1
	#	device = random.randint(0, n_devices -1)
	#	stuck_at = random.randint(0, 1)
	#	selected_str =  str(threadID) +  " " + str(reg) + " " + str(mask) + " " + str(device) + " " + str(stuck_at)
	#	sel_str = " Threadid "+str(threadID)+" REG "+ str(reg)+" MASK "+str(mask)+" SM "+str(device)+" STUCKAT "+ str(stuck_at)
	#	print ("%d/%d: Selected: %s" %(num_injections,MAX_INJ, sel_str))
	#	logf.write(selected_str + "\n") # print injection site information
	#	num_injections -= 1	

################ Inject in first 10 registers for 512 random threads with random mask and random stuck-at-model and random SM(in jetson is just 1
	NumberOfFaults = maxregs * num_threads * n_devices * 32 * 2
	print(NumberOfFaults)
	max_injections = (NumberOfFaults/(1 +(0.01 * 0.01)*((NumberOfFaults-1)/((2.575829304*2.575829304)*(0.5*0.5)))))
	print(max_injections)
	j = 0
	while j<max_injections: #16533
		reg = 0	
		while reg < 10:		
			threadID = random.randint(0, num_threads -1)
			i = random.randint(0,32)		
			mask = 1<<i #one faulty bit for each injection
			device = random.randint(0, n_devices -1)
			stuck_at = random.randint(0, 1)
			selected_str=str(threadID)+" "+str(reg)+" "+str(mask)+" "+str(device)+" "+str(stuck_at)+"\n"
			f = open(fName, "r")
			equal = 0		
			for line in f:
				if line == selected_str:
					equal = 1
					#print(equal)
			f.close()
			if equal == 0:
				logf = open(fName, "a")
				logf.write(selected_str)
				j +=1
				reg = reg + 1	
				logf.close()	
	
################ Inject 10 registers for all the threads ###################### 
#	i = 0	
#	threadID = 0	
#	while threadID < num_threads:		
#			reg = 0 
#			while reg < maxregs: 
#				if i == 31: 
#					i = 0 		
#				mask = 1<<i #one faulty bit for each injection
#				i = i + 1
#				device = random.randint(0, n_devices -1)
#				stuck_at = random.randint(0, 1)
#				selected_str=str(threadID)+" "+str(reg)+" "+str(mask)+" "+str(device)+" "+str(stuck_at)
#				logf.write(selected_str + "\n")
#				reg = reg + 16	
#			threadID = threadID + 1 #injected just few threads for each register: 17 threads for eache register

################ Inject 10 random registers for all the threads ###################### 
#	i = 0
#	i = 32	
#	threadID = 0	
#	while threadID < num_threads:		
#			j = 0 
#			while j < 10: 
				#if i == 31: 
				#	i = 0 		
				#mask = 1<<i #one faulty bit for each injection
				#i = i + 1
#				mask = 1				
#				device = random.randint(0, n_devices -1)
#				stuck_at = random.randint(0, 1)
#				reg = random.randint(0, maxregs -1)
#				selected_str=str(threadID)+" "+str(reg)+" "+str(mask)+" "+str(device)+" "+str(stuck_at)
#				logf.write(selected_str + "\n")
#				
#				j += 1
#			threadID = threadID + 1 #injected just few threads for each register: 17 threads for eache register			



#################################################################
# Starting point of the script
#################################################################
def main():
	
	for app in p.apps:
		print ("\nCreating list for %s ... " %(app))
		profile_name_file = p.script_dir[app] + "/" + p.nvbit_profile_log_RF		
		[maxregs, num_threads, n_devices] = read_profile(profile_name_file)
		print ("\nFound for %s ... number of reg %s number of threads %s number of used SMs %s" %(app, maxregs,  num_threads, n_devices))
		generate_fault_list(app, MAX_INJ, maxregs, num_threads, n_devices)

if __name__ == "__main__":
    main()
