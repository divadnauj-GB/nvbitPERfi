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


import os, sys, re, string, math, datetime, time, pkgutil
import params as p
import common_functions as cf 

output_format = []
for cat in range(p.NUM_CATS-1):
	output_format.append(p.CAT_STR[cat])

fname_prefix = ""

results_app_table = {} # app, igid, bfm, outcome, 
num_injections_app_table = {} # app, igid, bfm, num_injections
runtime_app_table = {} # app, igid, bfm, runtime
runtime_app_nt_table = {} # app, igid, bfm, runtime without Timeouts
results_kname_table = {} # app, kname, igid, bfm, outcome, 
results_kiid_table = {} # app, kname, kid, igid, bfm, outcome, 
















def check_and_create_nested_dict(dict_name, k1, k2, k3, k4="", k5="", k6="", k7="", k8="",k9="",k10=""):
	if k1 not in dict_name:
		dict_name[k1] = 0 if k2 == "" else {}
	if k2=="":
		return
	if k2 not in dict_name[k1]:
		dict_name[k1][k2] = 0 if k3 == "" else {}
	if k3=="":
		return
	if k3 not in dict_name[k1][k2]:
		dict_name[k1][k2][k3] = 0 if k4 == "" else {}
	if k4 == "":
		return
	if k4 not in dict_name[k1][k2][k3]:
		dict_name[k1][k2][k3][k4] = 0 if k5 == "" else {}
	if k5 == "":
		return
	if k5 not in dict_name[k1][k2][k3][k4]:
		dict_name[k1][k2][k3][k4][k5] = 0 if k6 == "" else {}
	if k6 == "":
		return
	if k6 not in dict_name[k1][k2][k3][k4][k5]:
		dict_name[k1][k2][k3][k4][k5][k6] = 0 if k7=="" else {}
	if k7 == "":
		return
	if k7 not in dict_name[k1][k2][k3][k4][k5][k6]:
		dict_name[k1][k2][k3][k4][k5][k6][k7] = 0 if k8=="" else {}
	if k8 == "":
		return
	if k8 not in dict_name[k1][k2][k3][k4][k5][k6][k7]:
		dict_name[k1][k2][k3][k4][k5][k6][k7][k8] = 0 if k9=="" else {}
	if k9 == "":
		return
	if k9 not in dict_name[k1][k2][k3][k4][k5][k6][k7][k8]:
		dict_name[k1][k2][k3][k4][k5][k6][k7][k8][k9] = 0 if k10=="" else {}
	if k10 == "":
		return
	if k10 not in dict_name[k1][k2][k3][k4][k5][k6][k7][k8][k9]:
		dict_name[k1][k2][k3][k4][k5][k6][k7][k8][k9][k10] = 0

###############################################################################
# Add the injection result to the results*table dictionary 
###############################################################################
def add(app, kname, kiid, igid, bfm, outcome, runtime):
    check_and_create_nested_dict(results_app_table, app, igid, bfm, outcome)
    results_app_table[app][igid][bfm][outcome] += 1

    check_and_create_nested_dict(num_injections_app_table, app, igid, bfm)
    num_injections_app_table[app][igid][bfm] += 1

    check_and_create_nested_dict(runtime_app_table, app, igid, bfm)
    runtime_app_table[app][igid][bfm] += runtime

    if outcome != p.TIMEOUT: 
        check_and_create_nested_dict(runtime_app_nt_table, app, igid, bfm)
        runtime_app_nt_table[app][igid][bfm] += runtime

    check_and_create_nested_dict(results_kname_table, app, kname, igid, bfm, outcome)
    results_kname_table[app][kname][igid][bfm][outcome] += 1

    check_and_create_nested_dict(results_kiid_table, app, kname, kiid, igid, bfm, outcome)
    results_kiid_table[app][kname][kiid][igid][bfm][outcome] += 1


###############################################################################
# inst_fraction contains the fraction of IADD, FADD, IMAD, FFMA, ISETP, etc. 
# instructions per application
###############################################################################
inst_fraction = {}
inst_count = {}
def populate_inst_fraction():
    global inst_fraction
    for app in results_app_table:
        inst_counts = cf.get_total_counts(cf.read_inst_counts(p.app_log_dir[app], app))
        total = cf.get_total_insts(cf.read_inst_counts(p.app_log_dir[app], app), False)
        inst_fraction[app] = [total] + [1.0*i/total for i in inst_counts]
        inst_count[app] = inst_counts         
###############################################################################
# Print instruction distribution to a txt file
###############################################################################
def print_inst_fractions_tsv():
	f = open(fname_prefix + "instruction-fractions.tsv", "w")
	f.write("\t".join(["App", "Total"] + cf.get_inst_count_format().split(':')[2:]) + "\n")
	for app in inst_fraction: 
		f.write("\t".join([app] + map(str, inst_fraction[app])) + "\n")
	f.close()


def parse_results_file(app, inj_mode, igid, bfm):
	try:
		rf = open(p.app_log_dir[app] + "results-mode" + inj_mode + "-igid" + str(igid) + ".bfm" + str(bfm) + "." + str(p.NUM_INJECTIONS) + ".txt", "r")
	except IOError: 
		print ("Error opening file: " + p.app_log_dir[app] + "results-mode" + inj_mode + "-igid" + str(igid) + ".bfm" + str(bfm) + "." + str(p.NUM_INJECTIONS) + ".txt")
		print ("It is possible that no injections were performed for app=%s, inj_mode=%s, igid=%s, bfm=%s " %(app, inj_mode, str(igid), str(bfm)))
		return 

	num_lines = 0
	for line in rf: # for each injection site 
		# print(line)
		#Example line: 1;_Z22bpnn_layerforward_CUDAPfS_S_S_ii;0;26605491;0.506809798834;0.560204950825:..:MOV:773546:17:0.759537:3:dmesg:value_before_value_after, 
		#inj_count;kname;kcount;iid;opid;bid:pc:opcode:tid:injBID:runtime_sec:outcome_category:dmesg:value_before:value_after
		words1 = line.split(";")
		words2 = words1[5].split(":")
		[kname, invocation_index, opcode, injBID, runtime, outcome] = [words1[1], int(words1[2]), words2[2], words2[4], float(words2[5]), int(words2[6])]
		add(app, kname, invocation_index, igid, bfm, outcome, runtime)
		num_lines += 1
	rf.close()

	if num_lines == 0 and app in results_app_table and os.stat(p.app_log_dir[app] + "injection-list/mode" + inj_mode + "-igid" + str(igid) + ".bfm" + str(bfm) + "." + str(p.NUM_INJECTIONS) + ".txt").st_size != 0: 
		print ("%s, inj_mode=%s, igid=%d, bfm=%d not done" %(app, inj_mode, igid, bfm))

###################################################################################
# Parse results files and populate summary to results table 
###################################################################################
def parse_results_apps(typ): 
	for app in p.parse_apps:
		if typ == p.INST_VALUE_MODE:
			for igid in p.parse_inst_value_igid_bfm_map:
				for bfm in p.parse_inst_value_igid_bfm_map[igid]:
					parse_results_file(app, typ, igid, bfm)
		elif typ == p.INST_ADDRESS_MODE:
			for igid in p.parse_inst_address_igid_bfm_map:
				for bfm in p.parse_inst_address_igid_bfm_map[igid]:
					parse_results_file(app, typ, igid, bfm)
		else:
			for bfm in p.parse_rf_bfm_list:
				parse_results_file(app, typ, "rf", bfm)

###############################################################################
# Convert a dictionary to list
# input: d (dictionary), s (size)
###############################################################################
def to_list(d, s):  
	# if a specific category is not found then make it zero
	l = []
	for i in range(1,s-1):
		if i not in d:
			d[i] = 0
		l.append(d[i])
	return l


###############################################################################
# Helper function
###############################################################################
def get_igid_list(inj_mode):
	if inj_mode == p.INST_VALUE_MODE:
		return p.parse_inst_value_igid_bfm_map 
	elif inj_mode == p.INST_ADDRESS_MODE:
		return p.parse_inst_address_igid_bfm_map 
	else: # if inj_mode == p.RF_MODE:
		return ["rf"]

def get_bfm_list(inj_mode, igid):
	if inj_mode == p.INST_VALUE_MODE:
		return p.parse_inst_value_igid_bfm_map[igid] 
	elif inj_mode == p.INST_ADDRESS_MODE:
		return p.parse_inst_address_igid_bfm_map[igid] 
	else: # if inj_mode == p.RF_MODE:
		return p.parse_rf_bfm_list

def get_igid_str(inj_mode, igid):
	if inj_mode == p.INST_VALUE_MODE or inj_mode == p.INST_ADDRESS_MODE:
		return p.IGID_STR[igid]
	else: # if inj_mode == p.RF_MODE:
		return "rf"

###############################################################################
# Print Stats to a txt file
###############################################################################
def print_stats_tsv(typ):
	f = open(fname_prefix + "stats.tsv", "w")
	f.write("\t".join(["App", "IGID", "Injection Model", "Num Jobs", "Total Runtime", "Total Runtime without Timeouts"]) + "\n")

	for app in num_injections_app_table: 
		f.write(app + "\t") 

		igid_list = get_igid_list(typ)
		for igid in igid_list: 
			f.write(get_igid_str(typ, igid) + "\t")

			bfm_list = get_bfm_list(typ, igid)
			for bfm in bfm_list: 
				if igid in num_injections_app_table[app]:
					if bfm in num_injections_app_table[app][igid]:
						f.write("\t".join([p.EM_STR[bfm], str(num_injections_app_table[app][igid][bfm]), str(runtime_app_table[app][igid][bfm]),  str(runtime_app_nt_table[app][igid][bfm])]) + "\n")
					else:
						f.write("\t".join([p.EM_STR[bfm], "0", "0"] + "\n"))
	f.close()


###############################################################################
# Print detailed NVBitFI results to a tsv file for analysis in a worksheet
###############################################################################
def print_detailed_results_tsv(typ):
	f = open(fname_prefix + "NVBitFI_details.tsv", "w")
	f.write("\t".join(["App", "IGID", "Injection Model"] + output_format) + "\n")

	for app in results_app_table: 
		f.write(app + "\t") # write app name

		igid_list = get_igid_list(typ)
		for igid in igid_list: 
			f.write(get_igid_str(typ, igid) + "\t")

			bfm_list = get_bfm_list(typ, igid)
			for bfm in bfm_list: 
				written = False
				if igid in results_app_table[app]:
					if bfm in results_app_table[app][igid]:
						f.write("\t".join([p.EM_STR[bfm]] + map(str,to_list(results_app_table[app][igid][bfm], p.NUM_CATS))) + "\n")
						written = True
				if not written:
					f.write("\t".join([p.EM_STR[bfm]] + map(str,to_list({}, p.NUM_CATS))))
	f.close()


###############################################################################
# 
# 
###############################################################################
IRA_report_reason={}
IRA_report_regsrc={}
IRA_report_tot={}
IRA_report={}

def parse_results_IRA(app,log_file):

	IR_report_reason={}
	IR_report_regsrc={}
	IRreport_tot={}
	
	if os.path.isfile(log_file):
		file_data=open(log_file,'r')
		for line in file_data:
			rep_fields=line.strip().split('$')	
			sim_runtime=float(rep_fields[5])
			Outcome=rep_fields[7].replace("Outcome","")
			Outcome=Outcome.replace("(","")
			Outcome=((Outcome.replace(")","")).strip()).lstrip()


			for fields in (rep_fields[8].split(';')):
				if("injSmID" in fields):
					injSMID=int(fields.split(':')[1])
				if("injSchID" in fields):
					injSchID=int(fields.split(':')[1])
				if("injWarpIDH" in fields):
					injWarpIDH=int(fields.split(':')[1])					
				if("injWarpIDL" in fields):
					injWarpIDL=int(fields.split(':')[1])
				if("injLaneID" in fields):
					injLaneID=int(fields.split(':')[1])
				if("injRegField" in fields):
					injRegField=int(fields.split(':')[1])
				if("TotErrAct" in fields):
					TotErrAct=int(fields.split(':')[1])
				if("resRegLoc" in fields):
					if("InsideLims" in fields):
						resRegLoc="IRA"
					else:
						resRegLoc="IR"	
				Cause=""
				if("SimEndRes" in fields):
					Cause=fields.split(":::")[1]

			DmesErrType=rep_fields[9].split(',')[0]
			DmesErr=-1
			if "Xid" in (DmesErrType):
				DmesErr=int(DmesErrType.split()[4])
			num_warps=bin(injWarpIDH).count('1')+bin(injWarpIDL).count('1')
			num_LinesPerWarp=bin(injLaneID).count('1')

			if("SDC" in Outcome.upper()):
				classE="SDC"
			elif("MASKED" in Outcome.upper()):
				classE="Masked"
			elif("TIMEOUT" in Outcome.upper()):
				classE="Timeout"
			elif("DUE" in Outcome.upper()):
				classE="DUE"
			else:
				classE="-"
			check_and_create_nested_dict(IRA_report,app,resRegLoc,classE)
			IRA_report[app][resRegLoc][classE]+=1

			check_and_create_nested_dict(IRA_report_tot,app,resRegLoc,Outcome)
			IRA_report_tot[app][resRegLoc][Outcome]+=1

			check_and_create_nested_dict(IRA_report_regsrc,app,resRegLoc,injRegField,Outcome)
			IRA_report_regsrc[app][resRegLoc][injRegField][Outcome]+=1
			
			if "DUE" in Outcome:
				if "TIMEOUT" not in Outcome.upper(): 
					check_and_create_nested_dict(IRA_report_reason,app,resRegLoc,injRegField,Outcome,Cause)
					IRA_report_reason[app][resRegLoc][injRegField][Outcome][Cause]+=1
				else:
					check_and_create_nested_dict(IRA_report_reason,app,resRegLoc,injRegField,Outcome,"Timeout")
					IRA_report_reason[app][resRegLoc][injRegField][Outcome]["Timeout"]+=1
			
		file_data.close()
		print(IRA_report_tot)
		print(IRA_report_regsrc)
		print(IRA_report_reason)	
		

	else:
		print(f"The file: {log_file}; doesn't exist..")
	return

IAT_report_reason={}
IAT_report_regsrc={}
IAT_report_tot={}
IAT_report={}
def parse_results_IAT(app,log_file):
	
	inj_err="IAT"
	if os.path.isfile(log_file):
		file_data=open(log_file,'r')
		for line in file_data:
			rep_fields=line.strip().split('$')	
			sim_runtime=float(rep_fields[5])
			Outcome=rep_fields[7].replace("Outcome","")
			Outcome=Outcome.replace("(","")
			Outcome=((Outcome.replace(")","")).strip()).lstrip()
			
			for fields in (rep_fields[8].split(';')):
				if("injSmID" in fields):
					injSMID=int(fields.split(':')[1])
				if("injSchID" in fields):
					injSchID=int(fields.split(':')[1])
				if("injWarpIDH" in fields):
					injWarpIDH=int(fields.split(':')[1])					
				if("injWarpIDL" in fields):
					injWarpIDL=int(fields.split(':')[1])
				if("injLaneID" in fields):
					injLaneID=int(fields.split(':')[1])
				if("injMaskSeed" in fields):
					injMaskSeed=int(fields.split(':')[1])
				if("InjDimention" in fields):
					InjDimention=int(fields.split(':')[1])
				if("injStuck-at" in fields):
					injStuck_at=int(fields.split(':')[1])
				if("TotErrAct" in fields):
					TotErrAct=int(fields.split(':')[1])			
				Cause=""
				if("SimEndRes" in fields):
					Cause=fields.split(":::")[1]
			DmesErrType=rep_fields[9].split(',')[0]
			DmesErr=-1
			if "Xid" in (DmesErrType):
				DmesErr=int(DmesErrType.split()[4])
			num_warps=bin(injWarpIDH).count('1')+bin(injWarpIDL).count('1')
			num_LinesPerWarp=bin(injLaneID).count('1')
			
			if("SDC" in Outcome.upper()):
				classE="SDC"
			elif("MASKED" in Outcome.upper()):
				classE="Masked"
			elif("TIMEOUT" in Outcome.upper()):
				classE="Timeout"
			elif("DUE" in Outcome.upper()):
				classE="DUE"
			else:
				classE="-"
			check_and_create_nested_dict(IAT_report,app,inj_err,classE)
			IAT_report[app][inj_err][classE]+=1

			check_and_create_nested_dict(IAT_report_tot,app,inj_err,Outcome)
			IAT_report_tot[app][inj_err][Outcome]+=1

			check_and_create_nested_dict(IAT_report_regsrc,app,inj_err,InjDimention,Outcome)
			IAT_report_regsrc[app][inj_err][InjDimention][Outcome]+=1
			
			if "DUE" in Outcome:
				if "TIMEOUT" not in Outcome.upper(): 
					check_and_create_nested_dict(IAT_report_reason,app,inj_err,InjDimention,Outcome,Cause)
					IAT_report_reason[app][inj_err][InjDimention][Outcome][Cause]+=1
				else:
					check_and_create_nested_dict(IAT_report_reason,app,inj_err,InjDimention,Outcome,"Timeout")
					IAT_report_reason[app][inj_err][InjDimention][Outcome]["Timeout"]+=1			
		file_data.close()
		print(IAT_report_tot)
		print(IAT_report_regsrc)
		print(IAT_report_reason)	
		

	else:
		print(f"The file: {log_file}; doesn't exist..")
	return


IAW_report_reason={}
IAW_report_regsrc={}
IAW_report_tot={}
IAW_report={}
def parse_results_IAW(app,log_file):

	inj_err="IAW"
	if os.path.isfile(log_file):
		file_data=open(log_file,'r')
		for line in file_data:
			rep_fields=line.strip().split('$')	
			sim_runtime=float(rep_fields[5])
			Outcome=rep_fields[7].replace("Outcome","")
			Outcome=Outcome.replace("(","")
			Outcome=((Outcome.replace(")","")).strip()).lstrip()
			
			for fields in (rep_fields[8].split(';')):
				if("injSmID" in fields):
					injSMID=int(fields.split(':')[1])
				if("injSchID" in fields):
					injSchID=int(fields.split(':')[1])
				if("injWarpIDH" in fields):
					injWarpIDH=int(fields.split(':')[1])					
				if("injWarpIDL" in fields):
					injWarpIDL=int(fields.split(':')[1])
				if("injLaneID" in fields):
					injLaneID=int(fields.split(':')[1])
				if("injMaskSeed" in fields):
					injMaskSeed=int(fields.split(':')[1])
				if("InjDimention" in fields):
					InjDimention=int(fields.split(':')[1])
				if("injStuck-at" in fields):
					injStuck_at=int(fields.split(':')[1])
				if("TotErrAct" in fields):
					TotErrAct=int(fields.split(':')[1])			
				Cause=""
				if("SimEndRes" in fields):
					Cause=fields.split(":::")[1]
			DmesErrType=rep_fields[9].split(',')[0]
			DmesErr=-1
			if "Xid" in (DmesErrType):
				DmesErr=int(DmesErrType.split()[4])
			num_warps=bin(injWarpIDH).count('1')+bin(injWarpIDL).count('1')
			num_LinesPerWarp=bin(injLaneID).count('1')
			
			if("SDC" in Outcome.upper()):
				classE="SDC"
			elif("MASKED" in Outcome.upper()):
				classE="Masked"
			elif("TIMEOUT" in Outcome.upper()):
				classE="Timeout"
			elif("DUE" in Outcome.upper()):
				classE="DUE"
			else:
				classE="-"
			check_and_create_nested_dict(IAW_report,app,inj_err,classE)
			IAW_report[app][inj_err][classE]+=1

			check_and_create_nested_dict(IAW_report_tot,app,inj_err,Outcome)
			IAW_report_tot[app][inj_err][Outcome]+=1

			check_and_create_nested_dict(IAW_report_regsrc,app,inj_err,InjDimention,Outcome)
			IAW_report_regsrc[app][inj_err][InjDimention][Outcome]+=1
			
			if "DUE" in Outcome:
				if "TIMEOUT" not in Outcome.upper(): 
					check_and_create_nested_dict(IAW_report_reason,app,inj_err,InjDimention,Outcome,Cause)
					IAW_report_reason[app][inj_err][InjDimention][Outcome][Cause]+=1
				else:
					check_and_create_nested_dict(IAW_report_reason,app,inj_err,InjDimention,Outcome,"Timeout")
					IAW_report_reason[app][inj_err][InjDimention][Outcome]["Timeout"]+=1			
		file_data.close()
		print(IAW_report_tot)
		print(IAW_report_regsrc)
		print(IAW_report_reason)	
		

	else:
		print(f"The file: {log_file}; doesn't exist..")
	return

IAC_report_reason={}
IAC_report_regsrc={}
IAC_report_tot={}
IAC_report={}
def parse_results_IAC(app,log_file):

	inj_err="IAC"
	if os.path.isfile(log_file):
		file_data=open(log_file,'r')
		for line in file_data:
			rep_fields=line.strip().split('$')	
			sim_runtime=float(rep_fields[5])
			Outcome=rep_fields[7].replace("Outcome","")
			Outcome=Outcome.replace("(","")
			Outcome=((Outcome.replace(")","")).strip()).lstrip()
			
			for fields in (rep_fields[8].split(';')):
				if("injSmID" in fields):
					injSMID=int(fields.split(':')[1])
				if("injSchID" in fields):
					injSchID=int(fields.split(':')[1])
				if("injWarpIDH" in fields):
					injWarpIDH=int(fields.split(':')[1])					
				if("injWarpIDL" in fields):
					injWarpIDL=int(fields.split(':')[1])
				if("injLaneID" in fields):
					injLaneID=int(fields.split(':')[1])
				if("injMaskSeed" in fields):
					injMaskSeed=int(fields.split(':')[1])
				if("InjDimention" in fields):
					InjDimention=int(fields.split(':')[1])
				if("injStuck-at" in fields):
					injStuck_at=int(fields.split(':')[1])
				if("TotErrAct" in fields):
					TotErrAct=int(fields.split(':')[1])			
				Cause=""
				if("SimEndRes" in fields):
					Cause=fields.split(":::")[1]
			DmesErrType=rep_fields[9].split(',')[0]
			DmesErr=-1
			if "Xid" in (DmesErrType):
				DmesErr=int(DmesErrType.split()[4])
			num_warps=bin(injWarpIDH).count('1')+bin(injWarpIDL).count('1')
			num_LinesPerWarp=bin(injLaneID).count('1')

			if("SDC" in Outcome.upper()):
				classE="SDC"
			elif("MASKED" in Outcome.upper()):
				classE="Masked"
			elif("TIMEOUT" in Outcome.upper()):
				classE="Timeout"
			elif("DUE" in Outcome.upper()):
				classE="DUE"
			else:
				classE="-"
			check_and_create_nested_dict(IAC_report,app,inj_err,classE)
			IAC_report[app][inj_err][classE]+=1

			check_and_create_nested_dict(IAC_report_tot,app,inj_err,Outcome)
			IAC_report_tot[app][inj_err][Outcome]+=1

			check_and_create_nested_dict(IAC_report_regsrc,app,inj_err,InjDimention,Outcome)
			IAC_report_regsrc[app][inj_err][InjDimention][Outcome]+=1
			
			if "DUE" in Outcome:
				if "TIMEOUT" not in Outcome.upper(): 
					check_and_create_nested_dict(IAC_report_reason,app,inj_err,InjDimention,Outcome,Cause)
					IAC_report_reason[app][inj_err][InjDimention][Outcome][Cause]+=1
				else:
					check_and_create_nested_dict(IAC_report_reason,app,inj_err,InjDimention,Outcome,"Timeout")
					IAC_report_reason[app][inj_err][InjDimention][Outcome]["Timeout"]+=1			
		file_data.close()
		print(IAC_report_tot)
		print(IAC_report_regsrc)
		print(IAC_report_reason)	
		

	else:
		print(f"The file: {log_file}; doesn't exist..")
	return

WV_report_reason={}
WV_report_regsrc={}
WV_report_tot={}
WV_report={}
def parse_results_WV(app,log_file):

	inj_err="WV"
	if os.path.isfile(log_file):
		file_data=open(log_file,'r')
		for line in file_data:
			rep_fields=line.strip().split('$')	
			sim_runtime=float(rep_fields[5])
			Outcome=rep_fields[7].replace("Outcome","")
			Outcome=Outcome.replace("(","")
			Outcome=((Outcome.replace(")","")).strip()).lstrip()
			
			for fields in (rep_fields[8].split(';')):
				if("injSmID" in fields):
					injSMID=int(fields.split(':')[1])
				if("injSchID" in fields):
					injSchID=int(fields.split(':')[1])
				if("injWarpIDH" in fields):
					injWarpIDH=int(fields.split(':')[1])					
				if("injWarpIDL" in fields):
					injWarpIDL=int(fields.split(':')[1])
				if("injLaneID" in fields):
					injLaneID=int(fields.split(':')[1])
				if("injPredReg" in fields):
					injPredReg=int(fields.split(':')[1])
				if("injMaskSeed" in fields):
					injMaskSeed=int(fields.split(':')[1])
				if("injStuck-at" in fields):
					injStuck_at=int(fields.split(':')[1])
				if("TotErrAct" in fields):
					TotErrAct=int(fields.split(':')[1])			
				Cause=""
				if("SimEndRes" in fields):
					Cause=fields.split(":::")[1]
			DmesErrType=rep_fields[9].split(',')[0]
			DmesErr=-1
			if "Xid" in (DmesErrType):
				DmesErr=int(DmesErrType.split()[4])
			num_warps=bin(injWarpIDH).count('1')+bin(injWarpIDL).count('1')
			num_LinesPerWarp=bin(injLaneID).count('1')

			if("SDC" in Outcome.upper()):
				classE="SDC"
			elif("MASKED" in Outcome.upper()):
				classE="Masked"
			elif("TIMEOUT" in Outcome.upper()):
				classE="Timeout"
			elif("DUE" in Outcome.upper()):
				classE="DUE"
			else:
				classE="-"
			check_and_create_nested_dict(WV_report,app,inj_err,classE)
			WV_report[app][inj_err][classE]+=1

			check_and_create_nested_dict(WV_report_tot,app,inj_err,Outcome)
			WV_report_tot[app][inj_err][Outcome]+=1

			check_and_create_nested_dict(WV_report_regsrc,app,inj_err,injPredReg,Outcome)
			WV_report_regsrc[app][inj_err][injPredReg][Outcome]+=1
			
			if "DUE" in Outcome:
				if "TIMEOUT" not in Outcome.upper(): 
					check_and_create_nested_dict(WV_report_reason,app,inj_err,injPredReg,Outcome,Cause)
					WV_report_reason[app][inj_err][injPredReg][Outcome][Cause]+=1
				else:
					check_and_create_nested_dict(WV_report_reason,app,inj_err,injPredReg,Outcome,"Timeout")
					WV_report_reason[app][inj_err][injPredReg][Outcome]["Timeout"]+=1			
		file_data.close()
		print(WV_report_tot)
		print(WV_report_regsrc)
		print(WV_report_reason)	
		

	else:
		print(f"The file: {log_file}; doesn't exist..")
	return
###############################################################################
# 
# 
###############################################################################
def parse_results_app(app,inj_mode):
	log_file=p.app_log_dir[app] + "/results-mode" + inj_mode + str(p.NUM_INJECTIONS) + ".txt"
	if inj_mode=="IRA" or inj_mode=="IR":
		parse_results_IRA(app,log_file)
	elif inj_mode=="IAT":
		parse_results_IAT(app,log_file)
	elif inj_mode=="IAW":
		parse_results_IAW(app,log_file)
	elif inj_mode=="IAC":
		parse_results_IAC(app,log_file)
	elif inj_mode=="WV":
		parse_results_WV(app,log_file)
	else:
		print("Oops: This error model is not available...")




	



###############################################################################
# Main function that processes files, analyzes results and prints them to an
# xlsx file
###############################################################################
def main():

	#inj_type = os.environ['nvbitPERfi']
	#app= os.environ['BENCHMARK']		
	ERR_MOD=["IRA","IR","IAT","IAW","IAC","WV"]
	for inj_type in ERR_MOD:
		os.environ['nvbitPERfi']=inj_type
		p.set_paths()
		for app in p.apps:
			print(app)
			cf.set_env(app,False,inj_type)
			parse_results_app(app,inj_type)		

	file_csv=open("FinalReport.csv",'w')
	line=f"app,err_type,Masked,SDC,DUE,Timeout\n"
	file_csv.write(line)
	for app in p.apps:
		for err_type in ERR_MOD:
			if err_type=="IRA":
				masked=0
				sdc=0
				due=0
				timeout=0								
				for clasE in IRA_report[app][err_type]:
					if "SDC" in clasE.upper():					
						sdc=IRA_report[app][err_type][clasE]
					if "DUE" in clasE.upper():					
						due=IRA_report[app][err_type][clasE]
					if "MASKED" in clasE.upper():					
						masked=IRA_report[app][err_type][clasE]
					if "TIMEOUT" in clasE.upper():					
						timeout=IRA_report[app][err_type][clasE]				
				line=f"{app},{err_type},{masked},{sdc},{due},{timeout}\n"
				file_csv.write(line)

			if err_type=="IR":
				masked=0
				sdc=0
				due=0
				timeout=0	
				#print(IRA_report[app]["IR"]	)
				for clasE in IRA_report[app][err_type]:
					#print(clasE)
					if "SDC" in clasE.upper():					
						sdc=IRA_report[app][err_type][clasE]
					if "DUE" in clasE.upper():					
						due=IRA_report[app][err_type][clasE]
					if "MASKED" in clasE.upper():					
						masked=IRA_report[app][err_type][clasE]
					if "TIMEOUT" in clasE.upper():					
						timeout=IRA_report[app][err_type][clasE]				
				line=f"{app},{err_type},{masked},{sdc},{due},{timeout}\n"
				file_csv.write(line)

			if err_type=="IAT":
				masked=0
				sdc=0
				due=0
				timeout=0								
				for clasE in IAT_report[app][err_type]:
					if "SDC" in clasE.upper():					
						sdc=IAT_report[app][err_type][clasE]
					if "DUE" in clasE.upper():					
						due=IAT_report[app][err_type][clasE]
					if "MASKED" in clasE.upper():					
						masked=IAT_report[app][err_type][clasE]
					if "TIMEOUT" in clasE.upper():					
						timeout=IAT_report[app][err_type][clasE]				
				line=f"{app},{err_type},{masked},{sdc},{due},{timeout}\n"
				file_csv.write(line)
			if err_type=="IAW":
				masked=0
				sdc=0
				due=0
				timeout=0								
				for clasE in IAW_report[app][err_type]:
					if "SDC" in clasE.upper():					
						sdc=IAW_report[app][err_type][clasE]
					if "DUE" in clasE.upper():					
						due=IAW_report[app][err_type][clasE]
					if "MASKED" in clasE.upper():					
						masked=IAW_report[app][err_type][clasE]
					if "TIMEOUT" in clasE.upper():					
						timeout=IAW_report[app][err_type][clasE]				
				line=f"{app},{err_type},{masked},{sdc},{due},{timeout}\n"
				file_csv.write(line)
			if err_type=="IAC":
				masked=0
				sdc=0
				due=0
				timeout=0								
				for clasE in IAC_report[app][err_type]:
					if "SDC" in clasE.upper():					
						sdc=IAC_report[app][err_type][clasE]
					if "DUE" in clasE.upper():					
						due=IAC_report[app][err_type][clasE]
					if "MASKED" in clasE.upper():					
						masked=IAC_report[app][err_type][clasE]
					if "TIMEOUT" in clasE.upper():					
						timeout=IAC_report[app][err_type][clasE]				
				line=f"{app},{err_type},{masked},{sdc},{due},{timeout}\n"
				file_csv.write(line)

			if err_type=="WV":
				masked=0
				sdc=0
				due=0
				timeout=0								
				for clasE in WV_report[app][err_type]:
					if "SDC" in clasE.upper():					
						sdc=WV_report[app][err_type][clasE]
					if "DUE" in clasE.upper():					
						due=WV_report[app][err_type][clasE]
					if "MASKED" in clasE.upper():					
						masked=WV_report[app][err_type][clasE]
					if "TIMEOUT" in clasE.upper():					
						timeout=WV_report[app][err_type][clasE]				
				line=f"{app},{err_type},{masked},{sdc},{due},{timeout}\n"
				file_csv.write(line)
	print(IRA_report)
	print(IAT_report)
	print(IAW_report)
	print(IAC_report)
	print(WV_report)
	file_csv.close()


if __name__ == "__main__":
    main()
