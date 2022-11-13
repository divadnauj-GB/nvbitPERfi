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
inj_mode = ""

#################################################################
# Generate injection list of each
#   - app
#   - instruction group 
#   - bit-flip model
#################################################################
def write_injection_list_file(app, inj_mode, igid, bfm, num_injections, total_count, countList):
    if verbose:
        print ("total_count = %d, num_injections = %d" %(total_count, num_injections))
    fName = p.app_log_dir[app] + "/injection-list/mode" + inj_mode + "-igid" + str(igid) + ".bfm" + str(bfm) + "." + str(num_injections) + ".txt"
    print (fName)
    f = open(fName, "w")

    while num_injections > 0 and  total_count != 0: # first two are kname and kcount
        num_injections -= 1
        injection_num = random.randint(0, total_count) # randomly select an injection index
        if igid == "rf":
            [inj_kname, inj_kcount, inj_icount] = cf.get_rf_injection_site_info(countList, injection_num, True) # convert injection index to [kname, kernel count, inst index]
            inj_op_id_seed = p.num_regs[app][inj_kname]*random.random() # register selection
        else:
            [inj_kname, inj_kcount, inj_icount] = cf.get_injection_site_info(countList, injection_num, igid) # convert injection index to [kname, kernel count, inst index]
            inj_op_id_seed = random.random()
        inj_bid_seed = random.random() 
        selected_str = inj_kname + " " + str(inj_kcount) + " " + str(inj_icount) + " " + str(inj_op_id_seed) + " " + str(inj_bid_seed) + " "
        if verbose:
            print ("%d/%d: Selected: %s" %(num_injections, total_count, selected_str))
        f.write(selected_str + "\n") # print injection site information
    f.close()

#################################################################
# Generate injection list of each app for 
# (1) RF AVF (for each error model)
# (2) instruction-level value injections (for each error model and instruction type)
# (3) instruction-level address injections (for each error model and instruction type)
#################################################################
def gen_lists(app, countList, inj_mode):
    if inj_mode == p.RF_MODE: # RF injection list
        total_count = cf.get_total_insts(countList, True) if inj_mode == p.RF_MODE else cf.get_total_insts(countList, False)
        for bfm in p.rf_bfm_list:
            write_injection_list_file(app, inj_mode, "rf", bfm, MAX_INJ, total_count, countList)
    elif inj_mode == p.INST_VALUE_MODE: # instruction value injections
        total_icounts = cf.get_total_counts(countList)
        for igid in p.inst_value_igid_bfm_map:
            for bfm in p.inst_value_igid_bfm_map[igid]: 
                write_injection_list_file(app, inj_mode, igid, bfm, MAX_INJ, total_icounts[igid - p.NUM_INST_GROUPS], countList)
    elif inj_mode == p.INST_ADDRESS_MODE: # instruction value injections
        total_icounts = cf.get_total_counts(countList)
        for igid in p.inst_address_igid_bfm_map:
            for bfm in p.inst_address_igid_bfm_map[igid]: 
                write_injection_list_file(app, inj_mode, igid, bfm, MAX_INJ, total_icounts[igid - p.NUM_INST_GROUPS], countList)


def get_MaxRegPerThread(app):
    maxreg = 0
    fName = p.app_log_dir[app] + "/" + p.nvbit_profile_log 
    if not os.path.exists(fName):
        print ("%s file not found!" %fName )
        return maxreg

    f = open(fName, "r")
    for line in f:
        # NVBit-igprofile; index: 0; kernel_name: _Z10simple_addi; ctas: 10; instrs: 409600; FADD: 0, FADD32I: 0, FCHK: 0, FCMP: 0, FFMA: 0, FFMA32I: 0, FMNMX: 0, FMUL: 0, FMUL32I: 0, FSET: 0, FSETP: 0, FSWZADD: 0, IPA: 0, MUFU: 0, RRO: 0, DADD: 0, DFMA: 0, DMNMX: 0, DMUL: 0, DSET: 0, DSETP: 0, HADD2: 0, HADD2_32I: 0, HFMA2: 0, HFMA2_32I: 0, HMUL2: 0, HMUL2_32I: 0, HSET2: 0, HSETP2: 0, IDP: 0, BFE: 0, BFI: 0, FLO: 10240, IADD: 0, IADD3: 0, IADD32I: 30720, ICMP: 0, IMAD: 0, IMAD32I: 0, IMADSP: 0, IMNMX: 0, IMUL: 0, IMUL32I: 0, ISCADD: 0, ISCADD32I: 0, ISET: 0, ISETP: 30720, LEA: 0, LOP: 0, LOP3: 0, LOP32I: 0, POPC: 10240, SHF: 0, SHL: 0, SHR: 0, XMAD: 30720, VABSDIFF: 40960, VADD: 0, VMAD: 0, VMNMX: 0, VSET: 0, VSETP: 0, VSHL: 0, VSHR: 0, VABSDIFF4: 0, F2F: 0, F2I: 0, I2F: 0, I2I: 0, MOV: 122880, MOV32I: 40960, PRMT: 0, SEL: 0, SHFL: 0, CSET: 0, CSETP: 0, PSET: 0, PSETP: 0, P2R: 0, R2P: 0, TEX: 0, TLD: 0, TLD4: 0, TMML: 0, TXA: 0, TXD: 0, TXQ: 0, TEXS: 0, TLD4S: 0, TLDS: 0, STP: 0, LD: 0, LDC: 0, LDG: 0, LDL: 0, LDS: 0, ST: 0, STG: 0, STL: 0, STS: 0, ATOM: 0, ATOMS: 0, RED: 320, CCTL: 0, CCTLL: 0, MEMBAR: 0, CCTLT: 0, SUATOM: 0, SULD: 0, SURED: 0, SUST: 0, BRA: 61120, BRX: 0, JMP: 0, JMX: 0, SSY: 0, SYNC: 0, CAL: 0, JCAL: 0, PRET: 0, RET: 0, BRK: 0, PBK: 0, CONT: 0, PCNT: 0, EXIT: 10240, PEXIT: 0, LONGJMP: 0, PLONGJMP: 0, KIL: 0, BPT: 0, IDE: 0, RAM: 0, RTT: 0, SAM: 0, NOP: 0, CS2R: 0, S2R: 10240, LEPC: 0, B2R: 0, BAR: 0, R2B: 0, VOTE: 10240, DEPBAR: 0, GETCRSPTR: 0, GETLMEMBASE: 0, SETCRSPTR: 0, SETLMEMBASE: 0, fp64: 0, fp32: 0, ld: 0, pr: 30720, nodest: 71680, others: 307200, gppr: 337920,
        fields=line.strip().split(';')
        tmp=0
        for field in fields:
            if "max_regcount" in field:
                tmp=int(field.strip().split(':')[1])
                break
        if (maxreg<tmp):
            maxreg=tmp        
    f.close()
    return maxreg 

def get_MaxRegOper(app):
    maxreg = 0
    fName = p.app_log_dir[app] + "/" + p.nvbit_profile_log 
    if not os.path.exists(fName):
        print ("%s file not found!" %fName )
        return maxreg

    f = open(fName, "r")
    for line in f:
        # NVBit-igprofile; index: 0; kernel_name: _Z10simple_addi; ctas: 10; instrs: 409600; FADD: 0, FADD32I: 0, FCHK: 0, FCMP: 0, FFMA: 0, FFMA32I: 0, FMNMX: 0, FMUL: 0, FMUL32I: 0, FSET: 0, FSETP: 0, FSWZADD: 0, IPA: 0, MUFU: 0, RRO: 0, DADD: 0, DFMA: 0, DMNMX: 0, DMUL: 0, DSET: 0, DSETP: 0, HADD2: 0, HADD2_32I: 0, HFMA2: 0, HFMA2_32I: 0, HMUL2: 0, HMUL2_32I: 0, HSET2: 0, HSETP2: 0, IDP: 0, BFE: 0, BFI: 0, FLO: 10240, IADD: 0, IADD3: 0, IADD32I: 30720, ICMP: 0, IMAD: 0, IMAD32I: 0, IMADSP: 0, IMNMX: 0, IMUL: 0, IMUL32I: 0, ISCADD: 0, ISCADD32I: 0, ISET: 0, ISETP: 30720, LEA: 0, LOP: 0, LOP3: 0, LOP32I: 0, POPC: 10240, SHF: 0, SHL: 0, SHR: 0, XMAD: 30720, VABSDIFF: 40960, VADD: 0, VMAD: 0, VMNMX: 0, VSET: 0, VSETP: 0, VSHL: 0, VSHR: 0, VABSDIFF4: 0, F2F: 0, F2I: 0, I2F: 0, I2I: 0, MOV: 122880, MOV32I: 40960, PRMT: 0, SEL: 0, SHFL: 0, CSET: 0, CSETP: 0, PSET: 0, PSETP: 0, P2R: 0, R2P: 0, TEX: 0, TLD: 0, TLD4: 0, TMML: 0, TXA: 0, TXD: 0, TXQ: 0, TEXS: 0, TLD4S: 0, TLDS: 0, STP: 0, LD: 0, LDC: 0, LDG: 0, LDL: 0, LDS: 0, ST: 0, STG: 0, STL: 0, STS: 0, ATOM: 0, ATOMS: 0, RED: 320, CCTL: 0, CCTLL: 0, MEMBAR: 0, CCTLT: 0, SUATOM: 0, SULD: 0, SURED: 0, SUST: 0, BRA: 61120, BRX: 0, JMP: 0, JMX: 0, SSY: 0, SYNC: 0, CAL: 0, JCAL: 0, PRET: 0, RET: 0, BRK: 0, PBK: 0, CONT: 0, PCNT: 0, EXIT: 10240, PEXIT: 0, LONGJMP: 0, PLONGJMP: 0, KIL: 0, BPT: 0, IDE: 0, RAM: 0, RTT: 0, SAM: 0, NOP: 0, CS2R: 0, S2R: 10240, LEPC: 0, B2R: 0, BAR: 0, R2B: 0, VOTE: 10240, DEPBAR: 0, GETCRSPTR: 0, GETLMEMBASE: 0, SETCRSPTR: 0, SETLMEMBASE: 0, fp64: 0, fp32: 0, ld: 0, pr: 30720, nodest: 71680, others: 307200, gppr: 337920,
        fields=line.strip().split(';')
        tmp=0
        for field in fields:
            if "max_reg_operands" in field:
                tmp=int(field.strip().split(':')[1])
                break
        if (maxreg<tmp):
            maxreg=tmp        
    f.close()
    return maxreg 

def get_BlockDim(app):
    maxreg = 0
    fName = p.app_log_dir[app] + "/" + p.nvbit_profile_log 
    if not os.path.exists(fName):
        print ("%s file not found!" %fName )
        return maxreg

    f = open(fName, "r")
    for line in f:
        # NVBit-igprofile; index: 0; kernel_name: _Z10simple_addi; ctas: 10; instrs: 409600; FADD: 0, FADD32I: 0, FCHK: 0, FCMP: 0, FFMA: 0, FFMA32I: 0, FMNMX: 0, FMUL: 0, FMUL32I: 0, FSET: 0, FSETP: 0, FSWZADD: 0, IPA: 0, MUFU: 0, RRO: 0, DADD: 0, DFMA: 0, DMNMX: 0, DMUL: 0, DSET: 0, DSETP: 0, HADD2: 0, HADD2_32I: 0, HFMA2: 0, HFMA2_32I: 0, HMUL2: 0, HMUL2_32I: 0, HSET2: 0, HSETP2: 0, IDP: 0, BFE: 0, BFI: 0, FLO: 10240, IADD: 0, IADD3: 0, IADD32I: 30720, ICMP: 0, IMAD: 0, IMAD32I: 0, IMADSP: 0, IMNMX: 0, IMUL: 0, IMUL32I: 0, ISCADD: 0, ISCADD32I: 0, ISET: 0, ISETP: 30720, LEA: 0, LOP: 0, LOP3: 0, LOP32I: 0, POPC: 10240, SHF: 0, SHL: 0, SHR: 0, XMAD: 30720, VABSDIFF: 40960, VADD: 0, VMAD: 0, VMNMX: 0, VSET: 0, VSETP: 0, VSHL: 0, VSHR: 0, VABSDIFF4: 0, F2F: 0, F2I: 0, I2F: 0, I2I: 0, MOV: 122880, MOV32I: 40960, PRMT: 0, SEL: 0, SHFL: 0, CSET: 0, CSETP: 0, PSET: 0, PSETP: 0, P2R: 0, R2P: 0, TEX: 0, TLD: 0, TLD4: 0, TMML: 0, TXA: 0, TXD: 0, TXQ: 0, TEXS: 0, TLD4S: 0, TLDS: 0, STP: 0, LD: 0, LDC: 0, LDG: 0, LDL: 0, LDS: 0, ST: 0, STG: 0, STL: 0, STS: 0, ATOM: 0, ATOMS: 0, RED: 320, CCTL: 0, CCTLL: 0, MEMBAR: 0, CCTLT: 0, SUATOM: 0, SULD: 0, SURED: 0, SUST: 0, BRA: 61120, BRX: 0, JMP: 0, JMX: 0, SSY: 0, SYNC: 0, CAL: 0, JCAL: 0, PRET: 0, RET: 0, BRK: 0, PBK: 0, CONT: 0, PCNT: 0, EXIT: 10240, PEXIT: 0, LONGJMP: 0, PLONGJMP: 0, KIL: 0, BPT: 0, IDE: 0, RAM: 0, RTT: 0, SAM: 0, NOP: 0, CS2R: 0, S2R: 10240, LEPC: 0, B2R: 0, BAR: 0, R2B: 0, VOTE: 10240, DEPBAR: 0, GETCRSPTR: 0, GETLMEMBASE: 0, SETCRSPTR: 0, SETLMEMBASE: 0, fp64: 0, fp32: 0, ld: 0, pr: 30720, nodest: 71680, others: 307200, gppr: 337920,
        fields=line.strip().split(';')
        blockDimX=0
        blockDimY=0
        blockDimZ=0
        tmp_blockDimX=0
        tmp_blockDimY=0
        tmp_blockDimZ=0
        for field in fields:
            if "blockDimX" in field:
                tmp_blockDimX=int(field.strip().split(':')[1])
            elif "blockDimY" in field:
                tmp_blockDimY=int(field.strip().split(':')[1])
            elif "blockDimZ" in field:
                tmp_blockDimZ=int(field.strip().split(':')[1])

        if(blockDimX<tmp_blockDimX):
            blockDimX=tmp_blockDimX
        if(blockDimY<tmp_blockDimY):
            blockDimY=tmp_blockDimY
        if(blockDimZ<tmp_blockDimZ):
            blockDimZ=tmp_blockDimZ
    f.close()
    return [blockDimX,blockDimY,blockDimZ] 


def get_GridDim(app):
    maxreg = 0
    fName = p.app_log_dir[app] + "/" + p.nvbit_profile_log 
    if not os.path.exists(fName):
        print ("%s file not found!" %fName )
        return maxreg

    f = open(fName, "r")
    for line in f:
        # NVBit-igprofile; index: 0; kernel_name: _Z10simple_addi; ctas: 10; instrs: 409600; FADD: 0, FADD32I: 0, FCHK: 0, FCMP: 0, FFMA: 0, FFMA32I: 0, FMNMX: 0, FMUL: 0, FMUL32I: 0, FSET: 0, FSETP: 0, FSWZADD: 0, IPA: 0, MUFU: 0, RRO: 0, DADD: 0, DFMA: 0, DMNMX: 0, DMUL: 0, DSET: 0, DSETP: 0, HADD2: 0, HADD2_32I: 0, HFMA2: 0, HFMA2_32I: 0, HMUL2: 0, HMUL2_32I: 0, HSET2: 0, HSETP2: 0, IDP: 0, BFE: 0, BFI: 0, FLO: 10240, IADD: 0, IADD3: 0, IADD32I: 30720, ICMP: 0, IMAD: 0, IMAD32I: 0, IMADSP: 0, IMNMX: 0, IMUL: 0, IMUL32I: 0, ISCADD: 0, ISCADD32I: 0, ISET: 0, ISETP: 30720, LEA: 0, LOP: 0, LOP3: 0, LOP32I: 0, POPC: 10240, SHF: 0, SHL: 0, SHR: 0, XMAD: 30720, VABSDIFF: 40960, VADD: 0, VMAD: 0, VMNMX: 0, VSET: 0, VSETP: 0, VSHL: 0, VSHR: 0, VABSDIFF4: 0, F2F: 0, F2I: 0, I2F: 0, I2I: 0, MOV: 122880, MOV32I: 40960, PRMT: 0, SEL: 0, SHFL: 0, CSET: 0, CSETP: 0, PSET: 0, PSETP: 0, P2R: 0, R2P: 0, TEX: 0, TLD: 0, TLD4: 0, TMML: 0, TXA: 0, TXD: 0, TXQ: 0, TEXS: 0, TLD4S: 0, TLDS: 0, STP: 0, LD: 0, LDC: 0, LDG: 0, LDL: 0, LDS: 0, ST: 0, STG: 0, STL: 0, STS: 0, ATOM: 0, ATOMS: 0, RED: 320, CCTL: 0, CCTLL: 0, MEMBAR: 0, CCTLT: 0, SUATOM: 0, SULD: 0, SURED: 0, SUST: 0, BRA: 61120, BRX: 0, JMP: 0, JMX: 0, SSY: 0, SYNC: 0, CAL: 0, JCAL: 0, PRET: 0, RET: 0, BRK: 0, PBK: 0, CONT: 0, PCNT: 0, EXIT: 10240, PEXIT: 0, LONGJMP: 0, PLONGJMP: 0, KIL: 0, BPT: 0, IDE: 0, RAM: 0, RTT: 0, SAM: 0, NOP: 0, CS2R: 0, S2R: 10240, LEPC: 0, B2R: 0, BAR: 0, R2B: 0, VOTE: 10240, DEPBAR: 0, GETCRSPTR: 0, GETLMEMBASE: 0, SETCRSPTR: 0, SETLMEMBASE: 0, fp64: 0, fp32: 0, ld: 0, pr: 30720, nodest: 71680, others: 307200, gppr: 337920,
        fields=line.strip().split(';')
        gridDimX=0
        gridDimY=0
        gridDimZ=0
        tmp_gridDimX=0
        tmp_gridDimY=0
        tmp_gridDimZ=0
        for field in fields:
            if "gridDimX" in field:
                tmp_gridDimX=int(field.strip().split(':')[1])
            elif "gridDimY" in field:
                tmp_gridDimY=int(field.strip().split(':')[1])
            elif "gridDimZ" in field:
                tmp_gridDimZ=int(field.strip().split(':')[1])
        if(gridDimX<tmp_gridDimX):
            gridDimX=tmp_gridDimX
        if(gridDimY<tmp_gridDimY):
            gridDimY=tmp_gridDimY
        if(gridDimZ<tmp_gridDimZ):
            gridDimZ=tmp_gridDimZ

    f.close()
    return [gridDimX,gridDimY,gridDimZ] 



def getMaxThreadPerSM(app):
    maxreg = 0
    fName = p.app_log_dir[app] + "/" + p.nvbit_profile_log 
    if not os.path.exists(fName):
        print ("%s file not found!" %fName )
        return maxreg

    f = open(fName, "r")
    for line in f:
        # NVBit-igprofile; index: 0; kernel_name: _Z10simple_addi; ctas: 10; instrs: 409600; FADD: 0, FADD32I: 0, FCHK: 0, FCMP: 0, FFMA: 0, FFMA32I: 0, FMNMX: 0, FMUL: 0, FMUL32I: 0, FSET: 0, FSETP: 0, FSWZADD: 0, IPA: 0, MUFU: 0, RRO: 0, DADD: 0, DFMA: 0, DMNMX: 0, DMUL: 0, DSET: 0, DSETP: 0, HADD2: 0, HADD2_32I: 0, HFMA2: 0, HFMA2_32I: 0, HMUL2: 0, HMUL2_32I: 0, HSET2: 0, HSETP2: 0, IDP: 0, BFE: 0, BFI: 0, FLO: 10240, IADD: 0, IADD3: 0, IADD32I: 30720, ICMP: 0, IMAD: 0, IMAD32I: 0, IMADSP: 0, IMNMX: 0, IMUL: 0, IMUL32I: 0, ISCADD: 0, ISCADD32I: 0, ISET: 0, ISETP: 30720, LEA: 0, LOP: 0, LOP3: 0, LOP32I: 0, POPC: 10240, SHF: 0, SHL: 0, SHR: 0, XMAD: 30720, VABSDIFF: 40960, VADD: 0, VMAD: 0, VMNMX: 0, VSET: 0, VSETP: 0, VSHL: 0, VSHR: 0, VABSDIFF4: 0, F2F: 0, F2I: 0, I2F: 0, I2I: 0, MOV: 122880, MOV32I: 40960, PRMT: 0, SEL: 0, SHFL: 0, CSET: 0, CSETP: 0, PSET: 0, PSETP: 0, P2R: 0, R2P: 0, TEX: 0, TLD: 0, TLD4: 0, TMML: 0, TXA: 0, TXD: 0, TXQ: 0, TEXS: 0, TLD4S: 0, TLDS: 0, STP: 0, LD: 0, LDC: 0, LDG: 0, LDL: 0, LDS: 0, ST: 0, STG: 0, STL: 0, STS: 0, ATOM: 0, ATOMS: 0, RED: 320, CCTL: 0, CCTLL: 0, MEMBAR: 0, CCTLT: 0, SUATOM: 0, SULD: 0, SURED: 0, SUST: 0, BRA: 61120, BRX: 0, JMP: 0, JMX: 0, SSY: 0, SYNC: 0, CAL: 0, JCAL: 0, PRET: 0, RET: 0, BRK: 0, PBK: 0, CONT: 0, PCNT: 0, EXIT: 10240, PEXIT: 0, LONGJMP: 0, PLONGJMP: 0, KIL: 0, BPT: 0, IDE: 0, RAM: 0, RTT: 0, SAM: 0, NOP: 0, CS2R: 0, S2R: 10240, LEPC: 0, B2R: 0, BAR: 0, R2B: 0, VOTE: 10240, DEPBAR: 0, GETCRSPTR: 0, GETLMEMBASE: 0, SETCRSPTR: 0, SETLMEMBASE: 0, fp64: 0, fp32: 0, ld: 0, pr: 30720, nodest: 71680, others: 307200, gppr: 337920,
        fields=line.strip().split(';')
        maxThreadsPerSM=0
        for field in fields:
            if "maxThreadsPerSM" in field:
                maxThreadsPerSM=int(field.strip().split(':')[1])       
    f.close()
    return maxThreadsPerSM


def getMaxWarpsPerSM(app):
    maxreg = 0
    fName = p.app_log_dir[app] + "/" + p.nvbit_profile_log 
    if not os.path.exists(fName):
        print ("%s file not found!" %fName )
        return maxreg

    f = open(fName, "r")
    for line in f:
        # NVBit-igprofile; index: 0; kernel_name: _Z10simple_addi; ctas: 10; instrs: 409600; FADD: 0, FADD32I: 0, FCHK: 0, FCMP: 0, FFMA: 0, FFMA32I: 0, FMNMX: 0, FMUL: 0, FMUL32I: 0, FSET: 0, FSETP: 0, FSWZADD: 0, IPA: 0, MUFU: 0, RRO: 0, DADD: 0, DFMA: 0, DMNMX: 0, DMUL: 0, DSET: 0, DSETP: 0, HADD2: 0, HADD2_32I: 0, HFMA2: 0, HFMA2_32I: 0, HMUL2: 0, HMUL2_32I: 0, HSET2: 0, HSETP2: 0, IDP: 0, BFE: 0, BFI: 0, FLO: 10240, IADD: 0, IADD3: 0, IADD32I: 30720, ICMP: 0, IMAD: 0, IMAD32I: 0, IMADSP: 0, IMNMX: 0, IMUL: 0, IMUL32I: 0, ISCADD: 0, ISCADD32I: 0, ISET: 0, ISETP: 30720, LEA: 0, LOP: 0, LOP3: 0, LOP32I: 0, POPC: 10240, SHF: 0, SHL: 0, SHR: 0, XMAD: 30720, VABSDIFF: 40960, VADD: 0, VMAD: 0, VMNMX: 0, VSET: 0, VSETP: 0, VSHL: 0, VSHR: 0, VABSDIFF4: 0, F2F: 0, F2I: 0, I2F: 0, I2I: 0, MOV: 122880, MOV32I: 40960, PRMT: 0, SEL: 0, SHFL: 0, CSET: 0, CSETP: 0, PSET: 0, PSETP: 0, P2R: 0, R2P: 0, TEX: 0, TLD: 0, TLD4: 0, TMML: 0, TXA: 0, TXD: 0, TXQ: 0, TEXS: 0, TLD4S: 0, TLDS: 0, STP: 0, LD: 0, LDC: 0, LDG: 0, LDL: 0, LDS: 0, ST: 0, STG: 0, STL: 0, STS: 0, ATOM: 0, ATOMS: 0, RED: 320, CCTL: 0, CCTLL: 0, MEMBAR: 0, CCTLT: 0, SUATOM: 0, SULD: 0, SURED: 0, SUST: 0, BRA: 61120, BRX: 0, JMP: 0, JMX: 0, SSY: 0, SYNC: 0, CAL: 0, JCAL: 0, PRET: 0, RET: 0, BRK: 0, PBK: 0, CONT: 0, PCNT: 0, EXIT: 10240, PEXIT: 0, LONGJMP: 0, PLONGJMP: 0, KIL: 0, BPT: 0, IDE: 0, RAM: 0, RTT: 0, SAM: 0, NOP: 0, CS2R: 0, S2R: 10240, LEPC: 0, B2R: 0, BAR: 0, R2B: 0, VOTE: 10240, DEPBAR: 0, GETCRSPTR: 0, GETLMEMBASE: 0, SETCRSPTR: 0, SETLMEMBASE: 0, fp64: 0, fp32: 0, ld: 0, pr: 30720, nodest: 71680, others: 307200, gppr: 337920,
        fields=line.strip().split(';')
        maxWarpsPerSM=0
        for field in fields:
            if "maxWarpsPerSM" in field:
                maxWarpsPerSM=int(field.strip().split(':')[1])          
    f.close()
    return maxWarpsPerSM


def getMaxDim(app):
    maxreg = 0
    fName = p.app_log_dir[app] + "/" + p.nvbit_profile_log 
    if not os.path.exists(fName):
        print ("%s file not found!" %fName )
        return maxreg

    f = open(fName, "r")
    for line in f:
        # NVBit-igprofile; index: 0; kernel_name: _Z10simple_addi; ctas: 10; instrs: 409600; FADD: 0, FADD32I: 0, FCHK: 0, FCMP: 0, FFMA: 0, FFMA32I: 0, FMNMX: 0, FMUL: 0, FMUL32I: 0, FSET: 0, FSETP: 0, FSWZADD: 0, IPA: 0, MUFU: 0, RRO: 0, DADD: 0, DFMA: 0, DMNMX: 0, DMUL: 0, DSET: 0, DSETP: 0, HADD2: 0, HADD2_32I: 0, HFMA2: 0, HFMA2_32I: 0, HMUL2: 0, HMUL2_32I: 0, HSET2: 0, HSETP2: 0, IDP: 0, BFE: 0, BFI: 0, FLO: 10240, IADD: 0, IADD3: 0, IADD32I: 30720, ICMP: 0, IMAD: 0, IMAD32I: 0, IMADSP: 0, IMNMX: 0, IMUL: 0, IMUL32I: 0, ISCADD: 0, ISCADD32I: 0, ISET: 0, ISETP: 30720, LEA: 0, LOP: 0, LOP3: 0, LOP32I: 0, POPC: 10240, SHF: 0, SHL: 0, SHR: 0, XMAD: 30720, VABSDIFF: 40960, VADD: 0, VMAD: 0, VMNMX: 0, VSET: 0, VSETP: 0, VSHL: 0, VSHR: 0, VABSDIFF4: 0, F2F: 0, F2I: 0, I2F: 0, I2I: 0, MOV: 122880, MOV32I: 40960, PRMT: 0, SEL: 0, SHFL: 0, CSET: 0, CSETP: 0, PSET: 0, PSETP: 0, P2R: 0, R2P: 0, TEX: 0, TLD: 0, TLD4: 0, TMML: 0, TXA: 0, TXD: 0, TXQ: 0, TEXS: 0, TLD4S: 0, TLDS: 0, STP: 0, LD: 0, LDC: 0, LDG: 0, LDL: 0, LDS: 0, ST: 0, STG: 0, STL: 0, STS: 0, ATOM: 0, ATOMS: 0, RED: 320, CCTL: 0, CCTLL: 0, MEMBAR: 0, CCTLT: 0, SUATOM: 0, SULD: 0, SURED: 0, SUST: 0, BRA: 61120, BRX: 0, JMP: 0, JMX: 0, SSY: 0, SYNC: 0, CAL: 0, JCAL: 0, PRET: 0, RET: 0, BRK: 0, PBK: 0, CONT: 0, PCNT: 0, EXIT: 10240, PEXIT: 0, LONGJMP: 0, PLONGJMP: 0, KIL: 0, BPT: 0, IDE: 0, RAM: 0, RTT: 0, SAM: 0, NOP: 0, CS2R: 0, S2R: 10240, LEPC: 0, B2R: 0, BAR: 0, R2B: 0, VOTE: 10240, DEPBAR: 0, GETCRSPTR: 0, GETLMEMBASE: 0, SETCRSPTR: 0, SETLMEMBASE: 0, fp64: 0, fp32: 0, ld: 0, pr: 30720, nodest: 71680, others: 307200, gppr: 337920,
        fields=line.strip().split(';')
        maxGridX=0
        maxGridY=0
        maxGridZ=0
        for field in fields:
            if "maxDimX" in field:
                maxGridX=int(field.strip().split(':')[1])  
            if "maxDimY" in field:
                maxGridY=int(field.strip().split(':')[1])   
            if "maxDimZ" in field:
                maxGridZ=int(field.strip().split(':')[1])   
    f.close()
    return [maxGridX, maxGridY, maxGridZ]

def getMaxGrid(app):
    maxreg = 0
    fName = p.app_log_dir[app] + "/" + p.nvbit_profile_log 
    if not os.path.exists(fName):
        print ("%s file not found!" %fName )
        return maxreg

    f = open(fName, "r")
    for line in f:
        # NVBit-igprofile; index: 0; kernel_name: _Z10simple_addi; ctas: 10; instrs: 409600; FADD: 0, FADD32I: 0, FCHK: 0, FCMP: 0, FFMA: 0, FFMA32I: 0, FMNMX: 0, FMUL: 0, FMUL32I: 0, FSET: 0, FSETP: 0, FSWZADD: 0, IPA: 0, MUFU: 0, RRO: 0, DADD: 0, DFMA: 0, DMNMX: 0, DMUL: 0, DSET: 0, DSETP: 0, HADD2: 0, HADD2_32I: 0, HFMA2: 0, HFMA2_32I: 0, HMUL2: 0, HMUL2_32I: 0, HSET2: 0, HSETP2: 0, IDP: 0, BFE: 0, BFI: 0, FLO: 10240, IADD: 0, IADD3: 0, IADD32I: 30720, ICMP: 0, IMAD: 0, IMAD32I: 0, IMADSP: 0, IMNMX: 0, IMUL: 0, IMUL32I: 0, ISCADD: 0, ISCADD32I: 0, ISET: 0, ISETP: 30720, LEA: 0, LOP: 0, LOP3: 0, LOP32I: 0, POPC: 10240, SHF: 0, SHL: 0, SHR: 0, XMAD: 30720, VABSDIFF: 40960, VADD: 0, VMAD: 0, VMNMX: 0, VSET: 0, VSETP: 0, VSHL: 0, VSHR: 0, VABSDIFF4: 0, F2F: 0, F2I: 0, I2F: 0, I2I: 0, MOV: 122880, MOV32I: 40960, PRMT: 0, SEL: 0, SHFL: 0, CSET: 0, CSETP: 0, PSET: 0, PSETP: 0, P2R: 0, R2P: 0, TEX: 0, TLD: 0, TLD4: 0, TMML: 0, TXA: 0, TXD: 0, TXQ: 0, TEXS: 0, TLD4S: 0, TLDS: 0, STP: 0, LD: 0, LDC: 0, LDG: 0, LDL: 0, LDS: 0, ST: 0, STG: 0, STL: 0, STS: 0, ATOM: 0, ATOMS: 0, RED: 320, CCTL: 0, CCTLL: 0, MEMBAR: 0, CCTLT: 0, SUATOM: 0, SULD: 0, SURED: 0, SUST: 0, BRA: 61120, BRX: 0, JMP: 0, JMX: 0, SSY: 0, SYNC: 0, CAL: 0, JCAL: 0, PRET: 0, RET: 0, BRK: 0, PBK: 0, CONT: 0, PCNT: 0, EXIT: 10240, PEXIT: 0, LONGJMP: 0, PLONGJMP: 0, KIL: 0, BPT: 0, IDE: 0, RAM: 0, RTT: 0, SAM: 0, NOP: 0, CS2R: 0, S2R: 10240, LEPC: 0, B2R: 0, BAR: 0, R2B: 0, VOTE: 10240, DEPBAR: 0, GETCRSPTR: 0, GETLMEMBASE: 0, SETCRSPTR: 0, SETLMEMBASE: 0, fp64: 0, fp32: 0, ld: 0, pr: 30720, nodest: 71680, others: 307200, gppr: 337920,
        fields=line.strip().split(';')
        maxGridX=0
        maxGridY=0
        maxGridZ=0
        for field in fields:
            if "maxGridX" in field:
                maxGridX=int(field.strip().split(':')[1])  
            if "maxGridY" in field:
                maxGridY=int(field.strip().split(':')[1])   
            if "maxGridZ" in field:
                maxGridZ=int(field.strip().split(':')[1])   
    f.close()
    return [maxGridX, maxGridY, maxGridZ]




def gen_IRA_fault_list(app,inj_mode,num_injections,regcount,opercount):
    MaxWarpSize=getMaxWarpsPerSM(app)
    NumSch=4
    error_list=[]
    Warps=[]

    if verbose:
        print ("num_injections = %d" %(num_injections))
    fName = p.app_log_dir[app] + "/injection-list/mode" + inj_mode+str(num_injections) + ".txt"
    print (fName)
    f = open(fName, "w")
    smid=int(os.environ['SMID'])
    schid=int(os.environ['SCHID'])

    for i in range(0,MaxWarpSize):
        if((i%4)==schid):
            Warps.append(1)
        else:
            Warps.append(0)
    WarpH=0
    WarpL=0
    for i in range(0,32):
        if(Warps[i]==1):
            tmp=1
            tmp=tmp<<i
            WarpL=WarpL | tmp
    for i in range(32,MaxWarpSize):
        if(Warps[i]==1):
            tmp=1
            tmp=tmp<<(i-32)
            WarpH=WarpH | tmp
    Threads=0xffffffff

    if num_injections==10: #debug purposes
        error =f"{smid} {schid} {WarpH} {WarpL} {Threads} {2} {2}\n"
        error_list.append(error)
        f.write(error) # print injection site information
        num_injections-=1

    for i in range(opercount):
        for j in range(256):
    #while num_injections>0:        
        #errMask=random.randint(0,255)
            errMask=j
            #errOperField=random.randint(0,(opercount-1))
            errOperField=i

            error =f"{smid} {schid} {WarpH} {WarpL} {Threads} {errMask} {errOperField}\n"
            #print(Warps,WarpH, WarpL, error)
            
            if error not in error_list:
                error_list.append(error)
                f.write(error) # print injection site information
                num_injections-=1
            if(num_injections<1):
                exit()
        if(num_injections<1):
            exit()

    f.close()


def gen_IAT_fault_list(app,inj_mode,num_injections,blockDim):
    MaxWarpSize=getMaxWarpsPerSM(app)
    [maxDimx, maxDimy, maxDimz] = getMaxDim(app)
    ValidWarps=int((max(blockDim)/32))
    ValidWarps=MaxWarpSize
    NumSch=4
    error_list=[]
    

    if verbose:
        print ("num_injections = %d" %(num_injections))
    fName = p.app_log_dir[app] + "/injection-list/mode" + inj_mode+str(num_injections) + ".txt"
    print (fName)
    f = open(fName, "w")
    smid=int(os.environ['SMID'])
    schid=int(os.environ['SCHID'])

    while num_injections>0:
        Warps=[0]*MaxWarpSize

        ErrWarps=random.randint(1,(ValidWarps)/4)
        while(ErrWarps>0):
            warp=random.randint(0,(ValidWarps-1))  
            while((warp%4)!=schid or Warps[warp]!=0):           
                warp=random.randint(0,(ValidWarps-1))
            Warps[warp]=1
            ErrWarps-=1

        WarpH=0
        WarpL=0

        for i in range(0,32):
            if(Warps[i]==1):
                tmp=1
                tmp=tmp<<i
                WarpL=WarpL | tmp
        for i in range(32,MaxWarpSize):
            if(Warps[i]==1):
                tmp=1
                tmp=tmp<<(i-32)
                WarpH=WarpH | tmp
        
        numThreads=random.randint(1,31)
        n=0
        val=0
        while(n!=numThreads):
            val=val | (1<<random.randint(0,31))
            n=bin(val).count("1")
        
        #print(bin(val))

        Threads=val

        #TargetWarp=random.randint(0,(MaxWarpSize-1))  
        #while((TargetWarp==warp)):           
        #    TargetWarp=random.randint(0,(MaxWarpSize-1))
        [DimIDx,DimIDy,DimIDz]=get_BlockDim(app)

        dim=random.randint(0,2)
           
        if(DimIDz>1):
            if(dim==2): # error in the dimention z
                MaskSeed=random.randint(1,maxDimz)
        else:
            dim=random.randint(0,1)

        if(DimIDy>1):
            if(dim==1): # error in the dimention Y
                MaskSeed=random.randint(1,maxDimy)
        else:
            dim=0

        if(DimIDx>1):   
            if(dim==0): #error in the dimention x
                MaskSeed=random.randint(1,maxDimx)
        else:
            dim=2

        if(DimIDz>1):
            if(dim==2): # error in the dimention z
                MaskSeed=random.randint(1,maxDimz)
        else:
            dim=0
            MaskSeed=random.randint(1,maxDimx)
            
        """
        dim=random.randint(0,2)
        if(dim==0): #error in the dimention x
            MaskSeed=random.randint(1,maxDimx-1)
        if(dim==1): # error in the dimention Y
            MaskSeed=random.randint(1,maxDimy-1)
        if(dim==2): # error in the dimention z
            MaskSeed=random.randint(1,maxDimz-1)
        """
        active=random.randint(0,1)  #Always Inactive thread
        error =f"{smid} {schid} {WarpH} {WarpL} {Threads} {MaskSeed} {dim} {active}\n"
        #print(Warps,WarpH, WarpL, error)
        
        if error not in error_list:
            error_list.append(error)
            f.write(error) # print injection site information
            num_injections-=1

    f.close()


def gen_IAW_fault_list(app,inj_mode,num_injections,blockDim):
    MaxWarpSize=getMaxWarpsPerSM(app)
    [maxDimx, maxDimy, maxDimz] = getMaxDim(app)
    ValidWarps=int((max(blockDim)/32))
    ValidWarps=MaxWarpSize
    NumSch=4
    error_list=[]
    

    if verbose:
        print ("num_injections = %d" %(num_injections))
    fName = p.app_log_dir[app] + "/injection-list/mode" + inj_mode+str(num_injections) + ".txt"
    print (fName)
    f = open(fName, "w")
    smid=int(os.environ['SMID'])
    schid=int(os.environ['SCHID'])

    while num_injections>0:
        Warps=[0]*MaxWarpSize

        ErrWarps=random.randint(1,(ValidWarps)/4)
        while(ErrWarps>0):
            warp=random.randint(0,(ValidWarps-1))  
            while((warp%4)!=schid or Warps[warp]!=0):           
                warp=random.randint(0,(ValidWarps-1))
            Warps[warp]=1
            ErrWarps-=1

        WarpH=0
        WarpL=0

        for i in range(0,32):
            if(Warps[i]==1):
                tmp=1
                tmp=tmp<<i
                WarpL=WarpL | tmp
        for i in range(32,MaxWarpSize):
            if(Warps[i]==1):
                tmp=1
                tmp=tmp<<(i-32)
                WarpH=WarpH | tmp
        
        Threads=0xffffffff

        #TargetWarp=random.randint(0,(MaxWarpSize-1))  
        #while((TargetWarp==warp)):           
        #    TargetWarp=random.randint(0,(MaxWarpSize-1))      
        [DimIDx,DimIDy,DimIDz]=get_BlockDim(app)

        dim=random.randint(0,2)
           
        if(DimIDz>1):
            if(dim==2): # error in the dimention z
                MaskSeed=random.randint(1,maxDimz)
        else:
            dim=random.randint(0,1)

        if(DimIDy>1):
            if(dim==1): # error in the dimention Y
                MaskSeed=random.randint(1,maxDimy)
        else:
            dim=0

        if(DimIDx>1):   
            if(dim==0): #error in the dimention x
                MaskSeed=random.randint(1,maxDimx)
        else:
            dim=2

        if(DimIDz>1):
            if(dim==2): # error in the dimention z
                MaskSeed=random.randint(1,maxDimz)
        else:
            dim=0
            MaskSeed=random.randint(1,maxDimx)

        """
        dim=random.randint(0,2)
        if(dim==0): #error in the dimention x
            MaskSeed=random.randint(1,maxDimx-1)
        if(dim==1): # error in the dimention Y
            MaskSeed=random.randint(1,maxDimy-1)
        if(dim==2): # error in the dimention z
            MaskSeed=random.randint(1,maxDimz-1)
        """

        active=random.randint(0,1)  #Always Inactive thread
        error =f"{smid} {schid} {WarpH} {WarpL} {Threads} {MaskSeed} {dim} {active}\n"
        #print(Warps,WarpH, WarpL, error)
        
        if error not in error_list:
            error_list.append(error)
            f.write(error) # print injection site information
            num_injections-=1
    f.close()


def gen_IAC_fault_list(app,inj_mode,num_injections,blockDim):
    MaxWarpSize=getMaxWarpsPerSM(app)
    [maxGridx, maxGridy, maxGridz] = getMaxGrid(app)
    ValidWarps=int((max(blockDim)/32))
    ValidWarps=MaxWarpSize
    NumSch=4
    error_list=[]
    

    if verbose:
        print ("num_injections = %d" %(num_injections))
    fName = p.app_log_dir[app] + "/injection-list/mode" + inj_mode+str(num_injections) + ".txt"
    print (fName)
    f = open(fName, "w")
    smid=int(os.environ['SMID'])
    schid=int(os.environ['SCHID'])

    while num_injections>0:
        Warps=[0]*MaxWarpSize

        ErrWarps=random.randint(1,(ValidWarps)/4)
        while(ErrWarps>0):
            warp=random.randint(0,(ValidWarps-1))  
            while((warp%4)!=schid or Warps[warp]!=0):           
                warp=random.randint(0,(ValidWarps-1))
            Warps[warp]=1
            ErrWarps-=1

        WarpH=0
        WarpL=0

        for i in range(0,32):
            if(Warps[i]==1):
                tmp=1
                tmp=tmp<<i
                WarpL=WarpL | tmp
        for i in range(32,MaxWarpSize):
            if(Warps[i]==1):
                tmp=1
                tmp=tmp<<(i-32)
                WarpH=WarpH | tmp
        
        Threads=0xffffffff

        #TargetWarp=random.randint(0,(MaxWarpSize-1))  
        #while((TargetWarp==warp)):           
        #    TargetWarp=random.randint(0,(MaxWarpSize-1))

        [gridIDx,gridIDy,gridIDz]=get_GridDim(app)

        dim=random.randint(0,2)
           
        if(gridIDz>1):
            if(dim==2): # error in the dimention z
                MaskSeed=random.randint(1,maxGridz)
        else:
            dim=random.randint(0,1)

        if(gridIDy>1):
            if(dim==1): # error in the dimention Y
                MaskSeed=random.randint(1,maxGridy)
        else:
            dim=0

        if(gridIDx>1):   
            if(dim==0): #error in the dimention x
                MaskSeed=random.randint(1,maxGridx)
        else:
            dim=2

        if(gridIDz>1):
            if(dim==2): # error in the dimention z
                MaskSeed=random.randint(1,maxGridz)
        else:
            dim=0
            MaskSeed=random.randint(1,maxGridx)

        active=random.randint(0,1)  #Always Inactive thread
        error =f"{smid} {schid} {WarpH} {WarpL} {Threads} {MaskSeed} {dim} {active}\n"
        #print(Warps,WarpH, WarpL, error)
        
        if error not in error_list:
            error_list.append(error)
            f.write(error) # print injection site information
            num_injections-=1

    f.close()

def gen_WV_fault_list(app,inj_mode,num_injections,blockDim):
    MaxWarpSize=getMaxWarpsPerSM(app)
    [maxGridx, maxGridy, maxGridz] = getMaxGrid(app)
    ValidWarps=int((max(blockDim)/32))
    ValidWarps=MaxWarpSize
    NumSch=4
    error_list=[]
    

    if verbose:
        print ("num_injections = %d" %(num_injections))
    fName = p.app_log_dir[app] + "/injection-list/mode" + inj_mode+str(num_injections) + ".txt"
    print (fName)
    f = open(fName, "w")
    smid=int(os.environ['SMID'])
    schid=int(os.environ['SCHID'])

    while num_injections>0:
        Warps=[0]*MaxWarpSize
        #for i in range(0,MaxWarpSize):
        #    if((i%4)==schid):
        #        Warps[i]=1

        ErrWarps=random.randint(1,(ValidWarps)/4)
        while(ErrWarps>0):
            warp=random.randint(0,(ValidWarps-1))  
            while((warp%4)!=schid or Warps[warp]!=0):           
                warp=random.randint(0,(ValidWarps-1))
            Warps[warp]=1
            ErrWarps=ErrWarps-1

        WarpH=0
        WarpL=0
        
        
        for i in range(0,32):
            if(Warps[i]==1):
                tmp=1
                tmp=tmp<<i
                WarpL=WarpL | tmp
        for i in range(32,MaxWarpSize):
            if(Warps[i]==1):
                tmp=1
                tmp=tmp<<(i-32)
                WarpH=WarpH | tmp
        
        numThreads=random.randint(1,31)
        n=0
        val=0
        while(n!=numThreads):
            val=val | (1<<random.randint(0,31))
            n=bin(val).count("1")

        Threads=val

        #TargetWarp=random.randint(0,(MaxWarpSize-1))  
        #while((TargetWarp==warp)):           
        #    TargetWarp=random.randint(0,(MaxWarpSize-1))

        predReg=random.randint(0,8)
        while(predReg==7):
            predReg=random.randint(0,8)

        if(predReg==8):
            MaskSeed=random.randint(0,255)
        else:
            MaskSeed=(1<<(predReg))
        
        active=random.randint(0,1)  #Always Inactive thread
        error =f"{smid} {schid} {WarpH} {WarpL} {Threads} {predReg} {MaskSeed} {active}\n"
        #print(Warps,WarpH, WarpL, error)
        
        if error not in error_list:
            error_list.append(error)
            f.write(error) # print injection site information
            num_injections-=1

    f.close()


def gen_ICOC_fault_list(app, inj_mode_str, num_injections):
    max_warp_per_sm = getMaxWarpsPerSM(app)
    scheduler, decoder, fetch = range(3)
    subpartitions = [scheduler, decoder, fetch]
    is_iio_fault_model = int(inj_mode_str == 'IIO')
    if verbose:
        print("num_injections =", num_injections)
    f_name = p.app_log_dir[app] + "/injection-list/mode" + inj_mode_str + str(num_injections) + ".txt"
    print(f_name)
    with open(f_name, "w") as f:
        sm_id, scheduler_id = int(os.environ['SMID']), int(os.environ['SCHID'])
        for _ in range(num_injections):
            warp_id = random.choice([w_id_i for w_id_i in range(max_warp_per_sm) if (w_id_i % 4) == scheduler_id])
            icoc_subpartition = random.choice(subpartitions)
            err_string = f"{sm_id} {scheduler_id} {icoc_subpartition} {warp_id} {is_iio_fault_model}\n"
            f.write(err_string)  # print injection site information


#################################################################
# Starting point of the script
#################################################################
def main():
    # if len(sys.argv) == 2: 
    # 	inj_mode = sys.argv[1] # rf or inst_value or inst_address
    # else:
    # 	print ("Usage: ./script-name <rf or inst>")
    # 	print ("Only one mode is currently supported: inst_value")
    # 	exit(1)

    inj_mode = os.environ['nvbitPERfi']     
    # actual code that generates list per app is here
    for app in p.apps:
        if(app==os.environ['BENCHMARK']): 
            print ("\nCreating list for %s ... " %(app))
            os.system("mkdir -p %s/injection-list" %(p.app_log_dir[app])) # create directory to store injection list

            if inj_mode in ['ICOC', 'IIO']:
                gen_ICOC_fault_list(app=app, inj_mode_str=inj_mode, num_injections=p.NUM_INJECTIONS)
            elif inj_mode=='IRA' or inj_mode=='IR':
                regcount =  get_MaxRegPerThread(app)
                opercount = get_MaxRegOper(app) 
                gen_IRA_fault_list(app,inj_mode,p.NUM_INJECTIONS,regcount,opercount)
            elif inj_mode=='IAT':
                blockDim=get_BlockDim(app)
                gen_IAT_fault_list(app,inj_mode,p.NUM_INJECTIONS,blockDim)
            elif inj_mode=='IAW':
                blockDim=get_BlockDim(app)
                gen_IAW_fault_list(app,inj_mode,p.NUM_INJECTIONS,blockDim)
            elif inj_mode=='IAC':
                gridkDim=get_BlockDim(app)
                gen_IAC_fault_list(app,inj_mode,p.NUM_INJECTIONS,gridkDim)
            elif inj_mode=='WV':
                gridkDim=get_BlockDim(app)
                gen_WV_fault_list(app,inj_mode,p.NUM_INJECTIONS,gridkDim)
            elif inj_mode=='IIO':
                print('Sorry! This error model is not implemented yet, give us a hand ;)')
            else:
                print(f"Ops.. the {inj_mode} error model does not exist, perhaps it is a new model you can implement in the future ;)")

            print ("Output: Check %s" %(p.app_log_dir[app] + "/injection-list/"))

if __name__ == "__main__":
    main()
