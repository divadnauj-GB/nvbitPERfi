import os
import logging
from inspect import getframeinfo, stack

OPCODES = ['FADD', 'FADD32I', 'FCHK', 'FCMP', 'FFMA', 'FFMA32I', 'FMNMX', 'FMUL', 'FMUL32I', 'FSEL', 'FSET',
           'FSETP', 'FSWZADD', 'IPA', 'MUFU', 'RRO', 'DADD', 'DFMA', 'DMNMX', 'DMUL', 'DSET', 'DSETP', 'HADD2',
           'HADD2_32I', 'HFMA2', 'HFMA2_32I', 'HMUL2', 'HMUL2_32I', 'HSET2', 'HSETP2', 'IDP', 'IDP4A', 'BFE',
           'BFI', 'BMSK', 'BREV', 'FLO', 'IADD', 'IADD3', 'IADD32I', 'ICMP', 'IMAD', 'IMAD32I', 'IMADSP',
           'IMNMX', 'IMUL', 'IMUL32I', 'ISCADD', 'ISCADD32I', 'ISET', 'ISETP', 'LEA', 'LOP', 'LOP3', 'LOP32I',
           'PLOP3', 'POPC', 'SHF', 'SHL', 'SHR', 'XMAD', 'IMMA', 'HMMA', 'VABSDIFF', 'VADD', 'VMAD', 'VMNMX',
           'VSET', 'VSETP', 'VSHL', 'VSHR', 'VABSDIFF4', 'F2F', 'F2I', 'I2F', 'I2I', 'I2IP', 'FRND', 'MOV',
           'MOV32I', 'PRMT', 'SEL', 'SGXT', 'SHFL', 'CSET', 'CSETP', 'PSET', 'PSETP', 'P2R', 'R2P', 'TEX',
           'TLD', 'TLD4', 'TMML', 'TXA', 'TXD', 'TXQ', 'TEXS', 'TLD4S', 'TLDS', 'STP', 'LD', 'LDC', 'LDG',
           'LDL', 'LDS', 'ST', 'STG', 'STL', 'STS', 'MATCH', 'QSPC', 'ATOM', 'ATOMS', 'RED', 'CCTL', 'CCTLL',
           'ERRBAR', 'MEMBAR', 'CCTLT', 'SUATOM', 'SULD', 'SURED', 'SUST', 'BRA', 'BRX', 'JMP', 'JMX', 'SSY',
           'SYNC', 'CAL', 'JCAL', 'PRET', 'RET', 'BRK', 'PBK', 'CONT', 'PCNT', 'EXIT', 'PEXIT', 'LONGJMP',
           'PLONGJMP', 'KIL', 'BSSY', 'BSYNC', 'BREAK', 'BMOV', 'BPT', 'IDE', 'RAM', 'RTT', 'SAM', 'RPCMOV',
           'WARPSYNC', 'YIELD', 'NANOSLEEP', 'NOP', 'CS2R', 'S2R', 'LEPC', 'B2R', 'BAR', 'R2B', 'VOTE',
           'DEPBAR', 'GETCRSPTR', 'GETLMEMBASE', 'SETCRSPTR', 'SETLMEMBASE', 'PMTRIG', 'SETCTAID']


def execute_cmd(cmd, return_error_code=False):
    logging.debug(f"Executing {cmd}")
    ret = os.system(cmd)
    caller = getframeinfo(stack()[1][0])
    if ret != 0:
        logging.error(f"ERROR AT: {caller.filename}:{caller.lineno} CMD: {cmd}")
        logging.error(f"Command was not correctly executed error code {ret}")
        if return_error_code:
            return ret

        raise ValueError()
