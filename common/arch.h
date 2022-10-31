/*
 * Copyright 2020, NVIDIA CORPORATION.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#ifndef ARCH_H
#define ARCH_H 

#define NUM_COUNTERS NUM_INST_GROUPS+NUM_ISA_INSTRUCTIONS

// new inst: FSEL
/*
enum InstructionType { 
 // Floating point instructions
 FADD = 0,
 FADD32I,
 FCHK,
 FCMP,
 FFMA,
 FFMA32I,
 FMNMX,
 FMUL,
 FMUL32I,
 FSEL,
 FSET,
 FSETP,
 FSWZADD,
 IPA,
 MUFU,
 RRO,
 DADD,
 DFMA,
 DMNMX,
 DMUL,
 DSET,
 DSETP,
 HADD2,
 HADD2_32I,
 HFMA2,
 HFMA2_32I,
 HMUL2,
 HMUL2_32I,
 HSET2,
 HSETP2,
 // Integer Instructions
 IDP,
 IDP4A,
 BFE,
 BFI,
 BMSK,
 BREV,
 FLO,
 IADD,
 IADD3,
 IADD32I,
 ICMP,
 IMAD,
 IMAD32I,
 IMADSP,
 IMNMX,
 IMUL,
 IMUL32I,
 ISCADD,
 ISCADD32I,
 ISET,
 ISETP,
 LEA,
 LOP,
 LOP3,
 LOP32I,
 PLOP3,
 POPC,
 SHF,
 SHL,
 SHR,
 XMAD,
 //MMA Instructions
 IMMA,
 HMMA,
 // Video Instructions
 VABSDIFF,
 VADD,
 VMAD,
 VMNMX, 
 VSET, 
 VSETP,
 VSHL, 
 VSHR, 
 VABSDIFF4,
 // Conversion Instructions
 F2F,
 F2I,
 I2F,
 I2I,
 I2IP,
 FRND,
 // Move Instructions
 MOV,
 MOV32I,
 PRMT,
 SEL,
 SGXT,
 SHFL,
 // Predicate/CC Instructions
 CSET,
 CSETP,
 PSET,
 PSETP,
 P2R,
 R2P,
 // Texture Instructions
 TEX,
 TLD,
 TLD4,
 TMML,
 TXA,
 TXD,
 TXQ,
 TEXS,
 TLD4S,
 TLDS,
 STP,
 // Load/Store Instructions
 LD,
 LDC,
 LDG,
 LDL,
 LDS,
 ST,
 STG,
 STL,
 STS,
 MATCH,
 QSPC,
 ATOM,
 ATOMS,
 RED,
 CCTL,
 CCTLL,
 ERRBAR,
 MEMBAR,
 CCTLT,
 SUATOM,
 SULD,
 SURED,
 SUST,
 // Control Instructions
 BRA,
 BRX,
 JMP,
 JMX,
 SSY,
 SYNC,
 CAL,
 JCAL,
 PRET,
 RET,
 BRK,
 PBK,
 CONT,
 PCNT,
 EXIT,
 PEXIT,
 LONGJMP,
 PLONGJMP,
 KIL,
 BSSY,
 BSYNC,
 BREAK,
 BMOV,
 BPT,
 IDE,
 RAM,
 RTT,
 SAM,
 RPCMOV,
 WARPSYNC,
 YIELD,
 NANOSLEEP,
 // Miscellaneous Instructions
 NOP,
 CS2R,
 S2R,
 LEPC,
 B2R,
 BAR,
 R2B,
 VOTE,
 DEPBAR,
 GETCRSPTR,
 GETLMEMBASE,
 SETCRSPTR,
 SETLMEMBASE,
 PMTRIG,
 SETCTAID,
 NUM_ISA_INSTRUCTIONS
 };
*/

enum InstructionType { 
 // Floating point 32 instructions (0-13)
 FADD = 0,
 FADD32I,
 FCHK,
 FCMP,
 FFMA,
 FFMA32I,
 FMNMX,
 FMUL,
 FMUL32I,
 FSEL,
 FSET,
 FSETP,
 FSWZ,
 FSWZADD,

 // Floating Point 16 Instructions (14-22)
 HADD2,
 HADD2_32I,
 HFMA2,
 HFMA2_32I,
 HMNMX2,
 HMUL2,
 HMUL2_32I,
 HSET2,
 HSETP2,


 // SFU (23-24)
 MUFU,
 RRO,	
 // Tensor Core Instructions (25-28)
 // Execute Tensor Core Instructions on SPECIALIZED_UNIT_3
 //MMA Instructions
 HMMA,
 IMMA,
 BMMA,
 DMMA,

 // Double Point Instructions (29-34)
 DADD,
 DFMA,
 DMNMX,
 DMUL,
 DSET,
 DSETP,

 // Integer Instructions (35-67)
 BFE,
 BFI,
 BMSK,
 BREV,
 FLO, 
 IABS,
 IADD,
 IADD3,
 IADD32I,
 ICMP,
 IDP,
 IDP4A,
 IMAD,
 IMAD32I,
 IMADSP,
 IMNMX,
 IMUL,
 IMUL32I,
 IPA,
 ISAD,
 ISCADD,
 ISCADD32I,
 ISET,
 ISETP,
 LEA,
 LOP,
 LOP3,
 LOP32I,
 POPC,
 SHF,
 SHL,
 SHR,
 XMAD,

 // Video Instructions (68-76)
 VABSDIFF,
 VABSDIFF4,
 VADD,
 VMAD,
 VMNMX, 
 VSET, 
 VSETP,
 VSHL, 
 VSHR, 

 // Conversion Instructions (77-84)
 F2F,
 F2I,
 I2F,
 I2I,
 I2IP,
 I2FP,
 F2IP,
 FRND,

 // Move Instructions (85-91)
 MOV,
 MOV32I,
 MOVM,
 PRMT,
 SEL,
 SGXT,
 SHFL,

 // Predicate/CC Instructions (92-98)
 PLOP3,
 CSET,
 CSETP,
 PSET,
 PSETP,
 P2R,
 R2P,

 // Load/Store Instructions (99-123)
 LD,
 LDC,
 LDG,
 LDGDEPBAR,
 LDGSTS,
 LDL,
 LDS,
 LDSM,
 LDSLK,
 ST,
 STG,
 STL,
 STS,
 STSCUL,
 MATCH,
 QSPC,
 ATOM,
 ATOMS,
 ATOMG,
 RED,
 CCTL,
 CCTLL,
 ERRBAR,
 MEMBAR,
 CCTLT,

 // Uniform Datapaht Instructions (124-153)
 R2UR,
 REDUX,
 S2UR,
 UBMSK,
 UBREV,
 UCLEA,
 UF2FP,
 UFLO,
 UIADD3,
 UIADD3_64,
 UIMAD,
 UISETP,
 ULDC,
 ULEA,
 ULOP,
 ULOP3,
 ULOP32I,
 UMOV,
 UP2UR,
 UPLOP3,
 UPOPC,
 UPRMT,
 UPSETP,
 UR2UP,
 USEL,
 USGXT,
 USHF,
 USHL,
 USHR,
 VOTEU,

 // Texture Instructions (154-164)
 TEX,
 TLD,
 TLD4,
 TMML,
 TXA,
 TXD,
 TXQ,
 TEXS,
 TLD4S,
 TLDS,
 STP,

 // surface Instructions (165-174)
 SUATOM,
 SUCLAMP,
 SUBFM,
 SUEAU,
 SULD,
 SULDGA,
 SUQUERY,
 SURED,
 SUST,
 SUSTGA,

 // Control Instructions (175-210)
 BMOV,
 BPT,
 BRA,
 BREAK,
 BRK,
 BRX,
 BRXU,
 BSSY,
 BSYNC,
 CALL,
 CAL,
 CONT,
 EXIT,
 IDE,
 JCAL,
 JMP,
 JMX,
 JMXU,
 KIL,
 KILL,
 LONGJMP,
 NANOSLEEP,
 PBK,
 PCNT,
 PEXIT,
 PLONGJMP,
 PRET,
 RAM,
 RET,
 RPCMOV,
 RTT,
 SAM,
 SSY,
 SYNC,
 WARPSYNC,
 YIELD,
  
 // Miscellaneous Instructions (211-227)
 B2R,
 BAR,
 CS2R,
 CSMTEST,
 DEPBAR,
 GETLMEMBASE,
 LEPC,
 NOP,
 PMTRIG,
 R2B,
 S2R,
 SETCTAID,
 SETLMEMBASE,
 VOTE,
 VOTE_VTG,
 GETCRSPTR,
 SETCRSPTR, 
 NUM_ISA_INSTRUCTIONS
 };

// List of instruction groups
enum GroupType { 
	G_FP64 = 0, // FP64 arithmetic instructions
	G_FP32, // FP32 arithmetic instructions 
	G_LD, // instructions that read from emory 
	G_PR, // instructions that write to PR registers only
	G_NODEST, // instructions with no destination register 
	G_OTHERS, 
	G_GPPR, // instructions that write to general purpose and predicate registers
		//  #GPPR registers = all instructions - G_NODEST
	G_GP, // instructions that write to general purpose registers 
		// #GP registers = all instructions - G_NODEST - G_PR
	NUM_INST_GROUPS
	};

// List of the Bit Flip Models
enum BitFlipModel {
	FLIP_SINGLE_BIT = 0,  // flip a single bit
	FLIP_TWO_BITS, // flip two adjacent bits
	RANDOM_VALUE,  // write a random value.
	ZERO_VALUE, // write value 0
	NUM_BFM_TYPES 
};



#endif
