from genericpath import isdir, isfile
from logging import critical
from sys import argv
import numpy as np
import os
import struct
import sys




def main():
    filename=sys.argv[1]
    print(filename)
    report_file=open(filename,'r')

    lines_file=report_file.readlines()
    report_file.close()
    image_dic={}
    for line in lines_file:
        if "ImageID" in line:
            fields=line.split(',')
            image=fields[1].split(':')[1].strip()
            if(image) not in image_dic:
                image_dic[image]=[]
                image_dic[image].append(line)
            else:
                image_dic[image].append(line)

    for key in sorted(image_dic):
        fault_class={}
        for line in image_dic[key]:
            fields=line.split(',')
            if "SDC" in line or "DUE" in line or "Masked" in line:
                if "SDC" in fields[3]:
                    init=fields[3].find("SDC")
                    error_code=str(fields[3][init:len(fields[3])])
                    if error_code not in fault_class:
                        fault_class[error_code]=1
                    else:
                        fault_class[error_code]+=1

                    if "T_SDC" not in fault_class:
                        fault_class["T_SDC"]=1
                    else:
                        fault_class["T_SDC"]+=1
                    
                elif "Masked" in fields[3]:
                    init=fields[3].find("Masked")
                    error_code=str(fields[3][init:len(fields[3])])
                    #print(fields[2])
                    ACT=int(fields[2].split(":")[1].strip())
                    if ACT==0:
                        if "Masked_ACT_0" not in fault_class:
                            fault_class["Masked_ACT_0"]=1
                        else:
                            fault_class["Masked_ACT_0"]+=1
                    else:
                        if error_code not in fault_class:
                            fault_class[error_code]=1
                        else:
                            fault_class[error_code]+=1
                    
                    
                    if "T_Masked" not in fault_class:
                        fault_class["T_Masked"]=1
                    else:
                        fault_class["T_Masked"]+=1


                elif "DUE" in fields[3]:
                    init=fields[3].find("DUE")
                    error_code=str(fields[3][init:len(fields[3])])
                    if error_code not in fault_class:
                        fault_class[error_code]=1
                    else:
                        fault_class[error_code]+=1

                    if "T_DUE" not in fault_class:
                        fault_class["T_DUE"]=1
                    else:
                        fault_class["T_DUE"]+=1
                
                if "TotalF" not in fault_class:
                    fault_class["TotalF"]=1
                else:
                    fault_class["TotalF"]+=1

        print(f"Image ID: {key}, Fault injection results..")
        for keyerror in sorted(fault_class):
            print(f"{keyerror}, Num ocurrences: {fault_class[keyerror]}")
        print("\n")
main()