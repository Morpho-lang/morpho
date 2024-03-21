#!/usr/bin/env python3
# runexamples.py
# runs all the example for morpho
# reporting any errors found

import os, glob, sys
from queue import Empty
import regex as rx
from functools import reduce
import operator
import colored
from colored import stylize


sys.path.append('../test')
ext = "morpho"
command = 'morpho5'
stk = '@stacktrace'
err = '@error'

def finderror(str):
    #return rx.findall(r'\/\/ expect ?(.*) error', str)
    #return rx.findall(r'.*[E|e]rror[ :].*?(.*)', str)
    return rx.findall(r'@error.*', str)


def simplify_errors(str):
    # this monster regex extraxts NAME from error messages of the form error ... 'NAME'
    return rx.sub('.*[E|e]rror[ :]*\'([A-z;a-z]*)\'.*', err+'[\\1]', str.rstrip())

# Simplify stacktrace
def simplify_stacktrace(str):
    return rx.sub(r'.*at line.*', stk, str.rstrip())

def iserror(str):
    #return rx.findall(r'\/\/ expect ?(.*) error', str)
    test=rx.findall(r'@error.*', str)
    return len(test)>0
def remove_control_characters(str):
    return rx.sub(r'\x1b[^m]*m', '', str.rstrip())

def isin(str):
    #return rx.findall(r'\/\/ expect ?(.*) error', str)
    test=rx.findall(r'.*in .*', str)
    return len(test)>0



def getoutput(filepath):
    # Load the file
    file_object = open(filepath, 'r')
    lines = file_object.readlines()
    file_object.close()
    # remove all control characters
    lines = list(map(remove_control_characters, lines))
    # Convert errors to our universal error code
    lines = list(map(simplify_errors, lines))
    # Identify stack trace lines
    lines = list(map(simplify_stacktrace, lines))
    for i in range(len(lines)-1):
        if (iserror(lines[i])):
            if (isin(lines[i+1])):
                lines[i+1]=stk
    # and remove them
    return list(filter(lambda x: x!=stk, lines))
def run(file,testLog,CI):
    ret = 1
    print(file+":", end=" ")


    # Create a temporary file in the same directory
    tmp = file + '.out'

    # Run the test
    os.system(command + ' ' +file + ' > ' + tmp)

    # If we produced output
    if os.path.exists(tmp):
        # Get the output
        out = getoutput(tmp)

		#look for erros
        for line in out:
            err = finderror(line)
            # Was it expected?

            if (iserror(line)):
                if not CI:
                    print("Failed") #stylize("Failed",colored.fg("red"))) // Temporarily disable this 6/19/23 due to colored module API change
                else:
                    print("::error file = {",file,"}::{",file," Failed}")

                #also print to the test log
                print(file+":", end=" ", file = testLog)
                print("Failed", end=" ", file = testLog)
                print("with error "+ err[0], file = testLog)
                print("\n",file = testLog)
                ret = 0
                break
        
        if (ret ==1):
            if not CI:
                print(file+":", end=" ")
                print("Passed") #stylize("Passed",colored.fg("green")))
        # Delete the temporary file
        os.system('rm ' + tmp)
    return ret



print('--Building Examples---------------------')


# open a test log
# write failures to log
success=0 # number of successful examples
total=0   # total number of examples

# look for a command line arguement that says
# this is being run for continous integration
CI = sys.argv == '-c'


files=glob.glob('**/**.'+ext, recursive=True)
with open("FailedExamples.txt",'w') as testLog:

    for f in files:
        success+=run(f,testLog,CI)
        total+=1


print('--End testing-----------------------')
print(success, 'out of', total, 'tests passed.')
if CI and success<total:
    exit(-1)
