#!/usr/bin/env python3
# Simple automated testing with valgrind
# T J Atherton Jan 2025

# import necessary modules
import os, glob, sys
import regex as rx
from functools import reduce
import operator
import colored
from colored import stylize

# define what command to use to invoke valgrind
valgrind = 'valgrind'

# define what command to use to invoke the interpreter
command = 'morpho6'

# define the file extension to test
ext = 'morpho'

def checkvalgrindlog(filepath):
    file_object = open(filepath, 'r')
    lines = file_object.readlines()
    file_object.close()
    for l in lines: 
        if (rx.search('in use at exit: 0 bytes in 0 blocks', l)):
            return True
    return False

# Test a file
def test(file, CI):
    ret = 0
    if not CI:
        print(file+":", end=" ")

    # Create a temporary file in the same directory
    tmp = file + '.valgrind'

    # Run the test
    os.system(valgrind + ' --log-file='+tmp + ' ' + command + ' ' +file + ' > /dev/null')

    valpass = False
    if os.path.exists(tmp):
        valpass = checkvalgrindlog(tmp)

        # Was it expected?
        if(valpass):
            if not CI:
                print(stylize("Passed",colored.fg("green")))

            ret = 1
        else:
            if not CI:
                print(stylize("Failed",colored.fg("red")))
            else:
                print(file + " Failed")


        # Delete the temporary files
        os.system('rm ' + tmp)

    return ret

print('--Begin testing---------------------')

# open a test log
# write failures to log
success=0 # number of successful tests
total=0   # total number of tests

# look for a command line arguement that says
# this is being run for continous integration
CI = False
for arg in sys.argv:
    if arg == '-c': # if the argument is -c, then we are running in CI mode
        CI = True

files=glob.glob('**/**.'+ext, recursive=True)
for f in files:
    success+=test(f,CI)
    total+=1

print('--End testing-----------------------')
print(success, 'out of', total, 'tests passed.')
if CI and success<total:
    exit(-1)
