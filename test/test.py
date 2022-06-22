#!/usr/bin/env python3
# Simple automated testing
# T J Atherton Sept 2020
#
# Each input file is supplied to a test command, the results
# are piped to a file and the output is compared with expectations
# extracted from the input file.
# Expectations are coded into comments in the input file as follows:

# import necessary modules
import os, glob, sys
import regex as rx
from functools import reduce
import operator
import colored
from colored import stylize

# define what command to use to invoke the interpreter
command = 'morpho5'

# define the file extension to test
ext = 'morpho'

# We reduce any errors to this value
err = '@error'

# We reduce any stacktrace lines to this values
stk = '@stacktrace'

# Removes control characters
def remove_control_characters(str):
    return rx.sub(r'\x1b[^m]*m', '', str.rstrip())

# Simplify error reports
def simplify_errors(str):
    # this monster regex extraxts NAME from error messages of the form error ... 'NAME'
    return rx.sub('.*[E|e]rror[ :]*\'([A-z;a-z]*)\'.*', err+'[\\1]', str.rstrip())

# Simplify stacktrace
def simplify_stacktrace(str):
    return rx.sub(r'.*at line.*', stk, str.rstrip())

# Find an expected value
def findvalue(str):
    return rx.findall(r'// expect: ?(.*)', str)

# Find an expected error
def finderror(str):
    #return rx.findall(r'\/\/ expect ?(.*) error', str)
    return rx.findall(r'.*[E|e]rror[ :].*?(.*)', str)

# Find an expected error
def iserror(str):
    #return rx.findall(r'\/\/ expect ?(.*) error', str)
    test=rx.findall(r'@error.*', str)
    return len(test)>0

# Find an expected error
def isin(str):
    #return rx.findall(r'\/\/ expect ?(.*) error', str)
    test=rx.findall(r'.*in .*', str)
    return len(test)>0

# Remove elements from a list
def remove(list, remove_list):
    test_list = list
    for i in remove_list:
        try:
            test_list.remove(i)
        except ValueError:
            pass
    return test_list

# Find what is expected
def findexpected(str):
    out = finderror(str) # is it an error?
    if (out!=[]):
        out = [simplify_errors(str)] # if so, simplify it
    else:
        out = findvalue(str) # or something else?
    return out

# Works out what we expect from the input file
def getexpect(filepath):
    # Load the file
    file_object = open(filepath, 'r')
    lines = file_object.readlines()
    file_object.close()
    #Find any expected values over all lines
    if (lines != []):
        out = list(map(findexpected, lines))
        out = reduce(operator.concat, out)
    else:
        out = []
    return out

# Gets the output generated
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

def gettesttype(file):
    # check if first line is the test header 
    file_object = open(file, 'r')
    lines = file_object.readlines()
    file_object.close()
    if len(lines)==0:
        return False
    match = rx.search('===================================',lines[0])
    return (not match is None)

def parseunittestresults(test,tmp,testLog,CI):
    file_object = open(tmp, 'r')
    lines = file_object.readlines()
    file_object.close()
    
    # check the last line to see if it indicates a pass
    match = rx.search('===================================',lines[-1])
    if match is not None:
        if not CI:
            print(test+":", end=" ")
            print(stylize("Passed",colored.fg("green")))
        return 1
    else:
        # The test has failed, find the failing test and the reason
        # Find a line beginning with Assertion Failed or Error Encountered
        # The rest of the line is the reason
        # The line above is "Running test test_that_failed:"

        if not CI:
            print(stylize("Failed",colored.fg("red")))
            print("Test suite:" + test + " failed", file = testLog)
        else:
            print("\n::error file = {",test,"}::{",test," Failed}")

        for i in range(len(lines)):
            reason = rx.findall(r'Assertion Failed (.*)\n', lines[i])
            if len(reason) > 0:
                test = rx.findall(r'Running test (.*):',lines[i-1])
                print("Test "+test[0]+" failed: " + reason[0])
                print("Test "+test[0]+" failed: " + reason[0],file = testLog)
        return 0

# Test a file
def test(file,testLog,CI):
    ret = 0
    if not CI:
        print(file+":", end=" ")


    # Create a temporary file in the same directory
    tmp = file + '.out'

    # Run the test
    os.system(command + ' ' +file + ' > ' + tmp)

    # Check if this is and expect test or a unittest 
    usingUnitTest = gettesttype(tmp)
    

    if usingUnitTest:
        ret = parseunittestresults(file,tmp,testLog,CI)
    
    else:

        #Get the expected output
        expected=getexpect(file)

        # If we produced output
        if os.path.exists(tmp):
            # Get the output
            out=getoutput(tmp)

            # Was it expected?
            if(expected==out):
                if not CI:
                    print(file+":", end=" ")
                    print(stylize("Passed",colored.fg("green")))
                ret = 1
            else:
                if not CI:
                    print(stylize("Failed",colored.fg("red")))
                    print("  Expected: ", expected)
                    print("    Output: ", out)
                else:
                    print("\n::error file = {",file,"}::{",file," Failed}")


                #also print to the test log
                print(file+":", end=" ",file = testLog)
                print("Failed", file = testLog)

                if len(out) == len(expected):
                    failedTests = list(i for i in range(len(out)) if expected[i] != out[i])
                    print("Tests " + str(failedTests) + " did not match expected results.", file = testLog)
                    for testNum in failedTests:
                        print("Test "+str(testNum), file = testLog)
                        print("  Expected: ", expected[testNum], file = testLog)
                        print("    Output: ", out[testNum], file = testLog)
                else:
                    print("  Expected: ", expected, file = testLog)
                    print("    Output: ", out, file = testLog)


                print("\n",file = testLog)


        # Delete the temporary file
        os.system('rm ' + tmp)

    return ret

print('--Begin testing---------------------')

# open a test log
# write failures to log
success=0 # number of successful tests
total=0   # total number of tests

# look for a command line argument that says
# this is being run for continuos integration
CI = False
if (len(sys.argv) > 1):
    CI = sys.argv[1] == '-c'

files=glob.glob('**/**.'+ext, recursive=True)
with open("FailedTests.txt",'w') as testLog:

    for f in files:
        # print(f)
        success+=test(f,testLog,CI)
        total+=1

if (not CI) and (not success == total):
    os.system("emacs FailedTests.txt &")

print('--End testing-----------------------')
print(success, 'out of', total, 'tests passed.')
if CI and success<total:
    exit(-1)