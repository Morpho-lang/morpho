# Simple automated benchmarking
# T J Atherton Mar 2021

# import necessary modules
import os, glob
import re
import colored
from colored import stylize
import subprocess

"""
languages = { "morpho": "morpho5",
              "m3": "morpho3",
              "fe": "evolver",
              "lua": "lua",
              "wren": "wren",
              "lox": "clox",
              "py": "python3",
              "rb": "ruby",
              "dart": "dart"
            }
"""

languages = { "morpho": "morpho5" }
samples = 10

# Gets the output generated
def getoutput(filepath):
    # Load the file
    file_object = open(filepath, 'r')
    lines = file_object.readlines()
    file_object.close()
    # Extract the timing numbers [minutes, seconds]
    times = re.findall(r"[-+]?\d*\.\d+|\d+", lines[0])

    return float(times[0])

    return -1

# Runs a given command with a file and return the time in s.
def run(command, file):
    out = -1

    print(command + ' ' + file)
    # Create a temporary file in the same directory
    tmp = file + '.out'

    # Run the test
    exec = '( /usr/bin/time ' + command + ' ' + file + ' 1> /dev/null ) 2> ' + tmp
    os.system(exec)

    # If we produced output
    if os.path.exists(tmp):
        out=getoutput(tmp)

        # Delete the temporary file
        os.system('rm ' + tmp)

    return out

# Perform a benchmark
def benchmark(folder):
    dict = {};
    print(stylize(folder[:-1],colored.fg("green")))
    for lang in languages.keys():
        test = glob.glob(folder + '**.' + lang, recursive=False)
        if (len(test)>0):
            time = []
            for i in range(1,samples):
                time.append(run(languages[lang], test[0]))
            dict[lang]=min(time)
    return dict

print('--Begin testing---------------------')

success=0 # number of successful tests
total=0   # total number of tests

benchmarks=glob.glob('**/', recursive=False)

out = []

for f in benchmarks:
    times=benchmark(f)
    out.append(times)

# Display output
str="{:<15}".format("")
for lang in languages.keys():
    str+=" "+"{:<8}".format(lang)
print(str)

for i, results in enumerate(out):
    str="{:<15}".format(benchmarks[i][:-1])
    for lang in languages.keys():
        if lang in results:
            str+=" "+"{:<8}".format(results[lang])
        else:
            str+=" "+"{:<8}".format("-")
    print(str)


print('--End testing-----------------------')
