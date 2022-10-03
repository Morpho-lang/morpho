docsfolder = '../../morpho5/docs/'
reffolder = 'Reference/'

import os, glob

print('--Creating figures-------------------------------')

files=glob.glob('**/**.morpho', recursive=True)

for f in files:
    print(f)
    # Run the program
    os.system('morpho5 ' + f)

print('--Building reference section---------------------')

files=glob.glob(docsfolder+'**.md', recursive=True)

for f in files:
    filename=reffolder + f.split('/')[-1].split('.')[0]
    print(filename)
    os.system(f'pandoc -o {filename}.tex {f}')
    os.system(f'sed -i \'\' -e \'s/verbatim/lstlisting/g\' {filename}.tex')
    os.system(f'sed -i \'\' -e \'s/\\\\subsection/\\\\subsection/g\' {filename}.tex')
    os.system(f'sed -i \'\' -e \'s/\\\\tightlist//g\' {filename}.tex')

print('-------------------------------------------------')
