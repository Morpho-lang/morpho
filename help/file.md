[comment]: # (File class help)
[version]: # (0.5)

# File
[tagfile]: # (File)

The `File` class provides the capability to read from and write to files, or to obtain the contents of a file in convenient formats.

To open a file, create a File object with the filename as the argument

    var f = File("myfile.txt")

which opens `"myfile.txt"` for *reading*. To open a file for writing or appending, you need to provide a mode selector

    var g = File("myfile.txt", "write")

or

    var g = File("myfile.txt", "append")

Once the file is open, you can then read or write by calling appropriate methods:

    f.lines()            // reads the contents of the file into an array of lines.
    f.readline()         // reads a single line
    f.readchar()         // reads a single character.
    f.write(string)      // writes the arguments to the file.

After you're done with the file, close it with

    f.close()

[show]: # (subtopics)

## lines
[taglines]: # (lines)

Returns the contents of a file as an array of strings; each element corresponds to a single line.

Read in the contents of a file and print line by line:

    var f = File("input.txt")
    var s = f.lines()
    for (i in s) print i
    f.close()

## readline
[tagreadline]: # (readline)

Reads a single line from a file; returns the result as a string.

Read in the contents of a file and print each line:

    var f = File("input.txt")
    while (!f.eof()) {
      print f.readline()
    }
    f.close()

## readchar
[tagreadchar]: # (readchar)

Reads a single character from a file; returns the result as a string.

## write
[tagwrite]: # (write)

Writes to a file.

Write the contents of a list to a file:

    var f = File("output.txt", "w")
    for (k, i in list) f.write("${i}: ${k}")
    f.close()

## close
[tagclose]: # (close)

Closes an open file.

## eof
[tageof]: # (eof)

Returns true if at the end of the file; false otherwise

# Folder
[tagfolder]: # (Folder)

The `Folder` class enables you to find whether a filepath refers to a folder, and find the contents of that folder.

Find whether a path refers to a folder:

    print Folder.isfolder("path/folder")
    
Get a list of a folder's contents: 

    print Folder.contents("path/folder")
