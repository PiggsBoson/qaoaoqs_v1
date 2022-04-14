print("Here is the terminal output!")
with open(os.path.join('test_folder', "test.txt"),"w") as fp:
    fp.write('Inside the file!')
