import os 
import sys 
location = sys.argv[1]
path , folder = os.path.split(location)

def getformats(location):
	path , folder = os.path.split(location)
	formats = []
	print("the path given is "+ location)
	print("the directory chosen is "+ folder)
	print("current directory is "+ os.getcwd())
	print("changing the working directory")
	os.chdir(location)
	print("changed directory to "+ os.getcwd())
	for files in os.listdir(location):
			filename , ext = os.path.splitext(files)
			if len(ext) != 0:
				formats.append(ext)
			formats = list(set(formats))
	return formats
 
def making(location):
	empdirs = []
	formats = getformats(location)
	for f in set(formats):
		if os.path.exists(location+"\\"+f):
			print("its already a directory --> " + f )
		else:
			os.chdir(location)
			if not(os.path.exists(((f.replace("." , ""))+"s"))):
				os.mkdir(((f.replace("." , ""))+"s"))
				print("new empty " +((f.replace("." , ""))+"s") + " made" )
				empdirs.append(((f.replace("." , ""))+"s"))
			else:
				print("directory already exists")
				empdirs.append(((f.replace("." , ""))+"s"))
	return empdirs

def filetransfer(location):

	files = os.listdir(location)
	dirs = making(location)
	for f in files:
		if f.startswith("."):
			files.remove(f)
	formats = getformats(location)
	for f in files:
		for e in formats:
			if f.endswith(e):
				print(f , e)
				oldl = location+"\\"+f
				newl = location+"\\"+((e.replace("." , "")))+"s\\"+f
				os.rename(oldl , newl)
				print("file moved")
filetransfer(location)