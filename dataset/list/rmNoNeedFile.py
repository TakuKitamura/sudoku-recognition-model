# import subprocess
# import glob

# fileList = glob.glob("./*")
# try:
#     for filePath in fileList:
#         args = ['file', filePath]
#         res = subprocess.run(args, stdout=subprocess.PIPE)
#         if "image data" not in str(res):
#             args = ['rm', filePath]
#             res = subprocess.run(args, stdout=subprocess.PIPE)
# except:
#     print("Error.")