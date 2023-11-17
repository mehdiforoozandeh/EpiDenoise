from data import *
import sys, os


dl_dict={
    "url":sys.argv[1], 
    "save_dir_name":sys.argv[2], 
    "exp":sys.argv[3], 
    "bios":sys.argv[4]
}

single_download(dl_dict)