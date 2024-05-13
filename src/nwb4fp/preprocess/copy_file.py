import shutil
import sys

def main():
    print("y")

def copy_file(origin_p,destinate_p):

    # Copy the source file to the destination
    try:
        shutil.copy(origin_p + "/cluster_group.tsv", destinate_p + "/cluster_group.tsv")
        print(f"File copied successfully from {origin_p} to {destinate_p}")
    except IOError as e:
        print(f"Unable to copy file. {e}")
    except:
        print("Unexpected error:", sys.exc_info())
