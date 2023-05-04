import os

path = "/home/jawabreh/Desktop/sds/"

def rename_images(path):
    i = 30
    for filename in os.listdir(path):
        src = path + filename
        dst = path + str(i) + ".wav"
        os.rename(src, dst)
        i += 1

rename_images(path)
print("\n\n\tSUCCESS\n\n")