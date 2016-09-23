from  PIL import Image
import sys

im = Image.open(sys.argv[1])
out = im.rotate(180)
out.save('ans2.png')
