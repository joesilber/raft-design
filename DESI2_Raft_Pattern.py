# Joe Silber - 2022 - jhsilber@lbl.gov
# must be run within the FreeCAD gui, due to usage of ImportGui module for making the STEP file at the end

FREECADPATH = "C:/Program Files/FreeCAD0.19/bin" # path to your FreeCAD.so or FreeCAD.dll file
import sys
sys.path.append(FREECADPATH)
import math
import time
import FreeCAD
import Part
from FreeCAD import Base

script_title = "DESI2 Raft Patterning Script"
doc_name = "PatternDoc"
App.newDocument(doc_name)
AD = App.ActiveDocument		# just to make things more readable
starttime = time.clock()
print("\nBEGIN " + script_title + "...") # print the script name

# Paths to source model
homepath = "C:/Users/jhsilber/Documents/PDMWorks/"
source_model = "MM Raft Assembly - simplified - 2022-05-19.STEP"
base_name = "EnvelopesArray"

# Read in the source geometry
source_name  = "proto"
source      = AD.addObject("Part::Feature", source_name)
source.Shape = Part.read(homepath + source_model)

# Read in the hole positions
hole_loc_name = "pos_on_z1.txt"
hole_loc_file = open(homepath + hole_loc_name,"rU")
comment_line_symbol = "#"
i = 0
hole_number = []
hole_pos = []
precession = []
nutation = []
spin = []
for row in hole_loc_file:
    row_words = row.split()
    if comment_line_symbol not in row_words:
        is_symmetric_duplicate = bool(float(row_words[7]))
        # is_remove = bool(float(row_words[8]))
        is_remove = 0 # comment out to use is_remove flag in text file (and de-comment line above)
        if not is_symmetric_duplicate and not is_remove:
            hole_number = hole_number + [int(row_words[0])]
            x = float(row_words[1])
            y = float(row_words[2])
            z = float(row_words[3])
            hole_pos = hole_pos + [Base.Vector(x,y,z)]

    		# Euler angles
            precession = precession + [float(row_words[4])]
            nutation   = nutation   + [float(row_words[5])]
            spin       = spin       + [float(row_words[6])]
            i += 1

hole_loc_file.close()
steptime = time.clock()
print("..." + str(len(hole_pos)) + " hole positions read in %.2f" % ((steptime-starttime)/60) + " min")
lasttime = steptime

# Choose which subset of holes to pattern
holes_to_process = range(len(hole_number)) # can argue a smaller set, such as range(150), for testing

# Generate the holes
holes = []
for i in range(len(holes_to_process)):
	n = holes_to_process[i]
	hole_name = "hole" + str(hole_number[n])
	holes = holes + [AD.addObject("Part::Feature",hole_name)]
	holes[i].Shape = proto_hole.Shape

# Transform the holes
for i in range(len(holes)):
    h = holes_to_process[i]
    p = precession[h] * math.pi/180
    n = -(nutation[h] * math.pi/180)
    s = spin[h] * math.pi/180
    q1 = math.cos(s/2)*math.sin(n/2)*math.sin(p/2) - math.sin(s/2)*math.cos(p/2)*math.sin(n/2)
    q2 = -(math.cos(s/2)*math.cos(p/2)*math.sin(n/2) + math.sin(s/2)*math.sin(n/2)*math.sin(p/2))
    q3 = math.cos(s/2)*math.cos(n/2)*math.sin(p/2) + math.cos(n/2)*math.cos(p/2)*math.sin(s/2)
    q4 = math.cos(s/2)*math.cos(n/2)*math.cos(p/2) - math.sin(s/2)*math.cos(n/2)*math.sin(p/2)
    holes[i].Placement.Rotation = Base.Rotation(q1,q2,q3,q4)
    holes[i].Placement.Base = hole_pos[h]

steptime = time.clock()
print("..." + str(len(holes)) + " holes patterned in %.2f" % ((steptime-lasttime)/60) + " min")
lasttime = steptime

# Export hole array, using GUI module
export_name = base_name + "_" + str(len(holes)) + ".step"
import ImportGui # import GUI module
ImportGui.export(holes,homepath + export_name) # requires GUI, does the export of geometry
App.getDocument("PatternDoc").removeObject("proto_hole") # deletes the proto_hole, just to give user warm-fuzzies about what was exported
Gui.SendMsgToActiveView("ViewFit") # requires GUI, gives user warm-fuzzies
Gui.activeDocument().activeView().viewAxometric() # requires GUI, gives user warm-fuzzies
steptime = time.clock()
print("...export array in %.2f" % ((steptime-lasttime)/60) + " min")
lasttime = steptime

endtime = time.clock()
runtime = endtime-starttime
print("...DONE")
print("Total runtime = %.2f" % (runtime/60) + " min\n")
