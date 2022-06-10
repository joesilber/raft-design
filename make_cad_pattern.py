# Joe Silber - 2022 - jhsilber@lbl.gov
# must be run within the FreeCAD gui, due to usage of ImportGui module for making the STEP file at the end

FREECAD_PATH = 'C:/Program Files/FreeCAD 0.19/bin' # path to your FreeCAD.so or FreeCAD.dll file
OTHER_PKGS_PATH = 'C:/Users/joe/AppData/Local/Programs/Python/Python38/Lib/site-packages'
import sys
sys.path.append(FREECAD_PATH)
sys.path.append(OTHER_PKGS_PATH)
import math
import time
import os
import FreeCAD
import Part
from FreeCAD import Base
from astropy.table import Table
import tkinter.filedialog as filedialog

script_title = "Raft Patterning Script"
doc_name = "PatternDoc"
App.newDocument(doc_name)
AD = App.ActiveDocument		# just to make things more readable
starttime = time.time()
print("\nBEGIN " + script_title + "...") # print the script name

# Paths to source model
homepath = os.path.expanduser('~') #'C:/Users/joe/' #"C:/Users/jhsilber/Documents/PDMWorks/"
#source_model = "MM Raft Assembly - simplified - 2022-05-19.STEP"
#model_path = os.path.join(homepath, source_model)
model_path = filedialog.askopenfilename(title='Select CAD model to pattern...', initialdir=homepath, filetypes=[('STEP','*.STEP'), ('step', '*.step')])
model_dir, model_name = os.path.split(model_path)
base_name = "EnvelopesArray"

# Read in the source geometry
source_name  = "proto"
source       = AD.addObject("Part::Feature", source_name)
source.Shape = Part.read(model_path)

# Read in the raft positions
#pattern_name = '20220520T1631_desi2_layout_21rafts_1512robots.csv'
#pattern_path = os.path.join(homepath, pattern_name)
pattern_path = filedialog.askopenfilename(title='Select data table that specifies the pattern...', initialdir=model_dir, filetypes=[('csv', '*.csv')])
pattern_dir, pattern_name = os.path.split(pattern_path)
tbl = Table.read(pattern_path)

steptime = time.time()
print(f'... {len(tbl)} raft positions read')
lasttime = steptime

# Choose how many rafts to pattern (can argue a smaller number, i.e. for testing)
max_patterns = math.inf  # integer or math.inf
num_to_process = min(max_patterns, len(tbl))

rafts = []
for i in range(num_to_process):
    # Generate the raft
    raft_name = f'raft{i}'
    rafts += [AD.addObject("Part::Feature", raft_name)]
    rafts[-1].Shape = source.Shape

    # Transform the raft
    p = math.radians(tbl['precession'][i])
    n = math.radians(-tbl['nutation'][i])
    s = math.radians(tbl['spin'][i])
    q1 = math.cos(s/2)*math.sin(n/2)*math.sin(p/2) - math.sin(s/2)*math.cos(p/2)*math.sin(n/2)
    q2 = -(math.cos(s/2)*math.cos(p/2)*math.sin(n/2) + math.sin(s/2)*math.sin(n/2)*math.sin(p/2))
    q3 = math.cos(s/2)*math.cos(n/2)*math.sin(p/2) + math.cos(n/2)*math.cos(p/2)*math.sin(s/2)
    q4 = math.cos(s/2)*math.cos(n/2)*math.cos(p/2) - math.sin(s/2)*math.cos(n/2)*math.sin(p/2)
    rafts[-1].Placement.Rotation = Base.Rotation(q1, q2, q3, q4)
    x = float(tbl['x'][i])
    y = float(tbl['y'][i])
    z = float(tbl['z'][i])
    rafts[-1].Placement.Base = Base.Vector(x, y, z)

steptime = time.time()
print(f'... {len(rafts)} rafts patterned in {(steptime-lasttime)/60:.2f} min')
lasttime = steptime

# Export hole array, using GUI module
export_name = f'{os.path.splitext(pattern_name)[0]}.step'
#export_path = os.path.join(homepath, export_name)
export_path = filedialog.asksaveasfilename(title='Save patterned model as...', initialdir=model_dir, initialfile=export_name, defaultextension='.step')
import ImportGui # import GUI module
ImportGui.export(rafts, export_path) # requires GUI, does the export of geometry
App.getDocument("PatternDoc").removeObject("proto") # deletes the proto raft, just to give user warm-fuzzies about what was exported
Gui.SendMsgToActiveView("ViewFit") # requires GUI, gives user warm-fuzzies
Gui.activeDocument().activeView().viewAxometric() # requires GUI, gives user warm-fuzzies
steptime = time.time()
print(f'... exported array in {(steptime-lasttime)/60:.2f} min')
lasttime = steptime

endtime = time.time()
runtime = endtime-starttime
print('... DONE')
print(f'Total runtime = {runtime/60:.2f} min\n')
