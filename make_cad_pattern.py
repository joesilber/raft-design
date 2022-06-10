# Joe Silber - 2022 - jhsilber@lbl.gov
# Notes:
# 1. Must be run within the FreeCAD gui, due to usage of ImportGui module for making the STEP file at the end.
# 2. You also need FreeCAD to have an astropy module for reading in the csv file. In Windows:
#      - open a PowerShell terminal
#      - go to the FreeCAD install directory (like the one in POSSIBLE_FREECAD_PATHS below)
#      - ./python.exe -m pip install astropy

POSSIBLE_FREECAD_PATHs = ['C:/Program Files/FreeCAD 0.19/bin',
                          'C:/Users/jhsilber/AppData/Local/Programs/FreeCAD 0.19/bin',
                         ]  # add your path to your computer's FreeCAD.so or FreeCAD.dll file

import os
import sys
for path in POSSIBLE_FREECAD_PATHs:
    sys.path.append(path)
import math
import time
import FreeCAD
import Part
from FreeCAD import Base
from astropy.table import Table

import PySide
from PySide import QtGui ,QtCore
from PySide.QtGui import *
from PySide.QtCore import *

def get_open_name(title, initial_dir, ext):
	try:
	    OpenName = QFileDialog.getOpenFileName(None,QString.fromLocal8Bit(title), initial_dir, f'*.{ext}') # PyQt4
	except Exception:
	    OpenName, Filter = PySide.QtGui.QFileDialog.getOpenFileName(None, title, initial_dir, f'*.{ext}') #PySide
	return OpenName

def get_save_name(title, initial_path, ext):
	try:
	    SaveName = QFileDialog.getSaveFileName(None, QString.fromLocal8Bit(title), initial_path, f'*.{ext}') # PyQt4
	except Exception:
	    SaveName, Filter = PySide.QtGui.QFileDialog.getSaveFileName(None, title, initial_path, f'*.{ext}') # PySide
	return SaveName

script_title = "Raft Patterning Script"
doc_name = "PatternDoc"
App.newDocument(doc_name)
AD = App.ActiveDocument		# just to make things more readable
starttime = time.time()
print("\nBEGIN " + script_title + "...") # print the script name

# Paths to source model
homepath = os.path.expanduser('~')
model_path = get_open_name(title='Select CAD model to pattern...', initial_dir=homepath, ext='STEP')
print(model_path)
model_dir, model_name = os.path.split(model_path)
base_name = "EnvelopesArray"

# Read in the source geometry
source_name  = "proto"
source       = AD.addObject("Part::Feature", source_name)
source.Shape = Part.read(model_path)

# Read in the raft positions
#pattern_name = '20220520T1631_desi2_layout_21rafts_1512robots.csv'
#pattern_path = os.path.join(homepath, pattern_name)
pattern_path = get_open_name(title='Select data table that specifies the pattern...', initial_dir=model_dir, ext='csv')
pattern_dir, pattern_name = os.path.split(pattern_path)
tbl = Table.read(pattern_path)

steptime = time.time()
print(f'... {len(tbl)} raft positions read')
lasttime = steptime

# Choose how many rafts to pattern (can argue a smaller number, i.e. for testing)
max_patterns = math.inf  # integer or math.inf
num_to_process = min(max_patterns, len(tbl))

rafts = []
raft_name = os.path.splitext(model_name)[0]
for i in range(num_to_process):
    # Generate the raft
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
init_export_path = os.path.join(model_dir, export_name)
export_path = get_save_name(title='Save patterned model as...', initial_path=init_export_path, ext='STEP')
import ImportGui # import GUI module
ImportGui.export(rafts, export_path) # requires GUI, does the export of geometry
App.getDocument(doc_name).removeObject(source_name) # deletes the proto raft, just to give user warm-fuzzies about what was exported
Gui.SendMsgToActiveView("ViewFit") # requires GUI, gives user warm-fuzzies
Gui.activeDocument().activeView().viewAxometric() # requires GUI, gives user warm-fuzzies
steptime = time.time()
print(f'... exported array in {(steptime-lasttime)/60:.2f} min')
lasttime = steptime

endtime = time.time()
runtime = endtime-starttime
print('... DONE')
print(f'Total runtime = {runtime/60:.2f} min\n')
