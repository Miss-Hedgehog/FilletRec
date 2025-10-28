"""
A visualization tool for the machining feature dataset.

This module requires PythonOCC to run.
"""
import os
import random
import glob


from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.StepRepr import StepRepr_RepresentationItem
from OCC.Core.AIS import AIS_ColoredShape
from OCC.Display.SimpleGui import init_display
from OCC.Display.OCCViewer import rgb_color
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.StepRepr import StepRepr_RepresentationItem
from OCC.Core.AIS import AIS_ColoredShape
from OCC.Display.SimpleGui import init_display
from OCC.Display.OCCViewer import rgb_color


LABELS = [0, 1]
FEAT_NAMES = ["Non-fillet", "Fillet"]
COLORS = {
    "Non-fillet": rgb_color(1, 1, 1),  # White
    "Fillet": rgb_color(1, 0, 0),  # Red
}



def read_step_with_labels(filename):
    """Reads STEP file with labels on each B-Rep face."""
    if not os.path.exists(filename):
        print(filename, ' not exists')
        return

    reader = STEPControl_Reader()
    reader.ReadFile(filename)
    reader.TransferRoots()
    shape = reader.OneShape()

    treader = reader.WS().TransferReader()

    id_map = {}
    topo = TopologyExplorer(shape)
    faces = list(topo.faces())

    for face in faces:
        item = treader.EntityFromShapeResult(face, 1)
        if item is None:
            print(face)
            continue
        item = StepRepr_RepresentationItem.DownCast(item)
        name = item.Name().ToCString()
        if name:
            id_map[face] = int(name)

    return shape, id_map


def display():
    global shape_index
    global shape_paths

    shape, face_ids = read_step_with_labels(shape_paths[shape_index])

    if shape == None:
        return

    occ_display.EraseAll()
    AIS = AIS_ColoredShape(shape)

    for face, label in face_ids.items():
        feat_name = FEAT_NAMES[label]
        AIS.SetCustomColor(face, COLORS[feat_name])

    occ_display.Context.Display(AIS, True)
    occ_display.View_Iso()
    occ_display.FitAll()

    print(f"STEP: {shape_paths[shape_index]}")


def show_first():
    global shape_index
    shape_index = 0
    display()


def show_last():
    global shape_index
    global shape_paths

    shape_index = len(shape_paths) - 1
    display()


def show_next():
    global shape_index
    global shape_paths

    shape_index = (shape_index + 1) % len(shape_paths)
    display()


def show_previous():
    global shape_index
    global shape_paths

    shape_index = (shape_index - 1 + len(shape_paths)) % len(shape_paths)
    display()


def show_random():
    global shape_index
    global shape_paths

    shape_index = random.randrange(0, len(shape_paths))
    display()


if __name__ == '__main__':
    # User Defined
    
    dataset_dir = "save/"

    occ_display, start_occ_display, add_menu, add_function_to_menu = init_display()

    
    add_menu('explore')
    add_function_to_menu('explore', show_random)
    add_function_to_menu('explore', show_next)
    add_function_to_menu('explore', show_previous)
    add_function_to_menu('explore', show_first)
    add_function_to_menu('explore', show_last)

    #shape_paths = glob.glob(os.path.join(dataset_dir, '*.stp'))
    shape_paths = glob.glob(os.path.join(dataset_dir, '*.step'))

    print(len(shape_paths), 'shapes')

    if len(shape_paths) > 0:
        show_random()

    start_occ_display()
