"""This script allows for the running of Hierarchical CADNet on a single CAD model and saving the final result.

This script requires installing pythonocc: https://github.com/tpaviot/pythonocc.
"""

import tensorflow as tf
import numpy as np
import os

from collections import defaultdict

from OCC.Core.STEPConstruct import stepconstruct_FindEntity
from OCC.Core.TCollection import TCollection_HAsciiString
from OCC.Extend.DataExchange import read_step_file, STEPControl_Reader, STEPControl_Writer, STEPControl_AsIs
from OCC.Core.StepRepr import StepRepr_RepresentationItem
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.gp import gp_Pnt
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Surface
from OCC.Core.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Torus, GeomAbs_Cone, GeomAbs_Sphere, \
    GeomAbs_BezierSurface, GeomAbs_BSplineSurface, GeomAbs_SurfaceOfRevolution, GeomAbs_SurfaceOfExtrusion, \
    GeomAbs_OffsetSurface, GeomAbs_OtherSurface
from OCC.Core.GeomLProp import GeomLProp_SLProps
from OCC.Core.TopAbs import TopAbs_FORWARD, TopAbs_REVERSED
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps
from OCC.Core.TopoDS import topods_Face
from OCC.Core.gp import gp_Vec
from OCC.Core._BRepGProp import brepgprop_SurfaceProperties
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Extend.TopologyUtils import TopologyExplorer
from occwl.uvgrid import uvgrid
from OCC.Core.BRep import BRep_Tool
from occwl.face import Face

import json
from src.helper import normalize_curvature,normalize_attr,normalize_adj
from src.network import FilletRecGCN as FilletGCN

num_srf_U=5
num_srf_V=5


EPSILON = 1e-8


class WorkFace:
    def __init__(self, index, face):
        self.index = index
        self.hash = hash(face)
        self.face = face
        self.surface_area = None
        self.centroid = None
        self.face_type = None
        self.adjacent_angles=None
        self.adjacent_distance=None


class WorkEdge:
    def __init__(self, index, edge):
        self.index = index
        self.hash = hash(edge)
        self.edge = edge
        self.faces = []
        self.hash_faces = []
        self.face_tags = []
        # Convex = 0, Concave = 1, Other = 2
        self.convexity = None


class WorkFacet:
    """Stores information about each facet in mesh."""
    def __init__(self, facet_tag, face_tag, node_tags):
        self.facet_tag = facet_tag
        self.face_tag = face_tag
        self.node_tags = node_tags
        self.node_coords = []
        self.normal = None
        self.d_co = None
        self.centroid = None
        self.occ_face = None
        self.occ_hash_face = None

    def get_normal(self):
        vec1 = self.node_coords[1] - self.node_coords[0]
        vec2 = self.node_coords[2] - self.node_coords[1]
        norm = np.cross(vec1, vec2)
        self.normal = norm / np.linalg.norm(norm) + EPSILON

    def get_d_coefficient(self):
        self.d_co = -(self.normal[0] * self.node_coords[0][0] + self.normal[1] * self.node_coords[0][1]
                      + self.normal[2] * self.node_coords[0][2])

    def get_centroid(self):
        x = (self.node_coords[0][0] + self.node_coords[1][0] + self.node_coords[1][0]) / 3
        y = (self.node_coords[0][1] + self.node_coords[1][1] + self.node_coords[1][1]) / 3
        z = (self.node_coords[0][2] + self.node_coords[1][2] + self.node_coords[1][2]) / 3

        self.centroid = [x, y, z]

        
def get_faces(topo):
    work_faces = {}
    faces = list(topo.faces())

    for face in faces:
        wf = WorkFace(len(work_faces), face)
        wf.face_type = recognise_face_type(face)
        wf.surface_area = ask_surface_area(face)
        wf.centroid = ask_face_centroid(face)
        # wf.label = label_map[face]
        wf.adjacent_angles,wf.adjacent_distance=get_angle_between_adjacent_faces(face,topo)
        work_faces[wf.hash] = wf

    return work_faces, faces

def get_edge_midpoint_distance(edge1,edge2):
    curve1,first_param1,last_param1=BRep_Tool.Curve(edge1)
    curve2,first_param2,last_param2=BRep_Tool.Curve(edge2)
    
    point1=curve1.Value((first_param1+last_param1)/2.0).Coord()
    point2=curve2.Value((first_param2+last_param2)/2.0).Coord()
    
    p1=np.array(point1)
    p2=np.array(point2)
    
    distance=np.linalg.norm(p1-p2)
    
    return distance
    
    

def get_angle_between_adjacent_faces(face, topo):

    edges = list(topo.edges_from_face(face))
    edge_lengths = []
    
    for edge in edges:
        props = GProp_GProps()
        brepgprop.LinearProperties(edge, props)
        length = props.Mass()
        edge_lengths.append((edge, length))
    
    edge_lengths.sort(key=lambda x: x[1], reverse=True)
    
    if len(edge_lengths)>=2:
        main_edges = edge_lengths[:2]
        adjacent_faces=[]
        
        for edge in main_edges:
            adajcent_face=[f for f in topo.faces_from_edge(edge[0]) if not f.IsSame(face)]
            
            if len(adajcent_face)!=0:
                
                adjacent_faces.append(adajcent_face[0])
            else:
                adjacent_faces.append(face)
        
        angle=calculate_face_angle(adjacent_faces[0],main_edges[0][0],adjacent_faces[1],main_edges[1][0])
        distance=get_edge_midpoint_distance(main_edges[0][0],main_edges[1][0])
        
    else:
        angle=0.0
        props = GProp_GProps()
        brepgprop.LinearProperties(edge_lengths[0][0], props)
        distance= props.Mass()
        
    return angle,distance



def calculate_edge_centerpoint_normal(edge, face):
   
    curve, first_param, last_param = BRep_Tool.Curve(edge)
    
    start_point = curve.Value((first_param+last_param)/2.0)
    
    mid_coords = list(start_point.Coord())
    
    uv = ask_point_uv2(mid_coords, face)
    
    normal = ask_point_normal_face(uv, face)
    return normal


def calculate_face_angle(face1,edge1, face2,edge2):
    
    n1=calculate_edge_centerpoint_normal(edge1,face1)
    n2=calculate_edge_centerpoint_normal(edge2,face2)
   
    dot_product = np.dot(n1, n2)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
    
    return min(angle, np.pi - angle)


def get_brep_information(shape):
    topo = TopologyExplorer(shape)
    work_faces, faces = get_faces(topo)
    work_edges = get_edges(topo, faces)

    return work_faces, work_edges, faces


def ask_point_uv2(xyz, face):
    """
    This is a general function which gives the uv coordinates from the xyz coordinates.
    The uv value is not normalised.
    """
    gpPnt = gp_Pnt(float(xyz[0]), float(xyz[1]), float(xyz[2]))
    surface = BRep_Tool().Surface(face)

    sas = ShapeAnalysis_Surface(surface)
    gpPnt2D = sas.ValueOfUV(gpPnt, 0.01)
    uv = list(gpPnt2D.Coord())

    return uv


def ask_point_normal_face(uv, face):
    """
    Ask the normal vector of a point given the uv coordinate of the point on a face
    """
    face_ds = topods_Face(face)
    surface = BRep_Tool().Surface(face_ds)
    props = GeomLProp_SLProps(surface, uv[0], uv[1], 1, 1e-6)

    gpDir = props.Normal()
    if face.Orientation() == TopAbs_REVERSED:
        gpDir.Reverse()

    return gpDir.Coord()


def ask_edge_midpnt_tangent(edge):
    """
    Ask the midpoint of an edge and the tangent at the midpoint
    """
    result = BRep_Tool.Curve(edge)  # result[0] is the handle of curve;result[1] is the umin; result[2] is umax
    tmid = (result[1] + result[2]) / 2
    p = gp_Pnt(0, 0, 0)
    v1 = gp_Vec(0, 0, 0)
    result[0].D1(tmid, p, v1)  # handle.GetObject() gives Geom_Curve type, p:gp_Pnt, v1:gp_Vec

    return [p.Coord(), v1.Coord()]


def edge_dihedral(edge, faces):
    """
    Calculate the dihedral angle of an edge
    """
    [midPnt, tangent] = ask_edge_midpnt_tangent(edge)
    uv0 = ask_point_uv2(midPnt, faces[0])
    uv1 = ask_point_uv2(midPnt, faces[1])
    n0 = ask_point_normal_face(uv0, faces[0])
    n1 = ask_point_normal_face(uv1, faces[1])

    if edge.Orientation() == TopAbs_FORWARD:
        cp = np.cross(n0, n1)
        r = np.dot(cp, tangent)
        s = np.sign(r)

    else:
        cp = np.cross(n1, n0)
        r = np.dot(cp, tangent)
        s = np.sign(r)

    return s


def get_edges(topo, occ_faces):
    work_edges = {}

    edges = topo.edges()
    for edge in edges:
        faces = list(topo.faces_from_edge(edge))

        we = WorkEdge(len(work_edges), edge)

        if len(faces) > 1:
            s = edge_dihedral(edge, faces)
        else:
            s = 0

        if s == 1:
            # Convex
            edge_convexity = 0
        elif s == -1:
            # Concave
            edge_convexity = 1
        else:
            # Smooth (s==0) or other
            edge_convexity = 2

        we.convexity = edge_convexity
        we.faces = faces

        for face in faces:
            we.hash_faces.append(hash(face))
            we.face_tags.append(occ_faces.index(face))

        if len(faces) == 1:
            we.hash_faces.append(hash(faces[0]))
            we.face_tags.append(occ_faces.index(faces[0]))

        work_edges[we.hash] = we

    return work_edges


def ask_surface_area(f):
    props = GProp_GProps()

    brepgprop_SurfaceProperties(f, props)
    area = props.Mass()
    return area


def recognise_face_type(face):
    """Get surface type of B-Rep face"""
    #   BRepAdaptor to get the face surface, GetType() to get the type of geometrical surface type
    surf = BRepAdaptor_Surface(face, True)
    surf_type = surf.GetType()
    a = 0
    if surf_type == GeomAbs_Plane:
        a = 1
    elif surf_type == GeomAbs_Cylinder:
        a = 2
    elif surf_type == GeomAbs_Torus:
        a = 3
    elif surf_type == GeomAbs_Sphere:
        a = 4
    elif surf_type == GeomAbs_Cone:
        a = 5
    elif surf_type == GeomAbs_BezierSurface:
        a = 6
    elif surf_type == GeomAbs_BSplineSurface:
        a = 7
    elif surf_type == GeomAbs_SurfaceOfRevolution:
        a = 8
    elif surf_type == GeomAbs_OffsetSurface:
        a = 9
    elif surf_type == GeomAbs_SurfaceOfExtrusion:
        a = 10
    elif surf_type == GeomAbs_OtherSurface:
        a = 11

    return a


def ask_face_centroid(face):
    """Get centroid of B-Rep face."""
    mass_props = GProp_GProps()
    brepgprop.SurfaceProperties(face, mass_props)
    gPt = mass_props.CentreOfMass()

    return gPt.Coord()



def get_edge_dicts(facets):
    edge_dict = {}
    edge_facet_dict = {}

    for facet in facets.values():
        edge_1 = tuple(sorted((facet.node_tags[0], facet.node_tags[1])))
        edge_2 = tuple(sorted((facet.node_tags[0], facet.node_tags[2])))
        edge_3 = tuple(sorted((facet.node_tags[1], facet.node_tags[2])))

        edge_1_tag = len(edge_dict)
        edge_2_tag = edge_1_tag + 1
        edge_3_tag = edge_2_tag + 1

        edge_dict[edge_1_tag] = edge_1
        edge_dict[edge_2_tag] = edge_2
        edge_dict[edge_3_tag] = edge_3

        edge_facet_dict[edge_1_tag] = facet.facet_tag
        edge_facet_dict[edge_2_tag] = facet.facet_tag
        edge_facet_dict[edge_3_tag] = facet.facet_tag

    return edge_dict, edge_facet_dict




def get_sparse_tensor(adj_matrix, default_value=0.):
    idx = np.where(np.not_equal(adj_matrix, default_value))
    values = adj_matrix[idx]
    shape = np.shape(adj_matrix)

    idx = np.transpose(idx).astype(np.int32)
    values = values.astype(np.float32)
    shape = np.array(shape).astype(np.int32)

    return idx, values, shape


def get_face_features(faces):
    faces_list = []

    for face_tag, face in faces.items():
        face_list = [face.surface_area, face.centroid[0], face.centroid[1], face.centroid[2],
                     face.face_type]
        faces_list.append(face_list)

    return np.array(faces_list, dtype=np.float32)


def extract_face_curvature(face,num_srf_u,num_srf_v):
    new_face=Face(face.face)
    gaussian_curvatures=uvgrid(new_face, num_srf_u, num_srf_v, method="gaussian_curvature_v2")
    mean_curvatures=uvgrid(new_face, num_srf_v, num_srf_v, method="mean_curvature_v2")
    max_curvatures=uvgrid(new_face, num_srf_u, num_srf_v, method="max_curvature_v2")
    min_curvatures=uvgrid(new_face, num_srf_u, num_srf_v, method="min_curvature_v2")
    mask = uvgrid(new_face, num_srf_u, num_srf_v, method="inside")
    # return max_curvatures, min_curvatures, mask
    return gaussian_curvatures,mean_curvatures,mask


def get_face_features_fillet(faces):
    faces_list=[]
    for face_tag, face in faces.items():
        face_list=[]
        gaussian_curvature, mean_curvature,mask=extract_face_curvature(face,num_srf_U,num_srf_V)
        gaussian_curvature=gaussian_curvature.flatten().tolist()
        mean_curvature=mean_curvature.flatten().tolist()
        mask=mask.flatten().tolist()
        
        face_list.extend(gaussian_curvature)
        face_list.extend(mean_curvature)
        face_list.append(face.adjacent_distance)
        face_list.append(face.adjacent_angles)
        face_list.append(face.surface_area)
        # face_list.extend(mask) #this is for network3
        faces_list.append(face_list)
        
    return np.array(faces_list,dtype=np.float32)




def get_face_adj(edges, faces):
    brep_adj = np.zeros((len(faces), len(faces)))
    convex_adj = np.zeros((len(faces), len(faces)))
    concave_adj = np.zeros((len(faces), len(faces)))
    other_adj = np.zeros((len(faces), len(faces)))

    for edge in edges.values():
        a = edge.face_tags[0]
        b = edge.face_tags[1]

        brep_adj[a, b] = 1
        brep_adj[b, a] = 1

        if edge.convexity == 0:
            convex_adj[a, b] = 1
            convex_adj[b, a] = 1
        elif edge.convexity == 1:
            concave_adj[a, b] = 1
            concave_adj[b, a] = 1
        elif edge.convexity == 2:
            other_adj[a, b] = 1
            other_adj[b, a] = 1

    return brep_adj, convex_adj, concave_adj, other_adj

def get_graph_fillet(work_faces, work_facets, work_face_edges, work_facet_edges):
    V_1=get_face_features_fillet(work_faces)
    A_1, E_1, E_2, E_3 = get_face_adj(work_face_edges, work_faces)
    
    A_1_normalize=normalize_adj(A_1)
    
    curvature_data=tf.abs(V_1[:,:50])
    V_1_curvature=normalize_curvature(curvature_data)
    
    attr_data=V_1[:,50:51]#surface width
    V_1_attr=normalize_attr(attr_data)
    
    topo_data=V_1[:,51:52]#surface angle
    V_1_topo=normalize_attr(topo_data)

    
    return [V_1_curvature, V_1_attr, V_1_topo,A_1_normalize, E_1, E_2, E_3]
   
    
def read_step_file(filename):
    """Reads STEP file."""
    if not os.path.exists(filename):
        print(filename, ' not exists')
        return

    reader = STEPControl_Reader()
    reader.ReadFile(filename)
    reader.TransferRoots()
    shape = reader.OneShape()

    treader = reader.WS().TransferReader()

    topo = TopologyExplorer(shape)

    return shape, topo


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

    id_map = []
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
            nameid = name
            id_map.append(int(nameid))

    return shape, id_map, topo



def create_hier_graphs(step_path, with_labels=False):
    if with_labels:
        shape, labels, topo = read_step_with_labels(step_path)
    else:
        shape, topo = read_step_file(step_path)
        labels = None

    work_faces, work_edges, faces = get_brep_information(shape)
    facet_dict=None
    edge_facet_link=None
    graph=get_graph_fillet(work_faces,facet_dict,work_edges,edge_facet_link)

    return graph, shape, labels


def write_step_wth_prediction(filename, shape, prediction):
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)

    finderp = writer.WS().TransferWriter().FinderProcess()

    loc = TopLoc_Location()
    topo = TopologyExplorer(shape)
    faces = list(topo.faces())

    counter = 0
    for face in faces:
        item = stepconstruct_FindEntity(finderp, face, loc)
        if item is None:
            print(face)
            continue
        item.SetName(TCollection_HAsciiString(str(prediction[counter])))
        counter += 1

    writer.Write(filename)


def test_step(x):
    test_logits = model(x, training=False)
    y_pred = np.argmax(test_logits.numpy(), axis=1)
    return y_pred.tolist()


if __name__ == '__main__':
    with_labels = False
    
    step_dir = "cases/"
    save_dir="save/"
    
    step_name = "abc_613"
    checkpoint_path="checkpoint\\best.weights.h5" 
    

    num_classes = 2
    num_layers = 3
    dropout_rate = 0.3

    
    filters=[64,64,64]
    out_dim=32 #curvature feature
    units = 64#256
  
    
    model = FilletGCN(units=units,out_channel=out_dim,filter=filters,rate=dropout_rate, num_classes=num_classes, num_layers=num_layers)
    
    model.build(input_shape=[
    (None, 50),  # V_1
    (None,1),
    (None,1),
    (None,None), #A_1
    (None, None),              # E_1
    (None, None),              # E_2
    (None, None)           # E_3
])

    model.load_weights(checkpoint_path)


    loss_fn = tf.keras.losses.CategoricalCrossentropy()


    graph, shape, labels = create_hier_graphs(os.path.join(step_dir, f"{step_name}.step"), with_labels=with_labels)
    
    y_pred = test_step(graph)
    
    write_step_wth_prediction(os.path.join(save_dir, f"{step_name}_rec.step"), shape, y_pred)

    if with_labels:
        labels = np.array(labels)
        print(f"Predictions: {y_pred}")
        print(f"True labels: {labels}")

        print(f"Acc: {np.sum(np.where(y_pred == labels, 1, 0)) / labels.shape[0]}")
    
    
    
    
            

        