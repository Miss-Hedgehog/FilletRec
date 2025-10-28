
"""Mesh Extending"""

from OCC.Display.SimpleGui import init_display
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Core.StepRepr import StepRepr_RepresentationItem
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopTools import TopTools_IndexedDataMapOfShapeListOfShape,TopTools_ListOfShape
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.BRep import BRep_Tool
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop_LinearProperties
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Sewing
from OCC.Core.gp import gp_Vec, gp_Pnt
from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnCurve
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopAbs import TopAbs_FACE,TopAbs_WIRE,TopAbs_EDGE
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Surface
from OCC.Core.BRepTools import BRepTools_WireExplorer

from collections import defaultdict
import os
from write_read_vtk import write_mesh_to_vtk


#get face topology
def points_equal(p1, p2, tolerance=1e-6):
    return p1.Distance(p2) < tolerance


def get_edge_length(edge):
    props = GProp_GProps()
    brepgprop_LinearProperties(edge, props)
    return props.Mass()

def get_face_triangulation(face,deflection=0.1):
    
    #first get face's mesh ,then return triangulation data
    # mesh = BRepMesh_IncrementalMesh(face, deflection)  # 0.1 = mesh precision
    # mesh.Perform()
    location = TopLoc_Location()
    return BRep_Tool.Triangulation(face, location)

def is_point_on_edge(uv, cad_edge, face):
    surface = BRep_Tool.Surface(face)
    point = surface.Value(uv.X(), uv.Y())
    curve = BRep_Tool.Curve(cad_edge)[0]
    projector = GeomAPI_ProjectPointOnCurve(point, curve)
    return projector.NbPoints() > 0 and projector.LowerDistance() < 1e-3

def get_edge_wire_mapping(face, edges):
    wire_explorer = TopExp_Explorer(face, TopAbs_WIRE)
    wires = []
    while wire_explorer.More():
        wires.append(wire_explorer.Current())
        wire_explorer.Next()
        
    if not wires:
        return [None] * len(edges)
    
    edge_to_wire = {}
    external_wire = wires[0]
    wire_exp = BRepTools_WireExplorer(external_wire)
    while wire_exp.More():
        edge = wire_exp.Current()
        edge_to_wire[edge] = 1  #
        wire_exp.Next()
    
    for i, wire in enumerate(wires[1:], start=2):
        wire_exp = BRepTools_WireExplorer(wire)
        while wire_exp.More():
            edge = wire_exp.Current()
            edge_to_wire[edge] = i  # or a constant number for inner wire
            wire_exp.Next()
    
    results = []
    for edge in edges:
        found = False
        for mapped_edge in edge_to_wire.keys():
            # results.append(edge_to_wire[mapped_edge])
            if edge.IsSame(mapped_edge):
                results.append(edge_to_wire[mapped_edge])
                found = True
                break
        if not found:
            results.append(-1)
    return results


def read_step_with_labels(filename):
    """Reads STEP file with labels on each B-Rep face."""
    if not os.path.exists(filename):
        print(filename, ' not exists')
        return
    
    fillet_faces=[]
    non_fillet_faces=[]

    reader = STEPControl_Reader()
    reader.ReadFile(filename)
    reader.TransferRoots()
    shape = reader.OneShape()

    treader = reader.WS().TransferReader()

    id_map = {}
    topo = TopologyExplorer(shape)
    faces = list(topo.faces())
    
    face_counter=1

    for face in faces:
        item = treader.EntityFromShapeResult(face, 1)
        if item is None:
            print(face)
            continue
        item = StepRepr_RepresentationItem.DownCast(item)
        name = item.Name().ToCString()
        if name:
            if name=='1':
                fillet_faces.append(face)
            else:
                non_fillet_faces.append(face)
        face_counter+=1
    return shape, fillet_faces,non_fillet_faces



def remove_fillets_and_stitch(non_fillet_faces):
    """remove fillet faces and stitch to a new shape"""
   
    sewer = BRepBuilderAPI_Sewing(1e-6)
    for face in non_fillet_faces:
        sewer.Add(face)
 
    sewer.Perform()
    sewed_shape = sewer.SewedShape()
    
    return sewed_shape



def identify_boundary_edges(shape):
    """find boundary edges after removing fillet faces"""

    edge_map = TopTools_IndexedDataMapOfShapeListOfShape()
    
    face_exp = TopExp_Explorer(shape, TopAbs_FACE)
    while face_exp.More():
        face = face_exp.Current()
        edge_exp = TopExp_Explorer(face, TopAbs_EDGE)
        
        while edge_exp.More():
            edge = edge_exp.Current()
            if edge_map.Contains(edge):
                edge_map.ChangeFromKey(edge).Append(face)
            else:
                edge_map.Add(edge, TopTools_ListOfShape())
                edge_map.ChangeFromKey(edge).Append(face)
            edge_exp.Next()
        
        face_exp.Next()
    boundary_edges=[]
    for i in range(1, edge_map.Size() + 1):
        edge = edge_map.FindKey(i)
        if edge_map.FindFromIndex(i).Size() == 1:
            boundary_edges.append(edge)
    print(f"Has selected {len(boundary_edges)} boundary edges from the sewed_shape!")
    return boundary_edges


def find_existing_point(new_point,vertex_coord_to_index):
    for existing_point, idx in vertex_coord_to_index.items():
        if points_equal(existing_point, new_point):
            return idx
    return None


def get_edge_points(edge):
    try:
        curve, first, last = BRep_Tool.Curve(edge)
        return curve.Value(first), curve.Value(last)
    except :

        try:
            first_vertex = BRep_Tool.Pnt(BRep_Tool.FirstVertex(edge))
            last_vertex = BRep_Tool.Pnt(BRep_Tool.LastVertex(edge))
            return first_vertex, last_vertex
        except:
            adaptor = BRepAdaptor_Curve(edge)
            first_point = adaptor.Value(adaptor.FirstParameter())
            last_point = adaptor.Value(adaptor.LastParameter())
            return first_point, last_point
        
def get_edge_topology(edges):
    vertex_to_index={}
    point_coordinate=[]
    current_index=0
    edge_topology=[]
    for edge in edges:
        curve,first,last=BRep_Tool.Curve(edge)
        first_vertex=curve.Value(first)
        last_vertex=curve.Value(last)
        
        start_idx=find_existing_point(first_vertex,vertex_to_index)
        if start_idx is None:
            vertex_to_index[first_vertex]=current_index
            point_coordinate.append(first_vertex)
            start_idx=current_index
            current_index+=1
        
        end_idx=find_existing_point(last_vertex,vertex_to_index)
        if end_idx is None:
            vertex_to_index[last_vertex]=current_index
            point_coordinate.append(last_vertex)
            end_idx=current_index
            current_index+=1
        
        #sort edge vertex index 
        edge_topology.append(tuple(sorted((start_idx,end_idx))))
    
    return point_coordinate,edge_topology


def group_consistent_boundary_edges(boundary_edges, edges_topology, face_count):
    if not edges_topology:
        return []
    
    edges_topology_group = []
    topology_to_edge = {}  # Now maps (edge_topology, index) -> boundary_edge
    edges_group = []
    vertex_to_edges = defaultdict(list)  # Maps vertex -> list of (edge_topology, index)
    edge_set = set()  # Now stores (edge_topology, index) tuples
    
    # Assign unique IDs to edges using their indices
    for i, edge_topology in enumerate(edges_topology):
        edge_id = (edge_topology, i)  # Unique identifier for the edge
        vertex_to_edges[edge_topology[0]].append(edge_id)
        vertex_to_edges[edge_topology[1]].append(edge_id)
        topology_to_edge[edge_id] = boundary_edges[i]  # Store boundary edge with unique ID
        edge_set.add(edge_id)  # Track edges using their unique IDs
    
    while edge_set:
        start_edge = next(iter(edge_set))  # Get any remaining edge (edge_topology, index)
        edge_set.remove(start_edge)
        
        current_group = [start_edge]  # Track (edge_topology, index) pairs in this group
        edge_topology, _ = start_edge  # Extract the topology (v0, v1)
        end_points = list(edge_topology)  # Initialize with the edge's vertices
        
        while end_points:
            current_point = end_points.pop()
            
            # Iterate through all edges connected to current_point
            for edge_id in vertex_to_edges[current_point]:
                if edge_id in edge_set:  # Check if this edge is still unprocessed
                    edge_set.remove(edge_id)
                    current_group.append(edge_id)
                    
                    # Get the other vertex of the edge
                    other_point = edge_id[0][0] if edge_id[0][1] == current_point else edge_id[0][1]
                    end_points.append(other_point)
        
        edges_topology_group.append(current_group)
    
    # Convert (edge_topology, index) groups back to boundary edges
    for topology_group in edges_topology_group:
        edge_group = [topology_to_edge[edge_id] for edge_id in topology_group]
        edges_group.append(edge_group)
    
    print("Grouped edge topologies:", edges_topology_group)
    print(f"Face {face_count} boundary edges can be divided into {len(edges_group)} groups!")
    
    return edges_group


def find_vertices_on_cad_edge(cad_face,cad_edges,mesh):
    edge_vertices_wire_mappping={}
    edge_vertices=set()
    edge_wire_mapping=get_edge_wire_mapping(cad_face,cad_edges) #a list storing wire number for boundary edges
    uv_solver=ShapeAnalysis_Surface(BRep_Tool.Surface(cad_face))
    
    for i in range(1,mesh.NbNodes()+1):
        node=mesh.Nodes().Value(i)
        uv=uv_solver.ValueOfUV(node,1e-5)
        for j, cad_edge in enumerate(cad_edges):
            if is_point_on_edge(uv,cad_edge,cad_face) and i not in edge_vertices:
                edge_vertices.add(i)
                edge_vertices_wire_mappping[i]=edge_wire_mapping[j]
                
                break
    return edge_vertices,edge_vertices_wire_mappping


def build_mesh_topology(mesh):
    edge_to_tris=defaultdict(list)
    vertex_to_edges=defaultdict(list)
    triangles=mesh.Triangles()
    
    for t in range(1,mesh.NbTriangles()+1):
        tri=triangles.Value(t)
        v1,v2,v3=tri.Get()
        
        for edge in [tuple(sorted((v1,v2))),tuple(sorted((v2,v3))),tuple(sorted((v3,v1)))]:
            edge_to_tris[edge].append((v1,v2,v3))
            vertex_to_edges[edge[0]].append(edge)
            vertex_to_edges[edge[1]].append(edge)
    
    return edge_to_tris,vertex_to_edges   
    

def find_boundary_mesh_edges(cad_face,face_count,cad_edges,mesh,flag=False):
    #get boundary mesh edge vertices
    edge_vertices,edge_vertices_wire_mapping=find_vertices_on_cad_edge(cad_face,cad_edges,mesh)
    
    #get mesh topology
    edge_to_tris,vertex_to_edges=build_mesh_topology(mesh)
    
    #get boundary mesh edge
    boundary_edges=[]
    adjacent_tris=[]
    # boundary_edges_wire={}
    all_boundary_edges=defaultdict(list)
    
    for edge,tris in edge_to_tris.items():
        if len(tris)==1 and edge[0] in edge_vertices and edge[1] in edge_vertices:
            boundary_edges.append(edge)
            adjacent_tris.append(tris[0])
            # boundary_edges_wire[edge]=edge_vertices_wire_mapping[edge[0]] #use the first vertex wire num uniformly
        
        #record all boundary mesh facet
        if len(tris)==1:
            all_boundary_edges[edge[0]].append(edge)
            all_boundary_edges[edge[1]].append(edge)
            
    print(f"Find {len(boundary_edges)} boundary mesh edges and {len(adjacent_tris)} triangles!")
    
    if flag:
        
        #look for another two endpoints
        boundary_vertices=defaultdict(list)
        for i, (start_idx,end_idx) in enumerate(boundary_edges):
            boundary_vertices[start_idx].append(i)
            boundary_vertices[end_idx].append(i) #mappi

        # expand at two end point
        for point_idx,edges_indices in boundary_vertices.items():
            if len(edges_indices)==1:
                for edge in all_boundary_edges[point_idx]:
                    if edge not in boundary_edges:
                        boundary_edges.append(edge)
        
                        adjacent_tris.append(edge_to_tris[edge][0])
                        edge_vertices_wire_mapping[edge[0]]=edge_vertices_wire_mapping[point_idx]
                        edge_vertices_wire_mapping[edge[1]]=edge_vertices_wire_mapping[point_idx]
                        
                    # boundary_edges_wire[edge]= edge_vertices_wire_mapping[point_idx]
                    
                
    return boundary_edges,edge_vertices_wire_mapping,adjacent_tris



def calculate_length(boundary_cad_edges, num):
    edge_lengths = []
    for edge in boundary_cad_edges:
        length=get_edge_length(edge)
        edge_lengths.append((length, edge))
    edge_lengths.sort(key=lambda x: x[0])
    shortest_edges = [edge for length, edge in edge_lengths[:num]]
    shortest_lengths = [length for length, edge in edge_lengths[:num]]
    
    return shortest_edges
    


def add_mesh_at_boundary(face,face_count,boundary_cad_edges,mesh,node_count,offset_distance=10.0,inner_point_distance=8.0):
    
    flag=False

    boundary_edges,boundary_edges_wire,boundary_edges_tris=find_boundary_mesh_edges(face,face_count,boundary_cad_edges,mesh,flag)
    boundary_vertices=defaultdict(list)
    new_points_to_index={}
    new_points=[]
    new_tris=[]
    twoend_points=[]
    
    # if face_count in []:
    #     offset_distance = 
    
    for i, (start_idx,end_idx) in enumerate(boundary_edges):
        boundary_vertices[start_idx].append(i)
        boundary_vertices[end_idx].append(i) #mapping index of boundary_edges and boundary_edges_wire and boundary_edges_tris
    
    for point_idx, edges_indices in boundary_vertices.items():
        original_node=mesh.Nodes().Value(point_idx)
        
        avg_direction=gp_Vec(0,0,0)
        
        #look for two end points if have
        if len(edges_indices)==1:
            (start_idx,end_idx)=boundary_edges[edges_indices[0]]
            
            if start_idx==point_idx:
                twoend_points.append([start_idx,end_idx])
            else:
                twoend_points.append([end_idx,start_idx])
            
            # twoend_points.append(point_idx)
        
        for edge_index in edges_indices:
            (start_idx,end_idx)=boundary_edges[edge_index]
            (node_a,node_b,node_c)=boundary_edges_tris[edge_index]
            
            tri_nodes=[mesh.Nodes().Value(node_a),
                       mesh.Nodes().Value(node_b),
                       mesh.Nodes().Value(node_c)
                       ]
            
            vec1=gp_Vec(tri_nodes[0],tri_nodes[1])
            vec2=gp_Vec(tri_nodes[0],tri_nodes[2])
            
            
            facet_normal=vec1.Crossed(vec2).Normalized()
            
            
            if point_idx == start_idx:
                edge_tangent = gp_Vec(mesh.Nodes().Value(start_idx),
                                    mesh.Nodes().Value(end_idx)).Normalized()
            else:
                edge_tangent = gp_Vec(mesh.Nodes().Value(end_idx),
                                    mesh.Nodes().Value(start_idx)).Normalized()
            
            move_dir=facet_normal.Crossed(edge_tangent)
            
            face_center = gp_Pnt(*(sum(n.Coord()[i] for n in tri_nodes)/3 for i in range(3)))
                                 
            if gp_Vec(face_center, original_node).Dot(move_dir) < 0:
                move_dir = move_dir.Reversed()
           
                
            avg_direction+=move_dir
        
        avg_direction=avg_direction/len(edges_indices)
        translation_direction=avg_direction*offset_distance
        new_point=original_node.Translated(translation_direction)
        
        new_points.append((new_point.X(), new_point.Y(), new_point.Z()))
        new_points_to_index[point_idx]=len(new_points) #store original_point_idx--->new gP_Pnt index
    
    
    for i, (start_idx,end_idx) in enumerate(boundary_edges):
        new_start_idx=new_points_to_index[start_idx]+node_count
        new_end_idx=new_points_to_index[end_idx]+node_count
        new_tris.append((start_idx,new_start_idx,new_end_idx)) #add two new triangles for one edge
        new_tris.append((start_idx,end_idx,new_end_idx))
    

    return new_points,new_tris
    

def add_mesh_at_face(face,face_count,boundary_edges,offset):
    """extend mesh for a triangulated b-rep face"""
    old_triangulation=get_face_triangulation(face)
    new_triangulations=[]
    
    _,edge_topology=get_edge_topology(boundary_edges)
    print(edge_topology)
    boundary_edges_group=group_consistent_boundary_edges(boundary_edges,edge_topology,face_count)
    
    nodes_num=old_triangulation.NbNodes()
    nodes_tris=old_triangulation.NbTriangles()
    
    all_nodes=[]
    all_tris=[]
    all_labels=[]
    
    for i in range(1, old_triangulation.NbNodes() + 1):
        node = old_triangulation.Nodes().Value(i)
        all_nodes.append(node.Coord())
    
    for i in range(1, old_triangulation.NbTriangles() + 1):
        tri = old_triangulation.Triangle(i)
        all_tris.append((tri.Value(1), tri.Value(2), tri.Value(3)))
    
    all_labels.extend([0]*old_triangulation.NbTriangles())
    
    for group in boundary_edges_group:
        nodes,tris=add_mesh_at_boundary(face,face_count,group,old_triangulation,len(all_nodes),offset_distance=offset) #update node index 
        
        all_nodes.extend(nodes)
        all_tris.extend(tris)
        all_labels.extend([1]*len(tris))
       
    print(f"Face {face_count} originally has {nodes_tris} triangles, after extend it has {len(all_tris)} triangles!")
    return all_nodes,all_tris,all_labels


def simplify_fillet_with_mesh(shape,all_boundary_edges,offset_dis,deflection=0.1):
    """extend mesh along boundary edges"""
    
    print("Meshing the shape without fillet faces!")
    mesh = BRepMesh_IncrementalMesh(shape, deflection)  
    mesh.Perform()
    
    face_exp = TopExp_Explorer(shape, TopAbs_FACE)
    count=0
    
    combined_nodes = []
    combined_tris = []
    combined_labels=[]
    index_offset = 0
    
    while face_exp.More():
        
        face = face_exp.Current()
        
        edge_exp = TopExp_Explorer(face, TopAbs_EDGE)
        boundary_edges=[]
        while edge_exp.More():
            edge = edge_exp.Current()
            if edge in all_boundary_edges:
                boundary_edges.append(edge)
            edge_exp.Next()
            
        print("*"*50+f"Face {count}"+"*"*50)
        
        if len(boundary_edges)!=0:
            print(f"Face {count} has {len(boundary_edges)} boundary edges, need to be extended!")
            nodes,tris,labels = add_mesh_at_face(face,count,boundary_edges,offset_dis)
            
            adjusted_tris = [(a + index_offset, b + index_offset, c + index_offset) for (a, b, c) in tris]
            combined_nodes.extend(nodes)
            combined_tris.extend(adjusted_tris)
            combined_labels.extend(labels)
            
            index_offset += len(nodes)
            
        else:
            print(f"Face {count} doesn't has to be changed!")
            triangulation=get_face_triangulation(face)
            
            face_nodes = []
            for i in range(1, triangulation.NbNodes() + 1):
                node = triangulation.Nodes().Value(i)
                face_nodes.append(node.Coord())
            
            face_tris = []
            for i in range(1, triangulation.NbTriangles() + 1):
                tri = triangulation.Triangle(i)
                face_tris.append((tri.Value(1) + index_offset, 
                                tri.Value(2) + index_offset, 
                                tri.Value(3) + index_offset))
            
            labels=[0]*triangulation.NbTriangles()
            
            combined_nodes.extend(face_nodes)
            combined_tris.extend(face_tris)
            combined_labels.extend(labels)
            
            index_offset += len(face_nodes)
        
        
        count+=1
        face_exp.Next()
    
    return combined_nodes,combined_tris,combined_labels
    

def visualize(shape):
    """Visualize original and simplified models"""
    display, start_display, add_menu, add_function_to_menu = init_display()
    display.DisplayShape(shape, color='BLUE',transparency=0.5)
    display.FitAll()
    start_display()
        
    
def main():
    # 1.read step file and label
    filename="1333"
    step_file = f"save/abc_{filename}_rec.step" 
    original_shape,fillet_faces,non_fillet_faces=read_step_with_labels(step_file)
    print(f"step file {step_file} has {len(fillet_faces)} fillet faces and {len(non_fillet_faces)} non_fillet faces!")
    
    # 2.remove fillet faces and stitch a new shape
    sewed_shape=remove_fillets_and_stitch(non_fillet_faces)
    
    # 3.find boundary edges after removing fillet faces
    boundary_edges=identify_boundary_edges(sewed_shape) 
    
    # 4.extend mesh along boundary edges
    new_nodes,new_tris,new_labels=simplify_fillet_with_mesh(sewed_shape,boundary_edges,offset_dis=5.0,deflection=0.1) 
   
    # 5.write extended mesh to vtk file
    write_mesh_to_vtk(new_nodes,new_tris,new_labels,f"./simplify/abc_{filename}_sim.vtk")
    
    # visualize(sewed_shape)
    
     
if __name__ == "__main__":
    main()