"""Mesh Cleaning"""

from collections import defaultdict
import numpy as np
from write_read_vtk import write_mesh_to_vtk,write_mesh_to_vtk_v2,read_vtk_to_mesh


def get_mesh_topology(nodes,triangles):
    edge_to_tris=defaultdict(list)
    for i, [a,b,c] in enumerate(triangles):
        for edge in [tuple(sorted((a,b))),tuple(sorted((b,c))),tuple(sorted((a,c)))]:
            edge_to_tris[edge].append((a,b,c))
    
    return edge_to_tris

    
def delete_boundary_triangles(nodes,tris):
    """delete non-manifold structure"""
    edge_to_tris=get_mesh_topology(nodes,tris)
    
    new_tris=[]
    boundary_tris=set()
    deleted_tris=set()

    for edge,edge_tris in edge_to_tris.items():
        if len(edge_tris)==1:
            boundary_tris.add(edge_tris[0])
    
    while boundary_tris:
        tri=boundary_tris.pop()
        (a,b,c)=tri
        deleted_tris.add(tri)
        
        for edge in [tuple(sorted((a,b))),tuple(sorted((b,c))),tuple(sorted((a,c)))]:
            if tri in edge_to_tris[edge]:
                edge_to_tris[edge].remove(tri)
                
                if len(edge_to_tris[edge])==1:
                    remaining_tri=edge_to_tris[edge][0]
                    if remaining_tri not in deleted_tris:
                        boundary_tris.add(remaining_tri)
              
    for tri in tris:
        tri=tuple(tri)
        if tri not in deleted_tris:
            new_tris.append(tri)
            
    deleted_tris_list = [list(tri) for tri in deleted_tris]
    
    print(f"After delete boundary triangles, the mesh has {len(new_tris)} triangles...")
    return nodes,new_tris,deleted_tris_list


def remove_dangling_meshes(nodes, tris):
    """remove isolated component based on BFS"""
    if len(tris) == 0:
        return nodes, tris
    
    nodes = np.asarray(nodes)
    tris = np.asarray(tris)
    
    edge_to_tris = defaultdict(list)
    tri_edges = []
    

    for ti, tri in enumerate(tris):
        a, b, c = tri
        tri_edge = []
        for edge in [tuple(sorted((a, b))), tuple(sorted((b, c))), tuple(sorted((a, c)))]:
            edge_to_tris[edge].append(ti)
            tri_edge.append(edge)
        tri_edges.append(tri_edge)
    

    visited = [False] * len(tris)
    regions = []
    for ti in range(len(tris)):
        if not visited[ti]:
            region = []
            stack = [ti]
            visited[ti] = True
            while stack:
                current_ti = stack.pop()
                region.append(current_ti)
                
                for edge in tri_edges[current_ti]:
                    for neighbor_tri in edge_to_tris[edge]:
                        if not visited[neighbor_tri]:
                            visited[neighbor_tri] = True
                            stack.append(neighbor_tri)
            regions.append(region)
    
    if not regions:
        return nodes, np.zeros((0, 3))
    

    largest_region = max(regions, key=len)
    filtered_tris = tris[largest_region]
    

    all_indices = set(range(len(tris)))
    keep_indices = set(largest_region)
    delete_indices = list(all_indices - keep_indices)
    deleted_tris = tris[delete_indices]
    
    print(f"After remove {len(tris) - len(filtered_tris)} dangling triangles from the mesh, it has {len(filtered_tris)} triangles...")
    

    return nodes, filtered_tris,deleted_tris

def label_deleted_tris(tris, deleted_tris):
 
    deleted_set = set(tuple(tri) for tri in deleted_tris)
    labels = []
    for tri in tris:
        tri_tuple = tuple(tri)
        if tri_tuple in deleted_set:
            labels.append(1)
        else:
            labels.append(0)
    
    return labels

    
def main():
    filename="abc_1333"
    vtk_file=f"simplify/{filename}_autogrid.vtk"
    old_nodes,old_tris=read_vtk_to_mesh(vtk_file)
    new_nodes,new_tris,deleted_tris_1=delete_boundary_triangles(old_nodes,old_tris)
    new_nodes,new_tris,deleted_tris_2=remove_dangling_meshes(new_nodes,new_tris)
    
    deleted_tris_1.extend(deleted_tris_2)
    
    labels=label_deleted_tris(old_tris,deleted_tris_1)
    
    write_mesh_to_vtk(old_nodes,old_tris,labels,f"./simplify/{filename}_relabel_autogrid.vtk",minus_one=False)
    write_mesh_to_vtk_v2(new_nodes,new_tris,f"./simplify/{filename}_result.vtk",minus_one=False)
    
    

main()
    
    


