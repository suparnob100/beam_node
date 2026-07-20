import json
import math
import os

import numpy as np


def _boundary_edges_from_tris(tri_conn_zero_based):
    edge_count = {}
    for tri in tri_conn_zero_based:
        a, b, c = [int(v) for v in tri]
        edges = [(a, b), (b, c), (c, a)]
        for i, j in edges:
            key = tuple(sorted((i, j)))
            edge_count[key] = edge_count.get(key, 0) + 1
    return [edge for edge, count in edge_count.items() if count == 1]


def _two_point_gauss_segment_integral(x1, x2, gfun):
    # Integral over a line segment of N^T g(x) ds for linear edge shape functions.
    gp = (-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0))
    gw = (1.0, 1.0)
    length = abs(x2 - x1)
    jac = 0.5 * length
    out = np.zeros(2, dtype=float)
    for s, w in zip(gp, gw):
        n1 = 0.5 * (1.0 - s)
        n2 = 0.5 * (1.0 + s)
        x = n1 * x1 + n2 * x2
        g = gfun(x)
        out += np.array([n1, n2], dtype=float) * g * jac * w
    return out


def generate_common_mesh(
    out_dir='.',
    base_name='beam_ellipse_shared',
    L=1.0,
    H=0.1,
    a=0.08,
    b=0.025,
    mesh_size=0.006,
    n_top_sample_pts=101,
):
    try:
        import gmsh
    except ImportError as exc:
        raise ImportError('Gmsh Python API is required. Install with: pip install gmsh') from exc

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    msh_path = os.path.join(out_dir, base_name + '.msh')
    inp_path = os.path.join(out_dir, base_name + '.inp')
    json_path = os.path.join(out_dir, base_name + '.json')

    gmsh.initialize()
    gmsh.option.setNumber('General.Terminal', 0)

    try:
        gmsh.model.add(base_name)
        cx = 0.5 * L
        cy = 0.5 * H

        rect = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, L, H)
        hole = gmsh.model.occ.addDisk(cx, cy, 0.0, a, b)
        cut_out, _ = gmsh.model.occ.cut([(2, rect)], [(2, hole)], removeObject=True, removeTool=True)
        gmsh.model.occ.synchronize()

        if len(cut_out) != 1:
            raise RuntimeError('Expected exactly one remaining surface after cut.')

        surface_tag = cut_out[0][1]

        gmsh.option.setNumber('Mesh.CharacteristicLengthMin', mesh_size)
        gmsh.option.setNumber('Mesh.CharacteristicLengthMax', mesh_size)
        gmsh.option.setNumber('Mesh.Algorithm', 6)
        gmsh.option.setNumber('Mesh.ElementOrder', 1)

        gmsh.model.mesh.generate(2)
        gmsh.write(msh_path)
        gmsh.write(inp_path)

        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        coords = np.asarray(coords, dtype=float).reshape(-1, 3)
        node_tags = np.asarray(node_tags, dtype=int)
        node_id_to_idx = {int(tag): i for i, tag in enumerate(node_tags)}

        elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2, surface_tag)
        tri_conn_labels = None
        tri_elem_labels = None
        for etype, etags, conn in zip(elem_types, elem_tags, elem_node_tags):
            props = gmsh.model.mesh.getElementProperties(etype)
            _, dim, _, nnode, _, _ = props
            if dim == 2 and nnode == 3:
                tri_conn_labels = np.asarray(conn, dtype=int).reshape(-1, 3)
                tri_elem_labels = np.asarray(etags, dtype=int)
                break
        if tri_conn_labels is None:
            raise RuntimeError('No TRI3 elements found in Gmsh mesh.')

        tri_conn_zero_based = np.array(
            [[node_id_to_idx[int(n)] for n in elem] for elem in tri_conn_labels],
            dtype=int,
        )

        x = coords[:, 0]
        y = coords[:, 1]
        tol = max(1.0e-9, 1.0e-6 * max(L, H))

        left_nodes = node_tags[np.isclose(x, 0.0, atol=tol)].astype(int)
        right_nodes = node_tags[np.isclose(x, L, atol=tol)].astype(int)
        top_nodes = node_tags[np.isclose(y, H, atol=tol)].astype(int)
        bottom_nodes = node_tags[np.isclose(y, 0.0, atol=tol)].astype(int)

        # Identify boundary edges and keep only the outer top boundary edges.
        boundary_edges_zero_based = _boundary_edges_from_tris(tri_conn_zero_based)
        top_edges_zero_based = []
        for i0, i1 in boundary_edges_zero_based:
            if abs(coords[i0, 1] - H) <= tol and abs(coords[i1, 1] - H) <= tol:
                top_edges_zero_based.append((i0, i1))

        def gfun(xcoord):
            sigma_x = 0.025 * L
            return math.exp(-((xcoord - 0.5 * L) ** 2) / (2.0 * sigma_x ** 2))

        top_unit_nodal_load_coeff_y = np.zeros(len(node_tags), dtype=float)
        for i0, i1 in top_edges_zero_based:
            x0 = coords[i0, 0]
            x1 = coords[i1, 0]
            local = _two_point_gauss_segment_integral(x0, x1, gfun)
            top_unit_nodal_load_coeff_y[i0] += local[0]
            top_unit_nodal_load_coeff_y[i1] += local[1]

        # Top samples: nearest top-edge nodes to a uniform x-grid.
        top_nodes_sorted = np.array(sorted(top_nodes, key=lambda n: coords[node_id_to_idx[int(n)], 0]), dtype=int)
        top_x_sorted = np.array([coords[node_id_to_idx[int(n)], 0] for n in top_nodes_sorted], dtype=float)
        x_sample = np.linspace(0.0, L, int(n_top_sample_pts))
        top_sample_node_labels = []
        for xs in x_sample:
            idx = int(np.argmin(np.abs(top_x_sorted - xs)))
            top_sample_node_labels.append(int(top_nodes_sorted[idx]))

        payload = {
            'base_name': base_name,
            'L': float(L),
            'H': float(H),
            'a': float(a),
            'b': float(b),
            'mesh_size': float(mesh_size),
            'node_labels': [int(v) for v in node_tags.tolist()],
            'coords': coords.tolist(),
            'element_labels': [int(v) for v in tri_elem_labels.tolist()],
            'tri3_connectivity_labels': [[int(n) for n in row] for row in tri_conn_labels.tolist()],
            'boundary_node_labels': {
                'left': [int(v) for v in left_nodes.tolist()],
                'right': [int(v) for v in right_nodes.tolist()],
                'top': [int(v) for v in top_nodes.tolist()],
                'bottom': [int(v) for v in bottom_nodes.tolist()],
            },
            'top_sample_x': [float(v) for v in x_sample.tolist()],
            'top_sample_node_labels': [int(v) for v in top_sample_node_labels],
            'top_unit_nodal_load_coeff_y': [float(v) for v in top_unit_nodal_load_coeff_y.tolist()],
            'files': {
                'msh': os.path.abspath(msh_path),
                'inp': os.path.abspath(inp_path),
                'json': os.path.abspath(json_path),
            },
        }

        with open(json_path, 'w') as f:
            json.dump(payload, f, indent=2)

    finally:
        gmsh.finalize()

    return {
        'msh_path': os.path.abspath(msh_path),
        'inp_path': os.path.abspath(inp_path),
        'json_path': os.path.abspath(json_path),
    }



if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out = generate_common_mesh(out_dir=script_dir)
    print('Generated:')
    print('  %s' % out['msh_path'])
    print('  %s' % out['inp_path'])
    print('  %s' % out['json_path'])
