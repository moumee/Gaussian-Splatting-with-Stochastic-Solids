# adopted from https://github.com/autonomousvision/gaussian-opacity-fields/blob/main/extract_mesh.py
import os
from argparse import ArgumentParser

import numpy as np
import open3d as o3d
import torch
import trimesh
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import (
    GaussianModel,
    evaluate_transmittance,
    evaluate_sdf,
    evaluate_color,
)
from scene import Scene
from tetranerf.utils.extension import cpp
from utils.tetmesh import InsufficientMarchingMemory, marching_tetrahedra


def post_process_mesh(mesh, cluster_to_keep=1):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    import copy

    print(
        "post processing the mesh to have {} clusterscluster_to_kep".format(
            cluster_to_keep
        )
    )
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (
            mesh_0.cluster_connected_triangles()
        )

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50)  # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0


@torch.no_grad()
def evaluation_validation(view, points, inside):
    if view.gt_mask is None:
        return inside
    return (sample_mask(view, points) > 0.5) & inside


@torch.no_grad()
def dilated_mask(view, dilate_px):
    """마스크를 dilate_px 픽셀만큼 키운다. 마스크 경계 오차로 실제 표면이 깎이는 것을 막기 위함"""
    cached = getattr(view, "_dilated_mask", None)
    if cached is not None and cached[0] == dilate_px:
        return cached[1]
    mask = view.gt_mask.cuda()
    if dilate_px > 0:
        k = 2 * dilate_px + 1
        mask = torch.nn.functional.max_pool2d(mask[None], k, stride=1, padding=dilate_px)[0]
    view._dilated_mask = (dilate_px, mask)
    return mask


@torch.no_grad()
def sample_mask(view, points, mask=None):
    if mask is None:
        mask = view.gt_mask.cuda()
    points_cam = points @ view.R + view.T
    pts2d = points_cam[:, :2] / points_cam[:, 2:]
    pts2d = torch.addcmul(
        pts2d.new_tensor(
            [
                (view.Cx * 2.0 + 1.0) / view.image_width - 1.0,
                (view.Cy * 2.0 + 1.0) / view.image_height - 1.0,
            ]
        ),
        pts2d.new_tensor(
            [view.Fx * 2.0 / view.image_width, view.Fy * 2.0 / view.image_height]
        ),
        pts2d,
    )
    sampled_mask = torch.nn.functional.grid_sample(
        mask[None], pts2d[None, None], align_corners=True
    )
    return sampled_mask.squeeze()


@torch.no_grad()
def evaluate_alpha_cull(
    points, views, gaussians, pipeline, kernel_size, carve_views=2, mask_dilate=4
):
    """점마다 transmittance 기반 SDF를 구한다 (양수 = 빈 공간, 음수 = 물체 내부).

    carve_views > 0이면 화면 안에 들어오지만 (키운) 마스크 밖으로 투영되는 시점이
    carve_views개 이상인 점을 빈 공간으로 확정한다 (visual hull carving).
    이게 없으면 마스크 밖 시점이 그냥 무시되어, 어느 카메라에서도 직접 보이지 않는 곳
    (예: 위에서만 찍은 물체의 바닥 둘레)이 물체 내부로 남아 막이 생긴다.
    """
    final_sdf = []
    any_valid = []
    chunk_size = 10000000
    for point_chunk in torch.chunk(points, points.shape[0] // chunk_size + 1):
        final_weight_chunk = torch.zeros(
            point_chunk.shape[0], dtype=torch.float32, device="cuda"
        )
        any_valid_chunk = torch.zeros(
            point_chunk.shape[0], dtype=torch.bool, device="cuda"
        )
        carve_count_chunk = torch.zeros(
            point_chunk.shape[0], dtype=torch.int32, device="cuda"
        )
        for view in tqdm(views, desc="Rendering progress"):
            ret = evaluate_transmittance(
                point_chunk, view, gaussians, pipeline, kernel_size
            )
            inside = ret["inside"]
            if view.gt_mask is None:
                valid_points = inside
            else:
                valid_points = (sample_mask(view, point_chunk) > 0.5) & inside
                if carve_views > 0:
                    grown = sample_mask(view, point_chunk, dilated_mask(view, mask_dilate))
                    carve_count_chunk += ((grown <= 0.5) & inside).int()
            any_valid_chunk = torch.logical_or(any_valid_chunk, valid_points)
            final_weight_chunk = torch.where(
                valid_points,
                torch.max(ret["transmittance"], final_weight_chunk),
                final_weight_chunk,
            )
        final_weight_chunk[torch.logical_not(any_valid_chunk)] = 0
        if carve_views > 0:
            carved = carve_count_chunk >= carve_views
            final_weight_chunk[carved] = 1.0
            any_valid_chunk = torch.logical_or(any_valid_chunk, carved)
        final_sdf_chunk = final_weight_chunk - 0.5
        final_sdf.append(final_sdf_chunk)
        any_valid.append(any_valid_chunk)
    return torch.cat(final_sdf), torch.cat(any_valid)


@torch.no_grad()
def add_color(
    points, views, gaussians, pipeline, kernel_size, background, max_dist=0.008
):
    surface_points = []
    chunk_size = 10000000
    accumulated_colors = []
    for point_chunk in torch.chunk(points, points.shape[0] // chunk_size + 1):
        n_points = point_chunk.shape[0]
        accumulated_colors_chunk = torch.zeros(
            (n_points, 3), dtype=torch.float32, device="cuda"
        )
        accumulated_points_chunk = torch.zeros(
            n_points, dtype=torch.int32, device="cuda"
        )
        for view in tqdm(views, desc="Rendering progress"):
            transmittance = evaluate_transmittance(
                point_chunk, view, gaussians, pipeline, kernel_size
            )["transmittance"]
            ret = evaluate_sdf(point_chunk, view, gaussians, pipeline, kernel_size)
            sdf = ret["sdf"]
            valid_points = (
                evaluation_validation(view, point_chunk, ret["inside"])
                & (sdf.abs() < max_dist)
                & (transmittance > 0.2)
            )
            color = evaluate_color(
                point_chunk, view, gaussians, pipeline, kernel_size, background
            )["color"]
            accumulated_colors_chunk = torch.where(
                valid_points.unsqueeze(-1),
                accumulated_colors_chunk + color,
                accumulated_colors_chunk,
            )
            accumulated_points_chunk = torch.where(
                valid_points, accumulated_points_chunk + 1, accumulated_points_chunk
            )
        accumulated_colors.append(
            accumulated_colors_chunk
            / accumulated_points_chunk.unsqueeze(-1).type(torch.float32)
        )
        surface_points.append(accumulated_points_chunk > 0)
    return torch.cat(accumulated_colors), torch.cat(surface_points)


@torch.no_grad()
def add_color_silhouette(points, views, gaussians, pipeline, kernel_size, background):
    chunk_size = 10000000
    colors = []
    for point_chunk in torch.chunk(points, points.shape[0] // chunk_size + 1):
        n_points = point_chunk.shape[0]
        colors_chunk = torch.full(
            (n_points, 3), 0.5, dtype=torch.float32, device="cuda"
        )
        transmittance_chunk = torch.zeros(n_points, dtype=torch.int32, device="cuda")
        for view in tqdm(views, desc="Rendering progress"):
            ret = evaluate_transmittance(
                point_chunk, view, gaussians, pipeline, kernel_size
            )
            transmittance = ret["transmittance"]
            valid_points = evaluation_validation(view, point_chunk, ret["inside"])
            color = evaluate_color(
                point_chunk, view, gaussians, pipeline, kernel_size, background
            )["color"]
            update = (transmittance > transmittance_chunk) & valid_points
            colors_chunk = torch.where(update.unsqueeze(-1), color, colors_chunk)
            transmittance_chunk = torch.where(
                update, transmittance, transmittance_chunk
            )
        colors.append(colors_chunk)

    return torch.cat(colors)


def flatten_bottom(mesh, views, observed, quantile=0.002):
    """관측되지 않은 바닥 아래 정점을 관측된 가장 낮은 높이의 평면으로 눌러 평평하게 만든다.

    위에서만 찍은 물체는 바닥 밑이 어떤 시점에서도 마스크 밖으로 나오지 않아 carving이 안 되고,
    silhouette cone 모양의 볼록한 캡이 생긴다. 테이블 위에 놓인 물체라고 가정하고 그 캡을 평면으로 자른다.
    위쪽 방향은 카메라 -y축(화면 위쪽)의 평균으로 잡는다.
    """
    up = -torch.stack([view.R[:, 1] for view in views]).mean(0).cpu().numpy().astype(np.float64)
    up /= np.linalg.norm(up)
    verts = np.asarray(mesh.vertices)
    height = verts @ up
    if observed.sum() == 0:
        return
    floor = np.quantile(height[observed], quantile)
    below = (height < floor) & ~observed
    verts[below] += (floor - height[below])[:, None] * up[None]
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    print("flatten bottom: moved {} unobserved vertices onto the floor plane".format(below.sum()))


@torch.no_grad()
def marching_tetrahedra_with_binary_search(
    model_path,
    views,
    gaussians,
    pipeline,
    kernel_size,
    background,
    export_color,
    move_cpu,
    save_cells,
    num_cluster,
    carve_views=2,
    mask_dilate=4,
    flatten_unseen_bottom=False,
):
    # generate tetra points here
    points, points_scale = gaussians.get_tetra_points()

    print("construct cell")
    cells = cpp.triangulate(points)
    if move_cpu:
        cells = cells.cpu()
    if save_cells:
        torch.save(cells.cpu(), os.path.join(model_path, "cells.pt"))

    sdf, valid = evaluate_alpha_cull(
        points, views, gaussians, pipeline, kernel_size, carve_views, mask_dilate
    )

    torch.cuda.empty_cache()
    try:
        (end_points, end_sdf), end_scales, faces, _ = marching_tetrahedra(
            points, cells, sdf, points_scale, valid
        )
    except (InsufficientMarchingMemory, torch.cuda.OutOfMemoryError):
        if not cells.is_cuda:
            raise
        print("GPU edge sort exceeds available memory; streaming cells through CUDA instead")
        cells = cells.cpu()
        torch.cuda.empty_cache()
        (end_points, end_sdf), end_scales, faces, _ = marching_tetrahedra(
            points, cells, sdf, points_scale, valid
        )

    faces = faces.cpu().numpy()
    points = (end_points[:, 0, :] + end_points[:, 1, :]) * 0.5

    mesh = trimesh.Trimesh(vertices=points.cpu().numpy(), faces=faces, process=False)
    mesh.export(os.path.join(model_path, "recon_init.ply"))

    left_points = end_points[:, 0, :]
    right_points = end_points[:, 1, :]
    left_sdf = end_sdf[:, 0, :]
    right_sdf = end_sdf[:, 1, :]
    left_scale = end_scales[:, 0, 0]
    right_scale = end_scales[:, 1, 0]
    distance = torch.norm(left_points - right_points, dim=-1)
    scale = left_scale + right_scale

    n_binary_steps = 10
    for step in range(n_binary_steps):
        print("binary search in step {}".format(step))
        mid_points = (left_points + right_points) * 0.5
        mid_sdf, _ = evaluate_alpha_cull(
            mid_points, views, gaussians, pipeline, kernel_size, carve_views, mask_dilate
        )
        mid_sdf = mid_sdf.unsqueeze(-1)
        ind_low = ((mid_sdf < 0) & (left_sdf < 0)) | ((mid_sdf > 0) & (left_sdf > 0))
        left_sdf[ind_low] = mid_sdf[ind_low]
        right_sdf[~ind_low] = mid_sdf[~ind_low]
        left_points[ind_low.flatten()] = mid_points[ind_low.flatten()]
        right_points[~ind_low.flatten()] = mid_points[~ind_low.flatten()]

    points = (left_points + right_points) * 0.5
    mesh = trimesh.Trimesh(vertices=points.cpu().numpy(), faces=faces, process=False)
    vertice_mask = distance <= scale
    face_mask = vertice_mask.cpu().numpy()[faces].all(axis=1)
    mesh.update_faces(face_mask)
    mesh.remove_unreferenced_vertices()

    mesh.export(os.path.join(model_path, "recon.ply"))

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(
        np.asarray(mesh.vertices, dtype=np.float64)
    )
    o3d_mesh.triangles = o3d.utility.Vector3iVector(
        np.asarray(mesh.faces, dtype=np.int32)
    )

    print("remove flyers")
    mesh = post_process_mesh(o3d_mesh, num_cluster)

    if export_color or flatten_unseen_bottom:
        print("add color" if export_color else "find observed vertices")
        max_dist = 0.008
        v_np = np.asarray(mesh.vertices)
        v_np = torch.from_numpy(v_np).to("cuda", dtype=torch.float32)
        color, surface = add_color(
            v_np, views, gaussians, pipeline, kernel_size, background, max_dist
        )
        color[~surface], surface[~surface] = add_color(
            v_np[~surface],
            views,
            gaussians,
            pipeline,
            kernel_size,
            background,
            max_dist * 2,
        )
        observed = surface.cpu().numpy()
        if export_color:
            # A simple coloring step for points likely on the silhouette cone.
            color[~surface] = add_color_silhouette(
                v_np[~surface], views, gaussians, pipeline, kernel_size, background
            )
            mesh.vertex_colors = o3d.utility.Vector3dVector(color.cpu().numpy())
        if flatten_unseen_bottom:
            flatten_bottom(mesh, views, observed)

    o3d.io.write_triangle_mesh(os.path.join(model_path, "recon_post.ply"), mesh)
    print("done!")


def extract_mesh(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    export_color: bool,
    move_cpu: bool,
    save_cells: bool,
    num_cluster: int,
    carve_views: int = 2,
    mask_dilate: int = 4,
    flatten_unseen_bottom: bool = False,
):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, dataset.sg_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        kernel_size = dataset.kernel_size
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        cams = scene.getTrainCameras()
        marching_tetrahedra_with_binary_search(
            dataset.model_path,
            cams,
            gaussians,
            pipeline,
            kernel_size,
            background,
            export_color,
            move_cpu,
            save_cells,
            num_cluster,
            carve_views,
            mask_dilate,
            flatten_unseen_bottom,
        )


if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--num_cluster", default=1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--move_cpu", action="store_true", help="Keep cells and edge deduplication on the CPU")
    parser.add_argument("--save_cells", action="store_true", help="Save the large tetrahedral cell array to cells.pt")
    parser.add_argument("--export_color", action="store_true")
    parser.add_argument("--carve_views", default=2, type=int,
                        help="Treat points projecting outside the mask in at least this many views as empty space (0 disables)")
    parser.add_argument("--mask_dilate", default=4, type=int,
                        help="Dilate masks by this many pixels before carving")
    parser.add_argument("--flatten_unseen_bottom", action="store_true",
                        help="Flatten the unobserved cap under an object captured from above onto the lowest observed height")
    args = get_combined_args(parser)

    extract_mesh(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.export_color,
        args.move_cpu,
        args.save_cells,
        args.num_cluster,
        args.carve_views,
        args.mask_dilate,
        args.flatten_unseen_bottom,
    )
