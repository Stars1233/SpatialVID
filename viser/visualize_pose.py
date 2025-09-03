import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle as pkl
import plotly.graph_objs as go
import plotly.io as pio
from tqdm import tqdm
import pandas as pd

matplotlib.use("agg")


class Pose:
    """
    A class of operations on camera poses (numpy arrays with shape [...,3,4]).
    Each [3,4] camera pose takes the form of [R|t].
    """

    def __call__(self, R=None, t=None):
        """
        Construct a camera pose from the given rotation matrix R and/or translation vector t.

        Args:
            R: Rotation matrix [...,3,3] or None
            t: Translation vector [...,3] or None

        Returns:
            pose: Camera pose matrix [...,3,4]
        """
        assert R is not None or t is not None
        if R is None:
            if not isinstance(t, np.ndarray):
                t = np.array(t)
            R = np.eye(3).repeat(*t.shape[:-1], 1, 1)
        elif t is None:
            if not isinstance(R, np.ndarray):
                R = np.array(R)
            t = np.zeros(R.shape[:-1])
        else:
            if not isinstance(R, np.ndarray):
                R = np.array(R)
            if not isinstance(t, np.ndarray):
                t = np.array(t)
        assert R.shape[:-1] == t.shape and R.shape[-2:] == (3, 3)
        R = R.astype(np.float32)
        t = t.astype(np.float32)
        pose = np.concatenate([R, t[..., None]], axis=-1)  # [...,3,4]
        assert pose.shape[-2:] == (3, 4)
        return pose

    def invert(self, pose, use_inverse=False):
        """
        Invert a camera pose.

        Args:
            pose: Camera pose matrix [...,3,4]
            use_inverse: Whether to use matrix inverse instead of transpose

        Returns:
            pose_inv: Inverted camera pose matrix [...,3,4]
        """
        R, t = pose[..., :3], pose[..., 3:]
        R_inv = np.linalg.inv(R) if use_inverse else R.transpose(0, 2, 1)
        t_inv = (-R_inv @ t)[..., 0]
        pose_inv = self(R=R_inv, t=t_inv)
        return pose_inv

    def compose(self, pose_list):
        """
        Compose a sequence of poses together.
        pose_new(x) = poseN o ... o pose2 o pose1(x)

        Args:
            pose_list: List of camera poses to compose

        Returns:
            pose_new: Composed camera pose
        """
        pose_new = pose_list[0]
        for pose in pose_list[1:]:
            pose_new = self.compose_pair(pose_new, pose)
        return pose_new

    def compose_pair(self, pose_a, pose_b):
        """
        Compose two poses together.
        pose_new(x) = pose_b o pose_a(x)

        Args:
            pose_a: First camera pose
            pose_b: Second camera pose

        Returns:
            pose_new: Composed camera pose
        """
        R_a, t_a = pose_a[..., :3], pose_a[..., 3:]
        R_b, t_b = pose_b[..., :3], pose_b[..., 3:]
        R_new = R_b @ R_a
        t_new = (R_b @ t_a + t_b)[..., 0]
        pose_new = self(R=R_new, t=t_new)
        return pose_new

    def scale_center(self, pose, scale):
        """
        Scale the camera center from the origin.
        0 = R@c+t --> c = -R^T@t (camera center in world coordinates)
        0 = R@(sc)+t' --> t' = -R@(sc) = -R@(-R^T@st) = st

        Args:
            pose: Camera pose to scale
            scale: Scale factor

        Returns:
            pose_new: Scaled camera pose
        """
        R, t = pose[..., :3], pose[..., 3:]
        pose_new = np.concatenate([R, t * scale], axis=-1)
        return pose_new


def to_hom(X):
    """Get homogeneous coordinates of the input by appending ones."""
    X_hom = np.concatenate([X, np.ones_like(X[..., :1])], axis=-1)
    return X_hom


def cam2world(X, pose):
    """Transform points from camera to world coordinates."""
    X_hom = to_hom(X)
    pose_inv = Pose().invert(pose)
    return X_hom @ pose_inv.transpose(0, 2, 1)


def get_camera_mesh(pose, depth=1):
    """
    Create a camera mesh visualization.

    Args:
        pose: Camera pose matrix
        depth: Size of the camera frustum

    Returns:
        vertices: Camera mesh vertices
        faces: Camera mesh faces
        wireframe: Camera wireframe vertices
    """
    vertices = (
        np.array(
            [[-0.5, -0.5, 1], [0.5, -0.5, 1], [0.5, 0.5, 1], [-0.5, 0.5, 1], [0, 0, 0]]
        )
        * depth
    )  # [6,3]
    faces = np.array(
        [[0, 1, 2], [0, 2, 3], [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]
    )  # [6,3]
    vertices = cam2world(vertices[None], pose)  # [N,6,3]
    wireframe = vertices[:, [0, 1, 2, 3, 0, 4, 1, 2, 4, 3]]  # [N,10,3]
    return vertices, faces, wireframe


# def merge_xyz_indicators_plotly(xyz):
#     """Merge xyz coordinate indicators for plotly visualization."""
#     xyz = xyz[:, [[-1, 0], [-1, 1], [-1, 2]]]  # [N,3,2,3]
#     xyz_0, xyz_1 = unbind_np(xyz, axis=2)  # [N,3,3]
#     xyz_dummy = xyz_0 * np.nan
#     xyz_merged = np.stack([xyz_0, xyz_1, xyz_dummy], axis=2)  # [N,3,3,3]
#     xyz_merged = xyz_merged.reshape(-1, 3)
#     return xyz_merged


# def get_xyz_indicators(pose, length=0.1):
#     """Get xyz coordinate axis indicators for a camera pose."""
#     xyz = np.eye(4, 3)[None] * length
#     xyz = cam2world(xyz, pose)
#     return xyz


def merge_wireframes_plotly(wireframe):
    """Merge camera wireframes for plotly visualization."""
    wf_dummy = wireframe[:, :1] * np.nan
    wireframe_merged = np.concatenate([wireframe, wf_dummy], axis=1).reshape(-1, 3)
    return wireframe_merged


def merge_meshes(vertices, faces):
    """Merge multiple camera meshes into a single mesh."""
    mesh_N, vertex_N = vertices.shape[:2]
    faces_merged = np.concatenate([faces + i * vertex_N for i in range(mesh_N)], axis=0)
    vertices_merged = vertices.reshape(-1, vertices.shape[-1])
    return vertices_merged, faces_merged


def unbind_np(array, axis=0):
    """Split numpy array along specified axis into list."""
    if axis == 0:
        return [array[i, :] for i in range(array.shape[0])]
    elif axis == 1 or (len(array.shape) == 2 and axis == -1):
        return [array[:, j] for j in range(array.shape[1])]
    elif axis == 2 or (len(array.shape) == 3 and axis == -1):
        return [array[:, :, j] for j in range(array.shape[2])]
    else:
        raise ValueError("Invalid axis. Use 0 for rows or 1 for columns.")


def plotly_visualize_pose(
    poses, vis_depth=0.5, xyz_length=0.5, center_size=2, xyz_width=5, mesh_opacity=0.05
):
    """
    Create plotly visualization traces for camera poses.

    Args:
        poses: Camera poses to visualize [N,3,4]
        vis_depth: Size of camera frustum visualization
        xyz_length: Length of coordinate axis indicators
        center_size: Size of camera center markers
        xyz_width: Width of coordinate axis lines
        mesh_opacity: Opacity of camera frustum mesh

    Returns:
        plotly_traces: List of plotly visualization traces
    """
    N = len(poses)
    centers_cam = np.zeros([N, 1, 3])
    centers_world = cam2world(centers_cam, poses)
    centers_world = centers_world[:, 0]
    # Get the camera wireframes.
    vertices, faces, wireframe = get_camera_mesh(poses, depth=vis_depth)
    # xyz = get_xyz_indicators(poses, length=xyz_length)
    vertices_merged, faces_merged = merge_meshes(vertices, faces)
    wireframe_merged = merge_wireframes_plotly(wireframe)
    # xyz_merged = merge_xyz_indicators_plotly(xyz)
    # Break up (x,y,z) coordinates.
    wireframe_x, wireframe_y, wireframe_z = unbind_np(wireframe_merged, axis=-1)
    # xyz_x, xyz_y, xyz_z = unbind_np(xyz_merged, axis=-1)
    centers_x, centers_y, centers_z = unbind_np(centers_world, axis=-1)
    vertices_x, vertices_y, vertices_z = unbind_np(vertices_merged, axis=-1)
    # Set the color map for the camera trajectory and the xyz indicators.
    color_map = plt.get_cmap("gist_rainbow")  # red -> yellow -> green -> blue -> purple
    center_color = []
    faces_merged_color = []
    wireframe_color = []
    # xyz_color = []
    # x_color, y_color, z_color = *np.eye(3).T,

    # Determine the indices for the quarter positions
    quarter_indices = set()
    quarter_indices.add(0)
    if N >= 3:
        quarter_indices.add(N // 3)
        quarter_indices.add(2 * N // 3)
    quarter_indices.add(N - 1)

    for i in range(N):
        # Check if the current index is a quarter index
        is_quarter = i in quarter_indices

        # For quarter positions, use dark color with high opacity
        alpha = 6.0 if is_quarter else 0.4

        # Add alpha channel (RGBA)
        r, g, b, _ = color_map(i / (N - 1))
        rgb = np.array([r, g, b]) * 1.2 if is_quarter else np.array([r, g, b]) * 0.8
        rgba = np.concatenate([rgb, [alpha]])
        wireframe_color += [rgba] * 11
        center_color += [rgba]
        faces_merged_color += [rgba] * 6
        # xyz_color += [np.concatenate([x_color, [alpha]])] * 3 + \
        #              [np.concatenate([y_color, [alpha]])] * 3 + \
        #              [np.concatenate([z_color, [alpha]])] * 3

    # Plot in plotly.
    plotly_traces = [
        go.Scatter3d(
            x=wireframe_x,
            y=wireframe_y,
            z=wireframe_z,
            mode="lines",
            line=dict(color=wireframe_color, width=1),
        ),
        # go.Scatter3d(x=xyz_x, y=xyz_y, z=xyz_z, mode="lines", line=dict(color=xyz_color, width=xyz_width)),
        go.Scatter3d(
            x=centers_x,
            y=centers_y,
            z=centers_z,
            mode="markers",
            marker=dict(color=center_color, size=center_size, opacity=1),
        ),
        go.Mesh3d(
            x=vertices_x,
            y=vertices_y,
            z=vertices_z,
            i=[f[0] for f in faces_merged],
            j=[f[1] for f in faces_merged],
            k=[f[2] for f in faces_merged],
            facecolor=faces_merged_color,
            opacity=mesh_opacity,
        ),
    ]
    return plotly_traces


def compute_optimal_camera_view(poses):
    """
    计算最优的相机视角，确保整个轨迹都在视野内且视角美观
    
    Args:
        poses: Camera poses [N,3,4]
        
    Returns:
        dict: 包含camera参数、vis_depth等的字典
    """
    # 计算所有相机位置
    centers_cam = np.zeros([len(poses), 1, 3])
    centers_world = cam2world(centers_cam, poses)[:, 0]
    
    # 计算轨迹的包围盒
    min_coords = np.min(centers_world, axis=0)
    max_coords = np.max(centers_world, axis=0)
    ranges = max_coords - min_coords
    
    # 计算轨迹中心
    trajectory_center = (min_coords + max_coords) / 2
    
    # 计算最大范围，用于自适应缩放
    max_range = np.max(ranges)
    
    # 如果轨迹太小，设置最小范围避免除零
    if max_range < 1e-6:
        max_range = 1.0
        ranges = np.ones(3)
    
    # 计算轨迹的主方向（使用PCA）
    if len(centers_world) > 1:
        # 去中心化
        centered_points = centers_world - trajectory_center
        
        # 计算协方差矩阵
        cov_matrix = np.cov(centered_points.T)
        
        # 计算特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # 按特征值排序（降序）
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # 主方向是第一个特征向量
        main_direction = eigenvectors[:, 0]
        
        # 确保主方向指向轨迹的正方向
        start_to_end = centers_world[-1] - centers_world[0]
        if np.dot(main_direction, start_to_end) < 0:
            main_direction = -main_direction
            
    else:
        main_direction = np.array([1, 0, 0])
    
    # 计算最优相机距离
    # 基于轨迹范围和视野角度计算，使用更小的因子让轨迹更好填充画面
    fov_factor = 0.8  # 进一步减小视野因子，让轨迹占据更多画面空间
    base_distance = max_range * fov_factor
    
    # 考虑轨迹的纵横比，调整距离
    aspect_ratios = ranges / max_range
    distance_scale = 1.0 + 0.1 * np.std(aspect_ratios)  # 进一步减小距离调整幅度
    camera_distance = base_distance * distance_scale
    
    # 计算最优相机位置
    # 方法1：基于主方向的斜视角度
    up_vector = np.array([0, 0, 1])
    
    # 如果主方向接近垂直，调整策略
    if abs(np.dot(main_direction, up_vector)) > 0.9:
        # 主方向接近垂直，使用侧视角
        view_direction = np.cross(main_direction, np.array([1, 0, 0]))
        if np.linalg.norm(view_direction) < 0.1:
            view_direction = np.cross(main_direction, np.array([0, 1, 0]))
        view_direction = view_direction / np.linalg.norm(view_direction)
    else:
        # 计算垂直于主方向且倾斜的视角方向
        # 结合主方向的垂直分量和一个倾斜角度
        horizontal_component = main_direction - np.dot(main_direction, up_vector) * up_vector
        horizontal_component = horizontal_component / (np.linalg.norm(horizontal_component) + 1e-8)
        
        # 添加一些倾斜角度以获得更好的3D视角
        elevation_angle = np.pi / 6  # 30度仰角
        azimuth_offset = np.pi / 4   # 45度方位角偏移
        
        # 创建倾斜的视角方向
        view_direction = (
            horizontal_component * np.cos(azimuth_offset) * np.cos(elevation_angle) +
            np.cross(horizontal_component, up_vector) * np.sin(azimuth_offset) * np.cos(elevation_angle) +
            up_vector * np.sin(elevation_angle)
        )
    
    # 计算相机eye位置
    camera_eye = trajectory_center + view_direction * camera_distance
    
    # 微调相机位置，确保轨迹完全在视野内
    # 计算从相机位置到轨迹各点的向量
    view_vectors = centers_world - camera_eye
    view_distances = np.linalg.norm(view_vectors, axis=1)
    
    # 如果有些点太近，适度调整相机距离
    min_distance = camera_distance * 0.3  # 降低最小距离比例
    if np.min(view_distances) < min_distance:
        distance_adjustment = min_distance / np.min(view_distances)
        # 限制调整幅度，避免过度放大
        distance_adjustment = min(distance_adjustment, 1.2)  # 进一步限制调整幅度
        camera_eye = trajectory_center + view_direction * camera_distance * distance_adjustment
    
    # 计算自适应参数，使用更合适的比例
    auto_vis_depth = max_range * 0.08  # 适当减小相机锥体大小
    auto_center_size = max_range * 1.5  # 适当减小中心点大小
    
    # 确保参数在合理范围内
    auto_vis_depth = max(0.01, min(auto_vis_depth, max_range * 0.2))
    auto_center_size = max(0.1, min(auto_center_size, max_range * 2.0))
    
    return {
        'camera_eye': camera_eye,
        'trajectory_center': trajectory_center,
        'auto_vis_depth': auto_vis_depth,
        'auto_center_size': auto_center_size,
        'max_range': max_range,
        'ranges': ranges,
        'main_direction': main_direction
    }


def compute_multiple_camera_views(poses):
    """
    计算多个优化的相机视角，提供不同的观察选项
    
    Args:
        poses: Camera poses [N,3,4]
        
    Returns:
        dict: 包含多个视角选项的字典
    """
    base_params = compute_optimal_camera_view(poses)
    
    trajectory_center = base_params['trajectory_center']
    max_range = base_params['max_range']
    main_direction = base_params['main_direction']
    
    # 计算多个视角选项
    views = {}
    
    # 1. 最佳自动视角（原有的）
    views['optimal'] = base_params
    
    # 2. 正上方俯视视角
    top_distance = max_range * 1.5  # 进一步减小俯视距离
    views['top'] = {
        **base_params,
        'camera_eye': trajectory_center + np.array([0, 0, top_distance]),
        'description': 'Top-down view'
    }
    
    # 3. 侧视视角
    side_distance = max_range * 1.3  # 进一步减小侧视距离
    side_direction = np.cross(main_direction, np.array([0, 0, 1]))
    if np.linalg.norm(side_direction) < 0.1:
        side_direction = np.array([1, 0, 0])
    else:
        side_direction = side_direction / np.linalg.norm(side_direction)
    
    views['side'] = {
        **base_params,
        'camera_eye': trajectory_center + side_direction * side_distance,
        'description': 'Side view'
    }
    
    # 4. 对角线视角（45度仰角）
    diagonal_distance = max_range * 1.4  # 进一步减小对角视角距离
    elevation = np.pi / 4  # 45度
    azimuth = np.pi / 4    # 45度方位角
    
    diagonal_direction = np.array([
        np.cos(elevation) * np.cos(azimuth),
        np.cos(elevation) * np.sin(azimuth),
        np.sin(elevation)
    ])
    
    views['diagonal'] = {
        **base_params,
        'camera_eye': trajectory_center + diagonal_direction * diagonal_distance,
        'description': 'Diagonal view (45° elevation)'
    }
    
    # 5. 轨迹起点视角
    if len(poses) > 1:
        start_to_center = trajectory_center - base_params['camera_eye']  
        start_distance = max_range * 1.2  # 进一步减小起点视角距离
        start_direction = start_to_center / (np.linalg.norm(start_to_center) + 1e-8)
        
        views['trajectory_start'] = {
            **base_params,
            'camera_eye': trajectory_center + start_direction * start_distance,
            'description': 'View from trajectory start direction'
        }
    
    # 6. 紧凑视角 - 确保整个轨迹完全可见
    fit_distance = max_range * 0.6  # 非常紧凑的距离
    fit_direction = np.array([0.7, 0.7, 0.5])  # 稳定的视角方向
    fit_direction = fit_direction / np.linalg.norm(fit_direction)
    
    views['fit_all'] = {
        **base_params,
        'camera_eye': trajectory_center + fit_direction * fit_distance,
        'description': 'Fit all trajectory in view'
    }
    
    return views


def add_view_selector_to_html(html_str, views):
    """
    在HTML中添加视角选择器
    
    Args:
        html_str: 原始HTML字符串
        views: 视角字典
        
    Returns:
        str: 添加了视角选择器的HTML字符串
    """
    
    # 生成视角选择器的JavaScript代码
    view_selector_js = """
    <div id="view-selector" style="position: fixed; top: 10px; left: 10px; background: rgba(255,255,255,0.9); padding: 15px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); font-family: Arial, sans-serif; font-size: 12px; z-index: 1000; min-width: 120px;">
        <button onclick="autoRotate()" style="background: #ffc107; color: black; border: none; padding: 8px 12px; border-radius: 4px; cursor: pointer; width: 100%;">Auto Rotate</button>
    </div>
    
    <script>
    // 预定义的视角配置
    const views = {"""
    
    # 添加视角数据
    for view_name, view_data in views.items():
        eye = view_data['camera_eye']
        center = view_data['trajectory_center']
        view_selector_js += f"""
        {view_name}: {{
            eye: {{x: {eye[0]:.6f}, y: {eye[1]:.6f}, z: {eye[2]:.6f}}},
            center: {{x: {center[0]:.6f}, y: {center[1]:.6f}, z: {center[2]:.6f}}},
            up: {{x: 0, y: 0, z: 1}}
        }},"""
    
    view_selector_js += """
    };
    
    let rotationInterval = null;
    
    function autoRotate() {
        if (rotationInterval) {
            clearInterval(rotationInterval);
            rotationInterval = null;
            return;
        }
        
        var plotlyDiv = document.querySelector('.plotly-graph-div');
        if (!plotlyDiv) return;
        
        var currentView = views.fit_all;
        var center = currentView.center;
        var radius = Math.sqrt(
            Math.pow(currentView.eye.x - center.x, 2) + 
            Math.pow(currentView.eye.y - center.y, 2) + 
            Math.pow(currentView.eye.z - center.z, 2)
        );
        
        var angle = 0;
        rotationInterval = setInterval(function() {
            angle += 0.02; // 旋转速度
            
            var newEye = {
                x: center.x + radius * Math.cos(angle) * 0.7,
                y: center.y + radius * Math.sin(angle) * 0.7,
                z: center.z + radius * 0.5
            };
            
            var update = {
                'scene.camera.eye': newEye
            };
            
            Plotly.relayout(plotlyDiv, update);
        }, 50);
    }
    
    // 在页面加载完成后设置默认视角
    document.addEventListener('DOMContentLoaded', function() {
        setTimeout(function() {
            // 使用 Fit All 作为默认视角，无需按钮操作
            var plotlyDiv = document.querySelector('.plotly-graph-div');
            if (plotlyDiv && views.fit_all) {
                var update = {
                    'scene.camera': views.fit_all
                };
                Plotly.relayout(plotlyDiv, update);
            }
        }, 1000);
    });
    </script>
    """
    
    # 将视角选择器添加到HTML开头
    return view_selector_js + html_str


def write_html(poses, file, vis_depth=1, xyz_length=0.2, center_size=0.01, xyz_width=2):
    """Write camera pose visualization to HTML file with optimized camera view."""
    # 计算基础的最优视角
    base_view = compute_optimal_camera_view(poses)
    
    # 获取轨迹信息
    trajectory_center = base_view['trajectory_center']
    max_range = base_view['max_range']
    ranges = base_view['ranges']
    auto_vis_depth = base_view['auto_vis_depth']
    auto_center_size = base_view['auto_center_size']
    
    # 计算能看到整个轨迹的最佳视角
    # 使用更大的距离确保整个轨迹可见，并选择更好的角度
    optimal_distance = max_range * 1.8 * 10  # 增大10倍距离以获得更好的整体视角
    
    # 选择一个能看到轨迹全貌的理想角度
    # 使用45度仰角和45度方位角的组合，这通常能提供很好的3D视角
    elevation = np.pi / 4  # 45度仰角
    azimuth = np.pi / 4    # 45度方位角
    
    # 计算最佳观察方向
    optimal_direction = np.array([
        np.cos(elevation) * np.cos(azimuth),
        np.cos(elevation) * np.sin(azimuth),
        np.sin(elevation)
    ])
    
    # 计算最佳相机位置
    camera_eye = trajectory_center + optimal_direction * optimal_distance
    
    # 验证视野覆盖 - 确保所有轨迹点都在合理距离内
    centers_cam = np.zeros([len(poses), 1, 3])
    centers_world = cam2world(centers_cam, poses)[:, 0]
    
    # 计算从最佳相机位置到所有轨迹点的距离
    distances_to_points = np.linalg.norm(centers_world - camera_eye, axis=1)
    max_distance_to_point = np.max(distances_to_points)
    min_distance_to_point = np.min(distances_to_points)
    
    # 如果距离差异过大，说明视角可能不够理想，进行调整
    if max_distance_to_point / min_distance_to_point > 3.0:
        # 重新计算更平衡的距离
        optimal_distance = max_range * 2.2 * 10  # 进一步增大距离（10倍）
        camera_eye = trajectory_center + optimal_direction * optimal_distance
    
    # 创建视角字典，只包含最佳视角用于Auto Rotate
    views = {
        'fit_all': {
            'camera_eye': camera_eye,
            'trajectory_center': trajectory_center,
            'auto_vis_depth': auto_vis_depth,
            'auto_center_size': auto_center_size,
            'max_range': max_range,
            'ranges': ranges,
            'description': 'Optimal view to see entire trajectory'
        }
    }
    
    print(f"Trajectory ranges: x={ranges[0]:.3f}, y={ranges[1]:.3f}, z={ranges[2]:.3f}")
    print(f"Max range: {max_range:.3f}")
    print(f"Auto vis_depth: {auto_vis_depth:.3f}, center_size: {auto_center_size:.3f}")
    print(f"Trajectory center: ({trajectory_center[0]:.3f}, {trajectory_center[1]:.3f}, {trajectory_center[2]:.3f})")
    print(f"Optimal camera position for full trajectory view: ({camera_eye[0]:.3f}, {camera_eye[1]:.3f}, {camera_eye[2]:.3f})")
    print(f"Camera distance from trajectory center: {optimal_distance:.3f}")
    print(f"Distance range to trajectory points: {min_distance_to_point:.3f} - {max_distance_to_point:.3f}")

    xyz_length = xyz_length / 3
    xyz_width = xyz_width
    vis_depth = auto_vis_depth  # Use automatically computed depth
    center_size = auto_center_size  # Use automatically computed size

    traces_poses = plotly_visualize_pose(
        poses,
        vis_depth=vis_depth,
        xyz_length=xyz_length,
        center_size=center_size,
        xyz_width=xyz_width,
        mesh_opacity=0.05,
    )
    traces_all2 = traces_poses
    layout2 = go.Layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            dragmode="orbit",
            aspectratio=dict(x=1, y=1, z=1),
            aspectmode="data",
            # Set initial camera view to fully see the trajectory with optimized positioning
            camera=dict(
                eye=dict(x=camera_eye[0], y=camera_eye[1], z=camera_eye[2]),
                center=dict(x=trajectory_center[0], y=trajectory_center[1], z=trajectory_center[2]),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        height=800,
        width=1200,
        showlegend=False,
    )

    fig2 = go.Figure(data=traces_all2, layout=layout2)
    html_str2 = pio.to_html(fig2, full_html=False)

    # Add real-time camera view display functionality
    camera_info_html = """
    <div id="camera-info" style="position: fixed; top: 10px; right: 10px; background: rgba(255,255,255,0.9); padding: 15px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); font-family: monospace; font-size: 12px; z-index: 1000; min-width: 250px;">
        <h4 style="margin: 0 0 10px 0; color: #333;">Camera Info</h4>
        <div><strong>Eye:</strong></div>
        <div>x: <span id="eye-x">2.000</span></div>
        <div>y: <span id="eye-y">2.000</span></div>
        <div>z: <span id="eye-z">1.000</span></div>
        <br>
        <div><strong>Center:</strong></div>
        <div>x: <span id="center-x">0.000</span></div>
        <div>y: <span id="center-y">0.000</span></div>
        <div>z: <span id="center-z">0.000</span></div>
        <br>
        <div><strong>Up:</strong></div>
        <div>x: <span id="up-x">0.000</span></div>
        <div>y: <span id="up-y">0.000</span></div>
        <div>z: <span id="up-z">1.000</span></div>
        <br>
        <button onclick="copyToClipboard()" style="background: #007bff; color: white; border: none; padding: 5px 10px; border-radius: 4px; cursor: pointer; width: 100%;">Copy to Clipboard</button>
    </div>
    
    <script>
    function updateCameraInfo() {
        // Get Plotly chart
        var plotlyDiv = document.querySelector('.plotly-graph-div');
        if (!plotlyDiv) return;
        
        // Listen for camera change events
        plotlyDiv.on('plotly_relayout', function(eventData) {
            if (eventData['scene.camera']) {
                var camera = eventData['scene.camera'];
                updateCameraDisplay(camera);
            }
        });
        
        // Initial display
        setTimeout(function() {
            var gd = plotlyDiv;
            if (gd.layout && gd.layout.scene && gd.layout.scene.camera) {
                updateCameraDisplay(gd.layout.scene.camera);
            }
        }, 1000);
    }
    
    function updateCameraDisplay(camera) {
        if (camera.eye) {
            document.getElementById('eye-x').textContent = camera.eye.x.toFixed(3);
            document.getElementById('eye-y').textContent = camera.eye.y.toFixed(3);
            document.getElementById('eye-z').textContent = camera.eye.z.toFixed(3);
        }
        if (camera.center) {
            document.getElementById('center-x').textContent = camera.center.x.toFixed(3);
            document.getElementById('center-y').textContent = camera.center.y.toFixed(3);
            document.getElementById('center-z').textContent = camera.center.z.toFixed(3);
        }
        if (camera.up) {
            document.getElementById('up-x').textContent = camera.up.x.toFixed(3);
            document.getElementById('up-y').textContent = camera.up.y.toFixed(3);
            document.getElementById('up-z').textContent = camera.up.z.toFixed(3);
        }
    }
    
    function copyToClipboard() {
        var eyeX = document.getElementById('eye-x').textContent;
        var eyeY = document.getElementById('eye-y').textContent;
        var eyeZ = document.getElementById('eye-z').textContent;
        var centerX = document.getElementById('center-x').textContent;
        var centerY = document.getElementById('center-y').textContent;
        var centerZ = document.getElementById('center-z').textContent;
        var upX = document.getElementById('up-x').textContent;
        var upY = document.getElementById('up-y').textContent;
        var upZ = document.getElementById('up-z').textContent;
        
        var cameraConfig = `camera=dict(
    eye=dict(x=${eyeX}, y=${eyeY}, z=${eyeZ}),
    center=dict(x=${centerX}, y=${centerY}, z=${centerZ}),
    up=dict(x=${upX}, y=${upY}, z=${upZ})
)`;
        
        navigator.clipboard.writeText(cameraConfig).then(function() {
            alert('Copy to clipboard successful!');
        }).catch(function(err) {
            console.error('Copy failed:', err);
            // Fallback: Create a temporary textarea
            var textArea = document.createElement('textarea');
            textArea.value = cameraConfig;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);
            alert('Copy to clipboard successful!');
        });
    }

    // Initialize camera info display
    document.addEventListener('DOMContentLoaded', function() {
        updateCameraInfo();
    });

    // If the page has already loaded
    if (document.readyState === 'complete') {
        updateCameraInfo();
    }
    </script>
    """

    # 添加视角选择器和相机信息到HTML
    enhanced_html = add_view_selector_to_html(camera_info_html + html_str2, views)
    
    file.write(enhanced_html)

    print(f"Enhanced visualized poses are saved to {file.name}")
    # 移除了多余的视角选项打印


def plotly_visualize_pose_animated(
    poses_full,
    vis_depth=0.5,
    xyz_length=0.5,
    center_size=2,
    xyz_width=5,
    mesh_opacity=0.05,
):
    """
    Create plotly visualization traces for camera poses, frame by frame for animation.
    Now shows the full trajectory with future poses as completely transparent.

    Args:
        poses_full: All camera poses to visualize [N,3,4]
        vis_depth: Size of camera frustum visualization
        xyz_length: Length of coordinate axis indicators
        center_size: Size of camera center markers
        xyz_width: Width of coordinate axis lines
        mesh_opacity: Opacity of camera frustum mesh

    Returns:
        plotly_data: Initial plotly traces for the first frame
        plotly_frames: List of plotly frames for animation
    """
    N_total = len(poses_full)
    plotly_frames = []

    # Pre-compute data for all poses to ensure consistent layout
    centers_cam = np.zeros([N_total, 1, 3])
    centers_world = cam2world(centers_cam, poses_full)
    centers_world = centers_world[:, 0]
    # Get the camera wireframes for all poses
    vertices, faces, wireframe = get_camera_mesh(poses_full, depth=vis_depth)
    vertices_merged, faces_merged = merge_meshes(vertices, faces)
    wireframe_merged = merge_wireframes_plotly(wireframe)
    # Break up (x,y,z) coordinates.
    wireframe_x, wireframe_y, wireframe_z = unbind_np(wireframe_merged, axis=-1)
    centers_x, centers_y, centers_z = unbind_np(centers_world, axis=-1)
    vertices_x, vertices_y, vertices_z = unbind_np(vertices_merged, axis=-1)

    # Initial frame showing all poses with appropriate transparency
    initial_data = []

    for i in tqdm(range(1, N_total + 1), desc="Generating animation frames"):
        current_frame = i - 1  # Current frame index (0-based)
        
        # Set the color map for the camera trajectory
        color_map = plt.get_cmap("gist_rainbow")
        center_color = []
        faces_merged_color = []
        wireframe_color = []
        
        for k in range(N_total):  # Process all poses
            # Set the camera pose colors (with a smooth gradient color map).
            r, g, b, _ = color_map(k / (N_total - 1))
            rgb = np.array([r, g, b]) * 0.8
            
            # Set transparency based on current frame
            if k < current_frame:  # Past poses - visible with reduced opacity
                # 根据时间距离设置透明度，越远越透明
                time_distance = (current_frame - k) / max(current_frame, 1)
                alpha = 0.15 + 0.25 * (1 - time_distance)  # 0.15-0.4的透明度范围
                wireframe_alpha = alpha
                mesh_alpha = alpha * 0.4
            elif k == current_frame:  # Current pose - fully visible
                alpha = 0.8  # 完全不透明，深色显示
                wireframe_alpha = 0.8
                mesh_alpha = 0.6
            else:  # Future poses - completely transparent
                alpha = 0.0  # 完全透明
                wireframe_alpha = 0.0
                mesh_alpha = 0.0
            
            # 设置颜色和透明度
            wireframe_color += [np.concatenate([rgb, [wireframe_alpha]])] * 11
            center_color += [np.concatenate([rgb, [alpha]])]
            faces_merged_color += [np.concatenate([rgb, [mesh_alpha]])] * 6

        frame_data = [
            go.Scatter3d(
                x=wireframe_x,
                y=wireframe_y,
                z=wireframe_z,
                mode="lines",
                line=dict(color=wireframe_color, width=1),
            ),
            go.Scatter3d(
                x=centers_x,
                y=centers_y,
                z=centers_z,
                mode="markers",
                marker=dict(color=center_color, size=center_size),
            ),
            go.Mesh3d(
                x=vertices_x,
                y=vertices_y,
                z=vertices_z,
                i=[f[0] for f in faces_merged],
                j=[f[1] for f in faces_merged],
                k=[f[2] for f in faces_merged],
                facecolor=faces_merged_color,
                opacity=0.6,  # 为mesh设置基础不透明度
            ),
        ]
        
        if i == 1:  # Set initial data for the first frame
            initial_data = frame_data

        plotly_frames.append(go.Frame(data=frame_data, name=str(i)))

    return initial_data, plotly_frames


def write_html_animated(
    poses, file, vis_depth=1, xyz_length=0.2, center_size=0.01, xyz_width=2
):
    """Write camera pose visualization with animation to HTML file with optimized camera view."""
    # 计算基础的最优视角
    base_view = compute_optimal_camera_view(poses)
    
    # 获取轨迹信息
    trajectory_center = base_view['trajectory_center']
    max_range = base_view['max_range']
    ranges = base_view['ranges']
    auto_vis_depth = base_view['auto_vis_depth']
    auto_center_size = base_view['auto_center_size']
    
    # 计算能看到整个轨迹的最佳视角
    # 使用更大的距离确保整个轨迹可见，并选择更好的角度
    optimal_distance = max_range * 1.8 * 10  # 增大10倍距离以获得更好的整体视角
    
    # 选择一个能看到轨迹全貌的理想角度
    # 使用45度仰角和45度方位角的组合，这通常能提供很好的3D视角
    elevation = np.pi / 4  # 45度仰角
    azimuth = np.pi / 4    # 45度方位角
    
    # 计算最佳观察方向
    optimal_direction = np.array([
        np.cos(elevation) * np.cos(azimuth),
        np.cos(elevation) * np.sin(azimuth),
        np.sin(elevation)
    ])
    
    # 计算最佳相机位置
    camera_eye = trajectory_center + optimal_direction * optimal_distance
    
    # 验证视野覆盖 - 确保所有轨迹点都在合理距离内
    centers_cam = np.zeros([len(poses), 1, 3])
    centers_world = cam2world(centers_cam, poses)[:, 0]
    
    # 计算从最佳相机位置到所有轨迹点的距离
    distances_to_points = np.linalg.norm(centers_world - camera_eye, axis=1)
    max_distance_to_point = np.max(distances_to_points)
    min_distance_to_point = np.min(distances_to_points)
    
    # 如果距离差异过大，说明视角可能不够理想，进行调整
    if max_distance_to_point / min_distance_to_point > 3.0:
        # 重新计算更平衡的距离
        optimal_distance = max_range * 2.2 * 10  # 进一步增大距离（10倍）
        camera_eye = trajectory_center + optimal_direction * optimal_distance
    
    # 调整参数以适应动画
    xyz_length = xyz_length / 3
    xyz_width = xyz_width
    vis_depth = auto_vis_depth  # 使用自动计算的深度
    center_size = auto_center_size  # 使用自动计算的大小

    print(f"Animation - Trajectory ranges: x={ranges[0]:.3f}, y={ranges[1]:.3f}, z={ranges[2]:.3f}")
    print(f"Animation - Max range: {max_range:.3f}")
    print(f"Animation - Auto vis_depth: {auto_vis_depth:.3f}, center_size: {auto_center_size:.3f}")
    print(f"Animation - Trajectory center: ({trajectory_center[0]:.3f}, {trajectory_center[1]:.3f}, {trajectory_center[2]:.3f})")
    print(f"Animation - Optimal camera position for full trajectory view: ({camera_eye[0]:.3f}, {camera_eye[1]:.3f}, {camera_eye[2]:.3f})")
    print(f"Animation - Camera distance from trajectory center: {optimal_distance:.3f}")
    print(f"Animation - Distance range to trajectory points: {min_distance_to_point:.3f} - {max_distance_to_point:.3f}")

    initial_data, plotly_frames = plotly_visualize_pose_animated(
        poses,
        vis_depth=vis_depth,
        xyz_length=xyz_length,
        center_size=center_size,
        xyz_width=xyz_width,
        mesh_opacity=0.05,
    )

    layout = go.Layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            dragmode="orbit",
            aspectratio=dict(x=1, y=1, z=1),
            aspectmode="data",
            # 使用优化的相机视角设置（与write_html相同的10倍距离）
            camera=dict(
                eye=dict(x=camera_eye[0], y=camera_eye[1], z=camera_eye[2]),
                center=dict(x=trajectory_center[0], y=trajectory_center[1], z=trajectory_center[2]),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        height=800,  # 增加高度以更好地显示动画
        width=1200,  # 增加宽度以更好地显示动画
        showlegend=False,
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {"duration": 50, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    )
                ],
            )
        ],
    )

    fig = go.Figure(data=initial_data, layout=layout, frames=plotly_frames)
    html_str = pio.to_html(fig, full_html=False)
    file.write(html_str)

    print(f"Visualized poses are saved to {file}")


import einops
import torch


def quaternion_to_matrix(quaternions, eps: float = 1e-8):
    """
    Convert the 4-dimensional quaternions to 3x3 rotation matrices.
    This is adapted from Pytorch3D:
    https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
    """

    # Order changed to match scipy format!
    i, j, k, r = torch.unbind(quaternions, dim=-1)
    two_s = 2 / ((quaternions * quaternions).sum(dim=-1) + eps)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return einops.rearrange(o, "... (i j) -> ... i j", i=3, j=3)


def pose_from_quaternion(pose):  # input is w2c, pose(n,7) or (n,v,7)
    # tensor in https://github.com/pointrix-project/Geomotion/blob/6ab0c364f1b44ab4ea190085dbf068f62b42727c/geomotion/model/cameras.py#L6
    if type(pose) == np.ndarray:
        pose = torch.tensor(pose)
    if len(pose.shape) == 1:
        pose = pose[None]
    quat_t = pose[..., :3]
    quat_r = pose[..., 3:]
    w2c_matrix = torch.zeros((*list(pose.shape)[:-1], 3, 4), device=pose.device)
    w2c_matrix[..., :3, 3] = quat_t
    w2c_matrix[..., :3, :3] = quaternion_to_matrix(quat_r)
    return w2c_matrix


def viz_poses(i, pth, file, args):
    """Visualize camera poses for a sequence and write to HTML file."""
    file.write(f"<span style='font-size: 18pt;'>{i} {pth}</span><br>")

    pose = np.load(pth)  #
    poses = pose_from_quaternion(
        pose
    )  # pose: (N,7), after_pose (N,3,4) after_pose: w2c
    poses = poses.cpu().numpy()

    # Scale camera positions to reduce distance between pentagons
    scale_factor = getattr(
        args, "scale_factor", 0.3
    )  # Default scale factor 0.3, can be adjusted via parameter

    # Scale translation part (camera position)
    poses_scaled = poses.copy()
    poses_scaled[..., :3, 3] = poses[..., :3, 3] * scale_factor

    print(f"Original poses shape: {poses.shape}")
    print(f"Applied scale factor: {scale_factor}")

    if args.dynamic:
        write_html_animated(poses_scaled, file, vis_depth=args.vis_depth)
    else:
        write_html(poses_scaled, file, vis_depth=args.vis_depth)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datas", type=str, nargs="+", help="List of sequences to visualize poses for."
    )
    parser.add_argument(
        "--vis_depth",
        type=float,
        default=0.2,
        help="Depth of camera frustum visualization.",
    )
    parser.add_argument(
        "--scale_factor",
        type=float,
        default=0.3,
        help="Scale factor to reduce distance between cameras (smaller = closer cameras).",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="./visualize",
        help="Output directory to save HTML files.",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Whether to create an animated visualization showing camera trajectory over time.",
    )

    global args
    args = parser.parse_args()

    # define dataset paths
    os.makedirs(args.outdir, exist_ok=True)
    with open(f"{args.outdir}/visualize.html", "w") as file:
        for i, pth in enumerate(tqdm(args.datas)):
            if not os.path.exists(pth):
                print(f"Path {pth} does not exist, skipping.")
                continue
            print(pth, i)
            viz_poses(i, pth, file, args)
