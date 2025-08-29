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


def write_html(poses, file, vis_depth=1, xyz_length=0.2, center_size=0.01, xyz_width=2):
    """Write camera pose visualization to HTML file."""
    # Compute the bounding box of the trajectory to automatically adjust the view
    centers_cam = np.zeros([len(poses), 1, 3])
    centers_world = cam2world(centers_cam, poses)[:, 0]

    # Compute the trajectory range
    x_range = np.max(centers_world[:, 0]) - np.min(centers_world[:, 0])
    y_range = np.max(centers_world[:, 1]) - np.min(centers_world[:, 1])
    z_range = np.max(centers_world[:, 2]) - np.min(centers_world[:, 2])
    max_range = max(x_range, y_range, z_range)

    # Compute the center of the trajectory
    center_x = (np.max(centers_world[:, 0]) + np.min(centers_world[:, 0])) / 2
    center_y = (np.max(centers_world[:, 1]) + np.min(centers_world[:, 1])) / 2
    center_z = (np.max(centers_world[:, 2]) + np.min(centers_world[:, 2])) / 2

    # Adjust parameters based on the trajectory range
    auto_vis_depth = max_range * 0.1  # Adjust the size of the pentagon based on the trajectory range
    auto_center_size = max_range * 0.8  # Center point size

    # Set camera position to fully view the trajectory
    camera_distance = max_range * 2.5  # Camera distance
    camera_eye_x = center_x + camera_distance * 0.7
    camera_eye_y = center_y + camera_distance * 0.7
    camera_eye_z = center_z + camera_distance * 0.5

    print(f"Trajectory range: x={x_range:.3f}, y={y_range:.3f}, z={z_range:.3f}")
    print(f"Max range: {max_range:.3f}")
    print(f"Auto vis_depth: {auto_vis_depth:.3f}, center_size: {auto_center_size:.3f}")
    print(
        f"Camera position: ({camera_eye_x:.3f}, {camera_eye_y:.3f}, {camera_eye_z:.3f})"
    )

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
            # Set initial camera view to fully see the trajectory
            camera=dict(
                eye=dict(x=camera_eye_x, y=camera_eye_y, z=camera_eye_z),
                center=dict(x=center_x, y=center_y, z=center_z),
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
        <h4 style="margin: 0 0 10px 0; color: #333;"></h4>
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
        <button onclick="copyToClipboard()" style="background: #007bff; color: white; border: none; padding: 5px 10px; border-radius: 4px; cursor: pointer; width: 100%;">Copy to clipboard</button>
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

    # Insert camera info HTML
    html_str2 = camera_info_html + html_str2

    file.write(html_str2)

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
