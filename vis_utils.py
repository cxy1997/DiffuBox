from typing import Optional
import numpy as np
import torch
import matplotlib as mpl

import plotly.io as pio
pio.renderers.default = 'notebook'
import plotly.graph_objects as go


def get_layout_config():
    return {
        'title': {
            'text': '',
            'font': {
                'size': 20,
                'color': 'rgb(150,150,150)',
            },
            'xanchor': 'left',
            'yanchor': 'top'},
        'paper_bgcolor': 'rgb(255,255,255)',
        'width': 1000,
        'height': 800,
        'margin': {
            'l': 20,
            'r': 20,
            'b': 20,
            't': 20
        },
        'legend': {
            'font': {
                'size': 20,
                'color': 'rgb(150,150,150)',
            },
            'itemsizing': 'constant'
        },
        "hoverlabel": {
            "namelength": -1,
        },
        'showlegend': False,
        'coloraxis': {'showscale': False},
        'scene': {
            'aspectmode': 'manual',
            'aspectratio': {'x': 1, 'y': 1, 'z': 1},
            'camera': {'eye': {'x': 0, 'y': 0, 'z': 2}},
            'xaxis': {'color': 'rgb(150,150,150)',
                      'dtick': 1,
                      'gridcolor': 'rgb(100,100,100)',
                      'range': [-3, 3],
                      'showbackground': False,
                      'showgrid': True,
                      'showline': False,
                      'showticklabels': True,
                      'tickmode': 'linear',
                      'tickprefix': 'x:'},
            'yaxis': {'color': 'rgb(150,150,150)',
                      'dtick': 1,
                      'gridcolor': 'rgb(100,100,100)',
                      'range': [-3, 3],
                      'showbackground': False,
                      'showgrid': True,
                      'showline': False,
                      'showticklabels': True,
                      'tickmode': 'linear',
                      'tickprefix': 'y:'},
            'zaxis': {'color': 'rgb(150,150,150)',
                      'dtick': 1,
                      'gridcolor': 'rgb(100,100,100)',
                      'range': [-3, 3],
                      'showbackground': False,
                      'showgrid': True,
                      'showline': False,
                      'showticklabels': True,
                      'tickmode': 'linear',
                      'tickprefix': 'z:'}},
    }


scene_layout_config = {
    'title': {
        'text': 'test vis LiDAR',
        'font': {
            'size': 20,
            'color': 'rgb(150,150,150)',
        },
        'xanchor': 'left',
        'yanchor': 'top'},
    'paper_bgcolor': 'rgb(255,255,255)',
    'width' : 1000,
    'height' : 800,
    'margin' : {
        'l': 20,
        'r': 20,
        'b': 20,
        't': 20
    },
    'legend': {
        'font':{
            'size':20,
            'color': 'rgb(150,150,150)',
        },
        'itemsizing': 'constant'
    },
    "hoverlabel": {
        "namelength": -1,
    },
    'showlegend': False,
    'scene': {
          'aspectmode': 'manual',
          'aspectratio': {'x': 0.75, 'y': 0.25, 'z': 0.05},
          'camera': {'eye': {'x': 0, 'y': 0, 'z': 0.5}},
          'xaxis': {'color': 'rgb(150,150,150)',
                    'dtick': 10,
                    'gridcolor': 'rgb(100,100,100)',
                    'range': [-150, 150],
                    'showbackground': False,
                    'showgrid': True,
                    'showline': False,
                    'showticklabels': True,
                    'tickmode': 'linear',
                    'tickprefix': 'x:'},
          'yaxis': {'color': 'rgb(150,150,150)',
                    'dtick': 10,
                    'gridcolor': 'rgb(100,100,100)',
                    'range': [-50, 50],
                    'showbackground': False,
                    'showgrid': True,
                    'showline': False,
                    'showticklabels': True,
                    'tickmode': 'linear',
                    'tickprefix': 'y:'},
          'zaxis': {'color': 'rgb(150,150,150)',
                    'dtick': 10,
                    'gridcolor': 'rgb(100,100,100)',
                    'range': [-10, 10],
                    'showbackground': False,
                    'showgrid': True,
                    'showline': False,
                    'showticklabels': True,
                    'tickmode': 'linear',
                    'tickprefix': 'z:'}},
}


def cart2hom(pts_3d):
    n = pts_3d.shape[0]
    pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1), dtype=np.float32)))
    return pts_3d_hom


def transform_points(pts_3d_ref, Tr):
    pts_3d_ref = cart2hom(pts_3d_ref)  # nx4
    return np.dot(pts_3d_ref, np.transpose(Tr)).reshape(-1, 4)[:, 0:3]


def compute_box(trans_mat, shape):
    w, l, h = shape
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
    y_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
    z_corners = [-h/2,-h/2,-h/2,-h/2,h/2,h/2,h/2,h/2]
    corners_3d = np.vstack([x_corners,y_corners,z_corners]).T
    return transform_points(corners_3d, trans_mat)


def get_linemarks(trans_mat, shape):
    corners = compute_box(trans_mat, shape)
    mid_front = (corners[0] + corners[1]) / 2
    mid_left = (corners[0] + corners[3]) / 2
    mid_right = (corners[1] + corners[2]) / 2
    corners = np.vstack(
        (corners, np.vstack([mid_front, mid_left, mid_right])))
    idx = [0,8,9,10,8,1,2,3,0,4,5,1,5,6,2,6,7,3,7,4]
    return corners[idx, :]


def rotz(t):
    ''' Rotation about the z-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])


def get_bbox_lidar(box, name='bbox', color='yellow', width=3):
#     cmap = ['red', 'blue', 'yellow', 'green', 'orange', 'pink', 'cyan', 'grey', 'magenta', 'purple', 'lightgreen', 'lightblue']
    x, y, z, l, w, h, heading = box
    trans_mat = np.eye(4, dtype=np.float32)
    trans_mat[:3, 3] = np.array((x, y, z))
    trans_mat[:3, :3] = rotz(heading)
    markers = get_linemarks(trans_mat, (w, l, h))
    return go.Scatter3d(
        mode='lines',
        x=markers[:, 0],
        y=markers[:, 1],
        z=markers[:, 2],
        line=dict(color=color, width=width),
        name=name)


def showvelo(
    lidar: np.ndarray,  # (N, 3)
    p2_score: Optional[np.ndarray]=None,  # (N,)
    size: float=3,
):
    if torch.is_tensor(lidar):
        lidar = lidar.cpu().detach().numpy()
    if p2_score is not None and torch.is_tensor(p2_score):
        p2_score = p2_score.cpu().detach().numpy()
    while lidar.ndim > 2:
        assert lidar.shape[0] == 1
        lidar = lidar[0]
        if p2_score is not None:
            assert p2_score.shape[0] == 1
            p2_score = p2_score[0]
    if p2_score is not None and p2_score.ndim == 2:
        assert p2_score.shape[1] == 1
        p2_score = p2_score[:, 0]

    bbox = [get_bbox_lidar(np.array([0, 0, 0, 2, 2, 2, 0]), name=f'bbox', color='lightgreen')]
    vis_lidar = go.Scatter3d(
        x=lidar[:,0],
        y=lidar[:,1],
        z=lidar[:,2],
        mode='markers',
        marker_color=p2_score,
        hovertext=p2_score,
        marker_colorscale='Jet',
        marker_size=size)
    fig = go.Figure(data=[vis_lidar] + bbox, layout=get_layout_config())
    fig.show()


def showflow(
    points0: np.ndarray,  # (N, 3)
    points1: np.ndarray,  # (N, 3)
    p2_score: Optional[np.ndarray]=None,  # (N,)
    arrow_ratio: float=0.15,
    size: float=3,
):
    assert points0.shape == points1.shape
    if torch.is_tensor(points0):
        points0 = points0.cpu().detach().numpy()
    if torch.is_tensor(points1):
        points1 = points1.cpu().detach().numpy()
    if p2_score is not None and torch.is_tensor(p2_score):
        p2_score = p2_score.cpu().detach().numpy()
    while points0.ndim > 2:
        assert points0.shape[0] == 1
        points0 = points0[0]
        points1 = points1[0]
        if p2_score is not None:
            assert p2_score.shape[0] == 1
            p2_score = p2_score[0]
    if p2_score is not None and p2_score.ndim == 2:
        assert p2_score.shape[1] == 1
        p2_score = p2_score[:, 0]

    bbox = [get_bbox_lidar(np.array([0, 0, 0, 2, 2, 2, 0]), name=f'bbox', color='lightgreen')]
    vis_lidar = []
    lidar = np.stack([points0, points1], axis=1)
    cone_center = (0.75 * arrow_ratio) * points0 + (1 - 0.75 * arrow_ratio) * points1
    cone_dir = (points1 - points0) * arrow_ratio * 2
    if p2_score is not None:
        colors = (mpl.colormaps['jet'](p2_score)[:, :3] * 255).astype(np.uint8)
    for i in range(points0.shape[0]):
        color = 'magenta' if p2_score is None else f"rgb{tuple(colors[i])}"
        vis_lidar.append(go.Scatter3d(
            mode='lines',
            x=lidar[i, :, 0],
            y=lidar[i, :, 1],
            z=lidar[i, :, 2],
            line=dict(color=color, width=size)))
        vis_lidar.append(go.Cone(
            x=cone_center[i:i+1, 0],
            y=cone_center[i:i+1, 1],
            z=cone_center[i:i+1, 2],
            u=cone_dir[i:i+1, 0],
            v=cone_dir[i:i+1, 1],
            w=cone_dir[i:i+1, 2],
            colorscale=[[0, color], [1,color]],
            showscale=False,
        ))
    fig = go.Figure(data=vis_lidar + bbox, layout=get_layout_config())
    fig.show()


def show_scene(
        lidar=None,
        p2_score=None,
        labels=None,
        old_predictions=None,
        new_predictions=None,
        size=0.8):
    if torch.is_tensor(lidar):
        lidar = lidar.cpu().detach().numpy()
    if p2_score is not None and torch.is_tensor(p2_score):
        p2_score = p2_score.cpu().detach().numpy()
    if labels is not None and torch.is_tensor(labels):
        labels = labels.cpu().detach().numpy()
    if old_predictions is not None and torch.is_tensor(old_predictions):
        old_predictions = old_predictions.cpu().detach().numpy()
    if new_predictions is not None and torch.is_tensor(new_predictions):
        new_predictions = new_predictions.cpu().detach().numpy()
    gt_bboxes = [] if labels is None else [get_bbox_lidar(
        label, name=f'gt_bbox_{i}', color='lightgreen') for i, label in enumerate(labels)]
    old_pred_bboxes = [] if old_predictions is None else [get_bbox_lidar(
        pred, name=f'old_pred_bbox_{i}', color='pink') for i, pred in enumerate(old_predictions)]
    new_pred_bboxes = [] if new_predictions is None else [get_bbox_lidar(
        pred, name=f'new_pred_bbox_{i}', color='cyan') for i, pred in enumerate(new_predictions)]
    if lidar is None:
        vis_lidar = []
    else:
        vis_lidar = [go.Scatter3d(
            x=lidar[:,0],
            y=lidar[:,1],
            z=lidar[:,2],
            mode='markers',
            marker_color=p2_score,
            hovertext=p2_score,
            marker_colorscale='Jet',
            marker_size=size)]
    fig = go.Figure(data=vis_lidar + gt_bboxes + old_pred_bboxes + new_pred_bboxes, layout=scene_layout_config)
    return fig