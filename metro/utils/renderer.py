"""
Rendering tools for 3D mesh visualization on 2D image.

Parts of the code are taken from https://github.com/akanazawa/hmr
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import code
import os
import sys
import torch
import pytorch3d
import random
# Data structures and functions for rendering
from pytorch3d.transforms import (
    Transform3d
)
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PerspectiveCameras,
    PointLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)
os.environ["CUB_HOME"] = os.getcwd() + "/cub-1.10.0"
sys.path.append(os.path.abspath(''))

# Rotate the points by a specified angle.
def rotateY(points, angle):
    ry = np.array([
        [np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
        [-np.sin(angle), 0., np.cos(angle)]
    ])
    return np.dot(points, ry)

def draw_skeleton(input_image, joints, draw_edges=True, vis=None, radius=None):
    """
    joints is 3 x 19. but if not will transpose it.
    0: Right ankle
    1: Right knee
    2: Right hip
    3: Left hip
    4: Left knee
    5: Left ankle
    6: Right wrist
    7: Right elbow
    8: Right shoulder
    9: Left shoulder
    10: Left elbow
    11: Left wrist
    12: Neck
    13: Head top
    14: nose
    15: left_eye
    16: right_eye
    17: left_ear
    18: right_ear
    """

    if radius is None:
        radius = max(4, (np.mean(input_image.shape[:2]) * 0.01).astype(int))

    colors = {
        'pink': (197, 27, 125),  # L lower leg
        'light_pink': (233, 163, 201),  # L upper leg
        'light_green': (161, 215, 106),  # L lower arm
        'green': (77, 146, 33),  # L upper arm
        'red': (215, 48, 39),  # head
        'light_red': (252, 146, 114),  # head
        'light_orange': (252, 141, 89),  # chest
        'purple': (118, 42, 131),  # R lower leg
        'light_purple': (175, 141, 195),  # R upper
        'light_blue': (145, 191, 219),  # R lower arm
        'blue': (69, 117, 180),  # R upper arm
        'gray': (130, 130, 130),  #
        'white': (255, 255, 255),  #
    }

    image = input_image.copy()
    input_is_float = False

    if np.issubdtype(image.dtype, np.float):
        input_is_float = True
        max_val = image.max()
        if max_val <= 2.:  # should be 1 but sometimes it's slightly above 1
            image = (image * 255).astype(np.uint8)
        else:
            image = (image).astype(np.uint8)

    if joints.shape[0] != 2:
        joints = joints.T
    joints = np.round(joints).astype(int)

    jcolors = [
        'light_pink', 'light_pink', 'light_pink', 'pink', 'pink', 'pink',
        'light_blue', 'light_blue', 'light_blue', 'blue', 'blue', 'blue',
        'purple', 'purple', 'red', 'green', 'green', 'white', 'white',
        'purple', 'purple', 'red', 'green', 'green', 'white', 'white'
    ]

    if joints.shape[1] == 19:
        # parent indices -1 means no parents
        parents = np.array([
            1, 2, 8, 9, 3, 4, 7, 8, 12, 12, 9, 10, 14, -1, 13, -1, -1, 15, 16
        ])
        # Left is light and right is dark
        ecolors = {
            0: 'light_pink',
            1: 'light_pink',
            2: 'light_pink',
            3: 'pink',
            4: 'pink',
            5: 'pink',
            6: 'light_blue',
            7: 'light_blue',
            8: 'light_blue',
            9: 'blue',
            10: 'blue',
            11: 'blue',
            12: 'purple',
            17: 'light_green',
            18: 'light_green',
            14: 'purple'
        }
    elif joints.shape[1] == 14:
        parents = np.array([
            1,
            2,
            8,
            9,
            3,
            4,
            7,
            8,
            -1,
            -1,
            9,
            10,
            13,
            -1,
        ])
        ecolors = {
            0: 'light_pink',
            1: 'light_pink',
            2: 'light_pink',
            3: 'pink',
            4: 'pink',
            5: 'pink',
            6: 'light_blue',
            7: 'light_blue',
            10: 'light_blue',
            11: 'blue',
            12: 'purple'
        }
    elif joints.shape[1] == 21:  # hand
        parents = np.array([
            -1,
            0,
            1,
            2,
            3,
            0,
            5,
            6,
            7,
            0,
            9,
            10,
            11,
            0,
            13,
            14,
            15,
            0,
            17,
            18,
            19,
        ])
        ecolors = {
            0: 'light_purple',
            1: 'light_green',
            2: 'light_green',
            3: 'light_green',
            4: 'light_green',
            5: 'pink',
            6: 'pink',
            7: 'pink',
            8: 'pink',
            9: 'light_blue',
            10: 'light_blue',
            11: 'light_blue',
            12: 'light_blue',
            13: 'light_red',
            14: 'light_red',
            15: 'light_red',
            16: 'light_red',
            17: 'purple',
            18: 'purple',
            19: 'purple',
            20: 'purple',
        }
    else:
        print('Unknown skeleton!!')

    for child in range(len(parents)):
        point = joints[:, child]
        # If invisible skip
        if vis is not None and vis[child] == 0:
            continue
        if draw_edges:
            cv2.circle(image, (point[0], point[1]), radius, colors['white'],
                       -1)
            cv2.circle(image, (point[0], point[1]), radius - 1,
                       colors[jcolors[child]], -1)
        else:
            # cv2.circle(image, (point[0], point[1]), 5, colors['white'], 1)
            cv2.circle(image, (point[0], point[1]), radius - 1,
                       colors[jcolors[child]], 1)
            # cv2.circle(image, (point[0], point[1]), 5, colors['gray'], -1)
        pa_id = parents[child]
        if draw_edges and pa_id >= 0:
            if vis is not None and vis[pa_id] == 0:
                continue
            point_pa = joints[:, pa_id]
            cv2.circle(image, (point_pa[0], point_pa[1]), radius - 1,
                       colors[jcolors[pa_id]], -1)
            if child not in ecolors.keys():
                print('bad')
                import ipdb
                ipdb.set_trace()
            cv2.line(image, (point[0], point[1]), (point_pa[0], point_pa[1]),
                     colors[ecolors[child]], radius - 2)

    # Convert back in original dtype
    if input_is_float:
        if max_val <= 1.:
            image = image.astype(np.float32) / 255.
        else:
            image = image.astype(np.float32)

    return image

def draw_text(input_image, content):
    """
    content is a dict. draws key: val on image
    Assumes key is str, val is float
    """
    image = input_image.copy()
    input_is_float = False
    if np.issubdtype(image.dtype, np.float):
        input_is_float = True
        image = (image * 255).astype(np.uint8)

    black = (255, 255, 0)
    margin = 15
    start_x = 5
    start_y = margin
    for key in sorted(content.keys()):
        text = "%s: %.2g" % (key, content[key])
        cv2.putText(image, text, (start_x, start_y), 0, 0.45, black)
        start_y += margin

    if input_is_float:
        image = image.astype(np.float32) / 255.
    return image

def visualize_reconstruction(img, img_size, gt_kp, vertices, pred_kp, camera, renderer, color='pink', focal_length=1000):
    """Overlays gt_kp and pred_kp on img.
    Draws vert with text.
    Renderer is an instance of SMPLRenderer.
    camera size = [B, 3]
    """
    gt_vis = gt_kp[:, 2].astype(bool)
    loss = np.sum((gt_kp[gt_vis, :2] - pred_kp[gt_vis])**2)
    debug_text = {"sc": camera[0], "tx": camera[1], "ty": camera[2], "kpl": loss}
    # Fix a flength so i can render this with persp correct scale
    res = img.shape[1]
    camera_t = np.array([camera[1], camera[2], 2*focal_length/(res * camera[0] +1e-9)]) #camera_translation
    
    rend_img = renderer.visualize_mesh(vertices, camera_t, img) #Composed with pytorch3d

    rend_img = draw_text(rend_img, debug_text)

    # Draw skeleton
    gt_joint = ((gt_kp[:, :2] + 1) * 0.5) * img_size
    pred_joint = ((pred_kp + 1) * 0.5) * img_size
    img_with_gt = draw_skeleton( img, gt_joint, draw_edges=False, vis=gt_vis)
    skel_img = draw_skeleton(img_with_gt, pred_joint)

    combined = np.hstack([skel_img, rend_img])

    return combined

def visualize_reconstruction_test(img, img_size, gt_kp, vertices, pred_kp, camera, renderer, score, color='pink', focal_length=1000):
    """Overlays gt_kp and pred_kp on img.
    Draws vert with text.
    Renderer is an instance of SMPLRenderer.
    """
    gt_vis = gt_kp[:, 2].astype(bool)
    loss = np.sum((gt_kp[gt_vis, :2] - pred_kp[gt_vis])**2)
    debug_text = {"sc": camera[0], "tx": camera[1], "ty": camera[2], "kpl": loss, "pa-mpjpe": score*1000}
    # Fix a flength so i can render this with persp correct scale
    res = img.shape[1]
    camera_t = np.array([camera[1], camera[2], 2*focal_length/(res * camera[0] +1e-9)])

    rend_img = renderer.visualize_mesh(vertices, camera_t, img)#Composed with pytorch3d

    rend_img = draw_text(rend_img, debug_text)

    # Draw skeleton
    gt_joint = ((gt_kp[:, :2] + 1) * 0.5) * img_size
    pred_joint = ((pred_kp + 1) * 0.5) * img_size
    img_with_gt = draw_skeleton( img, gt_joint, draw_edges=False, vis=gt_vis)
    skel_img = draw_skeleton(img_with_gt, pred_joint)

    combined = np.hstack([skel_img, rend_img])

    return combined



def visualize_reconstruction_and_att(img, img_size, vertices_full, vertices, vertices_2d, camera, renderer, ref_points, attention, focal_length=1000):
    """Overlays gt_kp and pred_kp on img.
    Draws vert with text.
    Renderer is an instance of SMPLRenderer.
    """
    # Fix a flength so i can render this with persp correct scale
    res = img.shape[1]
    camera_t = np.array([camera[1], camera[2], 2*focal_length/(res * camera[0] +1e-9)])

    rend_img = renderer.visualize_mesh(vertices_full, camera_t, img)#Composed with pytorch3d

    heads_num, vertex_num, _ = attention.shape

    all_head = np.zeros((vertex_num,vertex_num))

    ###### find max
    # for i in range(vertex_num):
    #     for j in range(vertex_num):
    #         all_head[i,j] = np.max(attention[:,i,j])

    ##### find avg
    for h in range(4):
        att_per_img = attention[h]
        all_head = all_head + att_per_img   
    all_head = all_head/4

    col_sums = all_head.sum(axis=0)
    all_head = all_head / col_sums[np.newaxis, :]


    # code.interact(local=locals())

    combined = []
    if vertex_num>400:  # body
        selected_joints = [6,7,4,5,13] # [6,7,4,5,13,12] 
    else: # hand  
        selected_joints = [0, 4, 8, 12, 16, 20]
    # Draw attention
    for ii in range(len(selected_joints)):
        reference_id = selected_joints[ii]
        ref_point = ref_points[reference_id]
        attention_to_show = all_head[reference_id][14::] 
        min_v = np.min(attention_to_show)
        max_v = np.max(attention_to_show)
        norm_attention_to_show = (attention_to_show - min_v)/(max_v-min_v)

        vertices_norm = ((vertices_2d + 1) * 0.5) * img_size
        ref_norm = ((ref_point + 1) * 0.5) * img_size
        image = np.zeros_like(rend_img)

        for jj in range(vertices_norm.shape[0]):
            x = int(vertices_norm[jj,0]) 
            y = int(vertices_norm[jj,1])
            cv2.circle(image,(x,y), 1, (255,255,255), -1) 

        total_to_draw = []
        for jj in range(vertices_norm.shape[0]):
            thres = 0.0
            if norm_attention_to_show[jj]>thres:
                things = [norm_attention_to_show[jj], ref_norm, vertices_norm[jj]]
                total_to_draw.append(things)
                # plot_one_line(ref_norm, vertices_norm[jj], image, reference_id, alpha=0.4*(norm_attention_to_show[jj]-thres)/(1-thres)  )
        total_to_draw.sort()
        max_att_score = total_to_draw[-1][0]
        for item in total_to_draw:
            attention_score = item[0]
            ref_point = item[1]
            vertex = item[2]
            plot_one_line(ref_point, vertex, image, ii, alpha=(attention_score-thres)/(max_att_score-thres)  )
        # code.interact(local=locals())
        if len(combined)==0:
            combined = image
        else:
            combined = np.hstack([combined, image])
    

    final = np.hstack([img, combined, rend_img])

    return final


def visualize_reconstruction_and_att_local(img, img_size, vertices_full, vertices, vertices_2d, camera, renderer, ref_points, attention, color='light_blue', focal_length=1000):
    """Overlays gt_kp and pred_kp on img.
    Draws vert with text.
    Renderer is an instance of SMPLRenderer.
    """
    # Fix a flength so i can render this with persp correct scale
    res = img.shape[1]
    camera_t = np.array([camera[1], camera[2], 2*focal_length/(res * camera[0] +1e-9)])

    rend_img = renderer.visualize_mesh(vertices_full, camera_t, img) #Composed with pytorch3d

    heads_num, vertex_num, _ = attention.shape
    all_head = np.zeros((vertex_num,vertex_num))

    ##### compute avg attention for 4 attention heads
    for h in range(4):
        att_per_img = attention[h]
        all_head = all_head + att_per_img   
    all_head = all_head/4

    col_sums = all_head.sum(axis=0)
    all_head = all_head / col_sums[np.newaxis, :]

    combined = []
    if vertex_num>400:  # body
        selected_joints = [7]  # [6,7,4,5,13,12] 
    else: # hand  
        selected_joints = [0] # [0, 4, 8, 12, 16, 20] 
    # Draw attention
    for ii in range(len(selected_joints)):
        reference_id = selected_joints[ii]
        ref_point = ref_points[reference_id]
        attention_to_show = all_head[reference_id][14::] 
        min_v = np.min(attention_to_show)
        max_v = np.max(attention_to_show)
        norm_attention_to_show = (attention_to_show - min_v)/(max_v-min_v)
        vertices_norm = ((vertices_2d + 1) * 0.5) * img_size
        ref_norm = ((ref_point + 1) * 0.5) * img_size
        image = rend_img*0.4

        total_to_draw = []
        for jj in range(vertices_norm.shape[0]):
            thres = 0.0
            if norm_attention_to_show[jj]>thres:
                things = [norm_attention_to_show[jj], ref_norm, vertices_norm[jj]]
                total_to_draw.append(things)
        total_to_draw.sort()
        max_att_score = total_to_draw[-1][0]
        for item in total_to_draw:
            attention_score = item[0]
            ref_point = item[1]
            vertex = item[2]
            plot_one_line(ref_point, vertex, image, ii, alpha=(attention_score-thres)/(max_att_score-thres)  )

        for jj in range(vertices_norm.shape[0]):
            x = int(vertices_norm[jj,0])
            y = int(vertices_norm[jj,1])
            cv2.circle(image,(x,y), 1, (255,255,255), -1) 

        if len(combined)==0:
            combined = image
        else:
            combined = np.hstack([combined, image])

    final = np.hstack([img, combined, rend_img])

    return final


def visualize_reconstruction_no_text(img, img_size, vertices, camera, renderer, color='pink', focal_length=1000):
    """Overlays gt_kp and pred_kp on img.
    Draws vert with text.
    Renderer is an instance of SMPLRenderer.
    """
    # Fix a flength so i can render this with persp correct scale
    res = img.shape[1]
    camera_t = np.array([camera[1], camera[2], 2*focal_length/(res * camera[0] +1e-9)])

    rend_img = renderer.visualize_mesh(vertices, camera_t, img) #Composed with pytorch3d

    combined = np.hstack([img, rend_img])

    return combined


def plot_one_line(ref, vertex, img, color_index, alpha=0.0, line_thickness=None):
    # 13,6,7,8,3,4,5
    # att_colors = [(255, 221, 104), (255, 255, 0), (255, 215, 227),  (210, 240, 119), \
    #          (209, 238, 245), (244, 200, 243),  (233, 242, 216)] 
    att_colors = [(255, 255, 0), (244, 200, 243),  (210, 243, 119), (209, 238, 255), (200, 208, 255), (250, 238, 215)] 


    overlay = img.copy()
    # output = img.copy()
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness

    color = list(att_colors[color_index])
    c1, c2 = (int(ref[0]), int(ref[1])), (int(vertex[0]), int(vertex[1]))
    cv2.line(overlay, c1, c2, (alpha*float(color[0])/255,alpha*float(color[1])/255,alpha*float(color[2])/255) , thickness=tl, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)



def cam2pixel(cam_coord, f, c):
    x = cam_coord[:, 0] / (cam_coord[:, 2]) * f[0] + c[0]
    y = cam_coord[:, 1] / (cam_coord[:, 2]) * f[1] + c[1]
    z = cam_coord[:, 2]
    img_coord = np.concatenate((x[:,None], y[:,None], z[:,None]),1)
    return img_coord

class Renderer(object):

    def __init__(self, focal_length = 1000, img_res = 224, *args, faces):
        self.focal_length = ((focal_length, focal_length),)
        self.camera_center = ((img_res // 2 , img_res // 2),)
        self.img_res = img_res
        self.faces = faces

    def visualize_mesh(self, vertices, camera_t, img):
        device = torch.device('cuda:0')
        rend_img = self.__call__(vertices, camera_t, img, device).float() #returns [1, 224, 224, 4]
        rend_img = rend_img[0, ... , :3] #[224, 224, 3]
        rend_img_cpu = rend_img.cpu()
        rend_img = (cv2.rotate(rend_img_cpu.numpy(), cv2.ROTATE_180))
        rend_img = self.overlay_img(rend_img, img)

        return rend_img
    
    #Added function - Blends original image and rendered mesh
    def overlay_img(self, rend_img, img):
        mask = (rend_img == 1)[:,:,:,None]
        mask = torch.from_numpy(mask).squeeze()
        mask = mask.cpu().numpy()
        output = rend_img[:,:,:3] * ~mask + mask * img

        return output

    def __call__(self, vertices, camera_t, img, device):
        R, T = torch.from_numpy(np.eye(3)).unsqueeze(dim = 0), torch.from_numpy(camera_t).unsqueeze(dim = 0)

        cameras = PerspectiveCameras(device = device,focal_length=self.focal_length, principal_point = self.camera_center, R = R, T = T, image_size = ((self.img_res, self.img_res),), in_ndc=False)

        raster_settings = RasterizationSettings(
            image_size = self.img_res,
            blur_radius = 0.0)

        #lights = PointLights(device=device, location=[[2, 8, 2.1]])
        lights = PointLights(device=device, location=[[5, 5, -5]])
        renderer = MeshRenderer(
            rasterizer = MeshRasterizer(
                cameras = cameras,
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader( #change the shader
                device = device,
                cameras=cameras,
                lights=lights
            )
        )

        #Set reflected light color and shininess of mesh
        '''
        materials = Materials(
            device = device,
            specular_color=[[0.0, 1.0, 0.0]],
            shininess=10.0
        )
        '''
        #r, g, b = 1, 192/255.0, 203/255.0 #For pink mesh

        vertices = torch.from_numpy(vertices).to(device)
        verts_rgb = torch.ones_like(vertices)[None]
        #verts_rgb[:, :, 1] = g
        #verts_rgb[:, :, 2] = b
        textures = TexturesVertex(verts_features=verts_rgb.to(device)) #part segmentation

        vertices = vertices.reshape(-1, vertices.shape[0], vertices.shape[1])
        faces = torch.from_numpy(self.faces.astype(np.float32))
        faces = faces.unsqueeze(dim = 0).to(device)
        
        mesh = Meshes(
            verts = vertices,
            faces = faces,
            textures = textures
        )
        return renderer(mesh, cameras = cameras, lights = lights)
