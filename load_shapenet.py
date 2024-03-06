import imageio
import os
import glob
import torch
import numpy as np
from torchvision import transforms

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()

def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


image_to_tensor = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.0,), (1.0, ))
    ])
mask_to_tensor = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize((0.0,), (1.0, ))
])

def load_shapenet(rgb_paths, mask_paths, cam_path, sel_indices, scale_focal=False,):
    all_imgs, all_poses, all_masks, all_bboxes = [],[],[],[]
    focal = None
    all_cam = np.load(cam_path)         # NpzFile "... , keys : camera_mat_17, world_mat_inv_5, world_mat_0, world_mat_23, camera_mat_7...
    for idx, (rgb_path, mask_path) in enumerate(zip(rgb_paths, mask_paths)):
        i = sel_indices[idx]                             # 절대 idx
        img = imageio.imread(rgb_path)[..., :3]          # [H, W, 3]
        mask = imageio.imread(mask_path)[..., :1]        # [H, W, 3]

        H, W = img.shape[:2]
        if scale_focal:
            x_scale = W / 2.0
            y_scale = H / 2.0
            xy_delta = 1.0
        else:
            x_scale = y_scale = 1.0
            xy_delta = 0.0
        
        wmat_inv_key = "world_mat_inv_" + str(i)
        wmat_key = "world_mat_" + str(i)

        if wmat_inv_key in all_cam:
            extr_inv_mtx = all_cam[wmat_inv_key]
        else:
            extr_inv_mtx = all_cam[wmat_key]
            if extr_inv_mtx.shape[0] == 3:
                extr_inv_mtx = np.vstack((extr_inv_mtx, np.array([0, 0, 0, 1])))
            extr_inv_mtx = np.linalg.inv(extr_inv_mtx)

        intr_mtx = all_cam["camera_mat_" + str(idx)]
        fx, fy = intr_mtx[0, 0], intr_mtx[1, 1]
        assert abs(fx - fy) < 1e-9
        fx = fx * x_scale
        if focal is None:
            focal = fx
        else:
            assert abs(fx - focal) < 1e-5
        pose = extr_inv_mtx     # [4, 4]

        _coord_trans_world = torch.tensor([
        [1, 0, 0, 0], 
        [0, 0, -1, 0], 
        [0, 1, 0, 0], 
        [0, 0, 0, 1]])

        _coord_trans_cam = torch.tensor([
            [1, 0, 0, 0], 
            [0, -1, 0, 0], 
            [0, 0, -1, 0], 
            [0, 0, 0, 1]])
        
        pose = (_coord_trans_world @ torch.tensor(pose) @ _coord_trans_cam)    # c2w

        img_tensor = image_to_tensor(img)
        mask_tensor = mask_to_tensor(mask)

        rows = np.any(mask, axis=1) # [H, 1]
        cols = np.any(mask, axis=0) # [H, 1]
        rnz = np.where(rows)[0]
        cnz = np.where(cols)[0]
        if len(rnz) == 0:
            raise RuntimeError(
                "ERROR: Bad image at", rgb_path, "please investigate!"
            )
        rmin, rmax = rnz[[0, -1]]
        cmin, cmax = cnz[[0, -1]]

        all_imgs.append(img_tensor)
        all_poses.append(pose)

    imgs = torch.stack(all_imgs, 0)
    poses = torch.stack(all_poses, 0)

    render_poses = torch.stack([pose_spherical(angle, -30.0, 2.7) for angle in np.linspace(-180,180,40+1)[:-1]], 0)

    return imgs, poses, render_poses, [H, W, focal]

def load_nerf_shapenet_data(path, mae_input= 20, stage='train', exp= 1, sel_fix=True, scale_focal=1) :
    """
    path : direction of metadata.yaml 
    """
    # Choose expriment
    if exp == 1 :
        cats = []
        fix_cats = ["02958343", "03001627"]     # [car, chair]
        for i in fix_cats :
            cats.extend(list(glob.glob(os.path.join(path, i))))
    if exp == 2:
        cats = [x for x in glob.glob(os.path.join(path, "*")) if os.path.isdir(x)]

    # Split
    if stage == "train":
        file_lists = [os.path.join(x, "mae_train.lst") for x in cats]
    elif stage == "val":
        file_lists = [os.path.join(x, "mae_val.lst") for x in cats]
    elif stage == "test":
        file_lists = [os.path.join(x, "mae_test.lst") for x in cats]
    
    all_objs = []
    for file_list in file_lists:            # file_list : NMR_Dataset/02691156/test.lst
        if not os.path.exists(file_list):
            continue
        base_dir = os.path.dirname(file_list)   # NMR_Dataset/02691156
        cat = os.path.basename(base_dir)        # 02691156
        with open(file_list, "r") as f:
            objs = [(cat, os.path.join(base_dir, x.strip())) for x in f.readlines()] # /****/b7bd7a753da0aa113ce4891c7dcdfb1c
        all_objs.extend(objs) # [cat, path(image, cam_pose)]
    
    num_objects = len(all_objs)
    print("Load ShapeNet from", path, num_objects)
    
    # 
    train_imgs, train_poses, object_list = [], [], []
    for all_obj in all_objs :
        cat, root_dir = all_obj
        object_list.append(os.path.basename(root_dir))
        print("Category : ", cat)
        
        rgb_paths = [x for x in glob.glob(os.path.join(root_dir, 'image', '*'))
                    if (x.endswith('.jpg') or x.endswith('.png'))]
        mask_paths = [x for x in glob.glob(os.path.join(root_dir, 'mask', '*'))
                    if (x.endswith('.jpg') or x.endswith('.png'))]
        
        # select
        num_rgbs = len(rgb_paths)
        if sel_fix :
            sel_indices = range(num_rgbs-mae_input, num_rgbs)
        else :
            sel_indices = np.random.choice(len(rgb_paths), mae_input, replace=False)

        rgb_paths = [rgb_paths[i] for i in sel_indices]     # len : mae_input(20)
        mask_paths = [mask_paths[i] for i in sel_indices]   
        
        # camera
        cam_path = os.path.join(root_dir, "cameras.npz")
        
        imgs, poses, render_poses, hwf = load_shapenet(rgb_paths, mask_paths, cam_path, sel_indices, scale_focal=False,)

        train_imgs.append(imgs)          # [N, H, W, 3]
        train_poses.append(poses)        # [N, 4, 4]

    train_imgs = torch.stack(train_imgs, 0)      # [O, N, H, W, 3]    
    train_poses = torch.stack(train_poses, 0)    # [O, N, 4, 4]

    return train_imgs, train_poses, hwf, object_list

def sampling_pose(N, theta_range=[-180.+1.,180.-1.], phi_range=[-90., 0.], radius_range=[2.7, 3.0]) :
    """ sampling with sorting angle
    """
    theta = np.random.uniform(*theta_range, N)
    phi = np.random.uniform(*phi_range, N)
    radius = np.random.uniform(*radius_range, N)

    render_poses = torch.stack([pose_spherical(theta[i], phi[i], radius[i]) for i in range(N)], 0)    # [N, 4, 4]
    
    return render_poses