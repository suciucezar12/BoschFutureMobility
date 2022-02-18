# import json
import time
import cv2
# import matplotlib.pyplot as plt
import numpy as np


# =========================================================
# Geometry
# =========================================================
def rotation_from_euler(roll=1., pitch=1., yaw=1.):
    """
    Get rotation matrix
    Args:
        roll, pitch, yaw:       In radians

    Returns:
        R:          [4, 4]
    """
    si, sj, sk = np.sin(roll), np.sin(pitch), np.sin(yaw)
    ci, cj, ck = np.cos(roll), np.cos(pitch), np.cos(yaw)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    R = np.identity(4)
    R[0, 0] = cj * ck
    R[0, 1] = sj * sc - cs
    R[0, 2] = sj * cc + ss
    R[1, 0] = cj * sk
    R[1, 1] = sj * ss + cc
    R[1, 2] = sj * cs - sc
    R[2, 0] = -sj
    R[2, 1] = cj * si
    R[2, 2] = cj * ci
    return R


def translation_matrix(vector):
    """
    Translation matrix

    Args:
        vector list[float]:     (x, y, z)

    Returns:
        T:      [4, 4]
    """
    M = np.identity(4)
    M[:3, 3] = vector[:3]
    return M


def load_camera_params():
    """
    Get the intrinsic and extrinsic parameters
    Returns:
        Camera extrinsic and intrinsic matrices
    """

    p = {   # camera parameters

        'pitch': 0.488692,
        'roll': 0.0,
        'yaw': 0.0,

        'x': 0.0,
        'y': 0.0,
        'z': 31.,

        'fx': 505.59817269,
        'fy': 506.50082419,

        'u0': 315.86549494,
        'v0': 238.34556175
    }

    # with open(file, 'rt') as handle:
    #     p = json.load(handle)

    fx, fy = p['fx'], p['fy']
    u0, v0 = p['u0'], p['v0']

    pitch, roll, yaw = p['pitch'], p['roll'], p['yaw']
    x, y, z = p['x'], p['y'], p['z']

    # Intrinsic
    K = np.array([[fx, 0, u0, 0],
                  [0, fy, v0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])

    # Extrinsic
    R_veh2cam = np.transpose(rotation_from_euler(roll, pitch, yaw))
    T_veh2cam = translation_matrix((-x, -y, -z))

    # Rotate to camera coordinates
    R = np.array([[0., -1., 0., 0.],
                  [0., 0., -1., 0.],
                  [1., 0., 0., 0.],
                  [0., 0., 0., 1.]])

    RT = R @ R_veh2cam @ T_veh2cam
    return RT, K


# =========================================================
# Projections
# =========================================================
def perspective(cam_coords, proj_mat, h, w):
    """
    P = proj_mat @ (x, y, z, 1)
    Project cam2pixel

    Args:
        cam_coords:         [4, npoints]
        proj_mat:           [4, 4]

    Returns:
        pix coords:         [h, w, 2]
    """
    eps = 1e-7
    pix_coords = proj_mat @ cam_coords

    pix_coords = pix_coords[:2, :] / (pix_coords[2, :] + eps)
    pix_coords = np.reshape(pix_coords, (2, h, w))
    pix_coords = np.transpose(pix_coords, (1, 2, 0))
    return pix_coords


def bilinear_sampler(imgs, pix_coords):
    """
    Construct a new image by bilinear sampling from the input image.
    Args:
        imgs:                   [H, W, C]
        pix_coords:             [h, w, 2]
    :return:
        sampled image           [h, w, c]
    """
    img_h, img_w, img_c = imgs.shape
    pix_h, pix_w, pix_c = pix_coords.shape
    out_shape = (pix_h, pix_w, img_c)

    pix_x, pix_y = np.split(pix_coords, [1], axis=-1)  # [pix_h, pix_w, 1]
    pix_x = pix_x.astype(np.float32)
    pix_y = pix_y.astype(np.float32)

    # Rounding
    pix_x0 = np.floor(pix_x)
    pix_x1 = pix_x0 + 1
    pix_y0 = np.floor(pix_y)
    pix_y1 = pix_y0 + 1

    # Clip within image boundary
    y_max = (img_h - 1)
    x_max = (img_w - 1)
    zero = np.zeros([1])

    pix_x0 = np.clip(pix_x0, zero, x_max)
    pix_y0 = np.clip(pix_y0, zero, y_max)
    pix_x1 = np.clip(pix_x1, zero, x_max)
    pix_y1 = np.clip(pix_y1, zero, y_max)

    # Weights [pix_h, pix_w, 1]
    wt_x0 = pix_x1 - pix_x
    wt_x1 = pix_x - pix_x0
    wt_y0 = pix_y1 - pix_y
    wt_y1 = pix_y - pix_y0

    # indices in the image to sample from
    dim = img_w

    # Apply the lower and upper bound pix coord
    base_y0 = pix_y0 * dim
    base_y1 = pix_y1 * dim

    # 4 corner vertices
    idx00 = (pix_x0 + base_y0).flatten().astype(np.int)
    idx01 = (pix_x0 + base_y1).astype(np.int)
    idx10 = (pix_x1 + base_y0).astype(np.int)
    idx11 = (pix_x1 + base_y1).astype(np.int)

    # Gather pixels from image using vertices
    imgs_flat = imgs.reshape([-1, img_c]).astype(np.float32)
    im00 = imgs_flat[idx00].reshape(out_shape)
    im01 = imgs_flat[idx01].reshape(out_shape)
    im10 = imgs_flat[idx10].reshape(out_shape)
    im11 = imgs_flat[idx11].reshape(out_shape)

    # Apply weights [pix_h, pix_w, 1]
    w00 = wt_x0 * wt_y0
    w01 = wt_x0 * wt_y1
    w10 = wt_x1 * wt_y0
    w11 = wt_x1 * wt_y1
    output = w00 * im00 + w01 * im01 + w10 * im10 + w11 * im11
    return output


def warped(src_image, pix_coords):
    """
    Warp source image using transformed points.
    """
    src_h, src_w, src_c = src_image.shape
    # dst_h, dst_w = dst_size
    dst_h, dst_w, pix_c = pix_coords.shape

    # Discretize & get points within image frame
    xpoints, ypoints = pix_coords[..., 0], pix_coords[..., 1]
    ind = np.where((xpoints >= 0) & (xpoints < src_w) & (ypoints >= 0) & (ypoints < src_h))
    ypoints, xpoints = ypoints[ind].astype(np.int), xpoints[ind].astype(np.int)

    # Get the corresponding point
    xpix, ypix = np.meshgrid(np.linspace(0, dst_w - 1, dst_w), np.linspace(0, dst_h - 1, dst_h))
    ypix, xpix = ypix[ind].astype(int), xpix[ind].astype(int)

    out = np.zeros((dst_h, dst_w, src_c), dtype=src_image.dtype)
    out[ypix, xpix, :] = src_image[ypoints, xpoints, :]
    return out


class Plane:
    """
    Defines a plane in the world
    """

    def __init__(self, x, y, z, roll, pitch, yaw,
                 col, row, scale):
        self.x, self.y, self.z = x, y, z
        self.roll, self.pitch, self.yaw = roll, pitch, yaw

        self.col, self.row = col, row
        self.scale = scale

        self.xyz = self.xyz_coord()

    def xyz_coord(self):
        """
        Returns:
            Grid coordinate: [b, 3/4, row*cols]
        """
        xmin = self.x
        xmax = self.x + self.col * self.scale
        ymin = self.y
        ymax = self.y + self.row * self.scale
        return meshgrid(xmin, xmax, self.col,
                        ymin, ymax, self.row)


def meshgrid(xmin, xmax, num_x, ymin, ymax, num_y, is_homogeneous=True):
    """
    Grid is parallel to z-axis

    Returns:
        array x,y,z,[1] coordinate   [3/4, num_x * num_y]
    """
    x = np.linspace(xmin, xmax, num_x)
    y = np.linspace(ymin, ymax, num_y)
    x, y = np.meshgrid(x, y)
    x = x.flatten()
    y = y.flatten()
    z = np.zeros_like(x)

    if is_homogeneous:
        coords = np.stack([x, y, z, np.ones_like(x)], axis=0)
    else:
        coords = np.stack([x, y, z], axis=0)
    return coords

# =====================================================================================================
# =====================================================================================================
# =====================================================================================================

# from utils import perspective, Plane, load_camera_params, bilinear_sampler, warped
interpolation_fn = bilinear_sampler  # or warped
TARGET_H, TARGET_W = 500, 500


def ipm_from_parameters(image, xyz, K, RT, interpolation_fn):
    # Flip y points positive upwards
    xyz[1] = -xyz[1]

    P = K @ RT
    pixel_coords = perspective(xyz, P, TARGET_H, TARGET_W)
    image2 = interpolation_fn(image, pixel_coords)
    return image2.astype(np.uint8)


def ipm_from_opencv(image, source_points, target_points):
    # Compute projection matrix
    M = cv2.getPerspectiveTransform(source_points, target_points)
    # Warp the image
    warped = cv2.warpPerspective(image, M, (TARGET_W, TARGET_H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=0)
    return warped


if __name__ == '__main__':
    ################
    # Derived method
    ################
    # Define the plane on the region of interest (road)
    plane = Plane(0,-191, 0, 0, 0, 0, TARGET_H, TARGET_W, 0.76562)
    # Retrieve camera parameters
    extrinsic, intrinsic = load_camera_params()

    time.sleep(1)
    cap = cv2.VideoCapture(0)

    ret, frame = cap.read()

    while True:
        # Apply perspective transformation

        start = time.time()

        frame = frame[int(frame.shape[0] * 0.7), :]

        warped1 = ipm_from_parameters(frame, plane.xyz, intrinsic, extrinsic, interpolation_fn)

        # ################
        # # OpenCV
        # ################
        # # Vertices coordinates in the source image
        # s = np.array([[830, 598],
        #               [868, 568],
        #               [1285, 598],
        #               [1248, 567]], dtype=np.float32)
        #
        # # Vertices coordinates in the destination image
        # t = np.array([[177, 231],
        #               [213, 231],
        #               [178, 264],
        #               [216, 264]], dtype=np.float32)
        #
        # # Warp the image
        # # warped2 = ipm_from_opencv(image, s, t)

        # Draw results
        # fig, ax = plt.subplots(1, 3)
        #
        # ax[0].imshow(image)
        # ax[0].set_title('Front View')
        # ax[1].imshow(warped1)
        # ax[1].set_title('IPM')
        # ax[2].imshow(warped2)
        # ax[2].set_title('IPM from OpenCv')
        # plt.tight_layout()
        # plt.show()

        # cv2.imshow("Image", image)

        cv2.imshow("Frame", frame)

        cv2.imshow("IPM", warped1)
        cv2.waitKey(1)

        end = time.time()

        print(end - start)

        ret, frame = cap.read()
