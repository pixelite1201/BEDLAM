import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from .rotation_converter import batch_euler2matrix


def plot_3d_andCam(p3d, camWorld):
    from .kp_utils import get_smpl_joint_names, get_spin_joint_names
    from mpl_toolkits.mplot3d import Axes3D

    jn = get_spin_joint_names()[25:]
    import matplotlib.cm as cm
    colors = cm.tab20c(np.linspace(0, 1, p3d.shape[0]))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for j in range(p3d.shape[0]):
        ax.scatter(p3d[j, 0], p3d[j, 1], p3d[j, 2], c=colors[j], label=j)
        label = '(%d, %d, %d), %s' % (p3d[j, 0], p3d[j, 1], p3d[j, 2], j)
        ax.text(p3d[j, 0], p3d[j, 1], p3d[j, 2], jn[j])
    # ax.scatter(p3d[:, 0], p3d[:, 1], p3d[:, 2])


    ax.scatter(camWorld[0,0], camWorld[0,1], camWorld[0,2], c='black', label='Camera')
    ax.text(camWorld[0,0], camWorld[0,1], camWorld[0,2], 'Camera')
    # plt.legend()


    max_range = np.max(np.array([np.max(p3d[j, 0]) + np.min(p3d[j, 0]), np.max(p3d[j, 1]) + np.min(p3d[j, 1]), np.max(p3d[j, 2]) + np.min(p3d[j, 2])]))
    Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (np.max(p3d[j, 0]) + np.min(p3d[j, 0]))
    Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (np.max(p3d[j, 1]) + np.min(p3d[j, 1]))
    Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (np.max(p3d[j, 2]) + np.min(p3d[j, 2]))
    # Comment or uncomment following both lines to test the fake bounding box:
    # for xb, yb, zb in zip(Xb, Yb, Zb):
    #     ax.plot([xb], [yb], [zb], 'w')


    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()
    print('wop')


def focalLength_mm2px(focalLength,dslr_sens,  focalPoint):
    focal_pixel = (focalLength / dslr_sens) * focalPoint *2
    return focal_pixel


def toCamCoords(j3d, camPosWorld):
    # transform gt to camera coordinate frame
    j3d = j3d - camPosWorld
    return  j3d


def unreal2cv2(points):
    '''

    :param points: Nx3
    :return:
    '''
    # x --> y, y --> z, z --> x
    points = np.roll(points, 2, 1)
    # change direction of y
    points = points * np.array([1,-1,1])
    return points


# def smpl2opencv(j3d):
#     # swap axis 1 and axis 2
#     j3d[:, [1, 2]] = j3d[:, [2, 1]]
#     j3d = j3d * np.array([1,1,-1])
#     return j3d


def smpl2opencv(j3d):
    # change sign of axis 1 and axis 2
    j3d = j3d * np.array([1,-1,-1])
    return j3d


def project_point(joint, RT, KKK):

    P = np.dot(KKK,RT)
    joints_2d = np.dot(P, joint)
    joints_2d = joints_2d[0:2] / joints_2d[2]

    return joints_2d


def project_2d(args, df, i, pNum, joints3d, imgPath, imgHeight, imgWidth, constants, meanPose=False):
    if args.scene3d:
        camPosWorld = [df.iloc[i]['camX'], df.iloc[i]['camY'], df.iloc[i]['camZ']]
        camYaw = df.iloc[i]['camYaw']
    else:
        camPosWorld = constants['camPosWorld']
        camYaw = constants['camPitch']

    if meanPose:
        transl = np.array([0,0,0])
        yawSMPL = 0
        trans3d = [0,0,0]
    else:
        transl = df.iloc[i]['gt'][pNum]['transl']
        yawSMPL = df.iloc[i]['YawLocal'][pNum]
        trans3d = [df.iloc[i]['XLocal'][pNum],
                   df.iloc[i]['YLocal'][pNum],
                   df.iloc[i]['ZLocal'][pNum]]

    if not (args.scene3d):
        gt2d, gt3d_camCoord, valid, valJoints = project2d_hdri(joints3d, focalLength=constants['focalLength'],
                                                           trans3d=trans3d,
                                                           dslr_sens_width=constants['dslr_sens_width'],
                                                           dslr_sens_height=constants['dslr_sens_height'],
                                                           camPosWorld=camPosWorld,
                                                           cy=imgHeight / 2,
                                                           cx=imgWidth / 2,
                                                           imgPath=imgPath,
                                                           yawSMPL=yawSMPL,
                                                           transl=transl,
                                                           debug=False,
                                                           checkValid=True,
                                                           constants=constants,
                                                           meanPose=meanPose)
    else:
        gt2d, gt3d_camCoord, valid, valJoints = project2d_world( joints3d, focalLength=constants['focalLength'],
                                                                 trans3d=trans3d,
                                                                 dslr_sens_width=constants['dslr_sens_width'],
                                                                 dslr_sens_height=constants['dslr_sens_height'],
                                                                 camPosWorld=camPosWorld,
                                                                 cy=imgHeight / 2,
                                                                 cx=imgWidth / 2,
                                                                 imgPath=imgPath,
                                                                 yawSMPL=yawSMPL,
                                                                 transl=transl,
                                                                 debug=False,
                                                                 checkValid=True,
                                                                 constants=constants,
                                                                 camYaw=camYaw,
                                                                 meanPose=meanPose)

    # gt2d are 2d projections of gt3d_camCoord
    return gt2d, gt3d_camCoord, valid, valJoints


def project2d_hdri(j3d, trans3d, focalLength, dslr_sens_width, dslr_sens_height, cy, cx, camPosWorld, yawSMPL,
                   imgPath=None, transl=None, debug=False, checkValid=False, constants=None, meanPose=False):
    """

    :param j3d:
    :param trans3d:
    :param focalLength:
    :param dslr_sens_width:
    :param dslr_sens_height:
    :param cy:
    :param cx:
    :param camPosWorld:
    :param yawSMPL:
    :param imgPath:
    :param transl:
    :return: 2d projection and 3d joints in camera coordinate system
    """

    focalLength_x = focalLength_mm2px(focalLength, dslr_sens_width, cx)
    focalLength_y = focalLength_mm2px(focalLength, dslr_sens_height, cy)

    camMat =  np.array( [[focalLength_x, 0, cx],
                        [0, focalLength_y,  cy],
                        [0 ,0,            1]])


    #camPosWorld and trans3d are in cm. Transform to meter
    trans3d = np.array(trans3d) / 100
    trans3d = unreal2cv2(np.reshape(trans3d,(1,3)))
    camPosWorld = np.array(camPosWorld) / 100
    camPosWorld = unreal2cv2(np.reshape(camPosWorld, (1,3)))

    # only when doing that with gt (other not necessary)
    # j3d = toCamCoords(j3d, camPosWorld)


    #get points in camera coordinate system
    j3d = smpl2opencv(j3d)

    #scans have a 90deg rotation, but for mean pose from vposer there is no such rotation
    if meanPose:
        rotMat, _ = cv2.Rodrigues(np.array([[0, (yawSMPL) / 180 * np.pi, 0]], dtype=float))
    else:
        rotMat, _ = cv2.Rodrigues(np.array([[0, (yawSMPL-90)/180*np.pi, 0]], dtype=float))

    j3d = np.matmul(rotMat, j3d.T).T

    j3d = j3d + trans3d
    #j3d = j3d - np.array([0, 0.8, 0])

    # plot_3d_andCam(j3d[:22,:], camPosWorld)

    camera_rotationMatrix, _ = cv2.Rodrigues(np.array([constants['camPitch']/180 * np.pi, 0., 0.]).reshape(3,1))
    j3d_new = np.matmul(camera_rotationMatrix, j3d.T - camPosWorld.T).T
    camPos_new = np.matmul(camera_rotationMatrix, camPosWorld.T - camPosWorld.T).T

    # plot_3d_andCam(j3d_new[:22, :], camPos_new)


    #do projection
    #rt is cannonical, since coordinate system was already changed to cam coords
    #
    RT = np.concatenate((np.diag([1., 1., 1.]), np.zeros((3, 1))), axis=1)
    j2d = np.zeros((j3d_new.shape[0], 2))
    for i in range(j3d_new.shape[0]):
        j2d[i,:] = project_point(np.concatenate([j3d_new[i,:], np.array([1])]), RT, camMat)



    #prepare imgpath
    if not(imgPath is None):

        # imgPath = imgPath.split('/')
        # imgPath[-2] = 'images' #hardcoded
        # imgName = imgPath[-1].split('_')[:2]
        # imgPath = os.path.join(*imgPath[:-1])
        #
        # imgName = imgName[0] + '_' + imgName[1] + '.png'
        # imgPath = os.path.join(imgPath,imgName)
        #imgPath = os.path.join(*imgPath)

        img = cv2.imread('/'+imgPath)
        img = img[:,:,::-1]


    if debug:
        import matplotlib.cm as cm
        colors = cm.tab20c(np.linspace(0, 1, 25))
        fig = plt.figure(dpi=300)
        ax = fig.add_subplot(111)
        if not (imgPath is None):
            ax.imshow(img)
        for i in range(22):
            ax.scatter(j2d[i,0], j2d[i,1], c=colors[i], s=10)
            # ax.text(j2d[i,0], j2d[i,1], str(i))
        # plt.show()

        if not (imgPath is None):
            savename = imgPath.split('/')[-1]
            savename=savename.replace('.pkl','.jpg')
            plt.savefig('./figs/tmp'+savename)
            plt.close('all')


    #check if person is valid ground truth (50% of joints visible)
    if checkValid:
        #S: valid is a boolean scalar, valJoints is a boolean flag for valid joints
        valid, valJoints = checkValidGT(j2d, cx*2, cy*2, return_val_joints=True)
        return j2d, j3d_new, valid, valJoints

    return j2d, j3d_new


def translation_unreal2cv(translation):
    # translation are in cm. Transform to meter
    translation = np.array(translation) / 100
    return unreal2cv2(np.reshape(translation, (1, 3)))


def rotation_unreal2cv(pitch, yaw, roll):
    # Unreal camera parameters are euler angles not axis angles.
    rotmat = batch_euler2matrix(
        torch.tensor(
            [[-pitch / 180 * np.pi, -yaw / 180 * np.pi, -roll / 180 * np.pi]]
        )
    )[0].numpy()
    return rotmat


def project2d_world(j3d, smpl_trans, smpl_rotmat, cam_pos_world, cam_intrinsics, cam_rotation_mat,
                    img_path=None, debug=False, check_valid=False, mean_pose=False,
                    img_w=1280, img_h=720, debug_file_name=''):

    # only when doing that with gt (other not necessary)
    # j3d = toCamCoords(j3d, cam_pos_world)

    # get points in camera coordinate system
    j3d = smpl2opencv(j3d)

    j3d = np.matmul(smpl_rotmat, j3d.T).T

    j3d = j3d + smpl_trans

    # plot_3d_andCam(j3d[:22,:], cam_pos_world)

    j3d_new = np.matmul(cam_rotation_mat, j3d.T - cam_pos_world.T).T

    # do projection
    # rt is cannonical, since coordinate system was already changed to cam coords
    RT = np.concatenate((np.diag([1., 1., 1.]), np.zeros((3, 1))), axis=1)
    j2d = np.zeros((j3d_new.shape[0], 2))
    for i in range(j3d_new.shape[0]):
        j2d[i,:] = project_point(np.concatenate([j3d_new[i,:], np.array([1])]), RT, cam_intrinsics)


    if debug:
        from .kp_utils import get_smpl_joint_names, get_spin_joint_names
        import matplotlib.cm as cm

        img = cv2.imread(img_path)
        img = img[:, :, ::-1]

        cam_pos_new = np.matmul(cam_rotation_mat, cam_pos_world.T - cam_pos_world.T).T
        # plot_3d_andCam(j3d_new[25:, :], cam_pos_new)

        j2d[:, 0] = np.clip(j2d[:, 0], 0, img_w)
        j2d[:, 1] = np.clip(j2d[:, 1], 0, img_h)
        # breakpoint()
        jn = get_spin_joint_names() # get_smpl_joint_names()
        colors = cm.tab20c(np.linspace(0, 1, len(j2d[:,0])))
        fig = plt.figure(figsize=(19.2, 10.8))
        ax = fig.add_subplot(111)
        if not (img_path is None):
            ax.imshow(img)
        for i in range(len(j2d[:,0])):
            ax.scatter(j2d[i,0], j2d[i,1], s=10) # , c=colors[i])
            ax.text(j2d[i,0], j2d[i,1], jn[i], fontsize=5)

        if debug_file_name != '':
            plt.savefig(debug_file_name, bbox_inches='tight')
        else:
            plt.show()

        plt.close()


    # check if person is valid ground truth (50% of joints visible)
    # if check_valid:
    #     valid, valJoints = checkValidGT(j2d, cx*2, cy*2, return_val_joints=True)
    #     return j2d, j3d_new, valid, valJoints

    return j2d, j3d_new


def get_head_bbox_size(j2d):
    head2d = np.concatenate([j2d[55:60, :], j2d[76:,:]], axis=0)

    minPt = np.min(head2d, axis=0)
    maxPt = np.max(head2d, axis=0)
    bbxDiagSize = np.linalg.norm(maxPt-minPt,ord=2)

    return bbxDiagSize


def checkValidGT(j2d, resx, resy, threshold = 0.5, return_val_joints=False, numJointsToCheck=22):
    #todo: function not tested
    valX = np.logical_and(j2d[:, 0] > 0,  j2d[:,0] < resx )
    valY = np.logical_and(j2d[:, 1] > 0, j2d[:, 1] < resy)

    valJoint = np.logical_and(valX, valY)

    if np.sum(valJoint[:numJointsToCheck]) / numJointsToCheck < threshold:
        if return_val_joints:
            return False, valJoint
        else:
            return False
    else:
        if return_val_joints:
            return True, valJoint
        else:
            return True