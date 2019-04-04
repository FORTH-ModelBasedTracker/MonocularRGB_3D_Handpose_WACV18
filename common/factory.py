"""
Adapted from the MonoHand3D codebase for the MonocularRGB_3D_Handpose project (github release)

@author: Paschalis Panteleris (padeler@ics.forth.gr)
"""

import PyMBVCore as Core
import PyMBVDecoding
import PyMBVAcquisition
import PyMBVRendering
import PyMBVLibraries as Libraries
import PyMBVParticleFilter as pf
import PyCeresIK as IK
import numpy as np

mmanager = Core.MeshManager()

class Model(object):
    def __init__(self, model3d, decoder, bones_map, landmarks_vector):
        self.model3d = model3d
        self.decoder = decoder
        self.bones_map = bones_map
        self.landmarks_vector = landmarks_vector

        self.init_pose = Core.ParamVector(model3d.default_state)
        self.low_bounds = np.array(model3d.low_bounds)
        self.high_bounds = np.array(model3d.high_bounds)

        self.bounds_mask = Core.UIntVector(set(range(len(self.low_bounds))) - {3, 4, 5, 6})
        self.low_bounds[[0, 1, 2]] = [-3000, -3000, 50]
        self.high_bounds[[0, 1, 2]] = [3000, 3000, 4000]


    def reset_pose(self):
        self.init_pose = Core.ParamVector(self.model3d.default_state)


class HandPoseEstimator(object):

    def __init__(self, config):

        self.model = create_model(config["model"])
        custom_init = config.get("model_init_pose", None)
        if custom_init is not None:
            self.model.init_pose = Core.ParamVector(custom_init)

        self.ba = IK.ModelAwareBundleAdjuster()
        self.ba.model_to_keypoints = config["model_map"]
        self.ba.max_iterations = config["ba_iter"]

        self.ba.decoder = self.model.decoder
        self.ba.landmarks = self.model.landmarks_vector

        self.ba.bounds_mask = self.model.bounds_mask
        self.ba.low_bounds = Core.ParamVector(self.model.low_bounds)
        self.ba.high_bounds = Core.ParamVector(self.model.high_bounds)

        # self.ba.ceres_report = True
        self.last_result = self.last_score = None

    def estimate(self, obs_vec):

        score, res = self.ba.solve(obs_vec, Core.ParamVector(self.model.init_pose))

        self.last_score = score
        self.last_result = res

        return score, res

    def print_report(self):
        succ_steps = self.ba.summary.num_successful_steps
        fail_steps = self.ba.summary.num_unsuccessful_steps
        total_steps = self.ba.summary.num_iterations
        print("[", total_steps, succ_steps, fail_steps, "]", "[", self.last_score, "]", self.ba.summary.message)# , "==>", self.last_result)



def create_model(model_xml):
    model3d = pf.Model3dMeta.create(model_xml)
    model3d.setupMeshManager(mmanager)
    print('Model Factory, loaded model from <', model_xml, '>', ', bones:', model3d.n_bones, ', dims:', model3d.n_dims)

    decoder = model3d.createDecoder()
    decoder.loadMeshTickets(mmanager)

    # Create Landmarks
    model_parts = model3d.parts
    model_parts.genBonesMap()
    print("Model Factory, Parts Map:", model_parts.parts_map)

    names_l = names_g = model3d.parts.parts_map['all']

    init_positions = Core.Vector3fStorage([Core.Vector3(0, 0, 0)] * len(names_l))
    source_ref = pf.ReferenceFrame.RFGeomLocal
    dest_ref = pf.ReferenceFrame.RFModel
    bmap = model3d.parts.bones_map
    landmarks = pf.Landmark3dInfoSkinned.create_multiple(names_l, names_g, source_ref, init_positions, bmap)
    landmarks_decoder = pf.LandmarksDecoder()
    landmarks_decoder.convertReferenceFrame(dest_ref, decoder.kinematics, landmarks)

    landmarks_vector = IK.LandmarksVector()
    for l in landmarks:
        landmarks_vector.append(l)

    tickets = Core.MeshTicketList()
    mmanager.enumerateMeshes(tickets)
    last = 0
    for idx, t in enumerate(tickets):
        last = t

    bonesMap = Libraries.BonesMap({last: model3d.n_bones})

    return Model(model3d, decoder, bonesMap, landmarks_vector)