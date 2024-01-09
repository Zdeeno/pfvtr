import numpy as np
from base_classes import ProbabilityDistanceEstimator, DisplacementEstimator, AbsoluteDistanceEstimator, \
    RepresentationsCreator
import torch as t
from backends.siamese.siam_model import get_parametrized_model, load_model
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import rospy
import os
from pfvtr.msg import SensorsInput, ImageList, Features, Descriptor
from typing import List
from scipy import interpolate
import ros_numpy
import ros
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

bridge = CvBridge()


def check_xy(horizontal, vertical, allowed_shifts, vertical_limit=50):
    if abs(vertical) > vertical_limit:
        return False
    for shift in allowed_shifts:
        if shift[0] <= horizontal <= shift[1]:
            return True
    return False


class SiamFeature(DisplacementEstimator, ProbabilityDistanceEstimator,
                  AbsoluteDistanceEstimator, RepresentationsCreator):

    def __init__(self, padding: int = 32, resize_w: int = 512, descriptor: str = "BRISK",path_to_model=None):
        super(SiamFeature, self).__init__()
        self.supported_message_type = SensorsInput
        self.descriptor = descriptor
        self.device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
        rospy.logwarn("Neural net device " + str(self.device))
        self.featureTypes = {"BRISK": cv2.BRISK_create()}  # TODO make this more general
        # init neural network
        self.padding = padding
        self.resize_w = resize_w
        model = get_parametrized_model(False, 3, 16, 0, 3, self.device)
        if path_to_model is None:
            file_path = os.path.dirname(os.path.abspath(__file__))
            path_to_model = os.path.join(file_path, "./model_tiny.pt")
        self.model = load_model(model, path_to_model).to(self.device).float()
        self.model = self.model.eval()

        # if self.device == t.device("cuda"):
        #     from torch2trt import torch2trt
        #     rospy.loginfo("speeding up neural network")
        #     tmp = t.ones((1, 3, 384, 512)).cuda().float()
        #     self.model.backbone = torch2trt(self.model.backbone, [tmp])

        self.to_tensor = transforms.ToTensor()
        self.alignment_processing = False
        self.histograms = None
        self.distances_probs = None
        rospy.loginfo("Siamese-NN displacement/distance estimator successfully initialized!")

    def _displacement_message_callback(self, msg: SensorsInput) -> List[np.ndarray]:
        self.alignment_processing = True
        self.process_msg(msg)
        return self.histograms

    def _prob_dist_message_callback(self, msg: SensorsInput) -> List[float]:
        if not self.alignment_processing:
            self.process_msg(msg)
        return self.distances_probs

    def _abs_dist_message_callback(self, msg: SensorsInput) -> float:
        # TODO: msg.distances seems obsolete. Should be msg.map_distances or msg.map_transitions
        if not len(msg.distances) > 0:
            rospy.logerr("You cannot assign absolute distance to ")
            raise Exception("Absolute distant message callback for siamese network.")
        if not self.alignment_processing:
            self.process_msg(msg)
        return self.distances[np.argmax(self.distances_probs)]

    def _from_feature(self, msg: Features):
        a = t.stack([t.tensor(np.array(feature.values).reshape(feature.shape)) for feature in msg], dim=0).to(
            self.device)
        b = [np.array(feature.descriptors) for feature in msg]
        return a, b

    def _to_feature(self, msg: Image) -> Features:
        with t.no_grad():
            tensor_in = self.image_to_tensor(msg.data)
            reprs = self.model.get_repr(tensor_in.float())
            ret_features = []
            for repr in reprs:
                f = Features()
                f.shape = t.tensor(repr.shape).numpy()
                f.values = t.flatten(repr).detach().cpu().numpy()
                f.descriptors = self.get_descriptors(msg.data)
                ret_features.append(f)
            return ret_features

    def get_descriptors(self, msg: Image):
        descriptors = []
        if self.descriptor not in self.featureTypes:
            rospy.logwarn(
                "Feature type unknown or unavailable on this machine/installation. Not correcting heading!")
            return descriptors

        for i in msg:
            kps, des = self.featureTypes[self.descriptor].detectAndCompute(bridge.imgmsg_to_cv2(i), None)
            if des is not None:
                for kp, d in zip(kps, des):
                    descriptor = Descriptor()
                    descriptor.x = kp.pt[0]
                    descriptor.y = kp.pt[1]
                    descriptor.descriptor = d
                    descriptor.type = 4  # TODO make this more general
                    descriptors.append(descriptor)
            else:
                rospy.logwarn("No descriptors found!")

        return np.array(descriptors)

    def health_check(self) -> bool:
        return True

    def process_msg(self, msg):
        hist, FM1, FM2 = self.forward(msg.map_features, msg.live_features)
        if len(FM1) >= 1 and len(FM2) >= 1:
            fm_hist = self.matching(FM1, FM2, hist)
        print("fm_hist: ", np.shape(fm_hist))
        print("hist: ", np.shape(hist))
        f = interpolate.interp1d(np.linspace(0, self.resize_w, hist.shape[-1]), hist, kind="cubic")
        interp_hist = f(np.arange(0, self.resize_w))
        self.distances_probs = np.max(interp_hist, axis=1)
        ret = []
        for hist in interp_hist:
            zeros = np.zeros(np.size(hist) // 2)
            ret.append(
                np.concatenate([zeros, hist, zeros]))  # siam can do only -0.5 to 0.5 img so extend both sides by sth
        self.histograms = ret
        return ret

    def matching(self, map, querry, histograms):  ## recoded function from C++ to python
        # assert len(map[0]) == len(querry[0]) == len(histograms[0])
        shifts = []

        for h in histograms:
            shifts.append(self.histogram_single_sort(h))
        hist_out = []
        #print( "map len", len(map), "querry len", len(querry), "shifts len", len(shifts))
        for idx, m in enumerate(map):
            if idx >= len(querry):
                q = querry[0]
            else:
                q = querry[idx]
            if len(m) > 0 and len(q) > 0:
                hist_out.append(self.make_one_fm_hist(m, q, shifts[idx]))
            else:
                if len(m) == 0:
                    print("no map representation")
                if len(q) == 0:
                    print("no query representation")
        return np.array(hist_out)

    def make_one_fm_hist(self, descriptors1, descriptors2, allowed_shifts):
        bf = cv2.BFMatcher()
        bin_count = 65
        gran = 20
        hist_max = 0

        masks = self.make_mask(descriptors1, descriptors2, allowed_shifts)
        do1 = np.array([np.array(d.descriptor, dtype=np.uint8) for d in descriptors1])
        do2 = np.array([np.array(d.descriptor, dtype=np.uint8) for d in descriptors2])
        matches = bf.match(do1, do2, mask=masks)
        best_hist = np.zeros(bin_count)
        if len(matches) == 0:
            return best_hist
        for g in range(gran):
            hist = np.zeros(bin_count)
            for match in matches:
                i1 = descriptors1[match.queryIdx]
                i2 = descriptors2[match.trainIdx]
                dx = i1.x - i2.x + gran * bin_count / 2
                idx = (dx + g) / gran
                if idx < 0 or idx >= bin_count:
                    print("Should not happen?")
                    continue
                hist[int(idx)] += 1
            if hist_max < hist.max():
                hist_max = hist.max()
                best_hist = hist
        return best_hist

    def make_mask(self, descriptors1, descriptors2, allowed_shifts):
        mask = np.zeros((len(descriptors1), len(descriptors2)), dtype=np.uint8)
        if not descriptors1[0].type == descriptors2[0].type:
            return mask, descriptors1[0].descriptor, descriptors2[0].descriptor

        for i, d1 in enumerate(descriptors1):
            for j, d2 in enumerate(descriptors2):
                if i == j:
                    continue
                if check_xy(d1.x - d2.x, d1.y - d2.y, allowed_shifts):
                    mask[i, j] = 1
        return mask

    def histogram_single_sort(self, hist):
        # get indexes of values which are higher then half of maximum value of hist
        threshold_count = (hist > hist.max() // 2.0).sum() - 1
        # get indexes of threshold_count highest values
        indexes = np.argpartition(hist, threshold_count)[threshold_count:]
        permitted_shifts = []
        self.image_width = 512.0
        for e, i in enumerate(indexes):
            permitted_shifts.append(np.array([
                -((32.0 - i) * 8.0 + 4.0) * self.image_width / 512.0,
                -((32.0 - i) * 8.0 - 4.0) * self.image_width / 512.0
            ]))
        return permitted_shifts

    def forward(self, map_features, live_features):
        """
        map_images: list of Image messages (map images)
        live_images: list of Image messages (live feed) - right now is supported size 1
        """
        tensor1, FM1 = self._from_feature(map_features)
        #for i in FM1:
        #    print("FM1: ", np.shape(i))
        #print("live: ", np.shape(tensor1))
        tensor2, FM2 = self._from_feature(live_features)

        #for i in FM2:
        #    print("FM2: ", np.shape(i))
        #print("live: ", np.shape(tensor2))
        if tensor1.shape[0] != tensor2.shape[0]:
            tensor2 = tensor2.repeat(tensor1.shape[0], 1, 1, 1)
        with t.no_grad():
            # only the crosscorrelation here since the representations were already calculated!
            hists = self.model.match_corr(tensor1.float(), tensor2.float(), padding=self.padding)[:, 0, 0]
            mean = hists.mean()
            std = hists.std()
            hists = (hists - mean) / std
            hists = t.sigmoid(hists)
        return np.flip(hists.cpu().numpy(), axis=-1), FM1, FM2

    def image_to_tensor(self, imgs):
        desired_height = int(imgs[0].height * self.resize_w / imgs[0].width)
        image_list = [transforms.Resize(desired_height, interpolation=InterpolationMode.NEAREST)(
            self.to_tensor(ros_numpy.numpify(img)).to(self.device))
            for img in imgs]
        stacked_tensor = t.stack(image_list)
        return stacked_tensor
