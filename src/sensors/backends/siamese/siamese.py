import numpy as np
from base_classes import ProbabilityDistanceEstimator, DisplacementEstimator, AbsoluteDistanceEstimator, RepresentationsCreator
import torch as t
from backends.siamese.siam_model import get_parametrized_model, load_model
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import rospy
import os
from bearnav2.msg import SensorsInput, ImageList, Features
from typing import List
from scipy import interpolate
import ros_numpy
import ros
from sensor_msgs.msg import Image


class SiameseCNN(DisplacementEstimator, ProbabilityDistanceEstimator,
                 AbsoluteDistanceEstimator, RepresentationsCreator):

    def __init__(self, padding: int=32, resize_w: int=512):
        super(SiameseCNN, self).__init__()
        self.supported_message_type = SensorsInput
        self.device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
        rospy.logwarn("Neural net device " + str(self.device))
        
        # init neural network
        self.padding = padding
        self.resize_w = resize_w
        model = get_parametrized_model(False, 3, 16, 0, 3, self.device)
        file_path = os.path.dirname(os.path.abspath(__file__))
        self.model = load_model(model, os.path.join(file_path, "./model_tiny.pt")).to(self.device).float()
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
        return t.stack([t.tensor(np.array(feature.values).reshape(feature.shape)) for feature in msg], dim=0).to(self.device)

    def _to_feature(self, msg: Image) -> Features:
        with t.no_grad():
            tensor_in = self.image_to_tensor(msg.data)
            reprs = self.model.get_repr(tensor_in.float())
            ret_features = []
            for repr in reprs:
                f = Features()
                f.shape = t.tensor(repr.shape).numpy()
                f.values = t.flatten(repr).detach().cpu().numpy()
                ret_features.append(f)
            return ret_features

    def health_check(self) -> bool:
        return True

    def process_msg(self, msg):
        hist = self.forward(msg.map_features, msg.live_features)
        f = interpolate.interp1d(np.linspace(0, self.resize_w, hist.shape[-1]), hist, kind="cubic")
        interp_hist = f(np.arange(0, self.resize_w))
        self.distances_probs = np.max(interp_hist, axis=1)
        ret = []
        for hist in interp_hist:
            zeros = np.zeros(np.size(hist)//2)
            ret.append(np.concatenate([zeros, hist, zeros]))    # siam can do only -0.5 to 0.5 img so extend both sides by sth
        self.histograms = ret
        return ret

    def forward(self, map_features, live_features):
        """
        map_images: list of Image messages (map images)
        live_images: list of Image messages (live feed) - right now is supported size 1
        """
        tensor1 = self._from_feature(map_features)
        tensor2 = self._from_feature(live_features)
        if tensor1.shape[0] != tensor2.shape[0]:
            tensor2 = tensor2.repeat(tensor1.shape[0], 1, 1, 1)
        with t.no_grad():
            # only the crosscorrelation here since the representations were already calculated!
            hists = self.model.match_corr(tensor1.float(), tensor2.float(), padding=self.padding)[:, 0, 0]
            mean = hists.mean()
            std = hists.std()
            hists = (hists - mean) / std
            hists = t.sigmoid(hists)
        return np.flip(hists.cpu().numpy(), axis=-1)

    def image_to_tensor(self, imgs):
        desired_height = int(imgs[0].height * self.resize_w / imgs[0].width)
        image_list = [transforms.Resize(desired_height, interpolation=InterpolationMode.NEAREST)(self.to_tensor(ros_numpy.numpify(img)).to(self.device))
                      for img in imgs]
        stacked_tensor = t.stack(image_list)
        return stacked_tensor
