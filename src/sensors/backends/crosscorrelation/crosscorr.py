import numpy as np
from base_classes import DisplacementEstimator
import torch as t
from torch.nn import functional as F
from torchvision import transforms
import rospy
import os
from bearnav2.msg import SensorsInputImages
from typing import List
from scipy import interpolate
import ros_numpy


HEIGHT_TOLERANCE = 2


class CrossCorrelation(DisplacementEstimator):

    def __init__(self, padding: int = 32, network_division: int = 8, resize_w: int = 512):
        super(CrossCorrelation, self).__init__()
        self.supported_message_type = SensorsInputImages
        self.device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
        rospy.logwarn("Crosscorrelation running on device: " + str(self.device))
        # init neural network
        self.padding = padding
        self.network_division = network_division
        self.resize_w = resize_w
        self.to_tensor = transforms.ToTensor()
        self.alignment_processing = False
        self.histograms = None
        self.distances_probs = None
        rospy.loginfo("Cross correlation displacement estimator sucessfully initialized!")

    def _displacement_message_callback(self, msg: SensorsInputImages) -> List[np.ndarray]:
        self.alignment_processing = True
        self.process_msg(msg)
        return self.histograms

    def health_check(self) -> bool:
        return True

    def process_msg(self, msg):
        # TODO: check if it's working for multiple map images
        hist = self.forward(msg.map_images.data, msg.live_images.data)      # not sure about .data here
        f = interpolate.interp1d(np.linspace(0, self.resize_w, len(hist[0])), hist[0], kind="cubic")
        interp_hist = f(np.arange(0, self.resize_w))
        zeros = np.zeros(np.size(interp_hist)//2)
        ret = np.concatenate([zeros, interp_hist, zeros])    # siam can do only -0.5 to 0.5 img so extend both sides by sth
        self.histograms = [ret]

    def forward(self, map_images, live_images):
        """
        map_images: list of Image messages (map images)
        live_images: list of Image messages (live feed) - right now is supported size 1
        """
        tensor1 = self.image_to_tensor(map_images)
        tensor2 = self.image_to_tensor(live_images)
        rospy.logdebug(f"Aligning using crosscorr {tensor1.shape[0]} to {tensor2.shape[0]} images")
        tensor2 = tensor2.repeat(tensor1.shape[0], 1, 1, 1)
        with t.no_grad():
            hists = self._match_corr(tensor1, tensor2, padding=self.padding).cpu().numpy()
        return hists[0][0]

    def _match_corr(self, embed_ref, embed_srch, padding=None):
        if padding is None:
            padding = self.padding
        b, c, h, w = embed_srch.shape
        _, _, h_ref, w_ref = embed_ref.shape

        match_map = F.conv2d(F.pad(embed_srch.view(1, b * c, h, w),
                                   pad=(padding, padding, HEIGHT_TOLERANCE, HEIGHT_TOLERANCE), mode='circular'),
                             embed_ref, groups=b)

        match_map = t.max(match_map.permute(1, 0, 2, 3), dim=2)[0].unsqueeze(2)
        return match_map

    def image_to_tensor(self, imgs):
        desired_height = int(imgs[0].height * int(self.resize_w // self.network_division) / imgs[0].width)
        image_list = [transforms.Resize(desired_height)(self.to_tensor(ros_numpy.numpify(img)).to(self.device))
                      for img in imgs]
        stacked_tensor = t.stack(image_list)
        return stacked_tensor
