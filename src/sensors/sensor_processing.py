import numpy as np
from base_classes import DisplacementEstimator, RelativeDistanceEstimator, AbsoluteDistanceEstimator,\
    SensorFusion, ProbabilityDistanceEstimator, RepresentationsCreator
import rospy
from bearnav2.srv import Alignment, AlignmentResponse, SetDist, SetDistResponse
from bearnav2.msg import FloatList, SensorsInput, ImageList
from scipy import interpolate

"""
Here should be placed all classes for fusion of sensor processing
"""


class BearnavClassic(SensorFusion):

    def __init__(self, type_prefix: str,
                 abs_align_est: DisplacementEstimator, abs_dist_est: AbsoluteDistanceEstimator,
                 repr_creator: RepresentationsCreator, rel_align_est: DisplacementEstimator):
        super().__init__(type_prefix, abs_align_est=abs_align_est, abs_dist_est=abs_dist_est,
                         rel_align_est=rel_align_est, repr_creator=repr_creator)

    def _process_rel_alignment(self, msg):
        histogram = self.rel_align_est.displacement_message_callback(msg.input)
        out = AlignmentResponse()
        out.histograms = histogram
        return out

    def _process_abs_alignment(self, msg):
        if msg.map_features[0].shape[0] > 1:
            rospy.logwarn("Bearnav classic can process only one image")
        histogram = np.array(msg.map_features[0].values).reshape(msg.map_features[0].shape)
        self.alignment = (np.argmax(histogram) - np.size(histogram)//2) / (np.size(histogram)//2)
        rospy.loginfo("Current displacement: " + str(self.alignment))
        # self.publish_align()

    def _process_rel_distance(self, msg):
        rospy.logerr("This function is not available for this fusion class")
        raise Exception("Bearnav Classic does not support relative distance")

    def _process_abs_distance(self, msg):
        self.distance = self.abs_dist_est.abs_dist_message_callback(msg)
        # if we want to use this topic for recording we need the header for time sync
        self.header = self.abs_dist_est.header
        # self.publish_dist()

    def _process_prob_distance(self, msg):
        rospy.logerr("This function is not available for this fusion class")
        raise Exception("Bearnav Classic does not support probability of distances")


class VisualOnly(SensorFusion):

    def __init__(self, type_prefix: str,
                 abs_align_est: DisplacementEstimator, prob_dist_est: ProbabilityDistanceEstimator,
                 repr_creator: RepresentationsCreator):
        super().__init__(type_prefix, abs_align_est=abs_align_est, prob_dist_est=prob_dist_est,
                         repr_creator=repr_creator)

    def _process_rel_alignment(self, msg):
        rospy.logwarn("This function is not available for this fusion class")
        raise Exception("Visual only does not support relative alignment")

    def _process_abs_alignment(self, msg: SensorsInput):
        hists = np.array(msg.map_features[0].values).reshape(msg.map_features[0].shape)
        hist = np.max(hists, axis=0)
        half_size = np.size(hist) / 2.0
        self.alignment = float(np.argmax(hist) - (np.size(hist) // 2.0)) / half_size  # normalize -1 to 1
        # self.publish_align()

    def _process_rel_distance(self, msg):
        rospy.logwarn("This function is not available for this fusion class")
        raise Exception("Visual only does not support relative distance")

    def _process_abs_distance(self, msg):
        rospy.logwarn("This function is not available for this fusion class")
        raise Exception("Visual only does not support absolute distance")

    def _process_prob_distance(self, msg):
        # TODO: this method usually publishes with too low frequency to control the spot
        dists = msg.map_distances
        probs = self.prob_dist_est.prob_dist_message_callback(msg)
        # TODO: add some interpolation to more cleanly choose between actions - more fancy :)
        self.distance = max(dists[np.argmax(probs)], 0.05)
        rospy.loginfo("Predicted dist: " + str(self.distance) + " and alignment: " + str(self.alignment))
        # self.publish_dist()


class PF2D(SensorFusion):
    # First dimension - traveled distance
    # Second dimension - alignment

    def __init__(self, type_prefix: str, particles_num: int, odom_error: float, odom_init_std: float,
                 align_error: float, align_init_std: float, particles_frac: int, debug: bool,
                 abs_align_est: DisplacementEstimator, rel_align_est: DisplacementEstimator,
                 rel_dist_est: RelativeDistanceEstimator, repr_creator: RepresentationsCreator):
        super(PF2D, self).__init__(type_prefix, abs_align_est=abs_align_est,
                                   rel_align_est=rel_align_est, rel_dist_est=rel_dist_est,
                                   repr_creator=repr_creator)

        self.one_dim = False
        self.rng = np.random.default_rng()

        self.odom_error = odom_error
        self.align_error = align_error
        self.odom_init_std = odom_init_std
        self.align_init_std = align_init_std

        self.particles_num = particles_num
        self.particles_frac = particles_frac
        self.last_image = None
        self.last_odom = None
        self.particles = None
        self.last_time = None
        self.traveled_dist = 0.0
        self.particle_prob = None
        self.coords = None
        self.last_hists = None
        self.map = 0
        self.last_map_transition_time = rospy.Time.now()

        self._min_align_noise = 0.01
        self._clip_surround = 0.5

        # For debugging
        self.debug = debug
        if debug:
            self.particles_pub = rospy.Publisher("particles", FloatList, queue_size=1)

    def set_distance(self, msg: SetDist) -> SetDistResponse:
        ret = super(PF2D, self).set_distance(msg)
        var = (self.odom_init_std, self.align_init_std, 0)
        dst = self.distance
        self.particles = np.transpose(np.ones((3, self.particles_num)).transpose() * np.array((dst, 0, 0)) +\
                                      self.rng.normal(loc=(0, 0, 0), scale=var, size=(self.particles_num, 3)))
        self.particles = self.particles - np.mean(self.particles, axis=-1, keepdims=True)
        self.particles[2] = np.random.randint(low=0, high=3, size=(self.particles_num,))
        self.last_image = None
        self._get_coords()
        rospy.loginfo("Particles reinitialized at position " + str(self.distance) + "m" +
                      " with alignment " + str(self.alignment))
        return ret

    def _process_rel_alignment(self, msg):
        histogram = self.rel_align_est.displacement_message_callback(msg.input)
        out = AlignmentResponse()
        out.histograms = histogram
        return out

    def _create_trans_matrix(self, array, map_num):
        transitions = np.zeros((map_num, map_num))
        curr_idx = 0
        for i in range(map_num):
            for j in range(i, map_num):
                transitions[i, j] = array[curr_idx]
                transitions[j, i] = array[curr_idx]
                curr_idx += 1
        for i in range(map_num):
            transitions[i] = self._numpy_softmax(transitions[i])
        rospy.logwarn(transitions)
        return transitions

    def _process_abs_alignment(self, msg):
        # rospy.logwarn("PF obtained new input")
        # get everything
        curr_time = float(str(msg.header.stamp.secs).zfill(10)[-4:] + str(msg.header.stamp.nsecs).zfill(9)[:4])
        if self.last_time is None:
            self.last_time = curr_time
            return
        map_diffs = msg.maps[1]
        all_hists = np.array(msg.map_features[0].values).reshape(msg.map_features[0].shape)
        hists = all_hists[:-map_diffs]
        live_hist = np.array(msg.live_features[0].values).reshape(msg.live_features[0].shape)
        curr_img_diff = self._diff_from_hist(live_hist)
        curr_time_diff = curr_time - self.last_time
        trans = np.array(msg.map_transitions)
        dists = np.array(msg.map_distances)
        time_diffs = np.array(msg.time_transitions)
        traveled = self.traveled_dist

        rospy.logwarn(str(hists.shape) + ":" + str(all_hists.shape) + ":" + str(dists))

        # handle multiple maps
        map_vals = list(np.max(all_hists[-map_diffs:], axis=-1))
        map_matrix = self._create_trans_matrix(msg.map_similarity, msg.maps[1])
        rospy.logwarn(map_vals)

        # if len(hists) < 2 or len(trans) != len(hists) - 1 or len(dists) != len(hists) or len(trans) == 0:
        #     rospy.logwarn("Invalid input sizes for particle filter!")
        #     return

        if abs(traveled) < 0.001 and abs(curr_img_diff) < 0.001:
            # this is when odometry is slower than camera
            self.last_time = curr_time
            rospy.logwarn("Not enough movement detected for particle filter update!\n" + "traveled: " + str(traveled) + "," + str(curr_img_diff))
            return

        # motion step --------------------------------------------------------------------------------

        # get map transition for each particle
        mat_dists = np.transpose(np.matrix(dists))
        p_distances = np.matrix(self.particles[0, :])
        # rospy.logwarn(np.argmin(np.abs(mat_dists - p_distances)))
        closest_transition = np.transpose(np.clip(np.argmin(np.abs(mat_dists - p_distances), axis=0), 0, len(dists) - 2))

        traveled_fracs = float(curr_time_diff) / time_diffs
        # rospy.loginfo("traveled fracs:" + str(traveled_fracs))

        trans_cumsum_per_particle = trans[closest_transition]
        frac_per_particle = traveled_fracs[closest_transition]
        # generate new particles
        out = []
        trans_diff = None
        # rolls = np.random.rand(self.particles.shape[1])
        # indices = self._first_nonzero(np.matrix(trans_cumsum_per_particle) >= np.transpose(np.matrix(rolls)), 1)
        trans_diff = np.array(trans_cumsum_per_particle * frac_per_particle)
        align_shift = curr_img_diff + trans_diff

        # distance is not shifted because it is shifted already in odometry step
        particle_shifts = np.concatenate((np.zeros(trans_diff.shape), align_shift), axis=1)
        moved_particles = np.transpose(self.particles[:2]) + particle_shifts +\
                                       self.rng.normal(loc=(0, 0),
                                           scale=(self.odom_error * traveled, 0.025 + self.align_error * np.mean(np.abs(align_shift))),
                                           size=(self.particles.shape[1], 2))
        out.append(moved_particles)
        self.particles[:2] = moved_particles.transpose()

        if (rospy.Time.now() - self.last_map_transition_time).to_sec() > 5.0:
            random_indices = self.rng.choice(np.arange(self.particles_num),
                                             int(self.particles_num // 5))
            random_particles = self.particles[2, random_indices]
            # move particles in map space, but not too often
            self.last_map_transition_time = rospy.Time.now()
            for map_id in range(msg.maps[1]):
                random_particles[random_particles == map_id] = self.rng.choice(np.arange(msg.maps[1]),
                                                                               int(np.sum(random_particles == map_id)),
                                                                               p=map_matrix[map_id])
            self.particles[2, random_indices] = random_particles
        # rospy.logwarn("Motion step finished!")

        # sensor step -------------------------------------------------------------------------------
        # add new particles
        new = []
        tmp = np.zeros((3, int(self.particles_num / 10)))
        tmp[0, :] = self.rng.uniform(low=dists[0], high=dists[1], size=(1, int(self.particles_num / 10)))
        tmp[1, :] = self.rng.uniform(low=-0.5, high=0.5, size=(1, int(self.particles_num / 10)))
        tmp[2, :] = np.random.randint(low=0, high=len(map_vals), size=(1, int(self.particles_num / 10)))
        new.append(tmp.transpose())
        new.append(self.particles.transpose())
        self.particles = np.concatenate(new).transpose()
        if self.particle_prob is not None:
            # set the lowest possible probability to all added particles
            self.particle_prob = np.concatenate([self.particle_prob, np.zeros((self.particles[0].size - self.particles_num), )])

        # interpolate
        # maxs_pre = hists.max(axis=1)
        # rospy.logwarn(str(maxs_pre) + str(dists))
        # rospy.loginfo(hists[:, 250:260])

        self.particles[0] = np.clip(self.particles[0], dists[0] - self._clip_surround, dists[-1] + self._clip_surround)
        self.particles[1] = np.clip(self.particles[1], -1.0, 1.0)
        hist_width = np.shape(hists)[1]
        xs, ys = np.meshgrid(dists, np.linspace(-1.0, 1.0, hist_width))
        positions = np.vstack([xs.ravel(), ys.ravel()])
        idx, idy = np.meshgrid(np.arange(hists.shape[0]), np.arange(hists.shape[1]))
        indices = np.vstack([idx.ravel(), idy.ravel()])
        self.particle_prob = interpolate.griddata(np.transpose(positions),
                                             hists[indices[0], indices[1]],
                                             (self.particles[0], self.particles[1]),
                                             method="nearest")
        rospy.logwarn((np.array(map_vals)))

        # get probabilites of particles
        self.particle_prob = self.particle_prob * (np.array(map_vals))[np.array(self.particles[2], dtype=int)]
        # self.particle_prob = self._numpy_softmax(self.particle_prob)
        # particle_prob -= particle_prob.min()
        # particle_prob /= particle_prob.sum()
        # choose best candidates and reduce the number of particles
        chosen_indices = self.rng.choice(np.shape(self.particles)[1], int(self.particles_num),
                                         p=self.particle_prob/np.sum(self.particle_prob))
        # rospy.logwarn(self.particles[2, chosen_indices])

        self.particle_prob = self.particle_prob[chosen_indices]
        self.particles = self.particles[:, chosen_indices]

        self.last_image = msg.live_features
        self.last_time = curr_time
        self.traveled_dist = 0.0
        self.last_hists = hists
        self._get_coords()
        # self.publish_align()
        # self.publish_dist()

        # rospy.logwarn(np.array((dist_diff, hist_diff)))
        if self.debug:
            particles_out = self.particles.flatten()
            particles_out = np.concatenate([particles_out, self.coords.flatten()])
            self.particles_pub.publish(particles_out)
            # rospy.loginfo("Outputted position: " + str(np.mean(self.particles[0, :])) + " +- " + str(np.std(self.particles[0, :])))
            # rospy.loginfo("Outputted alignment: " + str(np.mean(self.particles[1, :])) + " +- " + str(np.std(self.particles[1, :])) + " with transitions: " + str(np.mean(curr_img_diff))
            #               + " and " + str(np.mean(trans_diff)))

        rospy.logwarn("Finished processing - everything took: " + str((rospy.Time.now() - msg.header.stamp).to_sec()) + " secs")

    def _process_rel_distance(self, msg):
        # only increment the distance
        dist = self.rel_dist_est.rel_dist_message_callback(msg)
        if dist is not None and dist >= 0.005:
            self.particles[0] += dist
            self._get_coords()
            self.traveled_dist += dist

    def _process_abs_distance(self, msg):
        rospy.logwarn("This function is not available for this fusion class")
        raise Exception("PF2D does not support absolute distance")

    def _process_prob_distance(self, msg):
        rospy.logwarn("This function is not available for this fusion class")
        raise Exception("PF2D does not support distance probabilities")

    def _numpy_softmax(self, arr):
        tmp = np.exp(arr) / np.sum(np.exp(arr))
        return tmp

    def _first_nonzero(self, arr, axis, invalid_val=-1):
        mask = arr != 0
        return np.array(np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val))

    def _get_coords(self):
        # coords = np.mean(self.particles, axis=1)
        tmp_particles = self.particles
        if self.particle_prob is not None:
            self.coords = self._get_weighted_mean_pos(tmp_particles)
            if self.one_dim:
                self.coords[1] = (np.argmax(np.max(self.last_hists, axis=0)) - self.last_hists[0].size//2) / self.last_hists[0].size
            maps = []
            for i in range(3):  # TODO: magic constant - number of maps
                maps.append(np.sum(self.particle_prob[self.particles[2] == i]))
            ind = np.argmax(maps)
            rospy.logwarn(maps)
            self.map = ind

        else:
            self.coords = [0.0, 0.0]
        if self.coords[0] < 0.0:
            # the estimated distance cannot really be less than 0.0 - fixing for action repeating
            rospy.logwarn("Mean of particles is less than 0.0 - moving them forwards!")
            self.particles[0, :] -= self.coords[0] - 0.01  # add one centimeter for numeric issues
        stds = np.std(tmp_particles, axis=1)
        self.distance = self.coords[0]
        self.alignment = self.coords[1]
        self.distance_std = stds[0]
        self.alignment_std = stds[1]


    def _diff_from_hist(self, hist):
        half_size = np.size(hist) / 2.0
        curr_img_diff = ((np.argmax(hist) - (np.size(hist) // 2.0)) / half_size)
        return curr_img_diff

    def _get_mean_pos(self):
        return np.mean(self.particles, axis=1)

    def _get_weighted_mean_pos(self, particles):
        weighted_particles = self.particles[:2] * np.tile(self.particle_prob, (2, 1))
        out = np.sum(weighted_particles, axis=1) / np.sum(self.particle_prob)
        return out

    def _get_median_pos(self):
        return np.median(self.particles, axis=1)
