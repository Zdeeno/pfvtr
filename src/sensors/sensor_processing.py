import numpy as np
from base_classes import DisplacementEstimator, RelativeDistanceEstimator, AbsoluteDistanceEstimator, \
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
        # if msg.map_features[0].shape[0] > 1:
        #     rospy.logwarn("Bearnav classic can process only one image")
        histogram = np.array(msg.live_features[0].values).reshape(msg.live_features[0].shape)
        self.alignment = (np.argmax(histogram) - np.size(histogram) // 2) / (np.size(histogram) // 2)
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
        hists = np.array(msg.live_features[0].values).reshape(msg.live_features[0].shape)
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
    # Third dimension - maps

    def __init__(self, type_prefix: str, particles_num: int, odom_error: float, odom_init_std: float,
                 align_beta: float, align_init_std: float, particles_frac: int, choice_beta: float,
                 add_random: float, debug: bool,
                 abs_align_est: DisplacementEstimator, rel_align_est: DisplacementEstimator,
                 rel_dist_est: RelativeDistanceEstimator, repr_creator: RepresentationsCreator):
        super(PF2D, self).__init__(type_prefix, abs_align_est=abs_align_est,
                                   rel_align_est=rel_align_est, rel_dist_est=rel_dist_est,
                                   repr_creator=repr_creator)

        self.zero_dim = False
        self.one_dim = False
        self.use_map_trans = False
        self.rng = np.random.default_rng()

        self.odom_error = odom_error
        self.odom_init_std = odom_init_std
        self.align_init_std = align_init_std

        self.particles_num = particles_num
        self.particles_frac = particles_frac
        self.add_rand = add_random
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
        self.map_num = 1

        self._min_align_noise = 0.01
        self._map_trans_time = 5.0

        self.BETA_align = align_beta
        self.BETA_choice = choice_beta

        # For debugging
        self.debug = debug
        if debug:
            self.particles_pub = rospy.Publisher("particles", FloatList, queue_size=1)

    def set_distance(self, msg: SetDist) -> SetDistResponse:
        ret = super(PF2D, self).set_distance(msg)
        var = (self.odom_init_std, self.align_init_std, 0)
        dst = self.distance
        self.particles = np.transpose(np.ones((3, self.particles_num)).transpose() * np.array((dst, 0, 0)) + \
                                      self.rng.normal(loc=(0, 0, 0), scale=var, size=(self.particles_num, 3)))
        self.particles = self.particles - np.mean(self.particles, axis=-1, keepdims=True)
        self.map_num = msg.map_num
        self.particles[2] = np.random.randint(low=0, high=self.map_num, size=(self.particles_num,))
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
            transitions[i] = self._numpy_softmax(transitions[i], self.BETA_choice)
        rospy.logwarn(transitions)
        return transitions

    def _get_time_diff(self, timestamps: list):
        out = []
        for i in range(len(timestamps) - 1):
            out.append((timestamps[i + 1] - timestamps[i]).to_sec())
        return out

    def _process_abs_alignment(self, msg):
        # Parse all data from the incoming message
        curr_time = msg.header.stamp
        self.header = msg.header
        if self.last_time is None:
            self.last_time = curr_time
            return
        hists = np.array(msg.map_features[0].values).reshape(msg.map_features[0].shape)
        self.last_hists = hists
        map_trans = np.array(msg.map_transitions[0].values).reshape(msg.map_transitions[0].shape)
        live_hist = np.array(msg.live_features[0].values).reshape(msg.live_features[0].shape)
        hist_width = hists.shape[-1]
        shifts = np.round(np.array(msg.map_offset) * (hist_width // 2)).astype(int)
        hists = np.roll(hists, shifts, -1)  # not sure if last dim should be rolled like this
        curr_img_diff = self._sample_hist([live_hist])
        curr_time_diff = (curr_time - self.last_time).to_sec()
        dists = np.array(msg.map_distances)
        timestamps = msg.map_timestamps
        traveled = self.traveled_dist

        # Divide incoming data according the map affiliation
        len_per_map = np.size(dists) // self.map_num
        trans_per_map = len_per_map - 1
        if len(dists) % msg.map_num > 0:
            # TODO: this assumes that there is same number of features comming from all the maps (this does not have to hold when 2*map_len < lookaround)
            # however the mapmaker was updated so that the new maps should have always the same number of images, unless some major error occurs
            rospy.logwarn("!!!!!!!!!!!!!!!!!! One map has more images than other !!!!!!!!!!!!!!!!")
            return
        map_trans = [map_trans[trans_per_map * map_idx:trans_per_map * (map_idx + 1)] for map_idx in
                     range(self.map_num)]
        hists = [hists[len_per_map * map_idx:len_per_map * (map_idx + 1)] for map_idx in range(self.map_num)]
        dists = np.array([dists[len_per_map * map_idx:len_per_map * (map_idx + 1)] for map_idx in range(self.map_num)])
        timestamps = [timestamps[len_per_map * map_idx:len_per_map * (map_idx + 1)] for map_idx in range(self.map_num)]
        time_diffs = [self._get_time_diff(timestamps[map_idx]) for map_idx in range(self.map_num)]
        if self.map_num > 1:
            # transition matrix for between maps
            map_matrix = msg.map_transitions

        # if len(hists) < 2 or len(trans) != len(hists) - 1 or len(dists) != len(hists) or len(trans) == 0:
        #     rospy.logwarn("Invalid input sizes for particle filter!")
        #     return

        if abs(traveled) < 0.001:
            # this is when odometry is slower than the estimator
            self.last_time = curr_time
            # rospy.logwarn(
            #     "Not enough movement detected for particle filter update!\n" + "traveled: " + str(traveled) + "," + str(
            #         np.var(curr_img_diff)))
            return

        if self.particles.shape[-1] > self.particles_num:
            rospy.logwarn("errorneous state - too much particles")
            self.particles = self.particles[:, :self.particles_num]

        # motion step --------------------------------------------------------------------------------

        for map_idx in range(self.map_num):
            # get map transition for each particle
            map_particle_mask = self.particles[2, :] == map_idx
            # centers of distances
            mat_dists = np.transpose(np.matrix((dists[map_idx, 1:] + dists[map_idx, :-1]) / 2.0))
            p_distances = np.matrix(self.particles[0, map_particle_mask])
            particles_in_map = np.sum(map_particle_mask)
            # rospy.logwarn(np.argmin(np.abs(mat_dists - p_distances)))
            closest_transition = np.transpose(np.argmin(np.abs(mat_dists - p_distances), axis=0))
            traveled_fracs = float(curr_time_diff) / np.array(time_diffs[map_idx])
            # rospy.loginfo("traveled fracs:" + str(traveled_fracs))
            # Monte carlo sampling of transitions
            trans = -self._sample_hist(map_trans[map_idx])
            trans_per_particle = trans[closest_transition.squeeze(), np.arange(particles_in_map)].transpose()
            frac_per_particle = traveled_fracs[closest_transition]
            # generate new particles
            out = []
            # rolls = np.random.rand(self.particles.shape[1])
            # indices = self._first_nonzero(np.matrix(trans_cumsum_per_particle) >= np.transpose(np.matrix(rolls)), 1)
            trans_diff = np.array(trans_per_particle * frac_per_particle)
            align_shift = curr_img_diff.transpose()[map_particle_mask] + trans_diff

            # debugging
            # rospy.logwarn(closest_transition)
            # rospy.logwarn("map_tans" + str(np.argmax(map_trans[0], axis=-1)))
            # rospy.logwarn("live: " + str(np.mean(curr_img_diff)))
            # rospy.logwarn("map: " + str(np.mean(trans_diff)))
            # rospy.logwarn("curr_time:" + str(curr_time_diff))
            # rospy.logwarn("map_time" + str(time_diffs[map_idx]))

            # distance is not shifted because it is shifted already in odometry step
            particle_shifts = np.concatenate((np.zeros(trans_diff.shape), align_shift), axis=1)

            moved_particles = np.transpose(self.particles[:2, map_particle_mask]) + particle_shifts + \
                              self.rng.normal(loc=(0, 0),
                                              scale=(self.odom_error * traveled, 0),
                                              size=(particles_in_map, 2))
            out.append(moved_particles)
            self.particles[:2, map_particle_mask] = moved_particles.transpose()

            # TODO: this has to be updated for joint map state
            if self.use_map_trans and (rospy.Time.now() - self.last_map_transition_time).to_sec() > self._map_trans_time \
                    and hists is not None:
                random_indices = self.rng.choice(np.arange(self.particles_num),
                                                 int(self.particles_num // 5))
                random_particles = self.particles[2, random_indices]
                # move particles in map space, but not too often - time limit
                self.last_map_transition_time = rospy.Time.now()
                for map_id in range(msg.maps[1]):
                    random_particles[random_particles == map_id] = self.rng.choice(np.arange(msg.maps[1]),
                                                                                   int(np.sum(
                                                                                       random_particles == map_id)),
                                                                                   p=map_matrix[map_id])
                self.particles[2, random_indices] = random_particles
            # rospy.logwarn("Motion step finished!")

        # add randomly spawned particles ------------------------------------------------------------

        if self.add_rand > 0:
            new = []
            tmp = np.zeros((3, int(self.particles_num * self.add_rand)))
            tmp[0, :] = self.rng.uniform(low=np.mean(dists[:, 0]), high=np.mean(dists[:, -1]),
                                         size=(1, int(self.particles_num * self.add_rand)))
            tmp[1, :] = self.rng.uniform(low=-0.5, high=0.5, size=(1, int(self.particles_num * self.add_rand)))
            tmp[2, :] = np.random.randint(low=0, high=self.map_num, size=(1, int(self.particles_num * self.add_rand)))
            new.append(tmp.transpose())
            new.append(self.particles.transpose())
            self.particles = np.concatenate(new).transpose()
            if self.particle_prob is not None:
                # set the lowest possible probability to all added particles
                self.particle_prob = np.concatenate(
                    [self.particle_prob, np.zeros((self.particles[0].size - self.particles_num), )])

        # sensor step -------------------------------------------------------------------------------
        particle_prob = np.zeros(self.particles.shape[-1])
        self.particles[1] = np.clip(self.particles[1], -1.0, 1.0)  # more than 0% overlap is nonsense
        for map_idx in range(self.map_num):
            map_particle_mask = self.particles[2] == map_idx
            map_masked_particles = self.particles[:, map_particle_mask]

            interp_f = interpolate.RectBivariateSpline(dists[map_idx], np.linspace(-1.0, 1.0, hist_width),
                                                       hists[map_idx], kx=1)
            particle_prob[map_particle_mask] = interp_f(map_masked_particles[0],
                                                        map_masked_particles[1], grid=False)
        self.particle_prob = particle_prob
        self.particle_prob[self.particle_prob < 0] = 0.0  # lower than 0.0 probability - should not happen though

        # perform some normalization and resample the particles via roulette wheel
        # particle_prob -= particle_prob.min()
        # particle_prob /= particle_prob.sum()
        softmaxed_probs = self._numpy_softmax(self.particle_prob, self.BETA_choice)
        chosen_indices = self.rng.choice(np.shape(self.particles)[1], int(self.particles_num),
                                         p=softmaxed_probs)
        # rospy.logwarn(self.particles[2, chosen_indices])

        self.particle_prob = self.particle_prob[chosen_indices]
        self.particles = self.particles[:, chosen_indices]

        # publish filtering output ------------------------------------------------------------------
        self.last_image = msg.live_features
        self.last_time = curr_time
        self.traveled_dist = 0.0
        self._get_coords()  # this updates the values which are published continuously

        # rospy.logwarn(np.array((dist_diff, hist_diff)))
        # visualization & debugging -----------------------------------------------------------------
        if self.debug:
            # for
            particles_out = self.particles.flatten()
            particles_out = np.concatenate([particles_out, self.coords.flatten()])
            self.particles_pub.publish(particles_out)
            # rospy.loginfo("Outputted position: " + str(np.mean(self.particles[0, :])) + " +- " + str(np.std(self.particles[0, :])))
            # rospy.loginfo("Outputted alignment: " + str(np.mean(self.particles[1, :])) + " +- " + str(np.std(self.particles[1, :])) + " with transitions: " + str(np.mean(curr_img_diff))
            #               + " and " + str(np.mean(trans_diff)))

        # rospy.logwarn(
        #     "Finished processing - everything took: " + str((rospy.Time.now() - msg.header.stamp).to_sec()) + " secs")

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

    def _numpy_softmax(self, arr, beta):
        tmp = np.exp(arr * beta) / np.sum(np.exp(arr * beta))
        return tmp

    def _first_nonzero(self, arr, axis, invalid_val=-1):
        mask = arr != 0
        return np.array(np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val))

    def _get_coords(self):
        # TODO: implement better estimate - histogram voting!
        # coords = np.mean(self.particles, axis=1)
        tmp_particles = self.particles
        if self.particle_prob is not None:
            self.coords = self._get_weighted_mean_pos(np.copy(self.particles), np.copy(self.particle_prob))
            # self.coords = self._histogram_voting()
            # self.coords = self.particles[:2, np.argmax(self.particle_prob)]
            if self.one_dim:
                # for testing not using 2nd dim
                self.coords[1] = (np.argmax(np.max(self.last_hists, axis=0)) - self.last_hists[0].size // 2) / \
                                 self.last_hists[0].size
            maps = []
            for i in range(self.map_num):
                maps.append(np.sum(self.particle_prob[self.particles[2] == i]))
            ind = np.argmax(maps)
            # rospy.logwarn(maps)
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

    def _sample_hist(self, hists: list):
        """
        sample from histogram per each particle ...
        """
        hist_size = hists[0].shape[-1]
        return np.array([self.rng.choice(np.linspace(start=-1, stop=1, num=hist_size), int(self.particles_num),
                                         p=self._numpy_softmax(hists[idx], self.BETA_align))
                         for idx, curr_trans_sample in enumerate(hists)])

    def _sample_maxs(self, hists: list):
        return np.array([np.ones(self.particles_num) * self._diff_from_hist(curr_trans_sample)
                         for idx, curr_trans_sample in enumerate(hists)])

    def _get_mean_pos(self):
        return np.mean(self.particles, axis=1)

    def _get_weighted_mean_pos(self, particles, particle_prob):
        # TODO: this method can yield an error when class variables are changed in process - make copies
        align_span = 0.5    # crop of particles to estimate alignment
        predictive = 0.0   # make alignment slightly predictive
        dist = np.sum(particles[0] * particle_prob) / np.sum(particle_prob)
        mask = (particles[0] < (dist + align_span + predictive)) \
               * (particles[0] > (dist - align_span + predictive))
        p_num = np.sum(mask)
        if p_num < 50:
            rospy.logwarn("Only " + str(p_num) + " particles used for alignment estimate - could be very noisy")
        align = np.sum(particles[1, mask] * particle_prob[mask]) / np.sum(particle_prob[mask])
        return np.array((dist, align))
        # weighted_particles = particles[:2] * np.tile(particle_prob, (2, 1))
        # out = np.sum(weighted_particles, axis=1) / np.sum(particle_prob)
        # return out

    def _histogram_voting(self):
        hist, x, y = np.histogram2d(self.particles[0], self.particles[1], bins=20, weights=self.particle_prob)
        indices = np.unravel_index(np.argmax(hist, axis=None), hist.shape)
        return np.array([x[indices[0]], y[indices[1]]])

    def _get_median_pos(self):
        return np.median(self.particles, axis=1)
