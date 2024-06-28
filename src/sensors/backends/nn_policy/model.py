import torch as t
from tensordict.nn.distributions import NormalParamExtractor
from tensordict.nn import TensorDictModule
import tensordict
from torchrl.data import UnboundedContinuousTensorSpec, BoundedTensorSpec, CompositeSpec, BinaryDiscreteTensorSpec
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
import os
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from tensordict import TensorDict


class PPOActorSimple(t.nn.Module):

    def __init__(self, lookaround: int, dist_window=8, hidden_size=512):
        super().__init__()
        self.lookaround = lookaround
        self.map_obs_size = lookaround * 2 + 1
        map_trans_size = lookaround * 2
        total_size = self.map_obs_size + map_trans_size + 1  # + 1 for last camera img vs current
        self.hist_size = 64
        input_size = total_size * self.hist_size
        self.dist_hist_size = dist_window * 10 + 1
        input_size += self.dist_hist_size * 2

        # histograms from visual data (1, 2, 5)
        self.ff = t.nn.Sequential(t.nn.Linear(self.hist_size * 9 + self.dist_hist_size, hidden_size),
                                  t.nn.Tanh(),
                                  t.nn.Linear(hidden_size, hidden_size),
                                  t.nn.Tanh(),
                                  t.nn.Linear(hidden_size, hidden_size),
                                  t.nn.Tanh(),
                                  t.nn.Linear(hidden_size, 4))

        self.norm = NormalParamExtractor()


    def pass_network(self, x):
        out = self.ff(x)
        return out

    def forward(self, x):
        normed_out = self.norm(self.pass_network(x))
        print(normed_out)
        return normed_out


class PolicyNet:

    def __init__(self):
        self.device = t.device('cuda' if t.cuda.is_available() else 'cpu')
        actor_net = PPOActorSimple(2, hidden_size=1024).float().to(self.device)
        HOME = os.path.expanduser('~')
        SAVE_DIR = HOME + "/.ros/models/"
        actor_net.load_state_dict(t.load(SAVE_DIR + "actor_net.pt"))
        policy_module = TensorDictModule(
            actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
        )

        action_spec = CompositeSpec(
            {"action": BoundedTensorSpec([[-0.25, -0.5]], [[0.25, 0.5]], t.Size([1, 2]), self.device)},
            shape=t.Size([1]))

        self.policy_module = ProbabilisticActor(
            module=policy_module,
            spec=action_spec,
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            distribution_kwargs={
                "min": t.tensor([-0.25, -0.5]),
                "max": t.tensor([0.25, 0.5]),
                # "event_dims": 2,
                # "tanh_loc": True
            },
            return_log_prob=True,
            default_interaction_type=tensordict.nn.InteractionType.MEAN,
            # we'll need the log-prob for the numerator of the importance weights
        )

    def get_action(self, obs):
        net_in = TensorDict({"observation": obs.unsqueeze(0),
                             "reward": t.tensor([0.0], device=self.device).unsqueeze(0).float(),
                             "done": t.tensor([self.finished], device=self.device).unsqueeze(0)},
                            batch_size=[1])
        with set_exploration_type(ExplorationType.MEAN), t.no_grad():
            action = self.policy_module.forward(net_in)
        return action