import lightning.pytorch as pl
import torch
import torch.nn.functional as F

import json
from typing import Union

from physo import utils
from physo.physym.batch import Batch
from physo.learn import rnn
from physo.learn import learn


class PhySO(pl.LightningModule):

    def __init__(self,
                 X: torch.Tensor,
                 y: torch.Tensor,
                 run_config: Union[str, dict],
                 verbose: bool = False,
                 candidate_wrapper: bool = None):
        super().__init__()

        # main params
        self.config = utils.load_config(run_config)
        self.X = X
        self.y = y
        self.verbose = verbose
        self.candidate_wrapper = candidate_wrapper

        # keep best scores
        self.overall_max_R_history = []
        self.hall_of_fame = []

        # initialize batch loader
        self.batch = self.batch_resetter()

        # frequently used hyperparameters
        self.batch_size = self.batch.batch_size
        self.max_time_step = self.batch.max_time_step
        self.ntrain = self.config["learning_config"]["risk_factor"] * self.batch_size
        self.gamma_decay = self.config["learning_config"]["gamma_decay"]
        self.entropy_weight = self.config["learning_config"]["gamma_decay"]
        self.lr = self.config["learning_config"]["learning_rate"]

        # initialize rnn cell
        self.model = rnn.Cell(input_size=self.batch.obs_size,
                              output_size=self.batch.n_choices,
                              **self.config["cell_config"])

    def batch_resetter(self):
        return Batch(self.config, self.X, self.y, self.candidate_wrapper)

    def time_step(self, states):
        # get observations
        # (batch_size, obs_size)
        observations = torch.tensor(self.batch.get_obs().astype(torch.float32), requires_grad=False)

        # run rnn model to get distributions and new state
        # (batch_size, ouput_size)
        output, states = self.model(input_tensor=observations, states=states)

        # get prior array and set zeros to lowest possible non-zero value
        # (batch_size, output_size)
        prior_array = self.batch.prior().astype(torch.float32)
        prior_array[prior_array == 0] = torch.finfo(torch.float32).eps
        prior = torch.tensor(prior_array, requires_grad=False)
        logprior = torch.log(prior)

        # sample
        # (batch_size, )
        logit = output + logprior
        action = torch.multinomial(torch.exp(logit), num_samples=1)[:, 0]
        self.batch.programs.append(action.detach().cpu().numpy())

        return logit, action, states

    def loss_func(self, logits, actions):
        # get rewards
        rewards = torch.tensor(self.batch.get_rewards())

        # get top rankings
        ranking = rewards.argsort()
        keep = ranking[-self.ntrain:]

        # get elite candidates
        actions_train = actions[:, keep]
        logits_train = logits[:, keep]
        rewards_train = torch.tensor(rewards[keep], requires_grad=False)

        # compute helper arrays
        rewards_min = rewards_train.min()
        lengths = self.batch.programs.n_lengths[keep]

        # length mask
        mask_length = torch.tile(torch.arange(self.max_time_step), (self.ntrain, 1))
        mask_length = mask_length < torch.tile(lengths, (self.max_time_step, 1)).T
        mask_length = mask_length.float()

        # entropy mask
        entropy_gamma_decay = torch.pow(self.gamma_decay, torch.arange(self.max_time_step))
        entropy_gamma_decay = torch.tile(entropy_gamma_decay, (self.ntrain, 1)).T

        # gradient policy
        probs = F.softmax(logits_train, dim=2)
        log_probs = F.log_softmax(logits_train, dim=2)

        # sum over action dim (nansum == safe_cross_entropy)
        neglogp_per_step = -torch.nansum(probs * log_probs, dim=2)
        neglogp = torch.sum(neglogp_per_step * mask_length, dim=0)
        loss_gp = torch.mean((rewards_train - rewards_min) * neglogp)

        # entropy loss
        entropy_per_step = -torch.nansum(probs * log_probs, dim=2)
        entropy = torch.sum(entropy_per_step * entropy_gamma_decay, dim=0)
        loss_entropy = -self.entropy_weight * torch.mean(entropy)

        return loss_gp + loss_entropy

    def train_step(self, batch, batch_idx):
        # get size and max length
        self.batch = self.batch_resetter()

        # initialize arrays
        logits, actions = [], []
        states = self.model.get_zeros_initial_state(self.batch_size)

        # run recurrent neural network
        for i in range(self.max_time_step):
            logit, action, states = self.time_step(states)
            logits.append(logit)
            actions.append(action)

        # form program arrays and get rewards
        logits = torch.stack(logits, dim=0)
        actions = torch.stack(actions, dim=0)

        # compute loss
        loss = self.loss_func(actions, logits)

        return loss

    def predict_step(self, batch, batch_idx):
        return

    def validation_step(self, batch, batch_idx):
        return

    def configure_optimizers(self):
        optimizer_class = getattr(torch.optim, self.config["learning_config"]["get_optimizer"])
        optimizer = optimizer_class(self.model.parameters(),
                                    lr=self.lr,
                                    **self.config["learning_config"]["optimizer_args"])
        return optimizer
