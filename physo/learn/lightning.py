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
                 candidate_wrapper: bool = None):
        super().__init__()

        # main params
        self.config = utils.load_config(run_config)
        self.X = X
        self.y = y
        self.candidate_wrapper = candidate_wrapper

        # keep best scores
        self.overall_max_R_history = []
        self.hall_of_fame = []

        # initialize batch loader
        self.batch = self.batch_resetter()

        # frequently used hyperparameters
        self.batch_size = self.batch.batch_size
        self.max_time_step = self.batch.max_time_step
        self.ntrain = int(self.config["learning_config"]["risk_factor"] * self.batch_size)
        self.gamma_decay = self.config["learning_config"]["gamma_decay"]
        self.entropy_weight = self.config["learning_config"]["gamma_decay"]
        self.lr = self.config["learning_config"]["learning_rate"]
        self.n_choices = int(self.batch.n_choices)

        # initialize rnn cell
        self.model = rnn.Cell(input_size=self.batch.obs_size,
                              output_size=self.batch.n_choices,
                              **self.config["cell_config"])
        
    def get_trainer(self):
        trainer = pl.Trainer(
            max_epochs = self.config["learning_config"]["n_epochs"],
        )
        return trainer

    def batch_resetter(self):
        return Batch(self.config, self.X, self.y, self.candidate_wrapper)
    
    def on_train_epoch_start(self):
        #print("on_train_epoch_start called")
        self.batch_resetter()

    def train_dataloader(self):
        return iter(range(1))
    
    def forward(self, states):
        # get observations
        # (batch_size, obs_size)
        observations = torch.tensor(self.batch.get_obs(), requires_grad=False)
        observations = observations.to(torch.float32).to(self.device)

        # run rnn model to get distributions and new state
        # (batch_size, ouput_size)
        output, states = self.model(input_tensor=observations, states=states)

        # get prior array and set zeros to lowest possible non-zero value
        # (batch_size, output_size)
        prior = torch.tensor(self.batch.prior(), requires_grad=False)
        prior = prior.to(torch.float32).to(self.device)
        prior[prior == 0] = torch.finfo(torch.float32).eps
        logprior = torch.log(prior)

        # sample
        # (batch_size, )
        logit = output + logprior
        action = torch.multinomial(torch.exp(logit), num_samples=1)[:, 0]

        # pytorch on mps has bug that only writes multinomial generated tensors to first row?!
        action = action.detach().cpu().numpy()
        self.batch.programs.append(action)
        action = torch.tensor(action)

        return logit, action, states

    def loss_func(self, logits, actions):
        # get rewards
        rewards = torch.tensor(self.batch.get_rewards()).to(torch.float32).to(self.device)

        # get top rankings
        ranking = rewards.argsort()
        keep = ranking[-self.ntrain:]

        # get elite candidates
        actions_train = actions[:, keep]
        logits_train = logits[:, keep]
        rewards_train = torch.tensor(rewards[keep], requires_grad=False)

        # compute helper arrays
        rewards_min = rewards_train.min()
        lengths = torch.tensor(self.batch.programs.n_lengths).to(self.device)[keep]

        # length mask
        mask_length = torch.tile(torch.arange(self.max_time_step), (self.ntrain, 1)).to(self.device)
        mask_length = mask_length < torch.tile(lengths, (self.max_time_step, 1)).T
        mask_length = mask_length.T.to(torch.float32)

        # entropy mask
        entropy_gamma_decay = torch.pow(self.gamma_decay, torch.arange(self.max_time_step))
        entropy_gamma_decay = torch.tile(entropy_gamma_decay, (self.ntrain, 1)).T.to(self.device)

        # gradient policy
        probs = F.softmax(logits_train, dim=2)
        log_probs = F.log_softmax(logits_train, dim=2)

        # sum over action dim (nansum == safe_cross_entropy)
        ideal_probs_train = torch.eye(self.n_choices).to(self.device)[actions_train]
        neglogp_per_step = -torch.nansum(ideal_probs_train * log_probs, dim=2)
        neglogp = torch.sum(neglogp_per_step * mask_length, dim=0)
        loss_gp = torch.mean((rewards_train - rewards_min) * neglogp)

        # entropy loss
        entropy_per_step = -torch.nansum(probs * log_probs, dim=2)
        entropy = torch.sum(entropy_per_step * entropy_gamma_decay, dim=0)
        loss_entropy = -self.entropy_weight * torch.mean(entropy)

        return loss_gp + loss_entropy

    def compute(self):
        # initialize arrays
        states = self.model.get_zeros_initial_state(self.batch_size).to(self.device)
        logits = torch.empty((self.max_time_step, self.batch_size, self.n_choices)).to(self.device)
        actions = torch.empty((self.max_time_step, self.batch_size), dtype=torch.int32).to(self.device)

        # run recurrent neural network
        for i in range(self.max_time_step):
            logits[i], actions[i], states = self.forward(states)

        # compute loss
        loss = self.loss_func(logits, actions)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.compute()
        self.log("val_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        return

    def configure_optimizers(self):
        optimizer_class = getattr(torch.optim, self.config["learning_config"]["optimizer"])
        optimizer = optimizer_class(self.model.parameters(),
                                    lr=self.lr,
                                    **self.config["learning_config"]["optimizer_args"])
        return optimizer
