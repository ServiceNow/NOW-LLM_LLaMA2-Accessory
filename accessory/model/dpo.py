from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.distributed as dist

from accessory.model.meta import MetaModel


class DPOModel(MetaModel):
    
    def __init__(self, beta: float, eps: float=0, dpop_lambda: Optional[float]=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.beta = beta
        self.eps = eps
        self.dpop_lambda = dpop_lambda
        
    
    def forward(self, examples: torch.tensor, labels: torch.tensor, masks: torch.tensor, ref_logps: torch.tensor) -> Tuple[Dict[str, Any]]:
        
        # truncate padding to longest in batch
        with torch.no_grad():
            non_zero_ = torch.count_nonzero(labels, dim=0)
            pos = non_zero_.shape[0] - 1
            while pos >= 0:
                if non_zero_[pos] == 0:
                    pos -= 1
                else:
                    break

            if pos == -1:  # nothing to predict in the whole batch
                print(f"[RANK {dist.get_rank()}] nothing to predict in the whole batch!", force=True)
                print(examples.cpu().tolist(), force=True)
                pos = 2
            examples = examples[:, :pos+1]
            labels = labels[:, :pos+1]
            masks = masks[:, :pos+1]

        output = self.llma(examples, image=None)
        if isinstance(output, tuple):
            output, additional_loss = output
        else:
            additional_loss = {}

        policy_logps = self.get_batch_logps(output, labels, masks)

        # we concat chosen and rejected so original bsize is half of the output bsize
        bsize = int(output.shape[0]/2)
        
        dpo_output = self.compute_loss(
            policy_logps[:bsize],
            policy_logps[bsize:],
            ref_logps[:bsize],
            ref_logps[bsize:]
        )

        return dpo_output, additional_loss


    def get_batch_logps(self, logits: torch.FloatTensor, labels: torch.LongTensor, loss_mask: Optional[torch.LongTensor]=None, average_log_prob: bool=False,) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        assert logits.shape[:-1] == labels.shape

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        
        if loss_mask is not None:
            loss_mask = loss_mask[:, 1:]
        else:
            # pad tokens and user input prompt are set to 0 in the labels
            loss_mask = (labels != 0)
        
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

    
    def compute_loss(self, policy_chosen_logps: torch.FloatTensor,
            policy_rejected_logps: torch.FloatTensor,
            reference_chosen_logps: torch.FloatTensor,
            reference_rejected_logps: torch.FloatTensor,
            reference_free: bool = False) -> Dict[str, Any]:
        """Compute the DPO / cDPO loss for a batch of policy and reference model log probabilities.
        
        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
            beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
            eps: The noise parameter for conservative DPO; should be in range (0, 0.5); interpreted as the fraction of preference pairs that are flipped. eps=0 is the original DPO loss in the DPO paper
            reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        if reference_free:
            ref_logratios = 0

        logits = pi_logratios - ref_logratios
        
        # Eq. 3 https://ericmitchell.ai/cdpo.pdf; eps=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
        losses = -F.logsigmoid(self.beta * logits) * (1 - self.eps) - F.logsigmoid(-self.beta * logits) * self.eps
        
        if self.dpop_lambda is not None:
            # loss
            inverse_reward = reference_chosen_logps - policy_chosen_logps
            penalty = self.dpop_lambda * torch.max(torch.zeros_like(inverse_reward), inverse_reward)
            losses -= penalty
            
            # rewards
            chosen_rewards = self.beta * ((policy_chosen_logps - reference_chosen_logps) - penalty)
            rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps)
        else:
            chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps)
            rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps)
        
        outputs = {
            "chosen_rewards": chosen_rewards.detach().mean().cpu(),
            "rejected_rewards": rejected_rewards.detach().mean().cpu(),
            "logratios": pi_logratios.detach().mean().cpu(),
            "logits": logits.detach().mean().cpu(),
            "loss": losses.mean()
        }
        outputs["margins"] = outputs["chosen_rewards"] - outputs["rejected_rewards"]
    
        return outputs