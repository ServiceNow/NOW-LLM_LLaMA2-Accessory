from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.distributed as dist

from accessory.model.meta import MetaModel
import accessory.util.misc as misc


class KTOModel(MetaModel):
    
    def __init__(self, beta: float, desirable_weight: float, undesirable_weight: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.beta = beta
        self.desirable_weight = desirable_weight
        self.undesirable_weight = undesirable_weight
        
    
    def forward(self, examples: torch.tensor, labels: torch.tensor, masks: torch.tensor, kl_examples: torch.tensor, kl_labels: torch.tensor, kl_masks: torch.tensor, ref_logps: torch.tensor, ref_kl_logps: torch.tensor, tags: List[str]) -> Tuple[Dict[str, Any]]:
        
        # truncate padding to longest in batch
        with torch.no_grad():
            non_zero_ = torch.count_nonzero(labels, dim=0)
            non_zero_kl = torch.count_nonzero(kl_examples, dim=0)
            pos = non_zero_.shape[0] - 1
            kl_pos = non_zero_kl.shape[0] - 1
            while pos >= 0:
                if non_zero_[pos] == 0:
                    pos -= 1
                else:
                    break
            
            while kl_pos >= 0:
                if non_zero_kl[kl_pos] == 0:
                    kl_pos -= 1
                else:
                    break

            if pos == -1:  # nothing to predict in the whole batch
                print(f"[RANK {dist.get_rank()}] nothing to predict in the whole batch!", force=True)
                print(examples.cpu().tolist(), force=True)
                pos = 2
                kl_pos = 2
                
            examples = examples[:, :pos+1]
            labels = labels[:, :pos+1]
            masks = masks[:, :pos+1]

            kl_examples = kl_examples[:, :kl_pos+1]
            kl_labels = kl_labels[:, :kl_pos+1]
            kl_masks = kl_masks[:, :kl_pos+1]
            

        output = self.llma(examples, image=None)
        kl_output = self.llma(kl_examples, image=None)
        
        # Can we possibly use the kl examples router loss somehow in the overall loss comp.?
        if isinstance(output, tuple):
            output, additional_loss = output
            kl_output, _ = kl_output
        else:
            additional_loss = {}

        policy_logps = self.get_batch_logps(output, labels, masks)
        policy_kl_logps = self.get_batch_logps(kl_output, kl_labels, kl_masks)
        
        chosen_idx, rejected_idx = [], []
        for i in range(policy_logps.shape[0]):
            if tags["tag"][i] == "chosen":
                chosen_idx.append(i)
            elif tags["tag"][i] == "rejected":
                rejected_idx.append(i)
        
        dpo_output = self.compute_loss(
            policy_logps[chosen_idx, ...],
            policy_logps[rejected_idx, ...],
            policy_kl_logps,
            ref_logps[chosen_idx, ...],
            ref_logps[rejected_idx, ...],
            ref_kl_logps)

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
            policy_kl_logps: torch.FloatTensor,
            reference_chosen_logps: torch.FloatTensor,
            reference_rejected_logps: torch.FloatTensor,
            reference_kl_logps: torch.FloatTensor) -> Dict[str, Any]:
        """Compute the Kahneman-Tversky loss for a batch of policy and reference model log probabilities.

        If generation y ~ p_desirable, we have the 'desirable' loss:
            L(x, y) := 1 - sigmoid(beta * ([log p_policy(y|x) - log p_reference(y|x)] - KL(p_policy || p_reference)))
        If generation y ~ p_undesirable, we have the 'undesirable' loss:
            L(x, y) := 1 - sigmoid(beta * (KL(p_policy || p_reference) - [log p_policy(y|x) - log p_reference(y|x)]))

        The desirable losses are weighed by config.loss.desirable_weight.
        The undesirable losses are weighed by config.loss.undesirable_weight.
        This should be used to address imbalances in the ratio of desirable:undesirable examples respectively.

        The KL term is estimated by matching x with unrelated outputs y', then calculating the average log ratio
        log p_policy(y'|x) - log p_reference(y'|x). Doing so avoids the requirement that there be equal numbers of 
        desirable and undesirable examples in the microbatch.
        
        Source: https://github.com/ContextualAI/HALOs/blob/main/trainers.py
        """
        kl = (policy_kl_logps - reference_kl_logps).mean().detach()
        
        # sum up the KL estimates across all devices (gradient will also be scaled by world size)
        dist.nn.all_reduce(kl, op=dist.ReduceOp.SUM)
        
        # take average (will also scale gradients appropriately)
        kl = (kl / misc.get_world_size()).clamp(min=0)
        
        if policy_chosen_logps.shape[0] != 0:
            chosen_logratios = (policy_chosen_logps - reference_chosen_logps)
            chosen_losses = 1 - F.sigmoid(self.beta * (chosen_logratios - kl))
            chosen_rewards = self.beta * chosen_logratios.detach()
        else:
            # important to cast to policy_dtype; otherwise error will occur during all_gather
            chosen_losses = torch.Tensor([]).to(policy_chosen_logps.dtype).to(policy_chosen_logps.device)
            chosen_rewards = torch.Tensor([]).to(policy_chosen_logps.dtype).to(policy_chosen_logps.device)
        
        if policy_rejected_logps.shape[0] != 0:
            rejected_logratios = (policy_rejected_logps - reference_rejected_logps)
            rejected_losses = 1 - F.sigmoid(self.beta * (kl - rejected_logratios))
            rejected_rewards = self.beta * rejected_logratios.detach()
        else:
            # important to cast to policy_dtype; otherwise error will occur during all_gather
            rejected_losses = torch.Tensor([]).to(policy_rejected_logps.dtype).to(policy_rejected_logps.device)
            rejected_rewards = torch.Tensor([]).to(policy_rejected_logps.dtype).to(policy_rejected_logps.device)

        losses = torch.cat((self.desirable_weight * chosen_losses, self.undesirable_weight * rejected_losses), 0)
        
        outputs = {
            "chosen_rewards": chosen_rewards.detach().mean().cpu(),
            "rejected_rewards": rejected_rewards.detach().mean().cpu(),
            # "logratios": pi_logratios.detach().mean().cpu(),
            # "logits": logits.detach().mean().cpu(),
            "loss": losses.mean()
        }
        outputs["margins"] = outputs["chosen_rewards"] - outputs["rejected_rewards"]
    
        return outputs
    
