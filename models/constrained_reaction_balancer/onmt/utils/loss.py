"""
This file handles the details of the loss function during training.

This includes: LossComputeBase and the standard NMTLossCompute, and
               sharded loss compute stuff.
"""
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F

import onmt
import onmt.inputters as inputters
from onmt.modules.sparse_losses import SparsemaxLoss

from onmt.utils.masking import create_atom_vocab_maps
from onmt.utils.logging import logger


def build_loss_compute(model, tgt_vocab, opt, train=True):
    """
    This returns user-defined LossCompute object, which is used to
    compute loss in train/validate process. You can implement your
    own *LossCompute class, by subclassing LossComputeBase.
    """
    device = torch.device("cuda" if onmt.utils.misc.use_gpu(opt) else "cpu")

    if opt.copy_attn:
        compute = onmt.modules.CopyGeneratorLossCompute(
            model.generator, tgt_vocab, opt.copy_attn_force,
            opt.copy_loss_by_seqlength)
    elif opt.atom_balance_loss and train:
        compute = AtomBalanceLossCompute(
            model.generator, tgt_vocab, device, opt.atom_balance_loss_weight)
    elif opt.atom_balance_loss_valid and not train:
        compute = AtomBalanceLossCompute(
            model.generator, tgt_vocab, device, opt.atom_balance_loss_weight)
    else:
        compute = NMTLossCompute(
            model.generator, tgt_vocab,
            label_smoothing=opt.label_smoothing if train else 0.0)
    compute.to(device)

    return compute


class LossComputeBase(nn.Module):
    """
    Class for managing efficient loss computation. Handles
    sharding next step predictions and accumulating mutiple
    loss computations


    Users can implement their own loss computation strategy by making
    subclass of this one.  Users need to implement the _compute_loss()
    and make_shard_state() methods.

    Args:
        generator (:obj:`nn.Module`) :
             module that maps the output of the decoder to a
             distribution over the target vocabulary.
        tgt_vocab (:obj:`Vocab`) :
             torchtext vocab object representing the target output
        normalzation (str): normalize by "sents" or "tokens"
    """

    def __init__(self, generator, tgt_vocab):
        super(LossComputeBase, self).__init__()
        self.generator = generator
        self.tgt_vocab = tgt_vocab
        self.padding_idx = tgt_vocab.stoi[inputters.PAD_WORD]

    def _make_shard_state(self, batch, output, range_, attns=None):
        """
        Make shard state dictionary for shards() to return iterable
        shards for efficient loss computation. Subclass must define
        this method to match its own _compute_loss() interface.
        Args:
            batch: the current batch.
            output: the predict output from the model.
            range_: the range of examples for computing, the whole
                    batch or a trunc of it?
            attns: the attns dictionary returned from the model.
        """
        return NotImplementedError

    def _compute_loss(self, batch, output, target, **kwargs):
        """
        Compute the loss. Subclass must define this method.

        Args:

            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            **kwargs(optional): additional info for computing loss.
        """
        return NotImplementedError

    def monolithic_compute_loss(self, batch, output, attns):
        """
        Compute the forward loss for the batch.

        Args:
          batch (batch): batch of labeled examples
          output (:obj:`FloatTensor`):
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict of :obj:`FloatTensor`) :
              dictionary of attention distributions
              `[tgt_len x batch x src_len]`
        Returns:
            :obj:`onmt.utils.Statistics`: loss statistics
        """
        range_ = (0, batch.tgt.size(0))
        shard_state = self._make_shard_state(batch, output, range_, attns)
        _, batch_stats = self._compute_loss(batch, **shard_state)

        return batch_stats

    def sharded_compute_loss(self, batch, output, attns,
                             cur_trunc, trunc_size, shard_size,
                             normalization):
        """Compute the forward loss and backpropagate.  Computation is done
        with shards and optionally truncation for memory efficiency.

        Also supports truncated BPTT for long sequences by taking a
        range in the decoder output sequence to back propagate in.
        Range is from `(cur_trunc, cur_trunc + trunc_size)`.

        Note sharding is an exact efficiency trick to relieve memory
        required for the generation buffers. Truncation is an
        approximate efficiency trick to relieve the memory required
        in the RNN buffers.

        Args:
          batch (batch) : batch of labeled examples
          output (:obj:`FloatTensor`) :
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict) : dictionary of attention distributions
              `[tgt_len x batch x src_len]`
          cur_trunc (int) : starting position of truncation window
          trunc_size (int) : length of truncation window
          shard_size (int) : maximum number of examples in a shard
          normalization (int) : Loss is divided by this number

        Returns:
            :obj:`onmt.utils.Statistics`: validation loss statistics

        """
        batch_stats = onmt.utils.Statistics()
        range_ = (cur_trunc, cur_trunc + trunc_size)
        shard_state = self._make_shard_state(batch, output, range_, attns)
        for i,shard in enumerate(shards(shard_state, shard_size)):
            loss, stats = self._compute_loss(batch, **shard, offset=i*shard_size)
            loss.div(float(normalization)).backward()
            batch_stats.update(stats)
        return batch_stats

    def _stats(self, loss, scores, target, loss_extra = None):
        """
        Args:
            loss (:obj:`FloatTensor`): the loss computed by the loss criterion.
            scores (:obj:`FloatTensor`): a score for each possible output
            target (:obj:`FloatTensor`): true targets

        Returns:
            :obj:`onmt.utils.Statistics` : statistics for this batch.
        """
        if loss_extra is not None:
            loss_return = torch.tensor([loss, loss_extra])
        else:
            loss_return = loss.item()

        pred = scores.max(1)[1]
        non_padding = target.ne(self.padding_idx)
        num_correct = pred.eq(target) \
                          .masked_select(non_padding) \
                          .sum() \
                          .item()
        num_non_padding = non_padding.sum().item()
        return onmt.utils.Statistics(loss_return, num_non_padding, num_correct)

    def _bottle(self, _v):
        return _v.view(-1, _v.size(2))

    def _unbottle(self, _v, batch_size):
        return _v.view(-1, batch_size, _v.size(1))


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.padding_idx = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.padding_idx] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.padding_idx).unsqueeze(1), 0)

        return F.kl_div(output, model_prob, reduction='sum')


class NMTLossCompute(LossComputeBase):
    """
    Standard NMT Loss Computation.
    """

    def __init__(self, generator, tgt_vocab, normalization="sents",
                 label_smoothing=0.0):
        super(NMTLossCompute, self).__init__(generator, tgt_vocab)
        self.sparse = not isinstance(generator[1], nn.LogSoftmax)
        if label_smoothing > 0:
            self.criterion = LabelSmoothingLoss(
                label_smoothing, len(tgt_vocab), ignore_index=self.padding_idx
            )
        elif self.sparse:
            self.criterion = SparsemaxLoss(
                ignore_index=self.padding_idx, size_average=False
            )
        else:
            self.criterion = nn.NLLLoss(
                ignore_index=self.padding_idx, reduction='sum'
            )

    def _make_shard_state(self, batch, output, range_, attns=None):
        return {
            "output": output,
            "target": batch.tgt[range_[0] + 1: range_[1]],
        }

    def _compute_loss(self, batch, output, target, **kwargs):
        bottled_output = self._bottle(output)
        if self.sparse:
            # for sparsemax loss, the loss function operates on the raw output
            # vector, not a probability vector. Hence it's only necessary to
            # apply the first part of the generator here.
            scores = self.generator[0](bottled_output)
        else:
            scores = self.generator(bottled_output)
        gtruth = target.view(-1)

        loss = self.criterion(scores, gtruth)
        stats = self._stats(loss.clone(), scores, gtruth)

        return loss, stats
    
class AtomBalanceLossCompute(LossComputeBase):
    """
    Compute the loss for the atom balance constraint.
    """

    def __init__(self, generator, tgt_vocab, device, atom_balance_loss_weight):
        super(AtomBalanceLossCompute, self).__init__(generator, tgt_vocab)
        self.sparse = not isinstance(generator[1], nn.LogSoftmax)
        if self.sparse:
            self.criterion = SparsemaxLoss(
                ignore_index=self.padding_idx, size_average=False
            )
        else:
            self.criterion = nn.NLLLoss(
                ignore_index=self.padding_idx, reduction='sum'
            )
        
        self.atom_balance_loss = AtomBalanceLoss(tgt_vocab, device, atom_balance_loss_weight)


    def _make_shard_state(self, batch, output, range_, attns=None):
        return {
            "output": output,
            "target": batch.tgt[range_[0] + 1: range_[1]],
        }

    def _compute_loss(self, batch, output, target, offset=0, **kwargs):
        bottled_output = self._bottle(output)
        if self.sparse:
            # for sparsemax loss, the loss function operates on the raw output
            # vector, not a probability vector. Hence it's only necessary to
            # apply the first part of the generator here.
            scores = self.generator[0](bottled_output)
        else:
            scores = self.generator(bottled_output)
        gtruth = target.view(-1)

        loss_standard = self.criterion(scores, gtruth)
        loss_extra = self.atom_balance_loss(scores, gtruth, batch, offset)
        loss = loss_standard + loss_extra
        stats = self._stats(loss_standard.clone(), scores, gtruth, loss_extra=loss_extra.clone())
        
        return loss, stats
    


class AtomBalanceLoss(nn.Module):
    """
    Compute the loss for the atom balance constraint.
    """
    def __init__(self, tgt_vocab, device, atom_balance_loss_weight):
        
        super(AtomBalanceLoss, self).__init__()
        self.vocab_atom_tensor, self.atom_vocab_tensor, self.vocab_charge_tensor, \
            self.atom_types_tensor, self.atom_types_indices, self.special_tokens_tensor, \
                self.arrow_token, self.eos_token, self.sos_token, self.sep_token, self.unk_token, self.pad_token = \
                    create_atom_vocab_maps(tgt_vocab, device)
        self.vocab = tgt_vocab
        self.device = device
        self.weight = atom_balance_loss_weight

    
    def forward(self, output, target, batch, offset=0):
        num_instances = batch.tgt.size(1)
        output_preds = torch.exp(output)  # (num_preds, vocab_size)
        num_preds = output_preds.size(0)

        # Create matching (row, col) indices
        device = self.device
        idx_pred = torch.arange(0, num_preds, device=device)
        row = idx_pred % num_instances
        col = idx_pred // num_instances + offset

        atom_fields = batch.atom_balance_fields  # shape: (batch_size, seq_len, C)
        current_sides = atom_fields[row, col, 0]         # shape: (num_preds,)
        p_contexts = atom_fields[row, col, 1:]           # shape: (num_preds, atom_types+2)

        mask_valid = (current_sides == 0) | (current_sides == 2)
        mask_reactants = current_sides == 0
        mask_no_atoms = p_contexts.sum(dim=1) == 0

        # Mask: only compute loss for valid steps with atoms
        mask_valid = mask_valid & ~(mask_reactants & mask_no_atoms)
        mask_balanced_reactants = ((current_sides == 0).repeat(self.atom_types_tensor.shape[0]+2,1).T) & (p_contexts==0)
        mask_balanced_reactants[:,-2:] = False

        # Compute predicted atom distributions
        atom_dists = self.get_atom_dist_output_batch(output_preds)  # (num_preds, atom_types+2)

        # Final filtered values
        atom_dists[mask_balanced_reactants] = 0
        atom_dists = torch.nn.functional.normalize(atom_dists, p=1, dim=1)
        
        pred_filtered = atom_dists[mask_valid]
        context_filtered = p_contexts[mask_valid]

        loss = torch.nn.functional.l1_loss(pred_filtered, context_filtered, reduction='sum') 

        del atom_dists, context_filtered, pred_filtered, mask_valid, mask_balanced_reactants, output_preds   

        return self.weight * loss

    
    def get_atom_dist_output_batch(self, output_preds):
        """
        Fully vectorized version of atom distribution projection without for loops.
        output_preds: (batch_size, vocab_size)
        Returns: (batch_size, num_atom_types + 2)
        """
        device = output_preds.device
        batch_size, vocab_size = output_preds.shape
        num_atom_types = len(self.atom_types_tensor)

        # Create projection matrix: (vocab_size, num_atom_types + 2)
        projection = torch.zeros(vocab_size, num_atom_types + 2, device=device)

        # Build a flat list of vocab indices and their corresponding atom type indices
        atom_vocab_tensor = self.atom_vocab_tensor[0:num_atom_types]  # shape: (num_atom_types, N)
        mask_valid = atom_vocab_tensor != -1

        vocab_indices = atom_vocab_tensor[mask_valid]
        atom_type_indices = self.atom_types_indices[self.vocab_atom_tensor[vocab_indices]]
        # Scatter 1s into the projection matrix at (vocab_index, atom_type_index)
        projection.index_put_((vocab_indices, atom_type_indices), torch.ones_like(vocab_indices, dtype=projection.dtype), accumulate=True)

        # Add special tokens (arrow at -1, eos at -2)
        projection[self.arrow_token, -1] = 1
        projection[self.eos_token, -2] = 1

        del atom_vocab_tensor, mask_valid, vocab_indices, atom_type_indices

        # Project the output_preds to atom space
        return output_preds @ projection
    

    def get_atom_balance_fields(self, batch):
        # batch.src[0]: (seq_len, batch_size) → transpose to (batch_size, seq_len)
        src_rxns = batch.src[0].t()
        tgt_rxns = batch.tgt.t()

        # Batched versions of balance computation
        src_balance = self.compute_atom_balance_batch(src_rxns)  # shape: (batch_size, atom_type_dim)
        tgt_balance, current_sides = self.compute_atom_balances_and_current_sides_per_token_batch(tgt_rxns)  # shapes: (batch_size, seq_len+1, atom_type_dim), (batch_size, seq_len+1)
        p_contexts = self.compute_p_context(tgt_balance, src_balance, current_sides)
        
        # append current_sides at the beginning of the p_contexts tensor
        fields = torch.cat((current_sides.unsqueeze(2), p_contexts), dim=2)
        del src_balance, tgt_balance, current_sides, p_contexts, src_rxns, tgt_rxns
        return fields.to(self.device)


    def compute_atom_balance_batch(self, rxns):
        # rxns: (batch_size, seq_len)
        device = self.device
        num_atom_types = len(self.atom_types_tensor)

        # Lookup atom types for all reactions
        atoms = self.vocab_atom_tensor[rxns]  # (batch_size, seq_len)

        # Create masks
        mask_arrow = atoms == -1
        mask_eos = atoms == -2
        mask_skip = (atoms == -3) | (atoms == -4) | (atoms == -5) | (atoms == -6) | (atoms == 0) | (atoms == 1)

        # Compute current side for each token
        # cumsum along seq_len dimension
        current_side = torch.cumsum(mask_arrow.long(), dim=1)  # (batch_size, seq_len)

        # Mask valid atom tokens for each side
        valid_mask = ~(mask_arrow | mask_eos | mask_skip)

        # Side 0: increment
        mask_side_0 = (current_side == 0) & valid_mask
        # Side 2: decrement
        mask_side_2 = (current_side == 2) & valid_mask

        # Get atom indices from atoms
        atom_indices = torch.where(valid_mask, self.atom_types_indices[atoms], 0)  # (batch_size, seq_len)

        # One-hot encode all atom indices
        atom_one_hot = torch.nn.functional.one_hot(atom_indices, num_classes=num_atom_types).long()  # (batch_size, seq_len, atom_types)
        
        # Apply masks
        inc_contrib = torch.where(mask_side_0.unsqueeze(-1), atom_one_hot, 0)
        dec_contrib = torch.where(mask_side_2.unsqueeze(-1), atom_one_hot, 0)

        # Sum across sequence to get final balance for each reaction
        balance = inc_contrib.sum(dim=1) - dec_contrib.sum(dim=1)  # (batch_size, atom_types)

        del atoms, mask_arrow, mask_eos, mask_skip, current_side, valid_mask, mask_side_0, mask_side_2, atom_indices, atom_one_hot, inc_contrib, dec_contrib

        return balance.to(device)


    def compute_atom_balance(self, rxn):
        #NOTE: Not used anymore, moved to batched version
        # Initialize balance as a tensor on the correct device
        balance = torch.zeros(len(self.atom_types_tensor), dtype=torch.long, device=self.device)

        # Get all atoms in the reaction (vectorized approach)
        atoms = self.vocab_atom_tensor[rxn]  # This assumes rxn is a tensor of token indices

        # Create a mask for each condition (e.g., -1 for arrows, -2 for eos, etc.)
        mask_arrow = (atoms == -1)
        mask_eos = (atoms == -2)
        mask_skip = (atoms == -3) | (atoms == -4) | (atoms==-5) | (atoms==-6) | (atoms == 0) | (atoms == 1)

        # Find positions where the arrow occurs and switch sides (side 0 -> side 2, then 2 -> 1)
        current_side = torch.cumsum(mask_arrow.long(), dim=0)  # Update sides based on arrows

        # Apply the atom balance logic directly using tensor indexing and masking
        # For side 0 (left side), increment atom counts
        mask_side_0 = (current_side == 0) & (~mask_arrow) & (~mask_eos) & (~mask_skip)
        balance.index_add_(0, self.atom_types_indices[atoms[mask_side_0]], torch.ones_like(atoms[mask_side_0], dtype=torch.long))

        # For side 2 (right side), decrement atom counts
        mask_side_2 = (current_side == 2) & (~mask_arrow) & (~mask_eos) & (~mask_skip)
        balance.index_add_(0, self.atom_types_indices[atoms[mask_side_2]], -torch.ones_like(atoms[mask_side_2], dtype=torch.long))

        del atoms, mask_arrow, mask_eos, mask_skip, current_side, mask_side_0, mask_side_2

        return balance
    
    def compute_atom_balances_and_current_sides_per_token_batch(self, rxns):
        device = self.device
        batch_size, seq_len = rxns.shape
        num_steps = seq_len + 1
        num_atom_types = len(self.atom_types_tensor)

        # Create padded input: shape (batch_size, num_steps, seq_len)
        step_indices = torch.arange(num_steps, device=device).unsqueeze(1)  # (num_steps, 1)
        token_indices = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, seq_len)
        mask = step_indices > token_indices  # (num_steps, seq_len)
        
        # Expand mask to batch and apply pad_token
        padded_rxns = torch.where(
            mask.unsqueeze(0), rxns.unsqueeze(1).expand(-1, num_steps, -1), self.pad_token
        )  # shape: (batch_size, num_steps, seq_len)

        atoms = self.vocab_atom_tensor[padded_rxns]  # (batch_size, num_steps, seq_len)

        # Get current_sides from last row
        last_row = atoms[:, -1, :]  # (batch_size, seq_len)
        mask_arrow = (last_row == -1)
        mask_eos = (last_row == -2)
        current_sides = torch.cumsum(mask_arrow.int() + mask_eos.int(), dim=1)  # (batch_size, seq_len)
        

        # Valid atoms
        valid_atom_mask = atoms > 1  # (batch_size, num_steps, seq_len)
        atom_indices = torch.where(valid_atom_mask, self.atom_types_indices[atoms], 0)  # (batch_size, num_steps, seq_len)

        # One-hot encode
        atom_one_hot = torch.nn.functional.one_hot(atom_indices, num_classes=num_atom_types).long()  # (batch, steps, seq_len, atom_types)
        atom_one_hot *= valid_atom_mask.unsqueeze(-1)

        # Expand current sides to match atoms
        current_sides_exp = current_sides.unsqueeze(1).expand(-1, num_steps, -1)  # (batch, steps, seq_len)

        side_0_mask = (current_sides_exp == 0) & valid_atom_mask
        side_2_mask = (current_sides_exp == 2) & valid_atom_mask

        # Apply side masks
        side_0_contrib = torch.where(side_0_mask.unsqueeze(-1), atom_one_hot, 0)
        side_2_contrib = torch.where(side_2_mask.unsqueeze(-1), atom_one_hot, 0)

        # Sum across seq_len
        balances = side_0_contrib.sum(dim=2) - side_2_contrib.sum(dim=2)  # (batch_size, num_steps, atom_types)

        current_sides = torch.cat([
            torch.zeros(batch_size, 1, dtype=torch.long, device=device),
            current_sides
        ], dim=1)  # (batch_size, num_steps)

        del padded_rxns, atoms, last_row, mask_arrow, mask_eos, valid_atom_mask, atom_indices, atom_one_hot, side_0_mask, side_2_mask, side_0_contrib, side_2_contrib

        return balances.to(device), current_sides.to(device)


    def compute_atom_balances_and_current_sides_per_token(self, rxn):
        #NOTE: Not used anymore, moved to batched version
        device = self.device
        num_steps = len(rxn) + 1
        num_atom_types = len(self.atom_types_tensor)

        # Prepare output tensor
        balances = torch.zeros((num_steps, num_atom_types), dtype=torch.long, device=device)

        # Get atom tokens with padding logic applied
        step_indices = torch.arange(num_steps, device=device).unsqueeze(1)
        token_indices = torch.arange(len(rxn), device=device)
        mask = step_indices > token_indices
        padded_rxns = torch.where(mask, rxn.expand(num_steps, -1), self.pad_token)
        
        atoms = self.vocab_atom_tensor[padded_rxns]  # shape: (num_steps, seq_len)

        # Get current_sides vector
        last_row = atoms[-1]  # the last reaction row, used for side tracking
        mask_arrow = (last_row == -1)
        mask_eos = (last_row == -2)
        current_sides = torch.cumsum(mask_arrow.int() + mask_eos.int(), dim=0).to(device)

        # Compute atom indices for valid atoms (>1)
        valid_atom_mask = atoms > 1
        atom_indices = torch.where(valid_atom_mask, self.atom_types_indices[atoms], 0)  # shape: (num_steps, seq_len)

        # One-hot encode atoms only where they are valid
        atom_one_hot = torch.nn.functional.one_hot(atom_indices, num_classes=num_atom_types).long()  # (num_steps, seq_len, atom_types)
        atom_one_hot *= valid_atom_mask.unsqueeze(-1)

        # Create masks for side 0 and side 2
        side_0_mask = (current_sides == 0).unsqueeze(0).expand(num_steps, -1) & valid_atom_mask
        side_2_mask = (current_sides == 2).unsqueeze(0).expand(num_steps, -1) & valid_atom_mask

        # Mask and sum contributions
        side_0_contrib = torch.where(side_0_mask.unsqueeze(-1), atom_one_hot, 0)
        side_2_contrib = torch.where(side_2_mask.unsqueeze(-1), atom_one_hot, 0)

        # Compute balances (add for side 0, subtract for side 2)
        balances = side_0_contrib.sum(dim=1) - side_2_contrib.sum(dim=1)

        current_sides = torch.cat((torch.zeros(1, dtype=torch.long, device=device), current_sides), dim=0)

        del padded_rxns, atoms, last_row, mask_arrow, mask_eos, valid_atom_mask, atom_indices, atom_one_hot, side_0_mask, side_2_mask, side_0_contrib, side_2_contrib

        return balances, current_sides



    def compute_p_context(self, tgt_balances, src_balances, current_sides):

        src_repeated = src_balances.repeat(tgt_balances.shape[1],1,1).permute(1,0,2)
        cur_sides_repeated = current_sides.repeat(tgt_balances.shape[2],1,1).permute(1,2,0)
        combined_balance = src_repeated + tgt_balances
        p_contexts_total_reactants = torch.where(cur_sides_repeated==0, combined_balance, 0).float()
        p_contexts_total_reactants = torch.where(p_contexts_total_reactants<0, -p_contexts_total_reactants, 0)
        p_contexts_total_products = torch.where(cur_sides_repeated==2, combined_balance, 0).float()
        # assert p_contexts_total_products contains only positive values
        assert (p_contexts_total_products >= 0).all(), f"p_contexts_total_products contains negative values: {p_contexts_total_products}"
         
        # add the two tensors together and append two more elements
        p_contexts_total = p_contexts_total_reactants+p_contexts_total_products
        p_contexts_total = torch.nn.functional.normalize(p_contexts_total, p=1, dim=2)
        # add two more elements in dim 2
        eos_indices = torch.zeros((src_balances.shape[0], current_sides.shape[1], 2), dtype=torch.float, device=self.device)
        eos_indices[:,:,0] = 1
        cur_sides_repeated = cur_sides_repeated[:,:,0:2]
        eos_indices = torch.where(cur_sides_repeated==2, eos_indices, 0)
        eos_indices[~((p_contexts_total == 0).all(dim=2)),:] = 0
        p_contexts_total = torch.cat((p_contexts_total, eos_indices), dim=2)

        del src_repeated, cur_sides_repeated, combined_balance, p_contexts_total_reactants, p_contexts_total_products, eos_indices

        return p_contexts_total



def filter_shard_state(state, shard_size=None):
    """ ? """
    for k, v in state.items():
        if shard_size is None:
            yield k, v

        if v is not None:
            v_split = []
            if isinstance(v, torch.Tensor):
                for v_chunk in torch.split(v, shard_size):
                    v_chunk = v_chunk.data.clone()
                    v_chunk.requires_grad = v.requires_grad
                    v_split.append(v_chunk)
            yield k, (v, v_split)


def shards(state, shard_size, eval_only=False):
    """
    Args:
        state: A dictionary which corresponds to the output of
               *LossCompute._make_shard_state(). The values for
               those keys are Tensor-like or None.
        shard_size: The maximum size of the shards yielded by the model.
        eval_only: If True, only yield the state, nothing else.
              Otherwise, yield shards.

    Yields:
        Each yielded shard is a dict.

    Side effect:
        After the last shard, this function does back-propagation.
    """
    if eval_only:
        yield filter_shard_state(state)
    else:
        # non_none: the subdict of the state dictionary where the values
        # are not None.
        non_none = dict(filter_shard_state(state, shard_size))

        # Now, the iteration:
        # state is a dictionary of sequences of tensor-like but we
        # want a sequence of dictionaries of tensors.
        # First, unzip the dictionary into a sequence of keys and a
        # sequence of tensor-like sequences.
        keys, values = zip(*((k, [v_chunk for v_chunk in v_split])
                             for k, (_, v_split) in non_none.items()))

        # Now, yield a dictionary for each shard. The keys are always
        # the same. values is a sequence of length #keys where each
        # element is a sequence of length #shards. We want to iterate
        # over the shards, not over the keys: therefore, the values need
        # to be re-zipped by shard and then each shard can be paired
        # with the keys.
        for shard_tensors in zip(*values):
            yield dict(zip(keys, shard_tensors))

        # Assumed backprop'd
        variables = []
        for k, (v, v_split) in non_none.items():
            if isinstance(v, torch.Tensor) and state[k].requires_grad:
                variables.extend(zip(torch.split(state[k], shard_size),
                                     [v_chunk.grad for v_chunk in v_split]))
        inputs, grads = zip(*variables)
        torch.autograd.backward(inputs, grads)
