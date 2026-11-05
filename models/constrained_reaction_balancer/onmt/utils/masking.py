import torch
import numpy as np
from onmt.utils.logging import logger


def create_atom_vocab_maps(vocab, device):
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')


    vocab_size = len(vocab.itos)
    vocab_atom_tensor = torch.full((vocab_size,), 0, dtype=torch.long)  
    vocab_charge_tensor = torch.full((vocab_size,), 0, dtype=torch.long) 
    atom_occurence = torch.full((119,), 0, dtype=torch.long)  # Atomic numbers range from 1 to 118

    for i, v in enumerate(vocab.itos):
        if v == ">":
            vocab_atom_tensor[i] = -1
            arrow_token = torch.tensor(i, dtype=torch.long)
            continue
        if v == "</s>":
            vocab_atom_tensor[i] = -2
            eos_token = torch.tensor(i, dtype=torch.long)
            continue
        if v == "<s>":
            vocab_atom_tensor[i] = -3
            sos_token = torch.tensor(i, dtype=torch.long)
            continue
        if v == ".":
            vocab_atom_tensor[i] = -4
            sep_token = torch.tensor(i, dtype=torch.long)
            continue
        if v == "<unk>":
            vocab_atom_tensor[i] = -5
            unk_token = torch.tensor(i, dtype=torch.long)
            continue
        if v == "<blank>":
            vocab_atom_tensor[i] = -6
            pad_token = torch.tensor(i, dtype=torch.long)
            continue

        mol = Chem.MolFromSmiles(v)
        if mol is not None:
            atomic_num = mol.GetAtoms()[0].GetAtomicNum()
            charge = Chem.GetFormalCharge(mol)
            vocab_atom_tensor[i] = atomic_num
            vocab_charge_tensor[i] = charge
            atom_occurence[atomic_num] += 1
        else:
            # Handle case where the molecule could not be parsed
            new_v = ''
            first_alpha = True
            for c in v:
                if first_alpha and c.isalpha():
                    new_v += c.upper()
                    first_alpha = False
                else:
                    new_v += c
            mol = Chem.MolFromSmiles(new_v)

            if mol is not None:
                atomic_num = mol.GetAtoms()[0].GetAtomicNum()
                charge = Chem.GetFormalCharge(mol)
                vocab_atom_tensor[i] = atomic_num
                vocab_charge_tensor[i] = charge
                atom_occurence[atomic_num] += 1
            else:
                # Defaulting to atomic number 0 for unmapped tokens
                assert vocab_atom_tensor[i] == 0

    # ignore hydrogen
    atom_occurence[1] = 0
    atom_types_tensor = torch.nonzero(atom_occurence).squeeze()  # Get the indices of non-zero elements
    special_tokens_tensor = torch.tensor([-6,-5,-4,-3,-2,-1], dtype=torch.long)
    max_occurence = atom_occurence.max().item()
    atom_vocab_tensor = torch.full((len(atom_types_tensor) + len(special_tokens_tensor), max_occurence), -1, dtype=torch.long)
    for i, atom in enumerate(atom_types_tensor):
        atom_vocab_tensor[i, :atom_occurence[atom]] = torch.nonzero(vocab_atom_tensor == atom).squeeze()

    atom_vocab_tensor[-6, 0] = pad_token
    atom_vocab_tensor[-5, 0] = unk_token
    atom_vocab_tensor[-4, 0] = sep_token
    atom_vocab_tensor[-3, 0] = sos_token
    atom_vocab_tensor[-2, 0] = eos_token
    atom_vocab_tensor[-1, 0] = arrow_token

    atom_types_indices = torch.full((atom_types_tensor.max().item()+1,), 0, dtype=torch.long)
    for i, atom in enumerate(atom_types_tensor):
        atom_types_indices[atom] = i
    
    del atom_occurence
    return vocab_atom_tensor.to(device), atom_vocab_tensor.to(device), vocab_charge_tensor.to(device), atom_types_tensor.to(device), atom_types_indices.to(device), special_tokens_tensor.to(device), arrow_token.to(device), eos_token.to(device), sos_token.to(device), sep_token.to(device), unk_token.to(device), pad_token.to(device)



class AtomBalanceMask(object):
    def __init__(self, vocab, beam_size, constrain_charge, device):
        self.vocab = vocab
        self.device = device
        logger.info(f"Using device {self.device} for AtomBalanceMask")
        self.vocab_atom_tensor, self.atom_vocab_tensor, self.vocab_charge_tensor, \
            self.atom_types_tensor, self.atom_types_indices, self.special_tokens_tensor, \
                self.arrow_token, self.eos_token, self.sos_token, self.sep_token,  \
                    self.unk_token, self.pad_token = create_atom_vocab_maps(vocab, device)
        self.beam_size = beam_size
        self.constrain_charge = constrain_charge
        
    
    def get_log_probs_masking_tensor(self, src_balance, prev_preds):
        """
        Make a matrix same beam * batch_size x vocab_size, where valid tokens are have entry 1 and other 1e15.
        Therefore, if this is subtracted from the log_prob matrix, only valid tokens are predicted.
        (with multiplication by 1e-15, problems can arise when the log_prob is 0)
        
        Args:
            src_balance: (B, D)
            prev_preds: (B, L)
        Returns:
            mask: (B, V) logit mask with large values for restricted tokens
        """

        assert len(src_balance) == len(prev_preds)
        B, V = src_balance.shape[0], len(self.vocab)
        A = len(self.atom_types_tensor)
        device = self.device
 
        prev_balance, cur_side = self.compute_atom_balance_batch(prev_preds, src_rxns=False)  # (B, D), (B,)
        total_balance = prev_balance + src_balance  # (B, D)

        mask = torch.full((B, V), 0, device=device)

        # ======= RULE 1: Add EOS token for cur_side == 1 or 0 =======
        eos_restricted = (cur_side == 0) | (cur_side == 1)
        mask[torch.arange(B, device=device)[eos_restricted], self.eos_token] = 1e15

        # ======= RULE 2: Restrict arrow if imbalance on reactants side (cur_side == 0) =======
        atom_ok = (total_balance[:, :A] >= 0)  # broadcast mask for hydrogen
        block_arrow = (~atom_ok.all(dim=1) & (cur_side == 0).unsqueeze(1).squeeze(1)) | ((cur_side==2).unsqueeze(1).squeeze(1))  # (B,)
        mask[torch.arange(B, device=device)[block_arrow], self.arrow_token] = 1e15

        # ======= RULE 3: Restrict EOS if imbalance on product side =======
        is_product_side = (cur_side == 2)
        if self.constrain_charge:
            imbalance = (total_balance[:, -1] != 0)
            mask[torch.arange(B, device=device)[imbalance & is_product_side], self.eos_token] = 1e15

        imbalance_atoms = (total_balance[:, :A] != 0) & (self.atom_types_tensor != 1)
        eos_blocked = imbalance_atoms.any(dim=1) & is_product_side
        mask[torch.arange(B, device=device)[eos_blocked], self.eos_token] = 1e15

        # ======= RULE 4: Restrict individual atom tokens if atom is balanced (product side) =======
        product_balanced_atoms = (total_balance[:, :A] == 0) & is_product_side.unsqueeze(1)  # (B, A)

        # (B, A, T) mask where True means token t for atom a is restricted in rxn b
        valid_tokens = self.atom_vocab_tensor.clone()[:A,:]  # (A, T)
        valid_tokens_mask = valid_tokens != -1  # (A, T)

        # Combine masks
        product_balanced_mask = product_balanced_atoms.unsqueeze(-1) & valid_tokens_mask.unsqueeze(0).expand(B, -1, -1)  # (B, A, T)

        # Flatten indices to apply to `mask`
        rows, cols, toks = torch.where(product_balanced_mask)

        if rows.numel() > 0:
            tokens = valid_tokens[cols, toks]

            # Filter out invalid tokens
            valid = tokens != -1
            rows = rows[valid]
            tokens = tokens[valid]

            mask[rows, tokens] = 1e15


        # ======= RULE 5: Restrict SEP token after SOS, SEP or arrow =======
        last_tokens = prev_preds[:, -1]
        add_sep = (last_tokens == self.sos_token) | (last_tokens == self.sep_token) | (last_tokens == self.arrow_token)
        mask[torch.arange(B, device=device)[add_sep], self.sep_token] = 1e15

        # ======= RULE 6: Restrict <unk> token =======
        mask[:, self.unk_token] = 1e15

        # ======= RULE 7: Restrict no tokens if <unk> is present in src reaction =======
        mask_unk = (src_balance == -10).all(dim=1)
        if mask_unk.any():
            mask[mask_unk,:] = 0

        return mask
        
    
    def compute_atom_balance_batch(self, rxns, src_rxns = True):
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

        if self.constrain_charge:
            charge = self.vocab_charge_tensor[rxns]
            chg_0 = torch.where(current_side == 0, charge, 0).sum(dim=1, keepdim=True)
            chg_2 = torch.where(current_side == 2, charge, 0).sum(dim=1, keepdim=True)
            balance = torch.cat((balance, chg_0 - chg_2), dim=1)  # (batch_size, atom_types + 1)

        # replace balance with -10 if <unk> token is present in src reactions
        if src_rxns:
            mask_unk = atoms == -5
            balance[mask_unk.any(dim=1), :] = -10
            unk_rxns = rxns[mask_unk.any(dim=1), :] 
            for unk_rxn in unk_rxns:
                logger.info(f"Reaction {''.join([self.vocab.itos[i] for i in unk_rxn if i != self.pad_token])} contains <unk> token, so mask cannot be computed")
            del atoms, mask_arrow, mask_eos, mask_skip, valid_mask, mask_side_0, mask_side_2, atom_indices, atom_one_hot, inc_contrib, dec_contrib, current_side
            return balance.to(device)
        
        del atoms, mask_arrow, mask_eos, mask_skip, valid_mask, mask_side_0, mask_side_2, atom_indices, atom_one_hot, inc_contrib, dec_contrib
        return balance.to(device), current_side[:, -1].to(device)
    
   

class ChemVocabMask(object):
    """
    Chemical reaction token mask. Can either be initialized from a saved state or
    using a vocab file. Used to hide tokens that do not appear in the source sequence.
    """
    def __init__(self, vocab=None, from_file=None):

        if from_file is not None and from_file != 'None':
            checkpoint = torch.load(from_file)
            self.always_active = checkpoint['always_active']
            self.atom_vocab_dict = checkpoint['atom_vocab_dict']
            self.vocab_atom_dict = checkpoint['vocab_atom_dict']
            self.vocab_vocab_dict = checkpoint['vocab_vocab_dict']
            self.vocab = checkpoint['vocab']

        elif vocab is not None:
            self.vocab = vocab
            self.initialize_dicts()

    def save_dicts(self, file_path):
        torch.save({
            'always_active': self.always_active,
            'atom_vocab_dict': self.atom_vocab_dict,
            'vocab_atom_dict': self.vocab_vocab_dict,
            'vocab_vocab_dict': self.vocab_vocab_dict,
            'vocab': self.vocab
        }, file_path)

    def initialize_dicts(self):
        """
        Use the vocab file, generated by the preprocess.py to initialize the
        atom_vocab, vocab_atom, and vocab_vocab dicts.
        :return:
        """
        from rdkit import Chem
        from rdkit import RDLogger
        RDLogger.DisableLog('rdApp.*')
        
        always_active = []
        atom_vocab_dict = {}
        for i, v in enumerate(self.vocab.itos):
            mol = Chem.MolFromSmiles(v)
            if mol is not None:
                atomic_num = mol.GetAtoms()[0].GetAtomicNum()

                if atomic_num in atom_vocab_dict.keys():
                    atom_vocab_dict[atomic_num].append(i)
                else:
                    atom_vocab_dict[atomic_num] = [i]
            else:
                new_v = ''
                first_alpha = True
                for c in v:
                    if first_alpha and c.isalpha():
                        new_v += c.upper()
                        first_alpha = False
                    else:
                        new_v += c
                mol = Chem.MolFromSmiles(new_v)

                if mol is not None:
                    atomic_num = mol.GetAtoms()[0].GetAtomicNum()

                    if atomic_num in atom_vocab_dict.keys():
                        atom_vocab_dict[atomic_num].append(i)
                    else:
                        atom_vocab_dict[atomic_num] = [i]
                else:
                    always_active.append(i)
        self.always_active = always_active
        self.atom_vocab_dict = atom_vocab_dict
        vocab_atom_dict = {}
        for k, v in atom_vocab_dict.items():
            for token in v:
                vocab_atom_dict[token] = k
        self.vocab_atom_dict = vocab_atom_dict

        vocab_vocab_dict = {}
        for k, v in atom_vocab_dict.items():
            for i in v:
                vocab_vocab_dict[i] = v
        for i in always_active:
            vocab_vocab_dict[i] = always_active
        self.vocab_vocab_dict = vocab_vocab_dict

    def _get_valid_tokens_per_src_seq_in_batch(self, src):
        valid_tokens_per_seq = [
            np.unique([vocab_list for voc in np.unique(s.cpu().numpy()) for vocab_list in self.vocab_vocab_dict[voc]]) for s
            in src.t()]
        for i, valid_token in enumerate(valid_tokens_per_seq):
            valid_tokens_per_seq[i] = np.int64(valid_token)
        return valid_tokens_per_seq

    def get_log_probs_masking_tensor(self, src, beam_size):
        """
        Make a matrix same beam * batch_size x vocab_size, where valid tokens are have entry 1 and other 1e-15.
        Therefore, if this multiplies the log_prob matrix, only valid tokens are predicted.
        """
        valid_tokens_per_seq = self._get_valid_tokens_per_src_seq_in_batch(src)
        mask = torch.stack([(torch.ones(len(self.vocab)).index_fill(0, torch.tensor(valid_tokens), 0) \
                             * 1e15).index_fill(0,torch.tensor(valid_tokens), 1)
                            for valid_tokens in valid_tokens_per_seq for i in range(beam_size)])
        mask = mask.to(src.device)
        return mask

    def _get_unique_vocab_counts_from_source(self, src):
        unique_counts_dicts = [dict(zip(*np.unique(s.cpu().numpy(), return_counts=True))) for s in src.t()]
        return unique_counts_dicts