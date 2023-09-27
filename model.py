import torch
from torch import nn
from transformers import BertModel
import torchtext.functional as F
import numpy as np
from tqdm import tqdm

class TagLSTM(nn.Module):
    """Models an LSTM on top of a transformer to predict POS in a Neural CRF."""

    def __init__(self, nb_labels, emb_dim,  hidden_dim=256):
        """Constructor.

        Parameters
        ---
        nb_labels : int
            Number of POS tags to be considered.

        emb_dim : int
            Input_size of the LSTM - effectively embedding dimension of our pretrained transformer.

        hidden_dim : int
            Hidden dimension of the LSTM.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            emb_dim, hidden_dim // 2, bidirectional=True, batch_first=True
        )
        self.tag = nn.Linear(hidden_dim, nb_labels)
        self.hidden = None

    def init_hidden(self, batch_size):
        return (
            torch.randn(2, batch_size, self.hidden_dim // 2),
            torch.randn(2, batch_size, self.hidden_dim // 2),
        )

    def forward(self, x):
        self.hidden = self.init_hidden(x.shape[0])
        x, self.hidden = self.lstm(x, self.hidden)
        x = self.tag(x)
        return x
    
class NeuralCRF(nn.Module):
    def __init__(
        self,
        pad_idx_word,
        pad_idx_pos,
        bos_idx,
        eos_idx,
        bot_idx,
        eot_idx,
        t_cal,
        transformer,
        lstm_hidden_dim=64,
        beta=0,
        
    ):

        super().__init__()
        self.pad_idx_word = pad_idx_word
        self.pad_idx_pos = pad_idx_pos
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.bot_idx = bot_idx
        self.eot_idx = eot_idx
        self.t_cal = t_cal
        self.transformer = transformer
        self.lstm_hidden_dim = lstm_hidden_dim
        self.beta = beta
        self.transitions = nn.Parameter(torch.empty(len(t_cal), len(t_cal)))
        self.emissions = TagLSTM(
            len(t_cal),
            transformer.config.to_dict()["hidden_size"],
            lstm_hidden_dim,
        )

        self.init_weights()

    def init_weights(self):
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def forward(self, W):
        """Decode each sentence within W and return predicted tagging.

        Parameters
        ---
        W : torch.tensor
            Word sequences of dimension batch size x max sentence length within batch + 2.

        Returns
        ---
        sequences : list
            List of tensors, each of which contains the predicted tag indices for a particular
            word sequence.
        """
        # Calculate scores.
        emissions = self.calculate_emissions(W)
        # Run viterbi sentence by sentence.
        sequences = []
        for sentence in range(W.shape[0]):
            # Exclude beginning and end markers from each word sequence.
            scores, backpointers = self.maximizer(
                W[sentence, 1:], emissions[sentence, :]
            )
            sequences += [self.get_max(backpointers)]
        return sequences
    
    
    def calculate_emissions(self, W):
        return self.emissions(self.transformer(W)[0])[:, 1:, :]

    
    def normalizer(self, W, emissions):
            '''Uses dynamic programming to compute the normalizer in log space. Since the score for one path is given by the product of the edges,
            the sum of the scores in log space will corerespond to the logexpsum of the sums.


            Parameters
            ---
            W : torch.tensor
                Words for each sequence within the batch.
                Of dimension batch size x longest sequence within batch + 1.
                Note the paddings, EOS and BOS that have been added to W
                for usage with BERT so we mask them out here. We expect
                W to already have the initial BOS word indices taken out.
            emissions : torch.tensor
                Word level scores for each tag of dimension batch_size x max
                sentence length within batch + 1 x |T| (scores for the BOS
                initial tag have already been removed since BOS is
                only needed for the transformer).

            Returns
            ---
            torch.tensor
                Log Z for each sample in W.
            '''


            '''we create a mask for eos and pad tokens so that we can parallelize with respect to the whole batch'''
            T = self.t_cal
            nb, nw, nt = W.shape[0], W.shape[1], len(T)
            Mask = torch.clone( ~ ( (W == self.pad_idx_word ) | (W == self.eos_idx))).to(torch.int)

            '''Beta[:,t] at the nth iteration represents the sum of the products of all edges terminating with tag t at position n '''
            Beta = torch.full(fill_value = 0, size = (nb,nt), dtype = torch.float64)
            transitions = self.transitions.reshape((1,nt,nt)).expand((nb, nt, nt))
            for n in np.arange(nw-2,-1,-1):
                scores = transitions + emissions[:,n+1,:].reshape((nb,1,nt)).expand((nb, nt, nt))
                addends = scores + Beta.unsqueeze(1).expand((nb, nt, nt))
                osum = torch.logsumexp(addends,dim=2)
                Beta = osum * Mask[:,n+1].reshape((nb,1)).expand((nb,nt)) + (1 - Mask[:,n+1]).reshape((nb,1)).expand((nb,nt)) * Beta
            ## Last Layer
            scores = self.transitions[self.bot_idx,:] + emissions[:,0,:]
            addends = scores + Beta
            osum = torch.logsumexp(addends, dim=1)
            return osum
    


    def get_max(self, backpointer_matrix):

        """Return the best tagging based on a backpointer matrix.

        Parameters
        ---
        backpointer_matrix : torch.tensor
            Backpointer matrix from Viterbi indicating which
            tag is the highest scoring for each element in the sequence.

        Returns
        ---
        torch.tensor
            Indices of the best tagging based on `backpointer_matrix`."""

        BP = backpointer_matrix
        N = BP.shape[0]
        best_tag = torch.tensor([-1 for i in range(N)])
        best = BP[0,0]
        best_tag[0] = best
        for i in range(1,N):
            best = BP[i,best]
            best_tag[i] = best
        return best_tag
    

    def maximizer(self, W, emissions):
        """Calculate the best tagging using the backward algorithm and return
        both the scoring matrix in log-space and the backpointer matrix.

        Parameters
        ---
        W : torch.tensor
            Of dimension longest sequence within batch + 2 or less.
            Note the paddings, EOS and BOS that have been added to W
            for usage with BERT so we manually remove them here if present.
        emissions : torch.tensor
            Word level scores for each tag of dimension max
            sentence length within batch + 1 x |T| (we assume scores for the BOT
            initial tag have already been removed since BOT/BOS is
            only needed for the transformer).

        Returns
        ---
        Tuple[torch.tensor, torch.tensor]
            Tuple containing the scoring matrix in log-space and the
            backpointer matrix for recovering the best tagging.
        """
        T = self.t_cal
        # Remove padding
        W = W[torch.where(W != self.pad_idx_word)[0]]

        # Remove EOS if present
        if torch.any(W == self.eos_idx):
            W = W[:-1]
        
        ## Add BOS if removed:
        if W[0]!=self.bos_idx:
            W=torch.cat((torch.tensor(self.bos_idx).unsqueeze(0),W))

        nw, nt = W.shape[0], len(T)
        Beta = torch.zeros(size = (nw,nt),dtype = torch.float64)
        BP = - torch.ones(size=(nw,nt), dtype=torch.int)
        for n in np.arange(nw-2,-1,-1):
            scores = self.transitions + emissions[n,:].reshape((1,nt)).expand((nt,nt))
            addends = scores + Beta[n+1,:].reshape((1,nt)).expand((nt,nt))
            Beta[n,:],BP[n,:] = torch.max(addends,dim = 1)
        if Beta.shape[0] == 1: return Beta,BP
        return Beta[:-1,:], BP[:-1,:] 
    
    
    def entropy(self, W, emissions):
            """Calculate the unnormalized entropy in log space using the backward algorithm.

            
            Parameters
            ---
            W : torch.tensor
                Words for each sequence within the batch.
                Of dimension batch size x longest sequence within batch + 1.
                Note the paddings, EOS and BOS that have been added to W
                for usage with BERT so we mask them out here. We expect
                W to already have the initial BOS word indices taken out.
            emissions : torch.tensor
                Word level scores for each tag of dimension batch_size x max
                sentence length within batch + 1 x |T| (scores for the EOS
                initial tag have already been removed since EOS is
                only needed for the transformer).

            Returns
            ---
            torch.tensor
                Unnormalized entropy for each sample in W."""
            T = self.t_cal
            nb, nw, nt = W.shape[0], W.shape[1], len(T)
            M = torch.clone( ~ ( (W == self.pad_idx_word) | (W == self.eos_idx))).to(torch.int).unsqueeze(2).expand((nb,nw,2))
            Beta=torch.zeros((nb,nt,2),dtype=torch.float64)
            Beta[:,:,0]=1
            Beta[:,:,1]=0
            transitions=self.transitions.unsqueeze(0).expand((nb,nt,nt))
            for n in np.arange(nw-2,-1,-1):
                scores=torch.exp(transitions + emissions[:,n+1,:].unsqueeze(1).expand((nb,nt,nt)))
                w=torch.zeros(size=(nb,nt,nt,2),dtype=torch.float64)
                w[:,:,:,0]=scores
                w[:,:,:,1]=-scores*torch.log(scores)
                addends=torch.zeros(size=(nb,nt,nt,2),dtype=torch.float64)
                addends[:,:,:,0]=w[:,:,:,0].clone()*Beta[:,:,0].clone().unsqueeze(1).expand((nb,nt,nt))
                addends[:,:,:,1]=w[:,:,:,0].clone()*Beta[:,:,1].clone().unsqueeze(1).expand((nb,nt,nt))+w[:,:,:,1].clone()*Beta[:,:,0].clone().unsqueeze(1).expand((nb,nt,nt))
                osum=torch.sum(addends,dim=2)
                Beta=osum*M[:,n+1,:].unsqueeze(1).expand((nb,nt,2)) +(1-M[:,n+1,:].unsqueeze(1).expand((nb,nt,2))) * Beta.clone()
            ## Last Layer
            scores= torch.exp(self.transitions[self.bot_idx,:]+emissions[:,0,:])
            w=torch.zeros(size=(nb,nt,2),dtype=torch.float64)
            w[:,:,0]=scores
            w[:,:,1]=-scores*torch.log(scores)
            addends=torch.zeros(size=(nb,nt,2),dtype=torch.float64)
            addends[:,:,0]=w[:,:,0]*Beta[:,:,0]
            addends[:,:,1]=w[:,:,0]*Beta[:,:,1]+w[:,:,1]*Beta[:,:,0]
            osum=torch.sum(addends,dim=1)
            return osum[:,1]



    def loss(self, T, W):
        """Calculate the loss for a batch.

        Parameters
        ---
        T : torch.tensor
            True taggings for each sequence within the batch.
            Of dimension batch size x longest sequence within batch + 2.
            Note the paddings, EOS and BOS that have been added to T
            for symmetry with W which needs this for BERT.
        W : torch.tensor
            Words for each sequence within the batch.
            Of dimension batch size x longest sequence within batch + 2.
            Note the paddings, EOS and BOS that have been added to W
            for usage with BERT.

        Returns
        ---
        torch.tensor
            Mean loss for the batch.
        """
        emissions = self.calculate_emissions(W)
        # Note that we have to handle paddings and EOS within the score
        # and backward functions, but we can already skip the BOS tokens
        # here.
        scores = self.score(emissions, W[:, 1:], T[:, 1:])
        log_normalizer = self.normalizer(W[:, 1:], emissions)
        # NB: If you don't calculate your normaliser in log-space,
        # you may have issues with normalisers being too large. 
        # Since this occurs very rarely, you can just skip those 
        # batches with a high pseudo normaliser (comment out the code below).

        if torch.any(torch.isinf(log_normalizer)) or torch.any(torch.isnan(log_normalizer)):
           log_normalizer = torch.log(torch.tensor(1e+200, dtype=torch.float64), device=self.dev)
        
        loss = torch.negative(torch.mean(scores - log_normalizer))
        if self.beta > 0.0:
            unnormalized_entropy = self.entropy(
                W[:, 1:], emissions
            )
            entropy = (
                (unnormalized_entropy / torch.exp(log_normalizer))
                + log_normalizer
            )
            if torch.isinf(torch.max(torch.exp(log_normalizer))):
                return loss
            else:
                return loss + torch.negative(self.beta * torch.mean(entropy))
        else:
            return loss

    def score(self, emissions, W, T):
        """Calculate scores for specified taggings and word sequences.

        Parameters
        ---
        emissions : torch.tensor
        T : torch.tensor
            Taggings for each sequence within the batch.
            Of dimension batch size x longest sequence within batch + 1.
            Note the paddings, EOS and BOS that have been added to T
            for symmetry with W which needs this for BERT.
            We expect T to already have the initial BOT tag indices removed
            (see `loss` for details).
        W : torch.tensor
            Words for each sequence within the batch.
            Of dimension batch size x longest sequence within batch + 1.
            Note the paddings, EOS and BOS that have been added to W
            for usage with BERT so we mask them out here. We expect
            W to already have the initial BOS word indices taken out
            (see `loss` for details).

        Returns
        ---
        scores : torch.tensor
            score(T, W) for all samples in W.
        """
        scores = (
            emissions[:, 0].gather(1, (T[:, 0]).unsqueeze(1)).squeeze()
            + self.transitions[self.bot_idx, T[:, 0]]
        )
        for word in range(1, emissions.shape[1]):
            mask = torch.where(
                W[:, word] == self.pad_idx_word, 0, 1
            ) * torch.where(W[:, word] == self.eos_idx, 0, 1)
            scores += mask * (
                emissions[:, word]
                .gather(1, (T[:, word]).unsqueeze(1))
                .squeeze()
                + self.transitions[T[:, word - 1], T[:, word]]
            )
        return scores

    
def train_model_report_accuracy(
    crf,
    lr,
    epochs,
    train_dataloader,
    dev_dataloader,
    pad_token_idx_word,
    pad_token_idx_tag,
    device
):

    """Train model for `epochs` epochs and report performance on 
        dev set after each epoch.

    Parameters
    ---
    crf : NeuralCRF
    lr : float
        Learning rate to train with.
    epochs : int
        For how many epochs to train.
    train_dataloader : torch.DataLoader
    dev_dataloder : torch.DataLoader
    pad_token_idx_word : int
        Index with which to pad the word indices.
    pad_token_idx_tag : int
        Index with which to pad the tag indices.
    """
    optimizer = torch.optim.Adam(crf.parameters(), lr=lr)
    n_batches = len([batch for batch in train_dataloader])
    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1} / {epochs}")
        crf.train()
        crf.transformer.train()
        for i, data in tqdm(enumerate(train_dataloader),total=n_batches):
            #if (i+1) %20==0: print('Batch: {} / {} completed'.format(i+1, n_batches))
            with torch.device(device):
                torch.autograd.set_detect_anomaly(True)
                W = F.to_tensor(data["words"], padding_value=pad_token_idx_word)
                T = F.to_tensor(data["pos"], padding_value=pad_token_idx_tag)
                for param in crf.parameters():
                    param.grad = None
                loss = crf.loss(T, W)
                loss.backward()
                optimizer.step()
        crf.eval()
        crf.transformer.eval()
        with torch.no_grad():
            predicted_sequences = []
            true_sequences = []
            for i_dev, data_dev in enumerate(dev_dataloader):
                W_dev = F.to_tensor(
                    data_dev["words"], padding_value=pad_token_idx_word
                )
                T_dev = F.to_tensor(
                    data_dev["pos"], padding_value=pad_token_idx_tag
                )
                sequence_viterbi = crf(W_dev)
                predicted_sequences += sequence_viterbi
                for ix in range(W_dev.shape[0]):
                    true_sequences += [
                        T_dev[ix, 1 : (sequence_viterbi[ix].shape[0] + 1)]
                    ]

            acc = torch.tensor(0.0)
            for ix in range(len(predicted_sequences)):
                acc += torch.mean(
                    (predicted_sequences[ix] == true_sequences[ix]).float()
                )
            acc = acc / len(predicted_sequences)
            print("-------------------------")
            print(f"Development set accuracy: {acc}")
            print("-------------------------")
        epoch += 1
    return acc
