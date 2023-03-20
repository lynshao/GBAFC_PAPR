import torch, time, pdb, os, random, math
import numpy as np
import torch.nn as nn
from utils import *
from phy import *
from nn_layers import *
from parameters import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.io

################## Optimizer Libraries ########################
#from apex.optimizers import FusedLAMB # Nvidia implementation of Bert with lamb
import torch_optimizer as optim
###############################################################


################################## Distributed training approach #######################################################
def ModelAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


########################## This is the overall AutoEncoder model ########################


class AE(nn.Module):
    def __init__(self, args):
        super(AE, self).__init__()
        self.args = args
        ################## We use learnable positional encoder which can be removed later ######################################
        #self.pe = PositionalEncoder(SeqLen=self.args.K+1, lenWord=args.d_model_trx) # learnable PE
        self.pe = PositionalEncoder_fixed()
        ########################################################################################################################
        self.Tmodel = BERT("trx", args.m+2*(args.T-1), args.m, args.d_model_trx, args.N_trx, args.heads_trx, args.dropout, args.custom_attn,args.multclass, args.NS_model, args.isrelu)
        self.Rmodel = BERT("rec", args.T, args.m, args.d_model_trx, args.N_trx+1, args.heads_trx, args.dropout, args.custom_attn,args.multclass, args.NS_model, args.isrelu)
        ########## Power Reallocation as in deepcode work ###############
        if self.args.reloc == 1:
            self.total_power_reloc = Power_reallocate(args)
        ########## Power Reallocation as in deepcode work ###############
        self.phylayer = PHY(args)

    def power_constraint(self, inputs, isTraining, eachbatch, idx = 0): # Normalize through batch dimension
        # this_mean = torch.mean(inputs, 0)
        # this_std  = torch.std(inputs, 0)
        if isTraining == 1:
            # training
            this_mean = torch.mean(inputs, 0)
            this_std  = torch.std(inputs, 0)
        elif isTraining == 0:
            # test
            if eachbatch == 0:
                this_mean = torch.mean(inputs, 0)
                this_std  = torch.std(inputs, 0)
                if not os.path.exists('statistics'):
                    os.mkdir('statistics')
                torch.save(this_mean, 'statistics/this_mean' + str(idx))
                torch.save(this_std, 'statistics/this_std' + str(idx))
                print('this_mean and this_std saved ...')
            else:
                this_mean = torch.load('statistics/this_mean' + str(idx))
                this_std = torch.load('statistics/this_std' + str(idx))

        outputs = (inputs - this_mean)*1.0/ (this_std + 1e-8)
        return outputs

    ########### IMPORTANT ##################
    # We use unmodulated bits at encoder
    #######################################
    def forward(self, eachbatch, bVec, isTraining = 1):
        combined_noise = torch.zeros(size = (args.batchSize, args.ell, args.T), requires_grad=False).to(self.args.device)

        bVec_md = 2*bVec-1
        for idx in range(self.args.T): # Go through T interactions
            if idx == 0: # phase 0 
            	src = torch.cat([bVec_md, torch.zeros(self.args.batchSize, self.args.ell, 2*(self.args.T-1)).to(self.args.device)],dim=2)
            elif idx == self.args.T-1:
            	src = torch.cat([bVec_md, parity_all, combined_noise[:,:,:idx]],dim=2)
            else:
            	src = torch.cat([bVec_md, parity_all, torch.zeros(self.args.batchSize, args.ell, self.args.T-(idx+1) ).to(self.args.device),combined_noise[:,:,:idx],torch.zeros(self.args.batchSize, args.ell, self.args.T-(idx+1) ).to(self.args.device)],dim=2)
            ############# Generate the output ###################################################
            output = self.Tmodel(src, None, self.pe)
            parity = self.power_constraint(output, isTraining, eachbatch, idx) 
            parity = self.total_power_reloc(parity,idx) # parity.shape = (BS, ell, 1)

            if self.args.train == 0:
                PAPR, noise_A, data_r_B = self.phylayer(parity)
            else:
                PAPR = torch.ones(size=(1,2)).to(self.args.device)
                noise_std1 = 10 ** (-self.args.snr1 * 1.0 / 10 / 2)
                noise_std2 = 10 ** (-self.args.snr2 * 1.0 / 10 / 2)
                noise1 = torch.normal(0, std=noise_std1, size=(self.args.batchSize, self.args.ell), requires_grad=False).to(self.args.device)
                noise2 = torch.normal(0, std=noise_std2, size=(self.args.batchSize, self.args.ell), requires_grad=False).to(self.args.device)
                if self.args.snr2 == 100:
                    noise_B = noise_B * 0
                noise_A = noise1 + noise2
                data_r_B = parity + noise1.unsqueeze(-1)

            combined_noise[:,:,idx] = noise_A

            if idx == 0:
                PAPRarray = PAPR
                parity_all = parity
                received = data_r_B
            else:
                PAPRarray = torch.cat([PAPRarray, PAPR],dim=0)
                parity_all = torch.cat([parity_all, parity], dim=2)
                received = torch.cat([received, data_r_B], dim = 2)

        # ------------------------------------------------------------ receiver
        #print(received.shape)
        decSeq = self.Rmodel(received, None, self.pe) # Decode the sequence

        return decSeq, PAPRarray




############################################################################################################################################################################
def train_model(model, args):
    print("-->-->-->-->-->-->-->-->-->--> start training ...")
    model.train()
    start = time.time()
    epoch_loss_record = []
    flag = 0
    map_vec = 2**(torch.arange(args.m))

    # in each run, randomly sample a batch of data from the training dataset
    for eachbatch in range(10*args.totalbatch):
        bVec = torch.randint(0, 2, (args.batchSize, args.ell, args.m))

        if np.mod(eachbatch, args.core) == 0:
            w_locals = []
            w0 = model.state_dict()
            w0 = copy.deepcopy(w0)
        else:
            # Use the common model to have a large batch strategy
            model.load_state_dict(w0)

        # feed into model to get predictions
        preds, PAPRarray = model(eachbatch, bVec.to(args.device), isTraining=1)

        PAPRloss = PAPRarray.mean()

        args.optimizer.zero_grad()
        if args.multclass:
            bVec_mc = torch.matmul(bVec,map_vec)
            ys = bVec_mc.long().contiguous().view(-1)
        else:
            # expand the labels (bVec) in a batch to a vector, each word in preds should be a 0-1 distribution
            ys = bVec.long().contiguous().view(-1)

        preds = preds.contiguous().view(-1, preds.size(-1)) #=> (Batch*K) x 2
        preds = torch.log(preds)

        # if args.lamb == 0.0:
        loss = F.nll_loss(preds, ys.to(args.device))
        # else:
        #     loss = F.nll_loss(preds, ys.to(args.device)) + args.lamb * PAPRloss
        loss.backward()
        ####################### Gradient Clipping optional ###########################
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_th)
        ##############################################################################
        args.optimizer.step()
        # Save the model
        w1 = model.state_dict()
        w_locals.append(copy.deepcopy(w1))
        ###################### untill core number of iterations are completed ####################
        if np.mod(eachbatch, args.core) != args.core - 1:
            continue
        else:
            ########### When core number of models are obtained #####################
            w2 = ModelAvg(w_locals) # Average the models
            model.load_state_dict(copy.deepcopy(w2))
            ##################### change the learning rate ##########################
            if args.use_lr_schedule:
                args.scheduler.step()
        ################################ Observe test accuracy ##############################
        with torch.no_grad():
            probs, decodeds = preds.max(dim=1)
            succRate = sum(decodeds == ys.to(args.device)) / len(ys)
            print('Idx,loss,BER,num,PAPR=', (
            eachbatch, round(loss.item(), 4), round(1 - succRate.item(), 6),
            sum(decodeds != ys.to(args.device)).item(), round((10 * torch.log10(PAPRloss).item()), 2)))
        ####################################################################################

        if np.mod(eachbatch, args.core * 2000) == args.core - 1 and eachbatch >= 10000:
            if not os.path.exists('weights'):
                os.mkdir('weights')
            saveDir = 'weights/model_weights' + str(eachbatch)
            torch.save(model.state_dict(), saveDir)

        # if np.mod(eachbatch, args.core * 10000) == args.core - 1 and eachbatch >= 40000:
        if eachbatch == args.totalbatch - 1:
            if not os.path.exists('weights'):
                os.mkdir('weights')
            # saveDir = 'weights/model_weights' + str(eachbatch)
            saveDir = 'weights/model_weights'
            torch.save(model.state_dict(), saveDir)


def EvaluateNets(model, args):
    checkpoint = torch.load(args.saveDir)
    # # ======================================================= load weights
    model.load_state_dict(checkpoint)
    model.eval()
    complexell = math.ceil(args.ell/2)
    map_vec = 2**(torch.arange(args.m))

    args.numTestbatch = 100000000

    # failbits = torch.zeros(args.K).to(args.device)
    bitErrors = 0
    pktErrors = 0
    for eachbatch in range(args.numTestbatch):
        # generate b sequence and zero padding
        bVec = torch.randint(0, 2, (args.batchSize, args.ell, args.m))

        # feed into model to get predictions
        with torch.no_grad():
            # preds = model(eachbatch, bVec.to(args.device), fwd_noise_par.to(args.device), fb_noise_par.to(args.device), fwd_chan, fb_chan, isTraining=0)
            preds, PAPRarray = model(eachbatch, bVec.to(args.device), isTraining=0)
            if args.multclass:
                bVec_mc = torch.matmul(bVec,map_vec)
                ys = bVec_mc.long().contiguous().view(-1)
            else:
                ys = bVec.long().contiguous().view(-1)
            preds1 =  preds.contiguous().view(-1, preds.size(-1))
            #print(preds1.shape)
            probs, decodeds = preds1.max(dim=1)
            decisions = decodeds != ys.to(args.device)
            bitErrors += decisions.sum()
            BER = bitErrors / (eachbatch + 1) / args.batchSize / args.ell
            pktErrors += decisions.view(args.batchSize, args.ell).sum(1).count_nonzero()
            PER = pktErrors / (eachbatch + 1) / args.batchSize
            PAPRarray = 10 * torch.log10(PAPRarray)
            PAPRarray = PAPRarray.cpu().numpy()
            if eachbatch == 0:
                scipy.io.savemat("PAPR.mat", {"data": PAPRarray})
            print('num, BER, errors, PER, PAPR = ', eachbatch, round(BER.item(), 10), bitErrors.item(),
                  round(PER.item(), 10), round(PAPRarray.mean().item(), 4))

    BER = bitErrors.cpu() / (args.numTestbatch * args.batchSize * args.m * args.ell)
    PER = pktErrors.cpu() / (args.numTestbatch * args.batchSize)
    print(BER)
    print("Final test BER = ", torch.mean(BER).item())
    pdb.set_trace()


if __name__ == '__main__':
    # ======================================================= parse args
    args = args_parser()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ########### path for saving model checkpoints ################################
    args.saveDir = 'weights/model_weights'  # path to be saved to
    ################## Model size part ###########################################
    args.N = math.ceil(args.ell/2)
    args.d_model_trx = args.heads_trx * args.d_k_trx # total number of features
    # ======================================================= Initialize the model
    model = AE(args).to(args.device)
    if args.device == 'cuda':
        model = torch.nn.DataParallel(model)
        torch.backends.cudnn.benchmark = True


    # ======================================================= run
    if args.train == 1:
        if args.opt_method == 'adamW':
        	args.optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.wd, amsgrad=False)
        elif args.opt_method == 'lamb':
        	args.optimizer = optim.Lamb(model.parameters(),lr= 1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.wd)
        else:
        	args.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
        if args.use_lr_schedule:
        	lambda1 = lambda epoch: (1-epoch/args.totalbatch)
        	args.scheduler = torch.optim.lr_scheduler.LambdaLR(args.optimizer, lr_lambda=lambda1)
        	######################## huggingface library ####################################################
        	#args.scheduler = get_polynomial_decay_schedule_with_warmup(optimizer=args.optimizer, warmup_steps=1000, num_training_steps=args.totalbatch, power=0.5)


        if 0:
            checkpoint = torch.load(args.saveDir)
            model.load_state_dict(checkpoint)
            print("================================ Successfully load the pretrained data!")

        train_model(model, args)
    else:
        EvaluateNets(model, args)