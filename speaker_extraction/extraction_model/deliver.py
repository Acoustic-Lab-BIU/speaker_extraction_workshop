import os
from data import  CreateFeatures_specific_sig
from utils import save_wave
import torch
from torch.utils.data import DataLoader
from model_def import Extraction_Model
from pathlib import Path
from omegaconf import OmegaConf

class Extractor:
    def __init__(self,parent_dir = '/home/bari/workspace/spring_winter_school/speaker_extraction',save_dir='') -> None:
        self.parent_dir = Path(parent_dir)
        self.ckpt_path = self.parent_dir/'epoch=374,val_loss=-12.62.ckpt'
        self.save_dir = self.parent_dir/'outputs/mono_s1' # dir of the signal
        self.path_mix = self.save_dir/'mix.wav'
        self.path_ref = self.save_dir/'ref2.wav'
        self.hp = OmegaConf.load(str(self.parent_dir/'extraction_model/config.yaml'))
        self.load_model()

    def load_model(self):
        ckpt = torch.load(self.ckpt_path, map_location="cpu")['state_dict']
        the_names = list(ckpt.keys())
        for name in the_names:
            new_name = name.split('.',1)[-1]
            ckpt[new_name] = ckpt.pop(name)

        self.model = Extraction_Model(self.hp)
        self.model.load_state_dict(ckpt)
        self.model.eval()

    def extract_wave(self,path_mix,path_ref):
        test_set = CreateFeatures_specific_sig(
                        self.hp,path_mix,path_ref, 1, train_mode=False)
        testloader = DataLoader(test_set, batch_size=1, shuffle=False,
                                num_workers=self.hp.dataloader.num_workers, pin_memory=self.hp.dataloader.pin_memory)
        # return testloader
        for (mixs,  ref1) in testloader: # mix: list[0-5,5-10,10-15,...]  ref: tensor
            i=0
            for mix in mixs:
                # mix/ref1 dims [1,2,129,-1]  -> [1,real-imaginary,frequncey,frames]
                Y_outputs,_, _, _ = self.model.forward(mix, ref1) #mix and ref1 same size
                y1_curr =  self.post_processing(Y_outputs)
        
                y1 = y1_curr if i==0 else torch.cat((y1,y1_curr),0)
                i +=1

            # ======== save results ========= # 
            save_wave(y1, os.path.join(self.save_dir, 'y_ckpt.wav'))
        
    def post_processing(self,Y_outputs):
        Y_output  = Y_outputs[-1]
        Y_com1 = Y_output[0,:,:] + 1j*Y_output[1,:,:]
        y1_curr = torch.istft(Y_com1, n_fft=self.hp.stft.fft_length,hop_length=self.hp.stft.fft_hop,window=torch.hamming_window(self.hp.stft.fft_length))
        return y1_curr

if __name__ == "__main__":
    e = Extractor()
    e.extract_wave(e.path_mix,e.path_ref)
# def pre_proccess(hp,ckpt_path,path_mix,path_ref):
#     #######load model########

#     # ckpt = torch.load(ckpt_path, map_location="cpu")['state_dict']
#     # the_names = list(ckpt.keys())
#     # for name in the_names:
#     #     new_name = name.split('.',1)[-1]
#     #     ckpt[new_name] = ckpt.pop(name)

#     # model = Extraction_Model(hp)
#     # model.load_state_dict(ckpt)
#     # model.eval()
    
#     #######load data########
#     # test_set = CreateFeatures_specific_sig(
#     #                 hp,path_mix,path_ref, 1, train_mode=False)
  
#     # testloader = DataLoader(test_set, batch_size=1, shuffle=False,
#     #                         num_workers=hp.dataloader.num_workers, pin_memory=hp.dataloader.pin_memory)

#     # return model,testloader


# # @hydra.main(config_path=str(parent_dir/'deliver'), config_name="config.yaml")
# def main(hp):
#     model,testloader =  pre_proccess(hp,str(ckpt_path),str(path_mix),str(path_ref))


#     for (mixs,  ref1) in testloader: # mix: list[0-5,5-10,10-15,...]  ref: tensor
#         i=0
#         for mix in mixs:
#             # mix/ref1 dims [1,2,129,-1]  -> [1,real-imaginary,frequncey,frames]
#             Y_outputs,_, _, _ = model.forward(mix, ref1) #mix and ref1 same size
#             y1_curr =  post_processing(hp,Y_outputs)
    
#             y1 = y1_curr if i==0 else torch.cat((y1,y1_curr),0)
#             i +=1

#         # ======== save results ========= # 
#         save_wave(y1, os.path.join(save_dir, 'y_ckpt.wav'))
        

# if __name__ == '__main__':

#     main(hp)
