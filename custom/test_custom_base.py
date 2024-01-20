import torch
from faster_rcnn.detectron2_extract import predict, build_model
import cv2
import sys
import os
base_path = '/'.join(sys.path[0].split('/')[:-1])
VLmodel_path = '/'.join([base_path,'VLModel'])
VLmodel_path_src = '/'.join([VLmodel_path,'src'])
sys.path.insert(0,base_path)
sys.path.insert(0,VLmodel_path)
sys.path.insert(0,VLmodel_path_src)
from VLModel.src.vrd_caption import Trainer
from utility.param import parse_args

############################################################################################
# img_path = '/home/zhangjx/All_model/VSD/custom/child_fireyhdrant.jpg'
# sub = 'surfboard'
# obj = 'person'
# relation_word = 'under'

# img_path = '/home/zhangjx/All_model/VSD/custom/women_dog.jpg'
# img_path = '/home/zhangjx/All_model/VSD/custom/woman_skate.jpg'
# img_path = '/home/zhangjx/All_model/VSD/custom/women_dog2.jpg'
# img_path = '/home/zhangjx/All_model/VSD/custom/person_surfboard.jpg'
def caption(args, gen_kwargs, faster_rcnn, vsd, prefix, caption_list):
    for img in caption_list:
        img_cv = cv2.imread(img[0])
        feature, boxes = predict(img_cv, faster_rcnn)
        boxes = boxes.tensor

        # vrd_predict = - predict_list.index(relation_word)  + vsd.tokenizer.convert_tokens_to_ids('<extra_id_1>')
        input_token = ' '.join([prefix, img[1], img[2]])
        # input_token_golden = ' '.join([prefix, img[1], f'<extra_id_{idx}>', img[2]])
        input_ids = vsd.tokenizer.encode(
                            input_token,
                            max_length=args.max_text_length, truncation=True)
        
        input_ids = torch.tensor(input_ids).to(vsd.model.device).unsqueeze(0)
        # input_ids_golden = torch.tensor(input_ids_golden).to(vsd.model.device).unsqueeze(0)
        
        # img_name = img[0].split('/')[-1][:-4]
        # s_box = box_dict[img_name][img[1]]
        # o_box = box_dict[img_name][img[2]]
        # input_ids_with_promt = predict_prompt(s_box, o_box, vsd, input_ids, feature, boxes)
        #########################################################################################
        # <extra_id_0>
        output = vsd.model.generate(
                            input_ids=input_ids,
                            vis_inputs=(feature.unsqueeze(0), boxes.unsqueeze(0)),
                            # return_dict_in_generate=True,
                            # output_attentions=True,
                            **gen_kwargs
                        )
        generated_sents = vsd.tokenizer.batch_decode(output, skip_special_tokens=True)
        print('input_id:',generated_sents)

        #########################################################################################
        # 给定方位提示词
        # output = vsd.model.generate(
        #                     input_ids=input_ids_golden,
        #                     vis_inputs=(feature.unsqueeze(0), boxes.unsqueeze(0)),
        #                     # return_dict_in_generate=True,
        #                     # output_attentions=True,
        #                     **gen_kwargs
        #                 )
        # generated_sents = vsd.tokenizer.batch_decode(output, skip_special_tokens=False)
        # print('input_ids_golden:',generated_sents)

        #########################################################################################
        # 预测方位提示词<extra_id_0>
        # output = vsd.model.generate(
        #                     input_ids=input_ids_with_promt,
        #                     vis_inputs=(feature.unsqueeze(0), boxes.unsqueeze(0)),
        #                     # return_dict_in_generate=True,
        #                     # output_attentions=True,
        #                     **gen_kwargs
        #                 )
        # generated_sents = vsd.tokenizer.batch_decode(output, skip_special_tokens=False)
        # print('input_ids_with_promt:',generated_sents)




if __name__ == '__main__':
    args = parse_args()
    gen_kwargs = {}
    gen_kwargs['num_beams'] = args.num_beams
    gen_kwargs['max_length'] = args.gen_max_length
    gen_kwargs['num_return_sequences'] = args.num_beams
    faster_rcnn = build_model()
    vsd = Trainer(args,train=False)
    prefix = 'describe image with tags and relation:'
    # extral_id_0 = '<extra_id_0>'
    caption_list = [
        ['/home/zhangjx/All_model/genration_scene/3DVSD/custom/0.jpg','Chopsticks','plate'],
        ['/home/zhangjx/All_model/genration_scene/3DVSD/custom/1.jpg','fan','books'],
        ['/home/zhangjx/All_model/genration_scene/3DVSD/custom/2.jpg','child', 'bag'],
        ['/home/zhangjx/All_model/genration_scene/3DVSD/custom/3.jpg','game mechine', 'light bar'],
        ['/home/zhangjx/All_model/genration_scene/3DVSD/custom/4.jpg','screen','mouse'],
        ['/home/zhangjx/All_model/genration_scene/3DVSD/custom/5.jpg','paper box','water pot'],
        ['/home/zhangjx/All_model/genration_scene/3DVSD/custom/6.jpg','toy pig','toy alligator'],
        ['/home/zhangjx/All_model/genration_scene/3DVSD/custom/7.jpg','bottle', 'screen'],
        ['/home/zhangjx/All_model/genration_scene/3DVSD/custom/8.jpg','keyboard', 'screen'],
        ['/home/zhangjx/All_model/genration_scene/3DVSD/custom/9.jpg','toy','stone'],
    ]

    caption(args, gen_kwargs, faster_rcnn, vsd, prefix, caption_list)