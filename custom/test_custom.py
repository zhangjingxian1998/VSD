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
args = parse_args()
gen_kwargs = {}
gen_kwargs['num_beams'] = args.num_beams
gen_kwargs['max_length'] = args.gen_max_length
faster_rcnn = build_model()
vsd = Trainer(args,train=False)
prefix = 'describe image with tags and relation:'
extral_id_0 = '<extral_id_0>'
predict_list = ["on", "to the left of", "under", "behind", "to the right of", "in", "next to", "in front of", "above"]

caption_list = [
    ['/home/zhangjx/All_model/VSD/custom/child_fireyhdrant.jpg','child','fire hydrant','to the right of'],
    ['/home/zhangjx/All_model/VSD/custom/women_dog.jpg','woman','dog','to the left of'],
    ['/home/zhangjx/All_model/VSD/custom/woman_skate.jpg','woman', 'skate', 'under'],
    ['/home/zhangjx/All_model/VSD/custom/woman_dog2.jpg','woman', 'dog', 'in front of'],
    ['/home/zhangjx/All_model/VSD/custom/person_surfboard.jpg','surfboard','person','under']
]
############################################################################################
# img_path = '/home/zhangjx/All_model/VSD/custom/child_fireyhdrant.jpg'
# sub = 'surfboard'
# obj = 'person'
# relation_word = 'under'

# img_path = '/home/zhangjx/All_model/VSD/custom/women_dog.jpg'
# img_path = '/home/zhangjx/All_model/VSD/custom/woman_skate.jpg'
# img_path = '/home/zhangjx/All_model/VSD/custom/women_dog2.jpg'
# img_path = '/home/zhangjx/All_model/VSD/custom/person_surfboard.jpg'
for img in caption_list:
    img_cv = cv2.imread(img[0])
    feature, boxes = predict(img_cv, faster_rcnn)
    boxes = boxes.tensor

    # vrd_predict = - predict_list.index(relation_word)  + vsd.tokenizer.convert_tokens_to_ids('<extra_id_1>')
    idx = predict_list.index(img[3]) + 1
    input_token = ' '.join([prefix, img[1], extral_id_0, img[2]])
    input_token_golden = ' '.join([prefix, img[1], f'<extral_id_{idx}>', img[2]])
    input_ids = vsd.tokenizer.encode(
                        input_token,
                        max_length=args.max_text_length, truncation=True)
    input_ids_golden = vsd.tokenizer.encode(
                        input_token_golden,
                        max_length=args.max_text_length, truncation=True)
    output = vsd.model.generate(
                        input_ids=torch.tensor(input_ids).to(vsd.model.device).unsqueeze(0),
                        vis_inputs=(feature.unsqueeze(0), boxes.unsqueeze(0)),
                        # return_dict_in_generate=True,
                        # output_attentions=True,
                        **gen_kwargs
                    )
    generated_sents = vsd.tokenizer.batch_decode(output, skip_special_tokens=True)
    print(generated_sents)
    output = vsd.model.generate(
                        input_ids=torch.tensor(input_ids_golden).to(vsd.model.device).unsqueeze(0),
                        vis_inputs=(feature.unsqueeze(0), boxes.unsqueeze(0)),
                        # return_dict_in_generate=True,
                        # output_attentions=True,
                        **gen_kwargs
                    )
    generated_sents = vsd.tokenizer.batch_decode(output, skip_special_tokens=True)
    print(generated_sents)
    pass