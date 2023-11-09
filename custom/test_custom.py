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

def box_transfer(box):
    '''
    [left_top, right_down] --> [miny, maxy, minx, maxx]
    '''
    miny = box[1]
    maxy = box[3]
    minx = box[0]
    maxx = box[2]
    
    return [miny, maxy, minx, maxx]

def bbox_embed(box):
    y,y2,x,x2 = box

    w = abs(x - x2)
    h = abs(y - y2)

    sr1 = 1.0 * x / w
    sr2 = 1.0 * y / h
    sr3 = 1.0 * x2 / w
    sr4 = 1.0 * y2 / h

    return [sr1, sr2, sr3, sr4]

def predict_prompt(s_box, o_box, vsd, input_ids, feature, boxes):
    s_box = box_transfer(s_box)
    s_box = bbox_embed(s_box)

    o_box = box_transfer(o_box)
    o_box = bbox_embed(o_box)
    extra_id_mask = input_ids>32000
    so_box = torch.zeros([1,input_ids.size(1),2,4])
    so_box[extra_id_mask] = torch.tensor([s_box, o_box])
    so_box = so_box.to(vsd.model.device)

    # so_box = torch.tensor([s_box, o_box]).unsqueeze(0).unsqueeze(0).to(vsd.model.device) # [1,1,2,4]

    s_bbox_h = vsd.model.sbbox_encode(so_box[:,:,0,:]) # in [1,1,4] out [1,1,64]
    o_bbox_h = vsd.model.obbox_encode(so_box[:,:,1,:])
    bbox_h = s_bbox_h + o_bbox_h
    bbox_h = vsd.model.bbox_cls(bbox_h)

    prompt_output = vsd.model(
            input_ids=input_ids,
            vis_inputs=(feature.unsqueeze(0), boxes.unsqueeze(0)),
            only_encoder=True
        )
    sequence_output = prompt_output["encoder_last_hidden_state"][:, :input_ids.size(1), :]
    vrd_h = torch.cat((sequence_output, bbox_h), dim=-1)
    vrd_logits = vsd.model.vrd_cls(vrd_h)

    vrd_predict = torch.argmax(vrd_logits, dim=-1)
    vrd_predict = - vrd_predict + vsd.tokenizer.convert_tokens_to_ids('<extra_id_1>')
    rel_mask = torch.zeros_like(input_ids)
    rel_mask[extra_id_mask] = 1
    input_ids_with_promt = vrd_predict * rel_mask + input_ids * (1 - rel_mask)

    return input_ids_with_promt


############################################################################################
# img_path = '/home/zhangjx/All_model/VSD/custom/child_fireyhdrant.jpg'
# sub = 'surfboard'
# obj = 'person'
# relation_word = 'under'

# img_path = '/home/zhangjx/All_model/VSD/custom/women_dog.jpg'
# img_path = '/home/zhangjx/All_model/VSD/custom/woman_skate.jpg'
# img_path = '/home/zhangjx/All_model/VSD/custom/women_dog2.jpg'
# img_path = '/home/zhangjx/All_model/VSD/custom/person_surfboard.jpg'
def caption(args, gen_kwargs, faster_rcnn, vsd, prefix, extral_id_0, predict_list, caption_list, box_dict):
    for img in caption_list:
        img_cv = cv2.imread(img[0])
        feature, boxes = predict(img_cv, faster_rcnn)
        boxes = boxes.tensor

        # vrd_predict = - predict_list.index(relation_word)  + vsd.tokenizer.convert_tokens_to_ids('<extra_id_1>')
        idx = predict_list.index(img[3]) + 1
        input_token = ' '.join([prefix, img[1], '<extra_id_0>', img[2]])
        input_token_golden = ' '.join([prefix, img[1], f'<extra_id_{idx}>', img[2]])
        input_ids = vsd.tokenizer.encode(
                            input_token,
                            max_length=args.max_text_length, truncation=True)
        input_ids_golden = vsd.tokenizer.encode(
                            input_token_golden,
                            max_length=args.max_text_length, truncation=True)
        
        input_ids = torch.tensor(input_ids).to(vsd.model.device).unsqueeze(0)
        input_ids_golden = torch.tensor(input_ids_golden).to(vsd.model.device).unsqueeze(0)
        
        img_name = img[0].split('/')[-1][:-4]
        s_box = box_dict[img_name][img[1]]
        o_box = box_dict[img_name][img[2]]
        input_ids_with_promt = predict_prompt(s_box, o_box, vsd, input_ids, feature, boxes)
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
        output = vsd.model.generate(
                            input_ids=input_ids_golden,
                            vis_inputs=(feature.unsqueeze(0), boxes.unsqueeze(0)),
                            # return_dict_in_generate=True,
                            # output_attentions=True,
                            **gen_kwargs
                        )
        generated_sents = vsd.tokenizer.batch_decode(output, skip_special_tokens=True)
        print('input_ids_golden:',generated_sents)

        #########################################################################################
        # 预测方位提示词<extra_id_0>
        output = vsd.model.generate(
                            input_ids=input_ids_with_promt,
                            vis_inputs=(feature.unsqueeze(0), boxes.unsqueeze(0)),
                            # return_dict_in_generate=True,
                            # output_attentions=True,
                            **gen_kwargs
                        )
        generated_sents = vsd.tokenizer.batch_decode(output, skip_special_tokens=True)
        print('input_ids_with_promt:',generated_sents)




if __name__ == '__main__':
    args = parse_args()
    gen_kwargs = {}
    gen_kwargs['num_beams'] = args.num_beams
    gen_kwargs['max_length'] = args.gen_max_length
    faster_rcnn = build_model()
    vsd = Trainer(args,train=False)
    prefix = 'describe image with tags and relation:'
    extral_id_0 = '<extra_id_0>'
    predict_list = ["on", "to the left of", "under", "behind", "to the right of", "in", "next to", "in front of", "above"]

    caption_list = [
        ['/home/zhangjx/All_model/VSD/custom/child_fireyhdrant.jpg','child','fire hydrant','to the right of'],
        ['/home/zhangjx/All_model/VSD/custom/women_dog.jpg','woman','dog','to the left of'],
        ['/home/zhangjx/All_model/VSD/custom/woman_skate.jpg','woman', 'skate', 'under'],
        ['/home/zhangjx/All_model/VSD/custom/woman_dog2.jpg','woman', 'dog', 'in front of'],
        ['/home/zhangjx/All_model/VSD/custom/person_surfboard.jpg','surfboard','person','under'],

        ['/home/zhangjx/All_model/VSD/custom/child_fireyhdrant.jpg','fire hydrant', 'child','to the left of'],
        ['/home/zhangjx/All_model/VSD/custom/women_dog.jpg','dog', 'woman','to the right of'],
        ['/home/zhangjx/All_model/VSD/custom/woman_skate.jpg','skate','woman', 'above'],
        ['/home/zhangjx/All_model/VSD/custom/woman_dog2.jpg','dog', 'woman', 'behind'],
        ['/home/zhangjx/All_model/VSD/custom/person_surfboard.jpg','person', 'surfboard','on']
    ]
    box_dict = {
        'child_fireyhdrant':{'child':[443,137,515,273],'fire hydrant':[296,223,361,309]},
        'women_dog':{'woman':[119,111,227,359],'dog':[362,275,200,411]},
        'woman_skate':{'woman':[385,425,235,117],'skate':[429,78,202,204]},
        'woman_dog2':{'woman':[335,80,169,585],'dog':[293,296,390,438]},
        'person_surfboard':{'person':[232,67,366,315],'surfboard':[412,350,280,278]},
    }

    caption(args, gen_kwargs, faster_rcnn, vsd, prefix, extral_id_0, predict_list, caption_list, box_dict)