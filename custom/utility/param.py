import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--load', type=str, default='/home/zhangjx/All_model/VSD/weights/end2end_t5_b80_<0>/BEST')
    # parser.add_argument('--load', type=str, default='/home/zhangjx/All_model/VSD/weights/end2end_t5_b80/BEST')
    parser.add_argument('--backbone', type=str, default='t5-base')
    parser.add_argument('--tokenizer', type=str, default='t5-base')
    parser.add_argument('--from_scratch', type=bool, default=False)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--fp16', type=bool, default=False)
    parser.add_argument('--multiGPU', type=bool, default=False)
    parser.add_argument('--distributed', type=bool, default=False)
    parser.add_argument('--num_beams', type=int, default=5)
    parser.add_argument('--gen_max_length', type=int, default=20)
    parser.add_argument('--feat_dim', type=int, default=2048)
    parser.add_argument('--pos_dim', type=int, default=4)
    parser.add_argument('--use_vis_order_embedding', type=bool, default=True)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--use_vis_layer_norm', type=bool, default=True)
    parser.add_argument('--individual_vis_layer_norm', type=bool, default=True)
    parser.add_argument('--losses', type=str, default='lm,obj,attr,feat')
    parser.add_argument('--share_vis_lang_layer_norm', type=bool, default=False)
    parser.add_argument('--classifier', type=bool, default=False)
    parser.add_argument('--use_vision', type=bool, default=True)
    parser.add_argument('--max_text_length', type=int, default=40)
    parser.add_argument('--do_lower_case', type=bool, default=False)





    parser = parser.parse_args()
    # parser = vars(parser)
    # parser = argparse.Namespace(**parser)
    # args = Config(**kwargs)
    return parser