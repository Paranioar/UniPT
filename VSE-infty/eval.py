import os
import argparse
import logging
import torch
from lib import evaluation
os.environ['CUDA_VISIBLE_DEVICES']='0'

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='coco',  #--------------------- need to adjust ----------------------#
                        help='coco or f30k')
    parser.add_argument('--data_path', default='./data/coco') #--------------------- need to adjust ----------------------#
    parser.add_argument('--save_results', action='store_false')
    parser.add_argument('--evaluate_cxc', action='store_true')
    opt = parser.parse_args()

    if opt.dataset == 'coco':
        weights_bases = [
            'runs/save_model',
        ]
    elif opt.dataset == 'f30k':
        weights_bases = [
            'runs/save_model',
        ]
    else:
        raise ValueError('Invalid dataset argument {}'.format(opt.dataset))

    for base in weights_bases:
        logger.info('Evaluating {}...'.format(base))
        model_path = os.path.join(base, 'model_best.pth')

        if opt.save_results:  # Save the final results for computing ensemble results
            save_path = os.path.join(base, 'results_{}.npy'.format(opt.dataset))
        else:
            save_path = None

        if opt.dataset == 'coco':
            if not opt.evaluate_cxc:
                # Evaluate COCO 5-fold 1K
                evaluation.evalrank(model_path, data_path=opt.data_path, split='testall', fold5=True)
                # Evaluate COCO 5K
                evaluation.evalrank(model_path, data_path=opt.data_path, split='testall', fold5=False, save_path=save_path)
            else:
                # Evaluate COCO-trained models on CxC
                evaluation.evalrank(model_path, data_path=opt.data_path, split='testall', fold5=True, cxc=True)
        elif opt.dataset == 'f30k':
            # Evaluate Flickr30K
            evaluation.evalrank(model_path, data_path=opt.data_path, split='test', fold5=False, save_path=save_path)
        
        if torch.cuda.is_available():
            MB = 1024.0 * 1024.0
            peak_memory = (torch.cuda.max_memory_allocated() / MB) / 1000 
            logger.info("Memory utilization: %.3f GB" % peak_memory)


if __name__ == '__main__':
    main()
    print('finished')
