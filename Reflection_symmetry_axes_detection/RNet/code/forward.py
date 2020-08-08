from libs.utils import *

class Forward_Model(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def run_detector(self, img, line_pts, step, model):
        out = create_test_dict()

        # semantic line detection (D-Net)
        feat1, feat2 = model.feature_extraction(img)

        for i in range(len(step) - 1):
            start, end = step[i], step[i + 1]
            batch_line_pts = line_pts[:, start:end, :]

            batch_out = model(img=img,
                              line_pts=batch_line_pts,
                              feat1=feat1, feat2=feat2,
                              is_training=False)

            out['cls'] = torch.cat((out['cls'], batch_out['cls']), dim=0)
            out['reg'] = torch.cat((out['reg'], batch_out['reg']), dim=0)

        return out

    def run_feature_extractor(self, img, line_pts, model):

        out = model(img=img,
                    line_pts=line_pts,
                    with_comparator=True,
                    is_training=False)

        return out

    def run_comparator(self, feat1, feat2, model):

        out = model(feat1, feat2)

        return out
