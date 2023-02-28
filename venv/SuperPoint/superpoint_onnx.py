import argparse
import glob
import numpy as np
import os
import time
import cv2
import onnx.helper
import torch
import matplotlib.pyplot as plt
import math
import pandas as pd
import collections
import torch.onnx
import onnxruntime

class PointTracker(object):
    """ Class to manage a fixed memory of points and descriptors that enables
    sparse optical flow point tracking.
    Internally, the tracker stores a 'tracks' matrix sized M x (2+L), of M
    tracks with maximum length L, where each row corresponds to:
    row_m = [track_id_m, avg_desc_score_m, point_id_0_m, ..., point_id_L-1_m].
    """

    def __init__(self, max_length, nn_thresh):
        if max_length < 2:
            raise ValueError('max_length must be greater than or equal to 2.')
        self.maxl = max_length
        self.nn_thresh = nn_thresh
        self.all_pts = []
        for n in range(self.maxl):
            self.all_pts.append(np.zeros((2, 0)))
        self.last_desc = None
        self.tracks = np.zeros((0, self.maxl + 2)) #(count, 7)
        self.track_count = 0
        self.max_score = 9999
        self.pt_list = []

    def nn_match_two_way(self, desc1, desc2, nn_thresh):
        """
        Performs two-way nearest neighbor matching of two sets of descriptors, such
        that the NN match from descriptor A->B must equal the NN match from B->A.
        Inputs:
          desc1 - NxM numpy matrix of N corresponding M-dimensional descriptors.
          desc2 - NxM numpy matrix of N corresponding M-dimensional descriptors.
          nn_thresh - Optional descriptor distance below which is a good match.
        Returns:
          matches - 3xL numpy array, of L matches, where L <= N and each column i is
                    a match of two descriptors, d_i in image 1 and d_j' in image 2:
                    [d_i index, d_j' index, match_score]^T
        """
        assert desc1.shape[0] == desc2.shape[0]
        if desc1.shape[1] == 0 or desc2.shape[1] == 0: #descriptor 정보가 없을 경우 matchs는 빈거 return
            return np.zeros((3, 0))
        if nn_thresh < 0.0:
            raise ValueError('\'nn_thresh\' should be non-negative')
        # Compute L2 distance. Easy since vectors are unit normalized.
        dmat = np.dot(desc1.T, desc2)
        dmat = np.sqrt(2 - 2 * np.clip(dmat, -1, 1))
        # Get NN indices and scores.
        idx = np.argmin(dmat, axis=1)
        scores = dmat[np.arange(dmat.shape[0]), idx]
        # Threshold the NN matches.
        keep = scores < nn_thresh
        # Check if nearest neighbor goes both directions and keep those.
        idx2 = np.argmin(dmat, axis=0)
        keep_bi = np.arange(len(idx)) == idx2[idx]
        keep = np.logical_and(keep, keep_bi)
        idx = idx[keep]
        scores = scores[keep]
        # Get the surviving point indices.
        m_idx1 = np.arange(desc1.shape[1])[keep]
        m_idx2 = idx
        # Populate the final 3xN match data structure.
        matches = np.zeros((3, int(keep.sum())))
        matches[0, :] = m_idx1
        matches[1, :] = m_idx2
        matches[2, :] = scores
        return matches

    def get_offsets(self):
        """ Iterate through list of points and accumulate an offset value. Used to
        index the global point IDs into the list of points.
        Returns
          offsets - N length array with integer offset locations.
        """
        # Compute id offsets.
        offsets = []
        offsets.append(0)
        for i in range(len(self.all_pts) - 1):  # Skip last camera size, not needed.
            offsets.append(self.all_pts[i].shape[1])
        offsets = np.array(offsets)
        offsets = np.cumsum(offsets)
        return offsets

    def update(self, pts, desc):
        """ Add a new set of point and descriptor observations to the tracker.
        Inputs
          pts - 3xN numpy array of 2D point observations.
          desc - DxN numpy array of corresponding D dimensional descriptors.
        """
        if pts is None or desc is None:
            print('PointTracker: Warning, no points were added to tracker.')
            return
        assert pts.shape[1] == desc.shape[1]
        # Initialize last_desc.
        if self.last_desc is None:
            self.last_desc = np.zeros((desc.shape[0], 0))
        # Remove oldest points, store its size to update ids later.
        remove_size = self.all_pts[0].shape[1]
        self.all_pts.pop(0)
        self.all_pts.append(pts)
        # Remove oldest point in track.
        self.tracks = np.delete(self.tracks, 2, axis=1)
        # Update track offsets.
        for i in range(2, self.tracks.shape[1]):
            self.tracks[:, i] -= remove_size
        self.tracks[:, 2:][self.tracks[:, 2:] < -1] = -1
        offsets = self.get_offsets()
        # Add a new -1 column.
        self.tracks = np.hstack((self.tracks, -1 * np.ones((self.tracks.shape[0], 1))))
        # Try to append to existing tracks.
        matched = np.zeros((pts.shape[1])).astype(bool)
        matches = self.nn_match_two_way(self.last_desc, desc, self.nn_thresh)
        for match in matches.T:
            # Add a new point to it's matched track.
            id1 = int(match[0]) + offsets[-2]
            id2 = int(match[1]) + offsets[-1]
            found = np.argwhere(self.tracks[:, -2] == id1) #id1의 index
            if found.shape[0] > 0:
                matched[int(match[1])] = True
                row = int(found)
                self.tracks[row, -1] = id2
                if self.tracks[row, 1] == self.max_score:
                    # Initialize track score.
                    self.tracks[row, 1] = match[2]
                else:
                    # Update track score with running average.
                    # NOTE(dd): this running average can contain scores from old matches
                    #           not contained in last max_length track points.
                    track_len = (self.tracks[row, 2:] != -1).sum() - 1.
                    frac = 1. / float(track_len)
                    self.tracks[row, 1] = (1. - frac) * self.tracks[row, 1] + frac * match[2]
        # Add unmatched tracks.
        new_ids = np.arange(pts.shape[1]) + offsets[-1]
        new_ids = new_ids[~matched] #여기서 descriptor 매칭된거만 남기는거쥐
        new_tracks = -1 * np.ones((new_ids.shape[0], self.maxl + 2))
        new_tracks[:, -1] = new_ids
        new_num = new_ids.shape[0]
        new_trackids = self.track_count + np.arange(new_num)
        new_tracks[:, 0] = new_trackids
        new_tracks[:, 1] = self.max_score * np.ones(new_ids.shape[0])
        self.tracks = np.vstack((self.tracks, new_tracks))
        self.track_count += new_num  # Update the track count.
        # Remove empty tracks.
        keep_rows = np.any(self.tracks[:, 2:] >= 0, axis=1)
        self.tracks = self.tracks[keep_rows, :]
        # Store the last descriptors.
        self.last_desc = desc.copy()
        return

    def get_tracks(self, min_length):
        """ Retrieve point tracks of a given minimum length.
        Input
          min_length - integer >= 1 with minimum track length
        Output
          returned_tracks - M x (2+L) sized matrix storing track indices, where
            M is the number of tracks and L is the maximum track length.
        """
        if min_length < 1:
            raise ValueError('\'min_length\' too small.')
        valid = np.ones((self.tracks.shape[0])).astype(bool)
        good_len = np.sum(self.tracks[:, 2:] != -1, axis=1) >= min_length
        # Remove tracks which do not have an observation in most recent frame.
        not_headless = (self.tracks[:, -1] != -1)
        keepers = np.logical_and.reduce((valid, good_len, not_headless))
        returned_tracks = self.tracks[keepers, :].copy()
        return returned_tracks

    def draw_tracks(self,tracks):
        """ Visualize tracks all overlayed on a single image.
        Inputs
          out - numpy uint8 image sized HxWx3 upon which tracks are overlayed.
          tracks - M x (2+L) sized matrix storing track info.
        """
        # Store the number of points per camera.
        pts_mem = self.all_pts
        N = len(pts_mem)  # Number of cameras/images.
        # Get offset ids needed to reference into pts_mem.
        offsets = self.get_offsets()
        # Width of track and point circles to be drawn.
        stroke = 1

        # Iterate through each track and draw it.
        for track in tracks:
            for i in range(N - 1):
                if track[i + 2] == -1 or track[i + 3] == -1:
                    continue
                offset1 = offsets[i]
                offset2 = offsets[i + 1]
                idx1 = int(track[i + 2] - offset1)
                idx2 = int(track[i + 3] - offset2)
                pt1 = pts_mem[i][:2, idx1]
                pt2 = pts_mem[i + 1][:2, idx2]
                p1 = (int(round(pt1[0])), int(round(pt1[1])))
                p2 = (int(round(pt2[0])), int(round(pt2[1])))
                self.pt_list.append((p1[0], p1[1], p2[0], p2[1]))

    def reset(self):
        self.last_desc = None
        self.tracks = np.zeros((0, self.maxl + 2))
        self.track_count = 0
        self.max_score = 9999
        self.pt_list = []

class Superpoint(object):

    def __init__(self, onnx_path, rsize, nn_thresh, conf_thresh, nms_dist):
        self.onnx_path = onnx_path
        self.rsize = rsize
        self.scale_h = 0
        self.scale_w = 0
        self.cell = 8
        self.border_remove = 4
        self.nn_thresh = nn_thresh
        self.conf_thresh = conf_thresh
        self.nms_dist = nms_dist
        self.top = None
        self.bot = None
        self.fh = 0
        self.fw = 0

    def pre_processing(self, img):
        self.fh, self.fw = fullimg.shape
        top = fullimg[:self.fw, :]
        bot = fullimg[-self.fw:, :]
        self.scale_h = self.rsize / self.fh
        self.scale_w = self.rsize / self.fw

        top_r = cv2.resize(top, (self.rsize, self.rsize), interpolation=cv2.INTER_AREA)
        bot_r = cv2.resize(bot, (self.rsize, self.rsize), interpolation=cv2.INTER_AREA)
        top_r = cv2.GaussianBlur(top_r, (3, 3), 2)
        bot_r = cv2.GaussianBlur(bot_r, (3, 3), 2)
        top_r = top_r.astype('float32') / 255.
        bot_r = bot_r.astype('float32') / 255.

        return top_r, bot_r

    def nms_fast(self, in_corners, H, W, dist_thresh):
        grid = np.zeros((H, W)).astype(int)  # Track NMS data.
        inds = np.zeros((H, W)).astype(int)  # Store indices of points.
        # Sort by confidence and round to nearest int.
        inds1 = np.argsort(-in_corners[2, :])
        corners = in_corners[:, inds1]
        rcorners = corners[:2, :].round().astype(int)  # Rounded corners.
        # Check for edge case of 0 or 1 corners.
        if rcorners.shape[1] == 0:
            return np.zeros((3, 0)).astype(int), np.zeros(0).astype(int)
        if rcorners.shape[1] == 1:
            out = np.vstack((rcorners, in_corners[2])).reshape(3, 1)
            return out, np.zeros((1)).astype(int)
        # Initialize the grid.
        for i, rc in enumerate(rcorners.T):
            grid[rcorners[1, i], rcorners[0, i]] = 1
            inds[rcorners[1, i], rcorners[0, i]] = i
        # Pad the border of the grid, so that we can NMS points near the border.
        pad = dist_thresh
        grid = np.pad(grid, ((pad, pad), (pad, pad)), mode='constant')
        # Iterate through points, highest to lowest conf, suppress neighborhood.
        count = 0
        for i, rc in enumerate(rcorners.T):
            # Account for top and left padding.
            pt = (rc[0] + pad, rc[1] + pad)
            if grid[pt[1], pt[0]] == 1:  # If not yet suppressed.
                grid[pt[1] - pad:pt[1] + pad + 1, pt[0] - pad:pt[0] + pad + 1] = 0
                grid[pt[1], pt[0]] = -1
                count += 1
        # Get all surviving -1's and return sorted array of remaining corners.
        keepy, keepx = np.where(grid == -1)
        keepy, keepx = keepy - pad, keepx - pad
        inds_keep = inds[keepy, keepx]
        out = corners[:, inds_keep] #nms로 남은애들
        values = out[-1, :]  #남은애들 중에서 score
        inds2 = np.argsort(-values) # 남은 score 내림차순 idx
        out = out[:, inds2]
        out_inds = inds1[inds_keep[inds2]]
        return out, out_inds

    def run(self, img):
        assert img.ndim == 2, 'Image must be grayscale.'
        assert img.dtype == np.float32, 'Image must be float32.'
        H, W = img.shape[0], img.shape[1]
        inp = img.copy()
        inp = (inp.reshape(1, 1, H, W))

        onnx_model = onnxruntime.InferenceSession(self.onnx_path, providers=["CUDAExecutionProvider"])
        model_pred = onnx_model.run(None, {'model_input': inp})
        semi, coarse_desc = model_pred[0], model_pred[1]

        semi = semi.squeeze()
        dense = np.exp(semi)  # softmax
        dense = dense / (np.sum(dense, axis=0) + 0.00001)
        # Remove dustbin.
        nodust = dense[:-1, :, :]
        # Reshape to get full resolution heatmap.
        Hc = int(H / self.cell)
        Wc = int(W / self.cell)
        nodust = nodust.transpose(1, 2, 0)
        heatmap = np.reshape(nodust, [Hc, Wc, self.cell, self.cell])
        heatmap = np.transpose(heatmap, [0, 2, 1, 3])
        heatmap = np.reshape(heatmap, [Hc * self.cell, Wc * self.cell])
        xs, ys = np.where(heatmap >= self.conf_thresh)  # Confidence threshold.
        if len(xs) == 0:
            return np.zeros((3, 0)), None, None
        pts = np.zeros((3, len(xs)))  # Populate point data sized 3xN.
        pts[0, :] = ys
        pts[1, :] = xs
        pts[2, :] = heatmap[xs, ys]
        pts, _ = self.nms_fast(pts, H, W, dist_thresh=self.nms_dist)  # Apply NMS.
        inds = np.argsort(pts[2, :])
        pts = pts[:, inds[::-1]]  # Sort by confidence.
        # Remove points along border.
        bord = self.border_remove
        toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W - bord))
        toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H - bord))
        toremove = np.logical_or(toremoveW, toremoveH)
        pts = pts[:, ~toremove]
        # --- Process descriptor.
        D = coarse_desc.shape[1]
        if pts.shape[1] == 0:
            desc = np.zeros((D, 0))
        else:
            # Interpolate into descriptor map using 2D point locations.
            samp_pts = pts[:2, :].copy()
            samp_pts[0, :] = (samp_pts[0, :] / (float(W) / 2.)) - 1.
            samp_pts[1, :] = (samp_pts[1, :] / (float(H) / 2.)) - 1.
            samp_pts = np.transpose(samp_pts)
            samp_pts = samp_pts.reshape(1, 1, -1, 2)
            samp_pts = samp_pts.astype(np.float32)

            # grid_sample 설명
            # https://stackoverflow.com/questions/73300183/understanding-the-torch-nn-functional-grid-sample-op-by-concrete-example

            grid_sample = onnxruntime.InferenceSession('./grid_sample.onnx', providers=["CUDAExecutionProvider"])
            desc = grid_sample.run(None, {'X': coarse_desc, 'Grid':samp_pts})[0]
            desc = desc.reshape(D, -1)
            desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]
        return pts, desc, heatmap

    def reset(self):
        self.scale_h = 0
        self.scale_w = 0
        self.cell = 8
        self.border_remove = 4
        self.top = None
        self.bot = None
        self.fh = 0
        self.fw = 0



if __name__ == '__main__':

    # Parse command line arguments.
    parser = argparse.ArgumentParser(description='PyTorch SuperPoint Demo.')
    parser.add_argument('--input', type=str, default='data/',
                        help='Image directory or movie file or "camera" (for webcam).')
    parser.add_argument('--onnx_path', type=str, default='SuperPoint_500.onnx',
                        help='Path to onnx model file (default: SuperPoint.onnx).')
    parser.add_argument('--rsize', type=int, default=500,
                        help='Image resize default : 500')
    parser.add_argument('--nms_dist', type=int, default=4,
                        help='Non Maximum Suppression (NMS) distance (default: 4).')
    parser.add_argument('--conf_thresh', type=float, default=0.015,
                        help='Detector confidence threshold (default: 0.015).')
    parser.add_argument('--nn_thresh', type=float, default=0.7,
                        help='Descriptor matching threshold (default: 0.7).')
    parser.add_argument('--write_dir', type=str, default='myoutput/',
                        help='Directory where to write output frames (default: tracker_outputs/).')
    parser.add_argument('--min_length', type=int, default=2,
                        help='Minimum length of point tracks (default: 2).')
    parser.add_argument('--max_length', type=int, default=5,
                        help='Maximum length of point tracks (default: 5).')

    opt = parser.parse_args()
    print(opt)

    sp = Superpoint(opt.onnx_path, opt.rsize, opt.nn_thresh, opt.conf_thresh, opt.nms_dist)
    tracker = PointTracker(opt.max_length, nn_thresh=sp.nn_thresh)

    #data_ 폴더에 원본 긴 이미지를 넣으면, data 폴더에 top,bot으로 crop되어 저장됩니다.
    img_file = os.listdir('./data_')
    img_list = [(os.sep.join(['./data_', filename]))
                for filename in img_file]
    for idx, i in enumerate(img_list):
        sp.pt_list = []
        fullimg = cv2.imread(i, 0)
        filename = img_file[idx][:-4]
        top, bot = sp.pre_processing(fullimg)

        pts, desc, heatmap = sp.run(top)
        tracker.update(pts, desc)
        tracks = tracker.get_tracks(opt.min_length)
        tracks[:, 1] /= float(sp.nn_thresh)  # Normalize track scores to [0,1].
        tracker.draw_tracks(tracks) #얘는 필요하다..

        # out2 = (np.dstack((top, top, top)) * 255.).astype('uint8')
        # for pt in pts.T:
        #     pt1 = (int(round(pt[0])), int(round(pt[1])))
        #     cv2.circle(out2, pt1, 1, (0, 255, 0), -1, lineType=16)
        # cv2.imwrite(f"./myoutput/test01/_topkeypoint.bmp", out2)


        pts, desc, heatmap = sp.run(bot)
        tracker.update(pts, desc)
        tracks = tracker.get_tracks(opt.min_length)
        tracks[:, 1] /= float(sp.nn_thresh)  # Normalize track scores to [0,1].
        tracker.draw_tracks(tracks)

        src_pts = np.zeros((len(tracker.pt_list), 2))
        dst_pts = np.zeros((len(tracker.pt_list), 2))
        for i in range(len(tracker.pt_list)):
            p1 = (tracker.pt_list[i][0], tracker.pt_list[i][1])
            p2 = (tracker.pt_list[i][2], tracker.pt_list[i][3])
            src_pts[i] = np.array((p1[0], p1[1])).astype(np.float32)
            dst_pts[i] = np.array((p2[0], p2[1])).astype(np.float32)

        total_y = sp.fh * sp.scale_w
        dx_ = -(src_pts[:, 0] - dst_pts[:, 0])
        y_coord = total_y - sp.rsize + dst_pts[:, 1]
        total_dst = dst_pts.copy()
        total_dst[:, 1] = y_coord

        dx_ = -(src_pts[:, 0] - dst_pts[:, 0])
        dy_ = total_y - sp.rsize + (src_pts[:, 1] - dst_pts[:, 1])
        data = np.stack((dx_, dy_), -1)
        degree_ = np.arctan2(dx_, dy_) * 180 / np.pi

        degree = np.round(degree_, 2)
        collect = collections.Counter(degree)
        print(collect)
        print(f"{filename} prediction 완료")

        sp.reset()
        tracker.reset()
        src_pts = 0
        dst_pts = 0


